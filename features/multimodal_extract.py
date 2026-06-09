"""
features/multimodal_extract.py
==============================

ParselyFi Phase-3 "Card -> Rows" tab.

Turn images of company information (pitch-deck slides, cap tables, CRM /
contact cards, screenshots of company lists) into STRUCTURED company rows that
feed the flagship List Intelligence pipeline.

Public entry point (the ONLY thing the main app calls):

    def render_multimodal_extract_tab() -> None

It is invoked inside an already-open ``st.tab`` context, so this module does
NOT call ``st.set_page_config`` and has no ``__main__`` run block.

Flow
----
1. INPUT   — ``st.file_uploader(accept_multiple_files=True)`` (png/jpg/jpeg/webp),
             capped at ``MAX_IMAGES`` with an honest notice when exceeded.
             Thumbnails are shown; an optional free-text extraction hint nudges
             the model ("these are portfolio companies", "pull company +
             valuation + round").
2. EXTRACT — for EACH image we call Gemini MULTIMODAL via
             ``common.get_gemini_client().aio.models.generate_content`` with a
             ``gt.Part.from_bytes(data, mime_type)`` part + a JSON-schema prompt,
             ``temperature=0.0`` for deterministic structured output. Wrapped in
             ``common.run_async`` and an ``asyncio.wait_for`` timeout. Target
             schema per image:
             ``{companies:[{name, website, industry, stage, valuation, round,
             investors, people, notes}]}`` bounded to ``MAX_ROWS_PER_IMAGE``.
             HONEST: only fields visible in the image; "" / omit when not
             present; never invent a company. A failed / empty model response is
             recorded as "no rows extracted" (NOT fabricated).
3. RESULTS — all images' rows merge into ONE editable ``st.data_editor`` so the
             user can correct OCR slips before sending. A ``ui.kpi_row`` shows
             [images, companies found, with website]; per-image expanders show
             exactly what was pulled (or the honest failure reason).
4. HANDOFF — a primary "Send N companies to List Intelligence" button writes the
             deduped, cleaned company-name list as a newline string to
             ``st.session_state["li_raw_text_input"]`` (the exact widget key the
             List Intelligence names ``text_area`` reads), fires ``st.toast`` and
             captions the user to open that tab. A "Download CSV" of the full
             extracted table is also offered.

Honest status / safety (per project CLAUDE.md)
----------------------------------------------
- Missing GEMINI key (or SDK) -> ``ui.inject_css()`` + ``ui.hero`` +
  ``render_gemini_health`` / warning + ``return``. Never a success-shaped
  failure.
- BOUNDED memory: ``MAX_IMAGES`` files, ``MAX_IMAGE_BYTES`` per file (larger are
  skipped with an honest notice), ``MAX_ROWS_PER_IMAGE`` rows per image, a hard
  ``MAX_TOTAL_ROWS`` ceiling on the merged grid, and run history capped via
  ``common.push_history`` (FIFO, ``common.MAX_HISTORY``).
- TIMEOUTS: every Gemini call is wrapped in ``asyncio.wait_for`` using the shared
  ``common._LLM_TIMEOUT_S`` budget.
- ERROR_BOUNDARY: one bad image never kills the batch — each is extracted in its
  own try/except and recorded honestly as a failure.
- Optional libs (pandas, PIL) are imported under try/except with availability
  flags and honest fallbacks.
- All client / LLM / token logic is REUSED from ``common.*`` and the look from
  ``ui.*`` — nothing is duplicated here.
- Session-state keys are all uniquely prefixed ``mm_`` so they never collide
  with the main app or the sibling feature tabs.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

from features import common, ui

try:
    import pandas as pd  # type: ignore
    PANDAS_AVAILABLE = True
except Exception:  # pragma: no cover - pandas is a hard dep of the grid UI
    pd = None  # type: ignore
    PANDAS_AVAILABLE = False

try:
    from PIL import Image  # type: ignore
    PIL_AVAILABLE = True
except Exception:
    Image = None  # type: ignore
    PIL_AVAILABLE = False

logger = logging.getLogger("parselyfi.features.multimodal_extract")

# ===========================================================================
# Constants / bounded-memory caps (all mm_-prefixed session keys)
# ===========================================================================
_PFX = "mm_"
SS_RESULTS = _PFX + "results"            # list[dict] per-image extraction records
SS_EDITED = _PFX + "edited_rows"         # the merged editable grid (DataFrame)
SS_HINT = _PFX + "hint"                  # the optional extraction hint text_area
SS_RUNNING = _PFX + "running"            # guard against double-submit
SS_HISTORY = _PFX + "run_history"        # bounded via common.push_history
SS_UPLOADER = _PFX + "uploader"          # file_uploader widget key

# The EXACT widget key the List Intelligence names text_area reads. Writing this
# pre-fills that tab's input on the next rerun. (Confirmed against
# list_intelligence.SS_RAW_INPUT == "li_" + "raw_text_input".)
LI_RAW_TEXT_KEY = "li_raw_text_input"

# Hard caps so nothing the user / model supplies can blow up memory.
MAX_IMAGES: int = 8                       # cap images per batch
MAX_ROWS_PER_IMAGE: int = 40              # cap companies pulled from one image
MAX_TOTAL_ROWS: int = MAX_IMAGES * MAX_ROWS_PER_IMAGE  # ceiling on merged grid
MAX_IMAGE_BYTES: int = 8 * 1024 * 1024    # ~8MB per file; larger are skipped
MAX_HINT_CHARS: int = 600                 # cap the free-text hint fed to model
MAX_FIELD_CHARS: int = 400                # cap a single extracted string field
BATCH_WALL_CLOCK_S: float = 600.0         # overall budget for one extract batch

REQUIRED_KEYS = ["GEMINI_API_KEY"]
_AGENT = "multimodal_extract"

# Accepted upload types -> a Gemini-friendly MIME type.
_ACCEPTED_TYPES = ["png", "jpg", "jpeg", "webp"]
_EXT_TO_MIME = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
}

# Canonical company-row fields, in display order. "name" is the only required
# field; everything else is optional and stays "" when not visible in the image.
_ROW_FIELDS = [
    "name", "website", "industry", "stage", "valuation",
    "round", "investors", "people", "notes",
]

# JSON shape we ask the model to fill — documented inline so the prompt is the
# single source of truth for the contract.
_SCHEMA_HINT = (
    '{"companies": [{'
    '"name": "string (company/org name EXACTLY as shown)", '
    '"website": "string (domain or URL if visible, else \\"\\")", '
    '"industry": "string (sector/category if visible, else \\"\\")", '
    '"stage": "string (e.g. Seed/Series A/Growth if visible, else \\"\\")", '
    '"valuation": "string (e.g. $1.2B if visible, else \\"\\")", '
    '"round": "string (round size/amount if visible, else \\"\\")", '
    '"investors": "string (comma-joined investors if visible, else \\"\\")", '
    '"people": "string (comma-joined people/contacts if visible, else \\"\\")", '
    '"notes": "string (any other short visible detail, else \\"\\")"'
    '}]}'
)


# ===========================================================================
# Helpers — MIME, sanitation, dedupe
# ===========================================================================

def _mime_for(filename: str, declared: Optional[str]) -> str:
    """Best-effort MIME type for an uploaded image.

    Prefers the uploader-declared type when it is a real image/* type, else
    maps the file extension, else defaults to image/png.
    """
    if declared and declared.lower().startswith("image/"):
        return declared.lower()
    ext = (filename.rsplit(".", 1)[-1] if "." in filename else "").lower()
    return _EXT_TO_MIME.get(ext, "image/png")


def _clean_field(value: Any) -> str:
    """Coerce any model-supplied field to a bounded, stripped string.

    Lists (e.g. investors/people) are comma-joined. None / missing -> "".
    Honest: we never inject a placeholder value, only normalize what was given.
    """
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        parts = [str(v).strip() for v in value if str(v).strip()]
        s = ", ".join(parts)
    elif isinstance(value, bool):
        s = "yes" if value else ""
    elif isinstance(value, (int, float)):
        s = str(value)
    else:
        s = str(value).strip()
    s = s.strip()
    # Strip values the model sometimes emits for "absent" so they don't masquerade
    # as real data downstream.
    if s.lower() in {"n/a", "na", "none", "null", "unknown", "-", "—"}:
        return ""
    if len(s) > MAX_FIELD_CHARS:
        s = s[:MAX_FIELD_CHARS].rstrip() + "…"
    return s


def _normalize_companies(raw: Any) -> List[Dict[str, str]]:
    """Turn a model ``{"companies": [...]}`` payload into bounded clean rows.

    - Drops rows with no visible company name (HONEST: never invent a name).
    - Coerces every field to a bounded string.
    - Caps at ``MAX_ROWS_PER_IMAGE``.
    """
    rows: List[Dict[str, str]] = []
    if not isinstance(raw, dict):
        return rows
    companies = raw.get("companies")
    if not isinstance(companies, list):
        return rows
    for item in companies:
        if not isinstance(item, dict):
            continue
        name = _clean_field(item.get("name"))
        if not name:
            # No identifiable company name -> skip (do not fabricate one).
            continue
        row = {f: _clean_field(item.get(f)) for f in _ROW_FIELDS}
        row["name"] = name
        rows.append(row)
        if len(rows) >= MAX_ROWS_PER_IMAGE:
            logger.info("Image hit MAX_ROWS_PER_IMAGE=%d cap; truncating.", MAX_ROWS_PER_IMAGE)
            break
    return rows


def _dedupe_names(names: List[str]) -> List[str]:
    """Case-insensitive de-dup preserving first-seen order; drops blanks."""
    seen: set = set()
    out: List[str] = []
    for n in names:
        key = (n or "").strip().lower()
        # Drop blanks AND pandas placeholder strings ("nan"/"none"/"null") that
        # leak in from emptied data_editor cells — never hand those off as companies.
        if not key or key in ("nan", "none", "null") or key in seen:
            continue
        seen.add(key)
        out.append(n.strip())
    return out


def _build_prompt(hint: str) -> str:
    """Compose the JSON-schema extraction prompt (with optional user hint)."""
    hint = (hint or "").strip()
    if len(hint) > MAX_HINT_CHARS:
        hint = hint[:MAX_HINT_CHARS]
    hint_block = (
        f"\n\nUSER HINT (use as guidance, do not let it override what is "
        f"actually visible): {hint}\n"
        if hint else "\n"
    )
    return (
        "You are a precise company-data extractor. Look at the provided IMAGE "
        "(it may be a pitch-deck slide, cap table, CRM/contact card, or a "
        "screenshot of a list of companies) and extract EVERY distinct company / "
        "organization that is visibly named in it.\n\n"
        "STRICT RULES (non-negotiable):\n"
        "- Extract ONLY what is actually visible in the image. Do NOT use outside "
        "knowledge to fill in fields.\n"
        "- NEVER invent a company that is not shown. If the image has no "
        "companies, return an empty companies list.\n"
        "- For any field not visible for a company, use an empty string \"\". Do "
        "NOT guess valuations, rounds, websites, or investors.\n"
        "- Transcribe names/numbers exactly as shown (do not normalize or "
        "abbreviate).\n"
        f"- Return at most {MAX_ROWS_PER_IMAGE} companies.\n"
        f"{hint_block}\n"
        "Respond ONLY as minified JSON with EXACTLY this shape (no prose, no "
        "markdown fences):\n"
        f"{_SCHEMA_HINT}"
    )


# ===========================================================================
# EXTRACT — one Gemini multimodal call per image (honest, bounded, timed)
# ===========================================================================

async def _extract_one_image(
    data: bytes, mime_type: str, hint: str,
) -> Tuple[List[Dict[str, str]], Optional[str]]:
    """Run a single Gemini multimodal extraction on one image's bytes.

    Returns ``(rows, error_reason)``:
    - ``(rows, None)`` on success (rows may legitimately be [] if the image had
      no companies — that is honest, not an error).
    - ``([], reason)`` on any failure (no client/SDK, timeout, exception, empty
      or unparseable response) so the caller can record it as "no rows
      extracted" WITHOUT fabricating data.
    """
    client = common.get_gemini_client()
    gt = common.genai_types
    if client is None or gt is None:
        return [], "Gemini multimodal unavailable (no client or SDK)."

    prompt = _build_prompt(hint)

    try:
        contents = [
            gt.Content(
                role="user",
                parts=[
                    gt.Part.from_bytes(data=data, mime_type=mime_type),
                    gt.Part.from_text(text=prompt),
                ],
            )
        ]
        # TIMEOUT: bound the multimodal call using the shared LLM budget knob.
        response = await asyncio.wait_for(
            client.aio.models.generate_content(
                model=common.GEMINI_MODEL,
                contents=contents,
                config=gt.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.0,
                ),
            ),
            timeout=common._LLM_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        return [], f"extraction timed out after {common._LLM_TIMEOUT_S}s"
    except Exception as e:  # noqa: BLE001 - degrade honestly, never crash the batch
        # Surface auth failures through the shared health banner like the text helpers.
        try:
            common._record_llm_error(_AGENT, e)
        except Exception:
            pass
        logger.info("multimodal image extraction failed: %s", e)
        return [], f"extraction failed: {e}"

    # Token accounting via the shared bounded ledger (honest cost tracking).
    try:
        common._record_usage_from_response(response, _AGENT, common.GEMINI_MODEL)
    except Exception:
        pass

    text = common._extract_text_from_response(response)
    if not text:
        return [], "model returned an empty response"

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return [], "model response was not valid JSON"
        try:
            parsed = json.loads(match.group(0))
        except Exception:
            return [], "model response was not valid JSON"

    rows = _normalize_companies(parsed)
    return rows, None


async def _extract_batch(
    images: List[Dict[str, Any]], hint: str,
) -> List[Dict[str, Any]]:
    """Extract every image sequentially under one wall-clock budget.

    Sequential (not fanned-out) on purpose: image calls are heavy and the
    per-call timeout already bounds latency; this keeps token spend predictable
    and avoids hammering the API. ERROR_BOUNDARY: one bad image is recorded as a
    failed record and never aborts the rest.

    Returns one record per image:
        {"filename", "rows": [...], "error": str|None}
    """
    records: List[Dict[str, Any]] = []

    async def _runner() -> None:
        for img in images:
            try:
                rows, err = await _extract_one_image(
                    img["data"], img["mime"], hint
                )
            except Exception as e:  # noqa: BLE001 - belt & suspenders boundary
                rows, err = [], f"unexpected error: {e}"
            records.append(
                {"filename": img["filename"], "rows": rows, "error": err}
            )

    try:
        await asyncio.wait_for(_runner(), timeout=BATCH_WALL_CLOCK_S)
    except asyncio.TimeoutError:
        logger.error("extract batch hit wall-clock budget (%ss).", BATCH_WALL_CLOCK_S)
        # Mark any images not yet processed as honestly unprocessed.
        done = {r["filename"] for r in records}
        for img in images:
            if img["filename"] not in done:
                records.append({
                    "filename": img["filename"],
                    "rows": [],
                    "error": "not processed (batch timeout)",
                })
    return records


# ===========================================================================
# Results -> DataFrame helpers
# ===========================================================================

def _records_to_rows(records: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Flatten per-image records into merged grid rows (bounded), with source."""
    merged: List[Dict[str, str]] = []
    for rec in records:
        src = rec.get("filename", "")
        for row in rec.get("rows", []):
            out = {"source_image": src}
            out.update({f: row.get(f, "") for f in _ROW_FIELDS})
            merged.append(out)
            if len(merged) >= MAX_TOTAL_ROWS:
                logger.info("Merged grid hit MAX_TOTAL_ROWS=%d cap.", MAX_TOTAL_ROWS)
                return merged
    return merged


def _rows_to_df(rows: List[Dict[str, str]]):
    """Build the editable grid DataFrame (column order: source, name, fields)."""
    cols = ["source_image"] + _ROW_FIELDS
    if not PANDAS_AVAILABLE:
        return rows
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows, columns=cols)


# ===========================================================================
# Public entry point
# ===========================================================================

def render_multimodal_extract_tab() -> None:
    """Render the "Card -> Rows" multimodal extraction tab.

    Honest status: if the Gemini key/SDK is missing we inject the CSS, render
    the hero + Gemini health banner, and return without fabricating data.
    """
    ui.inject_css()

    # ---- HONEST STATUS gate ----
    ok, missing = common.feature_available(REQUIRED_KEYS)
    if not ok:
        ui.hero(
            "Card → Rows",
            "Turn images of company info into structured rows for List Intelligence.",
        )
        common.render_gemini_health()
        st.caption(f"Missing secrets: {', '.join(missing)}")
        return

    # Key present but possibly rejected (leaked/invalid) — surface honestly.
    common.render_gemini_health()

    ui.hero(
        "Card → Rows",
        "Drop pitch-deck slides, cap tables, CRM cards or list screenshots — "
        "Gemini reads them into structured company rows you can correct and send "
        "to List Intelligence.",
        chips=[
            "<b>Multimodal</b> image → company rows",
            f"Up to <b>{MAX_IMAGES}</b> images / batch",
            "Only what's visible — never invented",
        ],
    )

    if not PANDAS_AVAILABLE:
        st.error("pandas is required for the extraction grid and CSV export.")
        return

    # ---- INPUT ----
    ui.section("1 · Upload images", "PNG / JPG / WEBP — pitch slides, cap tables, contact cards, list screenshots")
    uploaded = st.file_uploader(
        "Company info images",
        type=_ACCEPTED_TYPES,
        accept_multiple_files=True,
        key=SS_UPLOADER,
    )

    hint = st.text_area(
        "Optional extraction hint",
        key=SS_HINT,
        height=70,
        max_chars=MAX_HINT_CHARS,
        placeholder='e.g. "these are portfolio companies" or "pull company + valuation + round"',
        help="Nudges what to focus on. The model still extracts only what is visible.",
    )

    # ---- Validate / bound the uploads (honest skips + notices) ----
    images: List[Dict[str, Any]] = []
    skipped: List[str] = []
    files = list(uploaded or [])

    if len(files) > MAX_IMAGES:
        st.warning(
            f"⚠️ {len(files)} images uploaded — only the first {MAX_IMAGES} will be "
            f"processed (bounded for cost/latency). Remove some to choose which."
        )
        files = files[:MAX_IMAGES]

    for f in files:
        try:
            data = f.getvalue()  # bytes; UploadedFile supports getvalue()
        except Exception:
            try:
                data = f.read()
            except Exception as e:
                skipped.append(f"{getattr(f, 'name', 'file')} (unreadable: {e})")
                continue
        size = len(data or b"")
        if size == 0:
            skipped.append(f"{getattr(f, 'name', 'file')} (empty)")
            continue
        if size > MAX_IMAGE_BYTES:
            skipped.append(
                f"{getattr(f, 'name', 'file')} "
                f"({size / 1024 / 1024:.1f}MB > {MAX_IMAGE_BYTES // (1024 * 1024)}MB cap)"
            )
            continue
        # Disambiguate duplicate filenames so per-image provenance stays unique
        # (st.file_uploader allows two files sharing a basename).
        fname = getattr(f, "name", "image")
        _existing = {im["filename"] for im in images}
        if fname in _existing:
            _base, _n = fname, 2
            while fname in _existing:
                fname = f"{_base} ({_n})"
                _n += 1
        images.append({
            "filename": fname,
            "data": data,
            "mime": _mime_for(getattr(f, "name", ""), getattr(f, "type", None)),
            "size": size,
        })

    if skipped:
        st.warning(
            "⚠️ Skipped " + str(len(skipped)) + " file(s) (not processed): "
            + "; ".join(skipped)
        )

    # ---- Thumbnails ----
    if images:
        ui.section("2 · Preview", f"{len(images)} image(s) ready")
        thumb_cols = st.columns(min(4, len(images)))
        for i, img in enumerate(images):
            col = thumb_cols[i % len(thumb_cols)]
            with col:
                try:
                    st.image(img["data"], caption=img["filename"], use_container_width=True)
                except Exception:
                    st.caption(f"🖼️ {img['filename']} (preview unavailable)")

    # ---- Extract action ----
    run = st.button(
        f"🔎 Extract companies from {len(images)} image(s)",
        type="primary",
        disabled=(len(images) == 0 or st.session_state.get(SS_RUNNING, False)),
        use_container_width=True,
    )

    if run and images:
        st.session_state[SS_RUNNING] = True
        try:
            with st.status("Reading images with Gemini…", expanded=True) as status:
                status.write(f"Extracting from {len(images)} image(s) (temperature 0.0)…")
                records = common.run_async(_extract_batch(images, hint))
                total_rows = sum(len(r.get("rows", [])) for r in records)
                failed = sum(1 for r in records if r.get("error"))
                status.update(
                    label=(
                        f"Done — {total_rows} compan{'y' if total_rows == 1 else 'ies'} "
                        f"from {len(records)} image(s)"
                        + (f", {failed} with no rows" if failed else "")
                    ),
                    state="complete",
                    expanded=False,
                )
            st.session_state[SS_RESULTS] = records
            # Reset the editable grid to the freshly-extracted rows.
            st.session_state.pop(SS_EDITED, None)
            common.push_history(SS_HISTORY, {
                "ts": datetime.now().isoformat(timespec="seconds"),
                "images": len(records),
                "companies": total_rows,
            })
            st.toast(f"Extracted {total_rows} companies from {len(records)} image(s).")
        except Exception as e:  # noqa: BLE001 - never let the tab crash
            logger.error("extract batch failed: %s", e)
            st.error(f"Extraction failed: {e}")
        finally:
            st.session_state[SS_RUNNING] = False

    # ---- RESULTS ----
    records: List[Dict[str, Any]] = st.session_state.get(SS_RESULTS) or []
    if not records:
        st.info("Upload one or more images and click **Extract** to pull company rows.")
        _render_token_footer()
        return

    merged_rows = _records_to_rows(records)
    n_images = len(records)
    has_extracted = len(merged_rows) > 0

    # KPI counts reflect the CURRENT (post-edit) grid when present, so the headline
    # numbers match what will actually be exported / handed off.
    _edited_df = st.session_state.get(SS_EDITED)
    grid_rows = merged_rows
    if _edited_df is not None:
        try:
            grid_rows = _edited_df.to_dict("records")
        except Exception:
            grid_rows = merged_rows
    n_companies = sum(
        1 for r in grid_rows
        if str(r.get("name") or "").strip().lower() not in ("", "nan", "none", "null")
    )
    n_with_site = sum(1 for r in grid_rows if str(r.get("website") or "").strip())

    ui.section("3 · Extracted rows", "Correct OCR slips here before sending — only what was visible was filled")
    ui.kpi_row([
        ("Images", n_images),
        ("Companies found", n_companies),
        ("With website", n_with_site),
    ])

    # Honest truncation notice when the merged-grid cap clipped later images' rows.
    _raw_total = sum(len(rec.get("rows", [])) for rec in records)
    if _raw_total > len(merged_rows):
        st.warning(
            f"Showing {len(merged_rows)} of {_raw_total} extracted rows "
            f"(capped at {MAX_TOTAL_ROWS}). Rows from later images are not shown."
        )

    if not has_extracted:
        st.warning(
            "No companies were extracted from these images. Nothing was "
            "fabricated — the model did not find identifiable company names. "
            "Try a clearer image or a more specific hint."
        )

    # Editable grid (only when extraction produced rows).
    edited_names: List[str] = []
    if has_extracted:
        base_df = st.session_state.get(SS_EDITED)
        if base_df is None:
            base_df = _rows_to_df(merged_rows)

        col_cfg = {
            "source_image": st.column_config.TextColumn("Source", disabled=True, width="small"),
            "name": st.column_config.TextColumn("Company", required=True),
            "website": st.column_config.TextColumn("Website"),
            "industry": st.column_config.TextColumn("Industry"),
            "stage": st.column_config.TextColumn("Stage"),
            "valuation": st.column_config.TextColumn("Valuation"),
            "round": st.column_config.TextColumn("Round"),
            "investors": st.column_config.TextColumn("Investors"),
            "people": st.column_config.TextColumn("People"),
            "notes": st.column_config.TextColumn("Notes"),
        }

        edited = st.data_editor(
            base_df,
            key=_PFX + "editor",
            use_container_width=True,
            hide_index=True,
            num_rows="dynamic",
            column_config=col_cfg,
        )
        st.session_state[SS_EDITED] = edited

        # Pull cleaned, deduped names from the (possibly edited) grid. Filter NaN /
        # blank cells so emptied or newly-added rows never become a "nan" company.
        try:
            raw_names = [
                str(x).strip()
                for x in (edited["name"].tolist() if "name" in edited else [])
                if pd.notna(x) and str(x).strip()
            ]
        except Exception:
            raw_names = [r.get("name", "") for r in merged_rows]
        edited_names = _dedupe_names(raw_names)

    # ---- HANDOFF ----
    ui.section("4 · Send to List Intelligence", "Hand the company names to the MATCH → ENRICH → CLASSIFY → SCORE pipeline")
    send_col, dl_col = st.columns([2, 1])

    with send_col:
        n_send = len(edited_names)
        send = st.button(
            f"📋 Send {n_send} compan{'y' if n_send == 1 else 'ies'} to List Intelligence",
            type="primary",
            disabled=(n_send == 0),
            use_container_width=True,
        )
        if send and edited_names:
            # Hand off via a NON-widget relay key, then rerun. Writing the widget
            # key directly crashes because List Intelligence already instantiated
            # its text_area earlier in this same script run.
            st.session_state[LI_RAW_TEXT_KEY + "_pending"] = "\n".join(edited_names)
            st.toast(
                f"Sent {n_send} compan{'y' if n_send == 1 else 'ies'} to List "
                "Intelligence — open that tab (names are pre-filled)."
            )
            st.rerun()

    with dl_col:
        # Download CSV of the FULL extracted table (current grid state).
        df_for_csv = st.session_state.get(SS_EDITED)
        if df_for_csv is None:
            df_for_csv = _rows_to_df(merged_rows)
        try:
            csv_bytes = df_for_csv.to_csv(index=False).encode("utf-8")
        except Exception:
            csv_bytes = b""
        st.download_button(
            "⬇️ Download CSV",
            data=csv_bytes,
            file_name="extracted_companies.csv",
            mime="text/csv",
            disabled=(len(csv_bytes) == 0),
            use_container_width=True,
        )

    # ---- Per-image detail (honest: shows failures, not just successes) ----
    ui.section("Per-image detail", "Exactly what was pulled from each image")
    for rec in records:
        rows = rec.get("rows", [])
        err = rec.get("error")
        label = f"🖼️ {rec.get('filename', 'image')} — "
        if err:
            label += "no rows extracted"
        else:
            label += f"{len(rows)} compan{'y' if len(rows) == 1 else 'ies'}"
        with st.expander(label):
            if err:
                st.warning(f"No rows extracted: {err}")
            elif not rows:
                st.caption("No companies were visible in this image (nothing fabricated).")
            else:
                st.dataframe(
                    _rows_to_df(rows), use_container_width=True, hide_index=True
                )

    _render_token_footer()


def _render_token_footer() -> None:
    """Render the shared bounded token-usage ledger at the bottom of the tab."""
    st.divider()
    common.render_token_usage()


__all__ = ["render_multimodal_extract_tab"]
