"""
features/transcription.py
=========================

ParselyFi "Transcription & Summaries" feature tab.

Ported from reference_codes/test6_audio_text_anywidget_v3.py (the audio ->
transcript workflow + the AudioTranscriptWidget synced player) and finished by
ADDING the "& Summaries" half the placeholder promised: once a transcript
exists, a Gemini-backed "Summarize" action produces a concise summary, key
points, and action items that are displayed and downloadable.

Hard rules honored (see project CLAUDE.md):
- This module exposes exactly ``render_transcription_tab() -> None`` and is
  rendered inside a tab by the main app. It does NOT call st.set_page_config
  and has NO ``if __name__ == '__main__'`` block.
- HONEST STATUS: if the ElevenLabs key is missing we warn + offer the
  transcript-JSON upload path instead of inventing a transcript. If the Gemini
  key is missing we hide the summary action rather than faking a summary.
  Transcription / summarization failures surface as errors, never as success.
- BOUNDED MEMORY: the transcript-summary history list is capped (FIFO).
- TIMEOUTS + bounded reads: the ElevenLabs HTTP call uses an explicit httpx
  timeout with retries; the summarizer runs through common.gemini_generate_text
  which already enforces a timeout, token recording, and bounded output. The
  transcript text fed to the LLM is length-capped before the call.
- Session-state keys are ALL prefixed ``tr_`` so they never collide with the
  main app ("s3_file_manager" / "selected_files") or sibling tabs.

Secrets are read ONLY through features.common helpers (get_secret /
feature_available) — never directly from st.secrets here.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import tempfile
import time
import uuid
from io import BytesIO
from typing import Any, Dict, List, Optional

import streamlit as st

# --- optional heavy deps (import-safe) -------------------------------------
try:
    import httpx  # type: ignore
    HTTPX_AVAILABLE = True
except Exception:  # pragma: no cover - httpx is a hard dep of the reference
    httpx = None  # type: ignore
    HTTPX_AVAILABLE = False

try:
    from streamlit_anywidget import anywidget  # type: ignore
    from anywidget import AnyWidget  # type: ignore
    import traitlets  # type: ignore
    ANYWIDGET_AVAILABLE = True
except Exception:
    anywidget = None  # type: ignore
    AnyWidget = object  # type: ignore - fallback base so class def never crashes import
    traitlets = None  # type: ignore
    ANYWIDGET_AVAILABLE = False

# --- shared feature API (the ONLY secret/LLM entrypoint) -------------------
from features import common
from features.common import (
    feature_available,
    gemini_generate_text,
    get_secret,
    get_history,
    push_history,
    run_async,
)

logger = logging.getLogger("parselyfi.features.transcription")

# ---------------------------------------------------------------------------
# Constants / bounded-memory caps / safety budgets
# ---------------------------------------------------------------------------
_KEY = "tr_"  # session-state prefix for EVERY key this module sets

ELEVENLABS_KEY_NAME = "ELEVEN_LABS_API_KEY"
ELEVENLABS_STT_URL = "https://api.elevenlabs.io/v1/speech-to-text"

# Network safety budgets for the ElevenLabs speech-to-text call.
_STT_TIMEOUT_S: float = 300.0          # 5-min ceiling (long audio); explicit, bounded
_STT_MAX_RETRIES: int = 3
# Bounded read on the uploaded audio: refuse absurdly large files before we
# base64 them into the page / ship them to the API.
_MAX_AUDIO_BYTES: int = 50 * 1024 * 1024   # 50 MB
# Bounded read on the uploaded transcript JSON: refuse oversized payloads
# before json.loads (Streamlit's default upload cap is 200 MB; that's far too
# large to parse into the page).
_MAX_TRANSCRIPT_JSON_BYTES: int = 25 * 1024 * 1024   # 25 MB

# Bounded read on transcript text fed to the LLM (chars, not tokens — a coarse
# but safe cap so a huge transcript can't blow the prompt budget).
_MAX_SUMMARY_INPUT_CHARS: int = 60_000

# Bounded history of generated summaries (FIFO via common.push_history's cap).
_SUMMARY_HISTORY_KEY = _KEY + "summary_history"


# ===========================================================================
# Synced audio<->transcript widget (ported verbatim from the reference)
# ===========================================================================
# NOTE: kept as faithful as possible. Only the Python class scaffolding adapts
# to the import-safe fallback (so the module imports even if anywidget is
# absent); the _esm / _css payloads are unchanged from the reference.

if ANYWIDGET_AVAILABLE:

    class AudioTranscriptWidget(AnyWidget):  # type: ignore[misc]
        """
        A custom widget that synchronizes audio playback with transcript text.
        Uses anywidget to enable JavaScript-Python communication.
        """
        # Bi-directional data with synchronization
        audio_data_base64 = traitlets.Unicode(allow_none=True).tag(sync=True)
        transcript = traitlets.Dict().tag(sync=True)
        current_time = traitlets.Float(0.0).tag(sync=True)
        is_playing = traitlets.Bool(False).tag(sync=True)

        # JavaScript for the widget frontend
        _esm = """
        function render({ model, el }) {
            // Create container elements
            const container = document.createElement("div");
            container.className = "audio-transcript-container";

            // Audio player
            const audioContainer = document.createElement("div");
            audioContainer.className = "audio-player-container";

            const audioEl = document.createElement("audio");
            audioEl.controls = true;
            audioEl.id = "audio-player";

            // Play/pause button for custom control
            const controlsContainer = document.createElement("div");
            controlsContainer.className = "controls-container";

            const playBtn = document.createElement("button");
            playBtn.textContent = "Play";
            playBtn.className = "control-button play-button";

            const pauseBtn = document.createElement("button");
            pauseBtn.textContent = "Pause";
            pauseBtn.className = "control-button pause-button";
            pauseBtn.style.display = "none";

            const timeDisplay = document.createElement("div");
            timeDisplay.className = "time-display";
            timeDisplay.textContent = "0:00 / 0:00";

            // Progress bar
            const progressContainer = document.createElement("div");
            progressContainer.className = "progress-container";

            const progressBar = document.createElement("div");
            progressBar.className = "progress-bar";

            const progressIndicator = document.createElement("div");
            progressIndicator.className = "progress-indicator";
            progressBar.appendChild(progressIndicator);

            // Transcript container
            const transcriptContainer = document.createElement("div");
            transcriptContainer.className = "transcript-container";

            // Add elements to the DOM
            controlsContainer.appendChild(playBtn);
            controlsContainer.appendChild(pauseBtn);
            controlsContainer.appendChild(timeDisplay);

            progressContainer.appendChild(progressBar);

            audioContainer.appendChild(audioEl);
            audioContainer.appendChild(controlsContainer);
            audioContainer.appendChild(progressContainer);

            container.appendChild(audioContainer);
            container.appendChild(transcriptContainer);

            el.appendChild(container);

            // Function to format time in MM:SS format
            function formatTime(seconds) {
                const minutes = Math.floor(seconds / 60);
                const secs = Math.floor(seconds % 60);
                return `${minutes}:${secs < 10 ? '0' : ''}${secs}`;
            }

            // Function to update UI when audio is loaded
            function handleAudioLoaded() {
                const duration = audioEl.duration;
                timeDisplay.textContent = `0:00 / ${formatTime(duration)}`;
            }

            // Function to update transcript highlighting based on current time
            function updateTranscriptHighlighting(currentTime) {
                const transcript = model.get("transcript") || {};
                const segments = transcript.segments || [];

                // Clear existing transcript if needed
                if (transcriptContainer.innerHTML === "" && segments.length > 0) {
                    segments.forEach((segment, index) => {
                        const segmentEl = document.createElement("span");
                        segmentEl.setAttribute("data-start", segment.start);
                        segmentEl.setAttribute("data-end", segment.end);
                        segmentEl.setAttribute("data-index", index);

                        // Add speaker information if available
                        if (segment.speaker_id) {
                            segmentEl.setAttribute("data-speaker", segment.speaker_id);
                            segmentEl.classList.add("speaker-" + segment.speaker_id.replace("speaker_", ""));
                        }

                        segmentEl.textContent = segment.text + " ";
                        transcriptContainer.appendChild(segmentEl);
                    });
                }

                // Update highlighting
                const transcriptElements = transcriptContainer.querySelectorAll("span");
                transcriptElements.forEach(spanEl => {
                    const start = parseFloat(spanEl.getAttribute("data-start"));
                    const end = parseFloat(spanEl.getAttribute("data-end"));

                    if (currentTime >= start && currentTime <= end) {
                        spanEl.classList.add("highlighted");

                        // Auto-scroll to the highlighted text
                        spanEl.scrollIntoView({ behavior: "smooth", block: "center" });
                    } else {
                        spanEl.classList.remove("highlighted");
                    }
                });

                // Update time display
                if (audioEl.duration) {
                    timeDisplay.textContent = `${formatTime(currentTime)} / ${formatTime(audioEl.duration)}`;
                }

                // Update progress bar
                if (audioEl.duration) {
                    const progress = (currentTime / audioEl.duration) * 100;
                    progressIndicator.style.width = `${progress}%`;
                }
            }

            // Audio event listeners
            audioEl.addEventListener("loadedmetadata", handleAudioLoaded);

            audioEl.addEventListener("timeupdate", () => {
                const currentTime = audioEl.currentTime;
                model.set("current_time", currentTime);
                model.save_changes();

                updateTranscriptHighlighting(currentTime);
            });

            audioEl.addEventListener("play", () => {
                model.set("is_playing", true);
                model.save_changes();
                playBtn.style.display = "none";
                pauseBtn.style.display = "inline-block";
            });

            audioEl.addEventListener("pause", () => {
                model.set("is_playing", false);
                model.save_changes();
                playBtn.style.display = "inline-block";
                pauseBtn.style.display = "none";
            });

            audioEl.addEventListener("ended", () => {
                model.set("is_playing", false);
                model.save_changes();
                playBtn.style.display = "inline-block";
                pauseBtn.style.display = "none";
            });

            // Control button event listeners
            playBtn.addEventListener("click", () => {
                audioEl.play();
            });

            pauseBtn.addEventListener("click", () => {
                audioEl.pause();
            });

            // Progress bar click for seeking
            progressBar.addEventListener("click", (e) => {
                const rect = progressBar.getBoundingClientRect();
                const position = (e.clientX - rect.left) / rect.width;
                if (audioEl.duration) {
                    audioEl.currentTime = position * audioEl.duration;
                }
            });

            // Sync from Python to JavaScript
            model.on("change:audio_data_base64", () => {
                const audioDataBase64 = model.get("audio_data_base64");
                if (audioDataBase64) {
                    // Create a blob URL for the audio data from base64
                    const byteCharacters = atob(audioDataBase64);
                    const byteNumbers = new Array(byteCharacters.length);
                    for (let i = 0; i < byteCharacters.length; i++) {
                        byteNumbers[i] = byteCharacters.charCodeAt(i);
                    }
                    const byteArray = new Uint8Array(byteNumbers);
                    const blob = new Blob([byteArray], { type: "audio/mp3" });
                    const url = URL.createObjectURL(blob);

                    // Set the audio source
                    audioEl.src = url;
                    audioEl.load();
                }
            });

            model.on("change:transcript", () => {
                const transcript = model.get("transcript");

                // Clear existing transcript
                transcriptContainer.innerHTML = "";

                // Rebuild transcript
                if (transcript && transcript.segments) {
                    transcript.segments.forEach((segment, index) => {
                        const segmentEl = document.createElement("span");
                        segmentEl.setAttribute("data-start", segment.start);
                        segmentEl.setAttribute("data-end", segment.end);
                        segmentEl.setAttribute("data-index", index);

                        // Add speaker information if available
                        if (segment.speaker_id) {
                            segmentEl.setAttribute("data-speaker", segment.speaker_id);
                            segmentEl.classList.add("speaker-" + segment.speaker_id.replace("speaker_", ""));
                        }

                        segmentEl.textContent = segment.text + " ";
                        transcriptContainer.appendChild(segmentEl);
                    });
                }

                // Update highlighting based on current time
                updateTranscriptHighlighting(model.get("current_time") || 0);
            });

            model.on("change:current_time", () => {
                const currentTime = model.get("current_time");
                if (Math.abs(audioEl.currentTime - currentTime) > 0.5) {
                    audioEl.currentTime = currentTime;
                }
                updateTranscriptHighlighting(currentTime);
            });

            model.on("change:is_playing", () => {
                const isPlaying = model.get("is_playing");
                if (isPlaying && audioEl.paused) {
                    audioEl.play();
                } else if (!isPlaying && !audioEl.paused) {
                    audioEl.pause();
                }
            });

            // Initial sync of data
            if (model.get("audio_data_base64")) {
                const audioDataBase64 = model.get("audio_data_base64");
                const byteCharacters = atob(audioDataBase64);
                const byteNumbers = new Array(byteCharacters.length);
                for (let i = 0; i < byteCharacters.length; i++) {
                    byteNumbers[i] = byteCharacters.charCodeAt(i);
                }
                const byteArray = new Uint8Array(byteNumbers);
                const blob = new Blob([byteArray], { type: "audio/mp3" });
                const url = URL.createObjectURL(blob);
                audioEl.src = url;
                audioEl.load();
            }

            if (model.get("transcript")) {
                updateTranscriptHighlighting(model.get("current_time") || 0);
            }
        }

        export default { render };
        """

        # CSS for the widget with speaker diarization support
        _css = """
        .audio-transcript-container {
            display: flex;
            flex-direction: column;
            width: 100%;
            font-family: sans-serif;
            gap: 20px;
        }

        .audio-player-container {
            width: 100%;
            padding: 15px;
            background-color: #f0f0f0;
            border-radius: 8px;
        }

        .controls-container {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-top: 10px;
        }

        .control-button {
            padding: 8px 15px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
        }

        .play-button {
            background-color: #4CAF50;
        }

        .pause-button {
            background-color: #f44336;
        }

        .time-display {
            margin-left: auto;
            font-size: 14px;
            color: #555;
        }

        .progress-container {
            margin-top: 10px;
            width: 100%;
        }

        .progress-bar {
            width: 100%;
            height: 8px;
            background-color: #ddd;
            border-radius: 4px;
            cursor: pointer;
            overflow: hidden;
        }

        .progress-indicator {
            height: 100%;
            width: 0%;
            background-color: #4CAF50;
            border-radius: 4px;
            transition: width 0.1s linear;
        }

        .transcript-container {
            padding: 15px;
            background-color: white;
            border: 1px solid #ddd;
            border-radius: 8px;
            max-height: 300px;
            overflow-y: auto;
            line-height: 1.6;
        }

        .transcript-container span {
            display: inline;
            transition: background-color 0.3s ease;
        }

        .transcript-container .highlighted {
            background-color: #ffff99;
            font-weight: bold;
        }

        /* Speaker colors for diarization */
        .speaker-1 {
            color: #3366cc;
        }

        .speaker-2 {
            color: #dc3912;
        }

        .speaker-3 {
            color: #ff9900;
        }

        .speaker-4 {
            color: #109618;
        }

        .speaker-5 {
            color: #990099;
        }

        .speaker-6 {
            color: #0099c6;
        }

        audio {
            width: 100%;
            display: none; /* Hide default audio controls */
        }
        """

else:  # anywidget not installed -> keep a sentinel so the module still imports
    AudioTranscriptWidget = None  # type: ignore


# ===========================================================================
# Secrets / availability helpers (read ONLY via common)
# ===========================================================================

def _elevenlabs_key() -> Optional[str]:
    """ElevenLabs API key (top-level secret) via common.get_secret, or None."""
    return get_secret(ELEVENLABS_KEY_NAME)


def _elevenlabs_available() -> bool:
    return bool(_elevenlabs_key()) and HTTPX_AVAILABLE


def _gemini_available() -> bool:
    ok, _missing = feature_available(["GEMINI_API_KEY"])
    return ok and common.GENAI_AVAILABLE


# ===========================================================================
# ElevenLabs speech-to-text (ported from the reference, key read via common)
# ===========================================================================

def transcribe_audio(
    audio_bytes: bytes,
    model_id: str = "scribe_v1",
    language_code: Optional[str] = None,
    timestamps_granularity: str = "word",
    diarize: bool = False,
    tag_audio_events: bool = True,
    num_speakers: Optional[int] = None,
) -> Dict[str, Any]:
    """Transcribe audio via the ElevenLabs speech-to-text API.

    Honest status: returns ``{"error": ...}`` (never a fake transcript) when
    the key/dep is missing or the request fails after retries. Uses an
    explicit bounded httpx timeout. Cleans up its temp file in a finally.
    """
    api_key = _elevenlabs_key()
    if not api_key:
        return {"error": "ELEVEN_LABS_API_KEY not found in secrets"}
    if not HTTPX_AVAILABLE or httpx is None:
        return {"error": "httpx not installed; transcription unavailable"}

    temp_path: Optional[str] = None
    try:
        # Write to a uniquely-named temp file (mirrors the reference's safe
        # file handling to avoid locking issues on Windows).
        unique_id = str(uuid.uuid4())
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f"elevenlabs_audio_{unique_id}.mp3")

        with open(temp_path, "wb") as f:
            f.write(audio_bytes)
            f.flush()
            os.fsync(f.fileno())

        headers = {"xi-api-key": api_key}

        data = {
            "model_id": model_id,
            "timestamps_granularity": timestamps_granularity,
            "diarize": str(diarize).lower(),
            "tag_audio_events": str(tag_audio_events).lower(),
        }
        if language_code:
            data["language_code"] = language_code
        if num_speakers:
            data["num_speakers"] = str(num_speakers)

        # Read the file content into memory once (BytesIO upload avoids file
        # locking during the request). A fresh BytesIO is built per attempt
        # since the stream is consumed by each POST.
        with open(temp_path, "rb") as f:
            file_content = f.read()

        last_error: Dict[str, Any] = {"error": "transcription failed"}
        for attempt in range(_STT_MAX_RETRIES):
            try:
                with httpx.Client(timeout=_STT_TIMEOUT_S) as client:
                    response = client.post(
                        ELEVENLABS_STT_URL,
                        headers=headers,
                        data=data,
                        files={
                            "file": (
                                f"audio_{unique_id}.mp3",
                                BytesIO(file_content),
                                "audio/mpeg",
                            )
                        },
                    )

                if response.status_code == 200:
                    return response.json()

                # Non-200 -> honest error, with retry for transient failures.
                content_type = response.headers.get("content-type", "")
                error_detail = (
                    response.json()
                    if "application/json" in content_type
                    else response.text
                )
                error_msg = f"API request failed with status {response.status_code}"
                logger.error("%s: %s", error_msg, error_detail)
                last_error = {"error": error_msg, "detail": error_detail}

                if attempt < _STT_MAX_RETRIES - 1:
                    time.sleep(1)
                    continue
                return last_error
            except Exception as e:  # network/transport error -> retry then fail
                logger.warning("Transcription attempt %d failed: %s", attempt + 1, e)
                last_error = {"error": str(e)}
                if attempt < _STT_MAX_RETRIES - 1:
                    time.sleep(1)
                    continue
                return last_error
        return last_error

    except Exception as e:
        logger.error("Error in transcription request: %s", e)
        return {"error": str(e)}

    finally:
        # Best-effort temp-file cleanup.
        try:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
        except Exception as cleanup_error:  # pragma: no cover
            logger.warning("Failed to clean up temp file: %s", cleanup_error)


def convert_elevenlabs_transcript_to_segments(
    transcript_data: Dict[str, Any],
    segment_strategy: str = "sentences",
) -> Dict[str, Any]:
    """Convert ElevenLabs speech-to-text response to widget segment format.

    Verbatim port of the reference's segmentation logic (sentences /
    time_blocks / speakers / word_by_word).
    """
    if "error" in transcript_data:
        return {"segments": [], "error": transcript_data["error"]}

    if "words" not in transcript_data:
        return {"segments": [], "error": "No word data found in transcript"}

    words = transcript_data["words"]

    def is_end_of_sentence(text: str) -> bool:
        return bool(text) and any(text.endswith(char) for char in [".", "!", "?", ":", ";"])

    if segment_strategy == "word_by_word":
        segments = []
        for word in words:
            if word.get("type") == "word":
                segments.append({
                    "start": word.get("start", 0),
                    "end": word.get("end", 0),
                    "text": word.get("text", ""),
                    "speaker_id": word.get("speaker_id", None),
                })

    elif segment_strategy == "speakers" and any("speaker_id" in word for word in words):
        segments = []
        current_segment = {"start": None, "end": None, "text": "", "speaker_id": None}

        for word in words:
            if word.get("type") == "spacing":
                continue

            current_speaker = word.get("speaker_id")

            if current_segment["start"] is None or current_speaker != current_segment["speaker_id"]:
                if current_segment["text"]:
                    segments.append(current_segment)
                current_segment = {
                    "start": word.get("start", 0),
                    "end": word.get("end", 0),
                    "text": word.get("text", ""),
                    "speaker_id": current_speaker,
                }
            else:
                current_segment["text"] += " " + word.get("text", "")
                current_segment["end"] = word.get("end", 0)

        if current_segment["text"]:
            segments.append(current_segment)

    elif segment_strategy == "time_blocks":
        block_duration = 5.0
        segments = []
        current_segment = {
            "start": words[0].get("start", 0) if words else 0,
            "end": None,
            "text": "",
            "speaker_id": words[0].get("speaker_id") if words and "speaker_id" in words[0] else None,
        }

        for word in words:
            if word.get("type") == "spacing":
                continue

            if word.get("end", 0) - current_segment["start"] > block_duration and current_segment["text"]:
                current_segment["end"] = word.get("start", 0)
                segments.append(current_segment)
                current_segment = {
                    "start": word.get("start", 0),
                    "end": None,
                    "text": word.get("text", ""),
                    "speaker_id": word.get("speaker_id") if "speaker_id" in word else None,
                }
            else:
                if current_segment["text"]:
                    current_segment["text"] += " "
                current_segment["text"] += word.get("text", "")
                current_segment["end"] = word.get("end", 0)

        if current_segment["text"]:
            segments.append(current_segment)

    else:  # Default to sentences
        segments = []
        current_segment = {"start": None, "end": None, "text": "", "speaker_id": None}

        for word in words:
            if word.get("type") == "spacing":
                continue

            if current_segment["start"] is None:
                current_segment["start"] = word.get("start", 0)
                current_segment["speaker_id"] = word.get("speaker_id") if "speaker_id" in word else None

            if current_segment["text"]:
                current_segment["text"] += " "

            current_segment["text"] += word.get("text", "")
            current_segment["end"] = word.get("end", 0)

            if is_end_of_sentence(word.get("text", "")):
                segments.append(current_segment)
                current_segment = {"start": None, "end": None, "text": "", "speaker_id": None}

        if current_segment["text"]:
            segments.append(current_segment)

    result: Dict[str, Any] = {"segments": segments}
    if "language_code" in transcript_data:
        result["language_code"] = transcript_data["language_code"]
    if "language_probability" in transcript_data:
        result["language_probability"] = transcript_data["language_probability"]

    return result


# ===========================================================================
# Summaries (the "& Summaries" half — Gemini via common.gemini_generate_text)
# ===========================================================================

def _safe_float(value: Any, default: float = 0.0) -> float:
    """Coerce an arbitrary value to float, returning ``default`` on failure.

    Uploaded transcript JSON is untrusted: start/end may be strings, null, or
    missing. This keeps downstream ``:.2f`` formatting from crashing the tab.
    """
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_uploaded_transcript(transcript_data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate/coerce an uploaded transcript dict into safe segment shape.

    Confirms ``segments`` is a list, drops non-dict entries, and coerces
    start/end via :func:`_safe_float`. Honest: never invents segments — a bad
    payload simply yields fewer (or zero) usable segments, never fake data.
    """
    raw_segments = transcript_data.get("segments")
    if not isinstance(raw_segments, list):
        raw_segments = []

    clean_segments: List[Dict[str, Any]] = []
    for seg in raw_segments:
        if not isinstance(seg, dict):
            continue
        clean_seg = dict(seg)
        clean_seg["start"] = _safe_float(seg.get("start"))
        clean_seg["end"] = _safe_float(seg.get("end"))
        clean_segments.append(clean_seg)

    coerced = dict(transcript_data)
    coerced["segments"] = clean_segments
    return coerced


def _transcript_to_plaintext(transcript: Dict[str, Any]) -> str:
    """Flatten a segment transcript into plain text for summarization.

    Includes speaker labels when present so the summary can attribute points.
    Bounded to ``_MAX_SUMMARY_INPUT_CHARS`` so a huge transcript can't blow
    the prompt budget.
    """
    segments = (transcript or {}).get("segments", []) or []
    lines: List[str] = []
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        speaker = seg.get("speaker_id")
        if speaker:
            lines.append(f"[{speaker}] {text}")
        else:
            lines.append(text)
    joined = "\n".join(lines).strip()
    if len(joined) > _MAX_SUMMARY_INPUT_CHARS:
        joined = joined[:_MAX_SUMMARY_INPUT_CHARS] + "\n[... transcript truncated for summarization ...]"
    return joined


def _build_summary_prompt(transcript_text: str) -> str:
    """Prompt asking for a concise summary + key points + action items."""
    return (
        "You are an expert meeting/audio analyst. Read the transcript below and "
        "produce a clear, well-structured Markdown brief with EXACTLY these three "
        "sections and headings:\n\n"
        "## Summary\n"
        "A concise 3-5 sentence overview of what the audio is about.\n\n"
        "## Key Points\n"
        "A bulleted list of the most important points, decisions, and facts.\n\n"
        "## Action Items\n"
        "A bulleted list of concrete next steps / tasks / follow-ups. Attribute "
        "to a speaker when the transcript makes the owner clear. If there are no "
        "action items, write '- None identified.'\n\n"
        "Be faithful to the transcript only — do not invent details. If the "
        "transcript is too short or empty to summarize, say so plainly.\n\n"
        "=== TRANSCRIPT START ===\n"
        f"{transcript_text}\n"
        "=== TRANSCRIPT END ==="
    )


def summarize_transcript(transcript: Dict[str, Any]) -> str:
    """Produce a Markdown summary of a transcript via Gemini.

    Returns "" on failure / no usable text (honest — caller treats "" as
    failure and surfaces an error, never a fake summary).
    """
    transcript_text = _transcript_to_plaintext(transcript)
    if not transcript_text:
        return ""
    prompt = _build_summary_prompt(transcript_text)
    try:
        return run_async(
            gemini_generate_text(prompt, agent_name="transcription_summarizer")
        )
    except Exception as e:  # run_async / event-loop edge cases -> honest empty
        logger.error("summarize_transcript failed: %s", e)
        return ""


# ===========================================================================
# Session-state init (all keys prefixed tr_)
# ===========================================================================

def _init_state() -> None:
    defaults = {
        _KEY + "audio_data": None,            # raw bytes
        _KEY + "audio_data_base64": None,     # base64 str for the widget
        _KEY + "audio_name": None,            # uploaded file name
        _KEY + "transcript": {"segments": []},
        _KEY + "transcript_raw": None,        # raw ElevenLabs response
        _KEY + "processing_complete": False,
        _KEY + "summary": None,               # last generated summary (Markdown)
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ===========================================================================
# UI sections
# ===========================================================================

def _render_transcribe_section() -> None:
    """Upload audio -> (ElevenLabs) -> transcript, with options + JSON upload."""
    eleven_ok = _elevenlabs_available()

    if not eleven_ok:
        # HONEST degradation: no key/dep -> warn + offer the JSON upload path.
        if not HTTPX_AVAILABLE:
            st.warning(
                "`httpx` is not installed, so live audio transcription is "
                "unavailable. You can still upload an existing transcript JSON "
                "below to use the synced player and summaries."
            )
        else:
            st.warning(
                "ElevenLabs API key not configured (`ELEVEN_LABS_API_KEY`), so "
                "live audio transcription is disabled. You can still upload an "
                "existing transcript JSON below to use the synced player and "
                "summaries."
            )

    st.subheader("1. Upload Audio")
    audio_file = st.file_uploader(
        "Upload an audio file (MP3, WAV, M4A, ...)",
        type=["mp3", "wav", "m4a", "mp4", "mpeg", "mpga", "webm"],
        key=_KEY + "audio_uploader",
        help="Audio is used for the synced player. With an ElevenLabs key it is "
             "also transcribed.",
    )

    if audio_file is not None:
        audio_bytes = audio_file.read()
        # BOUNDED read: refuse oversized audio before base64-ing into the page.
        if len(audio_bytes) > _MAX_AUDIO_BYTES:
            st.error(
                f"Audio file is too large "
                f"({len(audio_bytes) / 1_048_576:.1f} MB). "
                f"Limit is {_MAX_AUDIO_BYTES // 1_048_576} MB."
            )
            # Clear any stale audio from an earlier valid upload so a rejected
            # file can't leave bytes that the Generate-Transcript button would
            # silently transcribe (HONEST STATUS — no transcribing a hidden,
            # previously-uploaded file).
            st.session_state[_KEY + "audio_data"] = None
            st.session_state[_KEY + "audio_data_base64"] = None
            st.session_state[_KEY + "audio_name"] = None
            st.session_state[_KEY + "processing_complete"] = False
            st.session_state[_KEY + "summary"] = None
        else:
            st.session_state[_KEY + "audio_data"] = audio_bytes
            st.session_state[_KEY + "audio_data_base64"] = base64.b64encode(
                audio_bytes
            ).decode("utf-8")
            st.session_state[_KEY + "audio_name"] = audio_file.name
            st.success(f"Audio file '{audio_file.name}' loaded.")
            st.audio(audio_bytes)

    # --- Transcription options + run (only meaningful with a key) ----------
    if eleven_ok:
        st.subheader("2. Generate Transcript")
        with st.expander("Transcription Options", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                language_options = {
                    "": "Auto-detect",
                    "en": "English",
                    "es": "Spanish",
                    "fr": "French",
                    "de": "German",
                    "it": "Italian",
                    "pt": "Portuguese",
                    "pl": "Polish",
                }
                language_code = st.selectbox(
                    "Language:",
                    options=list(language_options.keys()),
                    format_func=lambda x: language_options[x],
                    key=_KEY + "language",
                )
                language_code = language_code if language_code else None

                segment_strategy = st.selectbox(
                    "Segmentation Strategy:",
                    options=["sentences", "time_blocks", "speakers", "word_by_word"],
                    help="How to group words into transcript segments",
                    key=_KEY + "segment_strategy",
                )
            with col2:
                timestamp_options = {
                    "word": "Word-level timestamps",
                    "character": "Character-level timestamps",
                }
                timestamps_granularity = st.selectbox(
                    "Timestamp Detail:",
                    options=list(timestamp_options.keys()),
                    format_func=lambda x: timestamp_options[x],
                    key=_KEY + "timestamp_granularity",
                )
                diarize = st.checkbox(
                    "Identify Speakers (Diarization)",
                    value=False,
                    key=_KEY + "diarize",
                )
                num_speakers = None
                if diarize:
                    num_speakers = st.slider(
                        "Maximum Number of Speakers:",
                        min_value=1,
                        max_value=32,
                        value=2,
                        key=_KEY + "num_speakers",
                    )

        audio_data = st.session_state.get(_KEY + "audio_data")
        if st.button(
            "Generate Transcript from Audio",
            key=_KEY + "btn_transcribe",
            disabled=audio_data is None,
            type="primary",
        ):
            if not audio_data:
                st.warning("Please upload an audio file first.")
            else:
                with st.spinner(
                    "Transcribing audio... this can take a while for long files."
                ):
                    transcript_data = transcribe_audio(
                        audio_bytes=audio_data,
                        model_id="scribe_v1",
                        language_code=language_code,
                        timestamps_granularity=timestamps_granularity,
                        diarize=diarize,
                        num_speakers=num_speakers,
                    )

                if "error" not in transcript_data:
                    st.session_state[_KEY + "transcript_raw"] = transcript_data
                    transcript_segments = convert_elevenlabs_transcript_to_segments(
                        transcript_data, segment_strategy=segment_strategy
                    )
                    st.session_state[_KEY + "transcript"] = transcript_segments
                    st.session_state[_KEY + "processing_complete"] = True
                    # New transcript invalidates any prior summary (honest state).
                    st.session_state[_KEY + "summary"] = None

                    st.success("Transcription completed.")
                    if "language_code" in transcript_data:
                        lang_code = transcript_data["language_code"]
                        lang_prob = transcript_data.get("language_probability", 0)
                        st.info(
                            f"Detected language: {lang_code} "
                            f"(confidence: {lang_prob:.2f})"
                        )
                    with st.expander("Full Transcript Text", expanded=True):
                        st.write(transcript_data.get("text", "No text available"))

                    st.download_button(
                        label="Download Transcript as JSON",
                        data=json.dumps(transcript_segments, indent=2),
                        file_name="transcript.json",
                        mime="application/json",
                        key=_KEY + "dl_transcript_inline",
                    )
                else:
                    st.error(f"Transcription failed: {transcript_data.get('error')}")
                    if "detail" in transcript_data:
                        st.code(str(transcript_data["detail"]))

    # --- Always offer the transcript-JSON upload path ----------------------
    header_n = "3" if eleven_ok else "2"
    st.subheader(f"{header_n}. Or Upload an Existing Transcript (JSON)")
    transcript_file = st.file_uploader(
        "Upload Transcript (JSON)",
        type=["json"],
        key=_KEY + "transcript_uploader",
        help="A JSON object with a 'segments' list (start/end/text per segment).",
    )
    if transcript_file is not None:
        # BOUNDED read: refuse oversized JSON before json.loads parses it into
        # the page (Streamlit's 200 MB upload default is far too large here).
        raw = transcript_file.read()
        if len(raw) > _MAX_TRANSCRIPT_JSON_BYTES:
            st.error(
                f"Transcript JSON is too large "
                f"({len(raw) / 1_048_576:.1f} MB). "
                f"Limit is {_MAX_TRANSCRIPT_JSON_BYTES // 1_048_576} MB."
            )
            return
        try:
            transcript_data = json.loads(raw)
            if not isinstance(transcript_data, dict) or "segments" not in transcript_data:
                st.error(
                    "Transcript JSON must be an object with a 'segments' list."
                )
            else:
                # Validate/coerce: confirm segments is a list, drop non-dict
                # entries, and coerce start/end to float so the player's
                # ':.2f' formatting can't crash on string/null timestamps.
                st.session_state[_KEY + "transcript"] = _coerce_uploaded_transcript(
                    transcript_data
                )
                st.session_state[_KEY + "processing_complete"] = True
                st.session_state[_KEY + "summary"] = None
                st.success(f"Transcript '{transcript_file.name}' loaded.")
        except json.JSONDecodeError:
            st.error("Invalid transcript file format. Please upload valid JSON.")


def _render_player_section() -> None:
    """Render the synced audio<->transcript player + transcript export."""
    transcript = st.session_state.get(_KEY + "transcript", {"segments": []})
    audio_b64 = st.session_state.get(_KEY + "audio_data_base64")
    has_segments = bool(transcript.get("segments"))

    if not has_segments:
        return

    st.subheader("Synchronized Audio-Text Player")

    if audio_b64 and ANYWIDGET_AVAILABLE and AudioTranscriptWidget is not None:
        try:
            widget = AudioTranscriptWidget(
                audio_data_base64=audio_b64,
                transcript=transcript,
            )
            anywidget(widget, key=_KEY + "audio_transcript")
        except Exception as e:
            logger.error("anywidget render failed: %s", e)
            st.warning(
                "Could not render the synced player; showing the transcript and "
                "a plain audio player instead."
            )
            audio_data = st.session_state.get(_KEY + "audio_data")
            if audio_data:
                st.audio(audio_data)
    else:
        # No audio (transcript-only) or anywidget missing -> degrade gracefully.
        if not audio_b64:
            st.info(
                "No audio loaded — showing the transcript only. Upload the "
                "matching audio file above to enable the synced player."
            )
        elif not ANYWIDGET_AVAILABLE:
            st.info(
                "`streamlit_anywidget` is not installed — showing a plain audio "
                "player and the transcript instead of the synced widget."
            )
            audio_data = st.session_state.get(_KEY + "audio_data")
            if audio_data:
                st.audio(audio_data)

    # Transcript text view (always available).
    with st.expander("Transcript Text"):
        for seg in transcript.get("segments", []):
            # Dict-guard + safe float coercion: uploaded transcripts may carry
            # non-dict segments or string/null start/end. Never crash the tab.
            if not isinstance(seg, dict):
                continue
            speaker_info = (
                f" [Speaker: {seg.get('speaker_id')}]" if seg.get("speaker_id") else ""
            )
            start_s = _safe_float(seg.get("start"))
            end_s = _safe_float(seg.get("end"))
            st.write(
                f"[{start_s:.2f}s - {end_s:.2f}s]"
                f"{speaker_info}: {seg.get('text', '')}"
            )

    # Export options.
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="Download Transcript as JSON",
            data=json.dumps(transcript, indent=2),
            file_name="transcript.json",
            mime="application/json",
            key=_KEY + "dl_transcript_player",
        )
    with col2:
        audio_data = st.session_state.get(_KEY + "audio_data")
        if audio_data:
            st.download_button(
                label="Download Audio (MP3)",
                data=audio_data,
                file_name=st.session_state.get(_KEY + "audio_name") or "audio.mp3",
                mime="audio/mp3",
                key=_KEY + "dl_audio_player",
            )


def _render_summary_section() -> None:
    """The "& Summaries" half: Gemini summary of the current transcript."""
    transcript = st.session_state.get(_KEY + "transcript", {"segments": []})
    has_segments = bool(transcript.get("segments"))
    if not has_segments:
        return

    st.divider()
    st.subheader("Summary & Key Points")

    # HONEST degradation: no Gemini key/SDK -> hide the action, explain why.
    if not _gemini_available():
        if not common.GENAI_AVAILABLE:
            st.info(
                "Summaries are unavailable because the `google-genai` SDK is not "
                "installed."
            )
        else:
            st.info(
                "Summaries are disabled because no Gemini API key is configured "
                "(`GEMINI_API_KEY` or legacy `GOOGLE_AI_STUDIO`). Add the key to "
                "enable AI summaries."
            )
        return

    if st.button(
        "Summarize Transcript",
        key=_KEY + "btn_summarize",
        type="primary",
    ):
        with st.spinner("Summarizing transcript with Gemini..."):
            summary = summarize_transcript(transcript)
        if summary:
            st.session_state[_KEY + "summary"] = summary
            # BOUNDED history (FIFO-capped in common.push_history).
            push_history(
                _SUMMARY_HISTORY_KEY,
                {
                    "name": st.session_state.get(_KEY + "audio_name")
                    or "transcript",
                    "summary": summary,
                },
            )
            st.success("Summary generated.")
        else:
            # HONEST: failure path surfaces an error, not a fake summary.
            st.error(
                "Could not generate a summary (the model returned nothing or the "
                "request failed). Check the Gemini key/quota and try again."
            )

    summary = st.session_state.get(_KEY + "summary")
    if summary:
        st.markdown(summary)
        st.download_button(
            label="Download Summary (Markdown)",
            data=summary,
            file_name="transcript_summary.md",
            mime="text/markdown",
            key=_KEY + "dl_summary",
        )

    # Show recent summaries (bounded history) for quick reference.
    history = get_history(_SUMMARY_HISTORY_KEY)
    if len(history) > 1:
        with st.expander(f"Recent summaries ({len(history)})"):
            for i, item in enumerate(reversed(history[-10:]), start=1):
                st.markdown(f"**{i}. {item.get('name', 'transcript')}**")
                st.markdown(item.get("summary", ""))
                if i < min(10, len(history)):
                    st.divider()


# ===========================================================================
# Public entrypoint
# ===========================================================================

def render_transcription_tab() -> None:
    """Render the Transcription & Summaries tab.

    Called by the main app inside a tab context. Takes no args, returns None,
    sets only ``tr_``-prefixed session-state keys, and degrades gracefully when
    ElevenLabs / Gemini keys or optional deps are missing.
    """
    _init_state()

    st.header("Transcription & Summaries")
    st.write(
        "Upload audio to transcribe it with ElevenLabs, watch the transcript "
        "highlight in sync with playback, then generate an AI summary with key "
        "points and action items."
    )

    _render_transcribe_section()

    if st.session_state.get(_KEY + "transcript", {}).get("segments"):
        st.divider()
        _render_player_section()
        _render_summary_section()
