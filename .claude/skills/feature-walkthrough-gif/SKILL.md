---
name: feature-walkthrough-gif
description: >-
  Produce polished, annotated workflow-walkthrough GIFs for a web app (built/tested
  on Streamlit, works for any browser UI): each GIF shows EVERY UI state of a feature
  — empty form → cursor gliding to the input → the click (with a ripple) → the loading
  state → the final result — plus a step caption and progress bar, so a viewer sees
  exactly what was clicked and what happened. Use when the user wants per-feature demo
  GIFs / README "skill preview" clips that actually walk through the end-to-end flow
  rather than just showing a single final-state "hero shot". Pipeline = Playwright
  capture → generated step data → Remotion render → ffmpeg GIF.
---

# Feature Walkthrough GIF

Turn a live feature into a short, looping, **annotated walkthrough GIF**: clean
state frames + an overlaid cursor that glides to each click (with a ripple) +
per-step captions + a progress bar. The opposite of a single Ken-Burns "hero"
frame — the viewer follows the whole flow.

## When to use
- "Make a GIF that walks through the feature" / "show every step / where the user clicked".
- Per-feature README previews, changelog clips, PR demos, onboarding.
NOT for: a static screenshot (use a screenshot), or a full narrated marketing reel
(that's a different, longer composition).

## Output, at a glance
For each feature → one `assets/feature-<name>.gif` (~0.5–1.5 MB, ~10 s, loops),
embedded in the README. Made of N "steps", each a captured UI state with a caption.

## Prerequisites (verify first)
- The app running locally in a **no-auth / demo state** (Streamlit: a `dev_preview`
  harness; set a `DEMO_CLEAN=1`-style flag to hide toolbars/chrome for clean frames).
- A **Remotion** project (`npm i remotion @remotion/cli @remotion/bundler react react-dom`)
  and **Playwright** (`npm i -D playwright && npx playwright install chromium`).
- **ffmpeg** on PATH.
- Real API keys if the feature calls live services (capture exercises the real app).

## The four-stage pipeline
1. **Spec** — for each feature, write an ordered list of ops (see format below):
   `cap` = capture this UI state (+ where the cursor points); `act` = perform an
   action to advance the UI.
2. **Capture** — `node walkthrough.mjs` drives the app per spec, screenshots a CLEAN
   frame at each `cap`, records the pointer target (element center, viewport px) and
   click flag, and writes `src/walkthrough.data.js` + frames under `public/wt/<id>/`.
3. **Render** — Remotion `Walkthrough.jsx` overlays an animated cursor (glides between
   targets), a click ripple, the step caption + progress dots over each frame.
   `npx remotion render src/index.js WT-<id> out/WT-<id>.mp4`.
4. **GIF** — `ffmpeg -i out/WT-<id>.mp4 -vf "fps=12,scale=760:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer:bayer_scale=3" -loop 0 assets/feature-<name>.gif`.

## Spec format (`walkthrough.specs.mjs`)
A spec is `{ id, title, accent, tab, steps: [...] }`. Each step is ONE of:
- **Capture**: `{ cap: "Caption text", cursor?: "<selector>", click?: true, hold?: 60 }`
  Captures a clean frame of the CURRENT state. `cursor` marks where the pointer
  glides to; `click:true` draws a ripple there; `hold` = frames to dwell (30 fps).
- **Action**: `{ act: "fill"|"click"|"upload"|"sleep"|"waitText"|"notRunning"|"scrollTop"|"scrollY"|"scrollText"|"scrollLastChat", ... }`
  Advances the UI so the NEXT `cap` shows the result.

Order so the story reads: capture empty → fill → capture (cursor on the button,
click:true) → click → sleep → capture the loading state → wait for result →
capture the result. See `scripts/walkthrough.specs.example.mjs` for 5 worked specs.

Selector shorthand (resolved against the **active tab panel** — see lesson #1):
`"textarea"`, `"input"`, `"file"`, `"drop"` (uploader dropzone), `"chat"` (chat input),
`"btn:Run pipeline"` (button by accessible-name regex), `"aria:Net income"` /
`"aria^:Revenue"` (input by aria-label exact / prefix), or any raw CSS selector.

## Hard-won capture lessons (THIS is why naive captures fail)
1. **Scope every locator to the ACTIVE tab panel.** Streamlit (and many tab UIs)
   render ALL tab panels in the DOM; an unscoped `.first()` silently matches a hidden
   tab's element. Use `[data-baseweb="tab-panel"]:visible` (the helper `panel(page)`).
2. **Await uploads until the widget registers** before clicking the action button
   (e.g. wait for "1 image(s) ready"); clicking too early hits a disabled/0-item button.
3. **Data-grids are CANVAS** (glide-data-grid): cell text is NOT in the DOM. Don't
   wait on row text — wait on a real DOM label (a KPI, a heading) instead.
4. **Capture the loading state on purpose**: after the click, `sleep ~1.5 s` then `cap`
   — that frame ("running…") is what makes the walkthrough show *what happened*.
5. **Avoid spinner-timing flakiness where you can**: e.g. for a deterministic form
   (a calculator), drive it via manual field entry instead of an LLM-extract call.
6. **Cursor coords are viewport CSS px** (`getBoundingClientRect` center) captured at
   the moment of the step; the cursor is NOT baked into the screenshot — Remotion
   overlays + animates it, so it can glide and ripple. Frames stay clean.
7. **Don't match caption text in your wait conditions**: e.g. waiting for "Adjusted
   EBITDA" fires instantly if it's in a page header. Wait for a post-result-only
   string (a computed value, a status="complete" label).

## Reuse in a new project
1. Copy `scripts/walkthrough.mjs` + `scripts/Walkthrough.jsx` into the Remotion project
   (`Walkthrough.jsx` → `src/`), and register it in `src/Root.jsx`:
   ```jsx
   import { Walkthrough, WT_FPS, WT_W, WT_H, wtDuration } from "./Walkthrough.jsx";
   import { WALKTHROUGHS } from "./walkthrough.data.js";   // generated by the capturer
   {WALKTHROUGHS.map((w) => (
     <Composition key={"WT-"+w.id} id={"WT-"+w.id} component={Walkthrough}
       durationInFrames={Math.max(1, wtDuration(w))} fps={WT_FPS} width={WT_W}
       height={WT_H} defaultProps={{ wt: w }} />
   ))}
   ```
   Create a stub `src/walkthrough.data.js` (`export const WALKTHROUGHS = [];`) so it
   imports before the first capture.
2. Write `walkthrough.specs.mjs` for your features (adapt the example).
3. Start the app's clean/demo harness, then: `node walkthrough.mjs` → render each
   `WT-<id>` → ffmpeg to `assets/feature-<name>.gif` → embed in the README.

## Gotchas
- Remotion composition ids **cannot contain `_`** (a-z A-Z 0-9 and `-` only) → use `WT-<id>`.
- Tune the GIF size with `fps` (10–15), `scale` (700–900 px = README width), and
  `max_colors` (96–128). Stepped frames + crossfades compress far better than
  continuous motion, so these GIFs are usually smaller than Ken-Burns ones.
- If a captured state is wrong, fix the SPEC (wait/scroll/selector), not the renderer.
