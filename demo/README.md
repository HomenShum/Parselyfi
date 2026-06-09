# demo/ — ParselyFi product demo reel

Feature-by-feature demo video, produced with **Playwright capture → edge-tts narration → Remotion**. The rendered reel lives at [`../assets/parselyfi-demo.mp4`](../assets/parselyfi-demo.mp4) (preview GIF: `../assets/parselyfi-demo.gif`).

## Pipeline

```
dev_preview_tabs.py (no-auth harness, real APIs)
        │  Playwright drives each feature, captures clean 1280x800 frames
        ▼
demo/recordings/*.png   →  curated frames copied to  demo/public/*.png
        │
edge-tts  →  demo/public/audio/*.mp3   (voiceover per segment)
        │
Remotion (src/Demo.jsx)  →  intro · 3 feature segments · outro
        │  browser-chrome framing, Ken Burns, lower-third captions, progress bar
        ▼
demo/out/parselyfi-demo.mp4  →  copied to ../assets/ + ffmpeg GIF
```

## Reproduce

```bash
cd demo
npm install
npx playwright install chromium

# 1. Start the no-auth harness (from the repo root, with secrets configured)
#    streamlit run dev_preview_tabs.py --server.port 8502
# 2. Capture frames
DEMO_URL=http://127.0.0.1:8502 node capture.mjs
DEMO_URL=http://127.0.0.1:8502 node capture_fix.mjs   # news result + transcription summary

# 3. Narration (edge-tts; free, no key)
python -m pip install edge-tts   # then see the edge-tts calls used for public/audio/*.mp3

# 4. Render
cp recordings/{s01_company_input,s02_company_results,s04_news_brief,s05_transcribe_ui,s07_transcribe_summary}.png public/
npm run render            # -> out/parselyfi-demo.mp4
npm run studio            # interactive preview/editing
```

## Files

| Path | Purpose |
| --- | --- |
| `capture.mjs` / `capture_fix.mjs` | Playwright capture scripts |
| `fixtures/sample_transcript.json` | Demo transcript for the Transcription segment |
| `src/Demo.jsx` | Remotion composition (intro/feature/outro) |
| `src/segments.js` | Segment config — images, captions, narration, durations |
| `public/` | Frames + narration the video reads via `staticFile` |

> The capture scripts hit real Gemini/LinkUp/ElevenLabs APIs through the harness. `recordings/`, `out/`, and `node_modules/` are gitignored; `public/` (frames + audio) is committed so the video re-renders without re-capturing.
