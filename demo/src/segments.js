// Segment config for the demo reel. `images` reference files in demo/public/
// (copied from recordings/). `narration` references demo/public/audio/.
// Durations are in frames at 30fps. Finalized after capture confirms images.
export const SEGMENTS = [
  {
    kind: "intro",
    durationInFrames: 280,
    narration: "audio/s0_intro.mp3",
    title: "ParselyFi",
    subtitle: "Source-backed financial research, in one app.",
  },
  {
    kind: "feature",
    step: "01",
    accent: "#34d399",
    durationInFrames: 285,
    narration: "audio/s1_company.mp3",
    images: ["s01_company_input.png", "s02_company_results.png"],
    title: "Company Search & Analysis",
    subtitle: "Resolve the right entity from real sources, then a 3-pass profile.",
  },
  {
    kind: "feature",
    step: "02",
    accent: "#0A7CFF",
    durationInFrames: 245,
    narration: "audio/s2_news.mp3",
    images: ["s04_news_brief.png"],
    title: "News & YouTube",
    subtitle: "A topic in → a cited briefing with current, real sources.",
  },
  {
    kind: "feature",
    step: "03",
    accent: "#8E75B2",
    durationInFrames: 250,
    narration: "audio/s3_transcribe.mp3",
    images: ["s05_transcribe_ui.png", "s07_transcribe_summary.png"],
    title: "Transcription & Summaries",
    subtitle: "Audio → synced transcript → one-click AI summary.",
  },
  {
    kind: "outro",
    durationInFrames: 210,
    narration: "audio/s4_outro.mp3",
    title: "ParselyFi",
    subtitle: "parselyfi.streamlit.app",
  },
];
