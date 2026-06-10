// Workflow-walkthrough specs for the 5 ParselyFi feature tabs.
// Each spec is an ORDERED list of ops:
//   { cap, cursor?, click?, hold? }  -> CAPTURE a clean frame of the current UI
//        state. `cursor` (a selector, see resolver in walkthrough.mjs) marks where
//        the pointer should glide to; `click:true` shows a click ripple there.
//        `hold` = frames to dwell on this state in the rendered walkthrough.
//   { act, ... }                     -> PERFORM an action to advance the UI.
// The capturer records each `cap` as {img, caption, cursor:{x,y}|null, click, hold}
// in walkthrough.data.js; Remotion (Walkthrough.jsx) overlays an animated cursor
// + step caption so the viewer sees exactly what was clicked and what happened.
export const SPECS = [
  {
    id: "ListIntel", title: "List Intelligence", accent: "#10b981", tab: "List Intelligence",
    steps: [
      { cap: "Paste a list of companies", cursor: "textarea", hold: 66 },
      { act: "fill", sel: "textarea", value: "Stripe\nRamp", commit: "Control+Enter" },
      { cap: "Click Run pipeline", cursor: "btn:Run pipeline", click: true, hold: 60 },
      { act: "click", sel: "btn:Run pipeline" },
      { act: "sleep", ms: 700 },
      { cap: "MATCH → ENRICH → CLASSIFY → SCORE — running live", burst: { ms: 3200, every: 300 }, hold: 78 },
      { act: "waitText", value: "Avg score|Results grid" }, { act: "notRunning" },
      { act: "sleep", ms: 1600 }, { act: "scrollEl", sel: "df", last: true },
      { cap: "Every company scored — evidence-backed", hold: 104 },
    ],
  },
  {
    id: "Graph", title: "Relationship Graph", accent: "#34d399", tab: "Relationship Graph",
    steps: [
      { cap: "Type a seed company", cursor: "input", hold: 60 },
      { act: "fill", sel: "input", value: "Stripe", commit: "Enter" },
      { cap: "Build the relationship graph", cursor: "btn:Build relationship graph", click: true, hold: 60 },
      { act: "click", sel: "btn:Build relationship graph" },
      { act: "sleep", ms: 700 },
      { cap: "Extracting source-backed relationships — live", burst: { ms: 3000, every: 300 }, hold: 74 },
      { act: "waitText", value: "Depth-1|Relationships|Entities" }, { act: "notRunning" },
      { act: "sleep", ms: 2800 }, { act: "scrollEl", sel: "iframe", last: true },
      { cap: "Interactive corporate lineage", hold: 110 },
    ],
  },
  {
    id: "Cards", title: "Card → Rows", accent: "#0A7CFF", tab: "Card",
    steps: [
      { cap: "Drop a cap table / deck slide image", cursor: "drop", hold: 64 },
      { act: "upload", sel: "file", file: "cap_table.png" },
      { act: "waitText", value: "1 image\\(s\\) ready|Extract companies from 1" }, { act: "sleep", ms: 700 },
      { cap: "Extract the companies", cursor: "btn:Extract companies from 1", click: true, hold: 60 },
      { act: "click", sel: "btn:Extract companies from 1" },
      { act: "sleep", ms: 600 },
      { cap: "Gemini reads the image — multimodal, live", burst: { ms: 2600, every: 280 }, hold: 66 },
      { act: "waitText", value: "Companies found" }, { act: "notRunning" },
      { act: "sleep", ms: 1400 }, { act: "scrollEl", sel: "df", last: true },
      { cap: "Structured rows — ready for List Intelligence", hold: 104 },
    ],
  },
  {
    id: "DocBrain", title: "Document Brain", accent: "#8E75B2", tab: "Document Brain",
    steps: [
      { cap: "Upload your own documents", cursor: "drop", hold: 60 },
      { act: "upload", sel: "file", file: "memo.md" }, { act: "sleep", ms: 900 },
      { cap: "Ingest & index", cursor: "btn:Ingest", click: true, hold: 56 },
      { act: "click", sel: "btn:Ingest" }, { act: "notRunning" }, { act: "sleep", ms: 1800 },
      { cap: "Ask a question about your files", cursor: "chat", hold: 60 },
      { act: "fill", sel: "chat", value: "Who founded Acme Robotics and how much did they raise?", commit: "Enter" },
      { act: "sleep", ms: 400 }, { act: "scrollLastChat" },
      { cap: "Searching your documents — grounded retrieval", burst: { ms: 2600, every: 280 }, hold: 62 },
      { act: "waitText", value: "Sequoia|Jane Doe|memo\\.md#chunk" }, { act: "notRunning" },
      { act: "sleep", ms: 1800 }, { act: "scrollLastChat" },
      { cap: "Grounded answer — with (file#chunk) citations", hold: 110 },
    ],
  },
  {
    id: "Ebitda", title: "EBITDA Bridge", accent: "#10b981", tab: "EBITDA",
    steps: [
      { cap: "Enter the income-statement figures", cursor: "aria:Net income", hold: 64 },
      { act: "fill", sel: "aria:Net income", value: "3200", commit: "Enter" },
      { act: "fill", sel: "aria:Interest expense", value: "1200", commit: "Enter" },
      { act: "fill", sel: "aria:Income taxes", value: "1100", commit: "Enter" },
      { cap: "…each line — no number is computed by the LLM", cursor: "aria:Depreciation", hold: 60 },
      { act: "fill", sel: "aria:Depreciation", value: "2500", commit: "Enter" },
      { act: "fill", sel: "aria:Amortization", value: "800", commit: "Enter" },
      { act: "fill", sel: "aria^:Revenue", value: "50000", commit: "Enter" },
      { act: "sleep", ms: 900 }, { act: "scrollEl", sel: "metric", last: false },
      { cap: "Python computes EBIT → EBITDA → Adjusted EBITDA", hold: 112 },
    ],
  },
];
