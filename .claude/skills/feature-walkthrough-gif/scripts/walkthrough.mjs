// Workflow-walkthrough capturer (reusable). Drives the no-auth Streamlit harness
// (dev_preview_tabs.py, DEMO_CLEAN=1) through each feature spec, capturing a CLEAN
// frame at every UI state plus the pointer target for each step. Emits:
//   demo/public/wt/<id>/NN.png        (one frame per `cap` op)
//   demo/src/walkthrough.data.js      (per-feature steps -> consumed by Walkthrough.jsx)
// Remotion then overlays an animated cursor + click ripple + step caption so the
// rendered GIF shows exactly where the user clicked and what happened next.
//
//   DEMO_CLEAN=1 streamlit run dev_preview_tabs.py --server.port 8502
//   node demo/walkthrough.mjs
import { chromium } from "playwright";
import { mkdirSync, writeFileSync, rmSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { SPECS } from "./walkthrough.specs.mjs";

const __dirname = dirname(fileURLToPath(import.meta.url));
const PUB = join(__dirname, "public", "wt");
const BASE = process.env.DEMO_URL || "http://127.0.0.1:8502";
const VW = 1280, VH = 800, DEFAULT_HOLD = 60;

const sleep = (p, ms) => p.waitForTimeout(ms);
const panel = (p) => p.locator('[data-baseweb="tab-panel"]:visible').first();
const notRunning = (p, t = 220000) => p.waitForFunction(() => !document.querySelector('[data-testid="stStatusWidget"]'), { timeout: t, polling: 1000 }).catch(() => {});
const waitText = (p, s, t = 220000) => p.waitForFunction((x) => new RegExp(x).test(document.body.innerText), s, { timeout: t, polling: 1500 }).catch(() => {});

// Resolve a spec selector to a Playwright Locator scoped to the ACTIVE tab panel.
const loc = (p, sel) => {
  const P = panel(p);
  if (sel === "textarea") return P.locator('[data-testid="stTextArea"] textarea').first();
  if (sel === "input") return P.locator('[data-testid="stTextInput"] input').first();
  if (sel === "file") return P.locator('input[type="file"]').first();
  if (sel === "drop") return P.locator('[data-testid="stFileUploaderDropzone"]').first();
  if (sel === "chat") return P.locator('[data-testid="stChatInput"] textarea').first();
  if (sel.startsWith("btn:")) return P.getByRole("button", { name: new RegExp(sel.slice(4), "i") }).first();
  if (sel.startsWith("aria^:")) return P.locator(`input[aria-label^="${sel.slice(6)}"]`).first();
  if (sel.startsWith("aria:")) return P.locator(`input[aria-label="${sel.slice(5)}"]`).first();
  return P.locator(sel).first();
};

// Viewport-relative center of an element (CSS px, clamped) — where the cursor points.
const cursorOf = async (p, sel) => {
  if (!sel) return null;
  try {
    const el = loc(p, sel);
    await el.scrollIntoViewIfNeeded({ timeout: 4000 }).catch(() => {});
    const box = await el.evaluate((n) => {
      const r = n.getBoundingClientRect();
      return { x: r.left + r.width / 2, y: r.top + Math.min(r.height / 2, 22) };
    });
    return { x: Math.max(8, Math.min(VW - 8, Math.round(box.x))), y: Math.max(8, Math.min(VH - 8, Math.round(box.y))) };
  } catch { return null; }
};

const clickTab = async (p, text) => { await p.locator('[data-baseweb="tab"]', { hasText: text }).first().click(); await sleep(p, 1200); };

const doAct = async (p, a) => {
  if (a.act === "fill") { const el = loc(p, a.sel); await el.click(); await el.fill(String(a.value)); if (a.commit) await el.press(a.commit); await sleep(p, 600); }
  else if (a.act === "click") { await loc(p, a.sel).click(); await sleep(p, 300); }
  else if (a.act === "upload") { await loc(p, a.sel).setInputFiles(join(__dirname, "fixtures", a.file)); }
  else if (a.act === "sleep") { await sleep(p, a.ms); }
  else if (a.act === "waitText") { await waitText(p, a.value); }
  else if (a.act === "notRunning") { await notRunning(p); }
  else if (a.act === "scrollTop") { await p.evaluate(() => window.scrollTo(0, 0)); await sleep(p, 300); }
  else if (a.act === "scrollY") { await p.evaluate((y) => window.scrollTo(0, y), a.y); await sleep(p, 300); }
  else if (a.act === "scrollText") { await p.evaluate((s) => { const rx = new RegExp(s); const el = [...document.querySelectorAll("*")].find((n) => rx.test(n.textContent || "") && n.children.length < 6); if (el) el.scrollIntoView({ block: "center" }); }, a.value); await sleep(p, 400); }
  else if (a.act === "scrollLastChat") { await p.evaluate(() => { const m = document.querySelectorAll('[data-testid="stChatMessage"]'); if (m.length) m[m.length - 1].scrollIntoView({ block: "center" }); }); await sleep(p, 500); }
  else if (a.act === "scrollEl") {
    // Center the RESULT WIDGET in the viewport (window.scrollTo doesn't move
    // Streamlit's inner scroll container — scrollIntoView on the element does).
    const map = { df: '[data-testid="stDataFrame"]', iframe: "iframe", metric: '[data-testid="stMetric"]' };
    const css = map[a.sel] || a.sel;
    const L = panel(p).locator(css);
    const el = a.last ? L.last() : L.first();
    await el.evaluate((n) => n.scrollIntoView({ block: "center", inline: "nearest" })).catch(() => {});
    await sleep(p, 600);
  }
};

const run = async () => {
  rmSync(PUB, { recursive: true, force: true });
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: VW, height: VH }, deviceScaleFactor: 2 });
  page.setDefaultTimeout(60000);
  await page.goto(BASE, { waitUntil: "networkidle" });
  await page.waitForFunction(() => [...document.querySelectorAll('[data-baseweb="tab"]')].some((t) => /List Intelligence/.test(t.innerText)), { timeout: 60000 });
  await sleep(page, 1500);

  const out = [];
  for (const spec of SPECS) {
    const dir = join(PUB, spec.id);
    mkdirSync(dir, { recursive: true });
    const steps = [];
    try {
      await clickTab(page, spec.tab);
      let n = 0;
      for (const op of spec.steps) {
        if (op.cap && op.burst) {
          // BURST: rapidly capture a SEQUENCE of the loading/streaming state so the
          // rendered clip shows real motion — the spinner spinning, the status text
          // updating, results streaming in — instead of a frozen snapshot.
          const every = op.burst.every || 320;
          const count = Math.max(2, Math.round((op.burst.ms || 2800) / every));
          const imgs = [];
          for (let b = 0; b < count; b++) {
            const fn = `${String(n).padStart(2, "0")}_${String(b).padStart(2, "0")}.png`;
            await page.screenshot({ path: join(dir, fn) });
            imgs.push(`wt/${spec.id}/${fn}`);
            if (b < count - 1) await sleep(page, every);
          }
          steps.push({ imgs, caption: op.cap, cursor: null, click: false, hold: op.hold || 72, burst: true });
          console.log(`  ${spec.id} burst ${n}: ${imgs.length} frames — ${op.cap}`);
          n++;
        } else if (op.cap) {
          const cur = await cursorOf(page, op.cursor);
          const name = String(n).padStart(2, "0") + ".png";
          await sleep(page, 350);
          await page.screenshot({ path: join(dir, name) });
          steps.push({ img: `wt/${spec.id}/${name}`, caption: op.cap, cursor: cur, click: !!op.click, hold: op.hold || DEFAULT_HOLD });
          console.log(`  ${spec.id} cap ${n}: ${op.cap}`);
          n++;
        } else {
          await doAct(page, op);
        }
      }
    } catch (e) { console.log(spec.id, "err", e.message); }
    out.push({ id: spec.id, title: spec.title, accent: spec.accent, steps });
  }
  await browser.close();

  const data = "// AUTO-GENERATED by demo/walkthrough.mjs — do not edit by hand.\n" +
    "export const WALKTHROUGHS = " + JSON.stringify(out, null, 2) + ";\n";
  writeFileSync(join(__dirname, "src", "walkthrough.data.js"), data);
  console.log("WALKTHROUGH_CAPTURE_DONE — wrote src/walkthrough.data.js");
};
run().catch((e) => { console.error(e); process.exit(1); });
