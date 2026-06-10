// Playwright capture for the 5 Phase 2-4 tabs (List Intelligence, Relationship
// Graph, Card -> Rows, Document Brain, EBITDA Bridge). Drives the no-auth harness
// (dev_preview_tabs.py, DEMO_CLEAN=1) with real APIs and saves clean 1280x800
// frames to recordings/; Remotion turns them into the per-feature README GIFs.
//
//   DEMO_CLEAN=1 streamlit run dev_preview_tabs.py --server.port 8502
//   node demo/capture_new.mjs
//
// KEY LESSON: Streamlit renders ALL tab panels in the DOM, so every locator is
// scoped to the ACTIVE (visible) tab panel, and uploads are awaited until the
// widget registers them before clicking the action button.
import { chromium } from "playwright";
import { mkdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const __dirname = dirname(fileURLToPath(import.meta.url));
const OUT = join(__dirname, "recordings");
mkdirSync(OUT, { recursive: true });
const BASE = process.env.DEMO_URL || "http://127.0.0.1:8502";

const shot = async (page, name) => { await page.waitForTimeout(400); await page.screenshot({ path: join(OUT, `${name}.png`) }); console.log("captured", name); };
const clickTab = async (page, text) => { await page.locator(`[data-baseweb="tab"]`, { hasText: text }).first().click(); await page.waitForTimeout(1200); };
const notRunning = async (page, t = 220000) => page.waitForFunction(() => !document.querySelector('[data-testid="stStatusWidget"]'), { timeout: t, polling: 1000 }).catch(() => {});
const panel = (page) => page.locator('[data-baseweb="tab-panel"]:visible').first();   // active tab only
const waitText = (page, s, t = 220000) => page.waitForFunction((x) => new RegExp(x).test(document.body.innerText), s, { timeout: t, polling: 1500 }).catch(() => {});
const scrollTo = (page, re) => page.evaluate((s) => {
  const rx = new RegExp(s);
  const el = [...document.querySelectorAll("*")].find((n) => rx.test(n.textContent || "") && n.children.length < 6);
  if (el) el.scrollIntoView({ block: "center" });
}, re.source);

const INCOME = [3200, 1200, 1100, 2500, 800, 50000]; // NI, interest, taxes, dep, amort, revenue

const run = async () => {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1280, height: 800 }, deviceScaleFactor: 2 });
  page.setDefaultTimeout(60000);
  await page.goto(BASE, { waitUntil: "networkidle" });
  await page.waitForFunction(() => [...document.querySelectorAll('[data-baseweb="tab"]')].some((t) => /List Intelligence/.test(t.innerText)), { timeout: 60000 });
  await page.waitForTimeout(1500);

  // 1. List Intelligence — run the pipeline on 2 companies, capture the scored grid.
  try {
    await clickTab(page, "List Intelligence");
    const ta = panel(page).locator('[data-testid="stTextArea"] textarea').first();
    await ta.click(); await ta.fill("Stripe\nRamp"); await ta.press("Control+Enter");
    await page.waitForTimeout(800);
    await panel(page).getByRole("button", { name: /Run pipeline/i }).first().click();
    await waitText(page, "Results grid|Avg score|Coverage");
    await notRunning(page); await page.waitForTimeout(1600);
    await page.evaluate(() => window.scrollTo(0, 0));
    await shot(page, "li_results");
  } catch (e) { console.log("li err", e.message); }

  // 2. Relationship Graph — build the pyvis graph for a seed, capture it.
  try {
    await clickTab(page, "Relationship Graph");
    const inp = panel(page).locator('[data-testid="stTextInput"] input').first();
    await inp.click(); await inp.fill("Stripe"); await inp.press("Enter"); await page.waitForTimeout(500);
    await panel(page).getByRole("button", { name: /Build relationship graph/i }).first().click();
    await waitText(page, "Depth-1|Relationships|Entities");
    await notRunning(page); await page.waitForTimeout(3000); // let physics settle
    await page.evaluate(() => window.scrollTo(0, 230));
    await shot(page, "graph_result");
  } catch (e) { console.log("graph err", e.message); }

  // 3. Card -> Rows — upload a cap-table image (wait until it registers), extract.
  try {
    await clickTab(page, "Card");
    await panel(page).locator('input[type="file"]').first().setInputFiles(join(__dirname, "fixtures", "cap_table.png"));
    await waitText(page, "1 image\\(s\\) ready|Extract companies from 1");
    await page.waitForTimeout(800);
    await panel(page).getByRole("button", { name: /Extract companies from 1/i }).first().click();
    await waitText(page, "Companies found");
    await notRunning(page); await page.waitForTimeout(1500);
    await scrollTo(page, /Companies found/);
    await shot(page, "cards_result");
  } catch (e) { console.log("cards err", e.message); }

  // 4. Document Brain — ingest a memo, ask a question, capture the cited answer.
  try {
    await clickTab(page, "Document Brain");
    await panel(page).locator('input[type="file"]').first().setInputFiles(join(__dirname, "fixtures", "memo.md"));
    await page.waitForTimeout(1000);
    await panel(page).getByRole("button", { name: /Ingest/i }).first().click();
    await notRunning(page); await page.waitForTimeout(2000);
    const chat = panel(page).locator('[data-testid="stChatInput"] textarea').first();
    await chat.click(); await chat.fill("Who founded Acme Robotics and how much did they raise?"); await chat.press("Enter");
    await waitText(page, "Sequoia|Jane Doe|memo\\.md#chunk");
    await notRunning(page); await page.waitForTimeout(2000);
    await page.evaluate(() => {
      const m = document.querySelectorAll('[data-testid="stChatMessage"]');
      if (m.length) m[m.length - 1].scrollIntoView({ block: "center" });
    });
    await page.waitForTimeout(700);
    await shot(page, "docbrain_result");
  } catch (e) { console.log("docbrain err", e.message); }

  // 5. EBITDA Bridge — deterministic MANUAL entry (no API spinner), capture bridge.
  try {
    await clickTab(page, "EBITDA");
    const labels = ['input[aria-label="Net income"]', 'input[aria-label="Interest expense"]',
      'input[aria-label="Income taxes"]', 'input[aria-label="Depreciation"]',
      'input[aria-label="Amortization"]', 'input[aria-label^="Revenue"]'];
    for (let i = 0; i < labels.length; i++) {
      const inp = panel(page).locator(labels[i]).first();
      await inp.click(); await inp.fill(String(INCOME[i])); await inp.press("Enter");
      await page.waitForTimeout(650);
    }
    await page.waitForTimeout(900);
    await scrollTo(page, /3 · Bridge/);
    await shot(page, "ebitda_result");
  } catch (e) { console.log("ebitda err", e.message); }

  await browser.close();
  console.log("CAPTURE_NEW_DONE");
};
run().catch((e) => { console.error(e); process.exit(1); });
