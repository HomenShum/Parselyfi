// Playwright capture for the ParselyFi demo reel.
// Drives the no-auth harness (dev_preview_tabs.py), pre-warms each feature with
// real APIs, and saves clean 1280x800 frames to recordings/. Trimming/pacing
// happens later in Remotion.
import { chromium } from "playwright";
import { mkdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const __dirname = dirname(fileURLToPath(import.meta.url));
const OUT = join(__dirname, "recordings");
mkdirSync(OUT, { recursive: true });

const BASE = process.env.DEMO_URL || "http://127.0.0.1:8502";
const VW = 1280, VH = 800;
const shot = async (page, name) => {
  await page.waitForTimeout(400);
  await page.screenshot({ path: join(OUT, `${name}.png`) });
  console.log("captured", name);
};
const clickTab = async (page, text) => {
  await page.locator(`[data-baseweb="tab"]`, { hasText: text }).first().click();
  await page.waitForTimeout(900);
};
const notRunning = async (page, timeout = 180000) =>
  page.waitForFunction(
    () => !document.querySelector('[data-testid="stStatusWidget"]'),
    { timeout, polling: 1000 }
  ).catch(() => {});

const run = async () => {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: VW, height: VH }, deviceScaleFactor: 2 });
  page.setDefaultTimeout(60000);

  console.log("goto", BASE);
  await page.goto(BASE, { waitUntil: "networkidle" });
  // Wait for the 3 tabs to render.
  await page.waitForFunction(
    () => [...document.querySelectorAll('[data-baseweb="tab"]')]
      .some((t) => /Company Search/.test(t.innerText)),
    { timeout: 60000 }
  );
  await page.waitForTimeout(1500);
  await shot(page, "s00_hero");

  // ---- Feature 1: Company Search ----
  try {
    await clickTab(page, "Company Search");
    const ta = page.locator('[data-testid="stTextArea"] textarea').first();
    await ta.click();
    await ta.fill("Anthropic");
    await ta.press("Control+Enter");
    await page.waitForTimeout(800);
    await shot(page, "s01_company_input");
    await page.getByRole("button", { name: "Search", exact: true }).first().click();
    // Real Gemini + LinkUp: wait for entity resolution to appear.
    await page.waitForFunction(
      () => /Entity Preview|Entity Selection|Selected Target|Selection Summary/.test(document.body.innerText),
      { timeout: 180000, polling: 1500 }
    ).catch(() => {});
    await notRunning(page);
    await page.waitForTimeout(1200);
    await shot(page, "s02_company_results");
    // Scroll to the rich entity description for a second framing.
    await page.evaluate(() => {
      const el = [...document.querySelectorAll("*")].find((n) => /Entity Preview/.test(n.textContent || "") && n.children.length < 6);
      (el || document.body).scrollIntoView({ block: "center" });
    });
    await page.waitForTimeout(700);
    await shot(page, "s02b_company_entity");
  } catch (e) { console.log("company err", e.message); }

  // ---- Feature 2: News & YouTube ----
  try {
    await clickTab(page, "News & YouTube");
    await page.waitForTimeout(800);
    await shot(page, "s03_news_landing");
    const inp = page.locator('input[aria-label="News topic / query"], [data-testid="stTextInput"] input').first();
    await inp.click();
    await inp.fill("NVIDIA quarterly earnings");
    await inp.press("Enter");
    await page.waitForTimeout(500);
    await page.getByRole("button", { name: /Generate News Report/i }).first().click();
    await page.waitForFunction(
      () => /Briefing|Report for:|Key points/.test(document.body.innerText),
      { timeout: 180000, polling: 1500 }
    ).catch(() => {});
    await notRunning(page);
    await page.waitForTimeout(1200);
    await page.evaluate(() => {
      const el = [...document.querySelectorAll("*")].find((n) => /Briefing/.test(n.textContent || "") && n.children.length < 4);
      (el || document.body).scrollIntoView({ block: "start" });
    });
    await page.waitForTimeout(700);
    await shot(page, "s04_news_brief");
  } catch (e) { console.log("news err", e.message); }

  // ---- Feature 3: Transcription & Summaries ----
  try {
    await clickTab(page, "Transcription");
    await page.waitForTimeout(900);
    await page.evaluate(() => window.scrollTo(0, 0));
    await shot(page, "s05_transcribe_ui");
    // Upload an existing transcript (JSON) -> then Summarize (reliable, no audio key needed).
    const jsonPath = join(__dirname, "fixtures", "sample_transcript.json");
    const fileInputs = page.locator('[data-testid="stFileUploader"] input[type="file"]');
    const count = await fileInputs.count();
    if (count >= 2) {
      await fileInputs.nth(1).setInputFiles(jsonPath); // 2nd uploader = transcript JSON
      await notRunning(page);
      await page.waitForTimeout(1500);
      await shot(page, "s06_transcribe_loaded");
      const sumBtn = page.getByRole("button", { name: /Summari/i }).first();
      if (await sumBtn.count()) {
        await sumBtn.click();
        await notRunning(page);
        await page.waitForTimeout(1500);
        await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
        await page.waitForTimeout(600);
        await shot(page, "s07_transcribe_summary");
      }
    }
  } catch (e) { console.log("transcribe err", e.message); }

  await browser.close();
  console.log("CAPTURE_DONE");
};
run().catch((e) => { console.error(e); process.exit(1); });
