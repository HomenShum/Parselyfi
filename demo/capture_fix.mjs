// Targeted re-capture: News briefing result + Transcription summary.
import { chromium } from "playwright";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const OUT = join(__dirname, "recordings");
const BASE = process.env.DEMO_URL || "http://127.0.0.1:8502";

const clickTab = async (page, text) => {
  await page.locator('[data-baseweb="tab"]', { hasText: text }).first().click();
  await page.waitForTimeout(1000);
};
const notRunning = async (page, t = 220000) =>
  page.waitForFunction(() => !document.querySelector('[data-testid="stStatusWidget"]'),
    { timeout: t, polling: 1000 }).catch(() => {});
const scrollToText = async (page, re) => {
  await page.evaluate((src) => {
    const rx = new RegExp(src);
    const el = [...document.querySelectorAll("h1,h2,h3,h4")].find((n) => rx.test(n.textContent || ""));
    if (el) el.scrollIntoView({ block: "start" });
    window.scrollBy(0, -24);
  }, re);
  await page.waitForTimeout(700);
};

const run = async () => {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1280, height: 800 }, deviceScaleFactor: 2 });
  page.setDefaultTimeout(60000);
  await page.goto(BASE, { waitUntil: "networkidle" });
  await page.waitForTimeout(1500);

  // ---- NEWS ----
  try {
    await clickTab(page, "News & YouTube");
    await page.waitForTimeout(800);
    const inp = page.locator('[data-testid="stTextInput"] input').first();
    await inp.click(); await inp.fill("NVIDIA quarterly earnings"); await inp.press("Enter");
    await page.waitForTimeout(400);
    await page.getByRole("button", { name: /Generate News Report/i }).first().click();
    await page.waitForFunction(
      () => /Report for:|Briefing/.test(document.body.innerText),
      { timeout: 220000, polling: 1500 }).catch(() => {});
    await notRunning(page);
    await page.waitForTimeout(1500);
    await scrollToText(page, "Report for:|Briefing");
    await page.screenshot({ path: join(OUT, "s04_news_brief.png") });
    console.log("news Report-for present:", await page.evaluate(() => /Report for:/.test(document.body.innerText)));
  } catch (e) { console.log("news err", e.message); }

  // ---- TRANSCRIPTION ----
  try {
    await clickTab(page, "Transcription");
    await page.waitForTimeout(900);
    const inputs = page.locator('[data-testid="stFileUploader"] input[type="file"]');
    const n = await inputs.count();
    console.log("file inputs:", n);
    await inputs.last().setInputFiles(join(__dirname, "fixtures", "sample_transcript.json"));
    await notRunning(page);
    await page.waitForTimeout(2000);
    await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight * 0.45));
    await page.waitForTimeout(500);
    await page.screenshot({ path: join(OUT, "s06_transcribe_loaded.png") });
    const sum = page.getByRole("button", { name: /Summari/i }).first();
    const has = await sum.count();
    console.log("summarize button:", has);
    if (has) {
      await sum.click();
      // Wait for the FINISHED summary (Download button only appears post-summary).
      await page.waitForFunction(
        () => /Download Summary|Summary generated/i.test(document.body.innerText),
        { timeout: 150000, polling: 1500 }).catch(() => {});
      await notRunning(page);
      await page.waitForTimeout(1800);
      await scrollToText(page, "Summary & Key Points");
      await page.screenshot({ path: join(OUT, "s07_transcribe_summary.png") });
      console.log("summary present:", await page.evaluate(() => /Key points|Action items|Summary/i.test(document.body.innerText)));
    }
  } catch (e) { console.log("transcribe err", e.message); }

  await browser.close();
  console.log("FIX_DONE");
};
run().catch((e) => { console.error(e); process.exit(1); });
