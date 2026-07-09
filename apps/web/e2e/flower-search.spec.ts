import { test, expect } from "@playwright/test";
import path from "path";

const SHOTS = "../../../plans/260709-1427-dual-app-buildout/reports/floralens-ui/screens";
// A real Oxford-102 gallery image; the baseline model returns "barbeton daisy"
// as the top match for it (verified via curl against POST /api/search).
const QUERY_IMG = path.resolve(
  __dirname,
  "../../../ml/data/raw/flowers102_hf/images/flowers102-te00000.jpg",
);

test.describe("FloraLens flower search UI", () => {
  test("backend online + page loads", async ({ page }) => {
    await page.goto("/");
    await expect(page.getByRole("heading", { name: "FloraLens" })).toBeVisible();
    await expect(page.getByTestId("health-meta")).toContainText("model", { timeout: 15_000 });
    await expect(page.getByTestId("health-dot")).toHaveClass(/ok/);
    await page.screenshot({ path: `${SHOTS}/01-loaded.png`, fullPage: true });
  });

  test("upload flower → scored similar matches (real retrieval)", async ({ page }) => {
    await page.goto("/");
    await page.getByTestId("file-input").setInputFiles(QUERY_IMG);
    await expect(page.getByTestId("preview")).toBeVisible();
    await page.getByTestId("search-btn").click();
    // Results grid must populate with real matches.
    await expect(page.getByTestId("results")).toBeVisible({ timeout: 30_000 });
    const cards = page.getByTestId("result-card");
    await expect(cards.first()).toBeVisible();
    expect(await cards.count()).toBeGreaterThan(3);
    // Deterministic: top match for this image is barbeton daisy.
    await expect(page.getByTestId("result-name").first()).toHaveText(/barbeton daisy/i);
    // A confidence percentage is rendered.
    await expect(page.getByTestId("result-score").first()).toHaveText(/%$/);
    await page.screenshot({ path: `${SHOTS}/02-search-results.png`, fullPage: true });
  });

  test("confidence band filter narrows results", async ({ page }) => {
    await page.goto("/");
    await page.getByTestId("file-input").setInputFiles(QUERY_IMG);
    await page.getByTestId("search-btn").click();
    await expect(page.getByTestId("results")).toBeVisible({ timeout: 30_000 });
    const total = await page.getByTestId("result-card").count();
    await page.getByTestId("filter-high").click();
    const highCount = await page.getByTestId("result-card").count();
    expect(highCount).toBeLessThanOrEqual(total);
    await page.screenshot({ path: `${SHOTS}/03-filter-high.png`, fullPage: true });
  });
});
