import { test, expect } from "@playwright/test";

const SHOTS = "../../../plans/260709-1427-dual-app-buildout/reports/floralens-ui/screens";

test("theme toggle + tabs + about render", async ({ page }) => {
  await page.goto("/app");
  await page.screenshot({ path: `${SHOTS}/theme-light.png`, fullPage: true });

  // Cycle theme to dark and verify it applied.
  const toggle = page.getByTestId("theme-toggle");
  await expect(toggle).toBeVisible();
  for (let i = 0; i < 3; i++) {
    const t = await page.evaluate(() => document.documentElement.getAttribute("data-theme"));
    if (t === "dark") break;
    await toggle.click();
  }
  await expect
    .poll(() => page.evaluate(() => document.documentElement.getAttribute("data-theme")))
    .toBe("dark");
  await page.screenshot({ path: `${SHOTS}/theme-dark.png`, fullPage: true });

  await page.getByTestId("tab-about").click();
  await expect(page.getByTestId("about-page")).toBeVisible();
  await page.screenshot({ path: `${SHOTS}/about-page.png`, fullPage: true });

  // Pipeline tab: wait for the real /api/pipeline data to render (not the
  // loading state), then verify live captured metrics are shown.
  await page.getByTestId("tab-pipeline").click();
  await expect(page.getByTestId("pipeline-page")).toBeVisible();
  const pipeline = page.getByTestId("pipeline-page");
  await expect(pipeline).toContainText(/dinov2/i, { timeout: 20_000 }); // backbone
  await expect(pipeline).toContainText(/PROMOTE/i); // promotion decision
  await expect(pipeline).toContainText(/exif|white.?balance|clahe/i); // preprocessing steps
  await page.screenshot({ path: `${SHOTS}/pipeline-page.png`, fullPage: true });

  // Persistence across reload.
  await page.reload();
  await expect
    .poll(() => page.evaluate(() => document.documentElement.getAttribute("data-theme")))
    .toBe("dark");
});
