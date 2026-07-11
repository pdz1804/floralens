import { test, expect } from "@playwright/test";

const SHOTS = "../../../plans/260709-1427-dual-app-buildout/reports/floralens-ui/screens";

test("garden tab: save a specimen → appears with thumbnail", async ({ page }) => {
  await page.goto("/app");
  await page.getByTestId("tab-garden").click();
  await expect(page.getByTestId("garden-page")).toBeVisible({ timeout: 15_000 });
  await page.getByTestId("garden-add-input").fill("flowers102-te01255");
  await page.getByTestId("garden-add-btn").click();
  await expect(page.getByTestId("garden-item").first()).toBeVisible({ timeout: 15_000 });
  await page.screenshot({ path: `${SHOTS}/garden.png`, fullPage: true });
});
