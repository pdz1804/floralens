import { test, expect } from "@playwright/test";

const SHOTS = "../../../plans/260709-1427-dual-app-buildout/reports/floralens-ui/screens";

test("galaxy tab renders point cloud", async ({ page }) => {
  await page.goto("/app");
  await page.getByTestId("tab-galaxy").click();
  await expect(page.getByTestId("galaxy-page")).toBeVisible({ timeout: 20_000 });
  const view = page.locator('[data-testid="galaxy-canvas"], [data-testid="galaxy-2d-fallback"]');
  await expect(view.first()).toBeVisible({ timeout: 20_000 });
  await page.waitForTimeout(2500); // let the cloud render
  await page.screenshot({ path: `${SHOTS}/galaxy.png`, fullPage: true });
});
