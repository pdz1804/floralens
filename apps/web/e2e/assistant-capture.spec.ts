import { test, expect } from "@playwright/test";

const SHOTS = "../../../plans/260709-1427-dual-app-buildout/reports/floralens-ui/screens";

test("naturalist assistant tab — ask + streamed answer", async ({ page }) => {
  await page.goto("/app");
  await page.getByTestId("tab-assistant").click();
  await expect(page.getByTestId("assistant-page")).toBeVisible({ timeout: 15_000 });
  await page.getByTestId("assistant-input").fill("How do I care for roses? Cite a source.");
  await page.getByTestId("assistant-send").click();
  await expect(page.getByTestId("assistant-answer").first()).toContainText(/water|prune|rose/i, {
    timeout: 80_000,
  });
  await page.screenshot({ path: `${SHOTS}/assistant.png`, fullPage: true });
});
