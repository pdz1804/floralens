import { defineConfig, devices } from "@playwright/test";

// Evidence lands in the shared plan reports folder next to AgentForge's.
const EVIDENCE = "../../../plans/260709-1427-dual-app-buildout/reports/floralens-ui";

export default defineConfig({
  testDir: "./e2e",
  timeout: 90_000,
  fullyParallel: false,
  retries: 0,
  reporter: [
    ["list"],
    ["html", { outputFolder: `${EVIDENCE}/html-report`, open: "never" }],
  ],
  outputDir: `${EVIDENCE}/artifacts`,
  use: {
    baseURL: process.env.WEB_BASE || "http://localhost:3100",
    screenshot: "on",
    video: "retain-on-failure",
    trace: "retain-on-failure",
    // SwiftShader gives headless Chromium a real WebGL context so the 3D galaxy
    // (instanced point cloud) renders with per-instance colors in screenshots.
    launchOptions: { args: ["--use-gl=angle", "--use-angle=swiftshader", "--ignore-gpu-blocklist"] },
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
});
