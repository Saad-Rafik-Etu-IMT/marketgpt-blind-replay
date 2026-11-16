import { defineConfig, devices } from "@playwright/test";
import path from "node:path";

const backendUrl = "http://localhost:8100";
const frontendUrl = "http://localhost:3100";
const python = process.env.MRTL_PYTHON ?? path.resolve(
  "..", ".venv", process.platform === "win32" ? "Scripts/python.exe" : "bin/python",
);

export default defineConfig({
  testDir: "./e2e",
  timeout: 60_000,
  fullyParallel: false,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  workers: 1,
  reporter: process.env.CI ? "github" : "list",
  use: {
    baseURL: frontendUrl,
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: [
    {
      command: `"${python}" -m uvicorn src.main:app --port 8100`,
      cwd: "../backend",
      url: `${backendUrl}/api/health`,
      reuseExistingServer: false,
      timeout: 120_000,
      env: {
        MRTL_DATABASE_URL: `sqlite:///./data/db/e2e-${Date.now()}-${process.pid}.sqlite`,
        MRTL_FRONTEND_ORIGIN: frontendUrl,
      },
    },
    {
      command: "npm run dev -- -p 3100",
      url: frontendUrl,
      reuseExistingServer: false,
      timeout: 120_000,
      env: {
        NEXT_PUBLIC_API_BASE_URL: backendUrl,
        NEXT_TELEMETRY_DISABLED: "1",
      },
    },
  ],
});
