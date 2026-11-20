import { expect, test } from "@playwright/test";
import { demoBundle } from "../lib/research/demo";

test("connected research workflow, mode isolation and responsive themes", async ({ page }, testInfo) => {
  const errors: string[] = [];
  page.on("pageerror", error => errors.push(error.message));
  await page.goto("/experiments");
  await expect(page).toHaveURL(/\/experiments$/);
  await expect(page.getByText("No Data · Real mode", { exact: true })).toBeVisible();
  await expect(page.getByText("No research runs loaded")).toBeVisible();
  await expect(page.getByRole("button", { name: "Load synthetic demo" })).toBeEnabled();
  await page.keyboard.press("Tab");
  await expect(page.getByRole("link", { name: "Skip to content" })).toBeFocused();
  await page.getByRole("button", { name: "Load synthetic demo" }).focus();
  await page.keyboard.press("Enter");
  await expect(page.getByText(/Synthetic demo data — not experimental results/)).toBeVisible();
  await expect(page.getByRole("img", { name: /Cross-entropy/ })).toBeVisible();
  await page.getByRole("link", { name: /Inspect prediction/ }).click();
  await expect(page.getByRole("heading", { name: "Model probabilities" })).toBeVisible();
  await expect(page.getByText("62.0%", { exact: true })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Synthetic explanation example" })).toBeVisible();
  await expect(page.getByText("Fact sheet & checks", { exact: true })).toBeVisible();
  await page.getByText("Template baseline", { exact: true }).focus();
  await page.keyboard.press("Enter");
  await expect(page.getByText(/^Predicted class: long/)).toBeVisible();
  await expect(page.getByText(/Synthetic next-bar return/)).toHaveCount(0);
  await page.getByRole("link", { name: /Review saved evaluation/ }).click();
  await expect(page.getByText("Misclassification", { exact: true })).toBeVisible();
  await page.getByRole("link", { name: /demo-sample-064 → Inspector/ }).click();
  await expect(page).toHaveURL(/run=demo-run-01&sample=demo-sample-064/);
  await expect(page.getByRole("heading", { name: "Synthetic explanation example" })).toBeVisible();

  for (const width of [1440, 390]) {
    await page.setViewportSize({ width, height: 1000 });
    for (const theme of ["light", "dark"]) {
      if (await page.locator("html").getAttribute("data-theme") !== theme) {
        await page.getByRole("button", { name: `Switch to ${theme} theme` }).click();
      }
      for (const name of ["Experiments", "Inspector", "Evaluation"]) {
        await page.getByRole("navigation", { name: "Main navigation" }).getByRole("link", { name: new RegExp(name) }).click();
        await expect(page.getByRole("heading", { name, exact: true })).toBeVisible();
        expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
        const path = testInfo.outputPath(`${name.toLowerCase()}-${theme}-${width}.png`);
        await page.screenshot({ path, fullPage: true });
        await testInfo.attach(`${name} ${theme} ${width}`, { path, contentType: "image/png" });
      }
    }
  }
  await page.getByRole("button", { name: "Real mode", exact: true }).click();
  await expect(page.getByText("No evaluation report loaded")).toBeVisible();
  await expect(page.getByText("Synthetic Demo", { exact: true })).toHaveCount(0);
  await page.getByRole("navigation").getByRole("link", { name: /Replay/ }).click();
  await page.getByRole("button", {name:"Manual simulation"}).click();
  await expect(page.getByRole("button", { name: "Start session", exact: true })).toBeVisible();
  expect(errors).toEqual([]);
});

test("browser import stays local, rejects invalid files and clears on reload", async ({ page }) => {
  const uploads: string[] = [];
  page.on("request", request => { if (request.method() !== "GET") uploads.push(request.url()); });
  await page.route("**/api/market/instruments", route => route.fulfill({ status: 503, contentType: "application/json", body: '{"error_code":"UNAVAILABLE","detail":"Offline test"}' }));
  await page.goto("/experiments");
  await expect(page.getByText(/Instrument inventory unavailable/)).toBeVisible();
  await expect(page.getByText("No Data · Real mode", { exact: true })).toBeVisible();
  const picker = page.getByLabel("Import research JSON", { exact: true });
  await picker.setInputFiles({ name: "unsupported.json", mimeType: "application/json", buffer: Buffer.from('{"schemaVersion":2}') });
  await expect(page.getByRole("alert", { name: "Artifact import error" })).toContainText("Unsupported schemaVersion");
  const artifact = structuredClone(demoBundle);
  artifact.origin = "experimental";
  artifact.runs[0].model = "<script>window.INJECTED=true</script>";
  await picker.setInputFiles({ name: "research.json", mimeType: "application/json", buffer: Buffer.from(JSON.stringify(artifact)) });
  await expect(page.getByText("Real · Imported artifacts", { exact: true })).toBeVisible();
  await expect(page.getByText("<script>window.INJECTED=true</script>", { exact: true })).toBeVisible();
  await page.getByRole("link", { name: /Inspect prediction/ }).click();
  await expect(page.getByRole("heading", { name: "Saved explanation", exact: true })).toBeVisible();
  await page.reload();
  await expect(page.getByText("No saved prediction loaded")).toBeVisible();
  expect(uploads).toEqual([]);
});
