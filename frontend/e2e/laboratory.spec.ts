import { expect, test } from "@playwright/test";

test("research laboratory runs a real synthetic campaign and preserves the selected evidence", async ({
  page,
}, testInfo) => {
  test.setTimeout(120_000);
  await page.goto("/");
  await expect(
    page.getByRole("heading", { name: "Research laboratory", exact: true }),
  ).toBeVisible();
  const tabs = page.getByRole("navigation", { name: "Laboratories" });
  await tabs.getByRole("button", { name: "Data", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "Point-in-time feature explorer" }),
  ).toBeVisible();
  await expect(
    page.getByRole("img", { name: /rsi_14.*development history/ }),
  ).toBeVisible();
  await tabs.getByRole("button", { name: "Training", exact: true }).click();
  await page.getByLabel("Campaign name").fill("E2E laboratory");
  await page.getByLabel("Maximum epochs").fill("1");
  await page.getByLabel("Automatic evaluation").selectOption("no");
  await page.getByRole("button", { name: "Estimate campaign" }).click();
  await expect(
    page.getByRole("button", { name: "Queue campaign" }),
  ).toBeEnabled();
  await page.getByRole("button", { name: "Queue campaign" }).click();
  await expect(
    page.getByRole("heading", { name: "Campaign monitor" }),
  ).toBeVisible();
  await expect(page.getByText("1/1", { exact: true })).toBeVisible({
    timeout: 60_000,
  });
  await expect(page.getByRole("img", { name: /Convergence/ })).toBeVisible();
  await tabs.getByRole("button", { name: "Evaluation", exact: true }).click();
  await page
    .getByRole("button", { name: "Generate evaluation report" })
    .click();
  await expect(
    page.getByRole("heading", { name: "Feature ablation" }),
  ).toBeVisible({ timeout: 60_000 });
  await expect(
    page.getByRole("img", { name: /Model versus statistical baselines/ }),
  ).toBeVisible();
  await tabs
    .getByRole("button", { name: "Explainability", exact: true })
    .click();
  await expect(
    page.getByRole("img", { name: /Feature sensitivity/ }),
  ).toBeVisible();
  await tabs.getByRole("button", { name: "Replay", exact: true }).click();
  await page.getByRole("button", { name: "Generate replay" }).click();
  await expect(
    page.getByRole("heading", { name: "Current blind decision" }),
  ).toBeVisible({ timeout: 60_000 });
  await page.getByRole("button", { name: "Reveal next session" }).click();
  await expect(page.getByRole("img", { name: /Marked equity/ })).toBeVisible();
  await page.getByRole("button", { name: "Measured facts" }).click();
  await expect(page.getByText(/fixed threshold policy selected/)).toBeVisible();
  const screenshot = testInfo.outputPath("laboratory.png");
  await page.screenshot({ path: screenshot, fullPage: true });
  await testInfo.attach("Laboratory replay", {
    path: screenshot,
    contentType: "image/png",
  });
});
