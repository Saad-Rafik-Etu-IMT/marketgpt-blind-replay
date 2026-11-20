import { expect, test } from "@playwright/test";

test("research laboratory runs a real synthetic campaign and preserves the selected evidence", async ({
  page,
}, testInfo) => {
  test.setTimeout(240_000);
  await page.goto("/");
  await expect(
    page.getByRole("heading", { name: "Research laboratory", exact: true }),
  ).toBeVisible();
  const tabs = page.getByRole("navigation", { name: "Laboratories" });
  await tabs.getByRole("link", { name: "Data", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "Point-in-time feature explorer" }),
  ).toBeVisible();
  await expect(
    page.getByRole("img", { name: /rsi_14.*development history/ }),
  ).toBeVisible();
  await page
    .getByRole("button", { name: "Zoom in Aligned underlying market history" })
    .click();
  await page
    .getByRole("button", { name: "Fit / reset", exact: true })
    .first()
    .click();
  await tabs.getByRole("link", { name: "Training", exact: true }).click();
  await page.getByLabel("Campaign name").fill("E2E laboratory");
  for (let i = 0; i < 3; i++)
    await page.getByRole("button", { name: "Continue", exact: true }).click();
  await page.getByLabel("Maximum epochs").fill("1");
  await page.getByLabel("Automatic evaluation").selectOption("no");
  await page.getByRole("button", { name: "Continue", exact: true }).click();
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
  await tabs.getByRole("link", { name: "Evaluation", exact: true }).click();
  await page
    .getByRole("button", { name: "Generate evaluation report" })
    .click();
  await expect(
    page.getByRole("heading", { name: "Feature ablation" }),
  ).toBeVisible({ timeout: 120_000 });
  await expect(
    page.getByRole("img", { name: /Model versus statistical baselines/ }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Open prediction explorer" }).click();
  await expect(
    page.getByRole("heading", { name: /Confusion matrix · actual/ }),
  ).toBeVisible({ timeout: 30_000 });
  await page.getByRole("button", { name: /False positive/ }).click();
  await page.getByRole("button", { name: "All prediction cases" }).click();
  await page
    .getByRole("button", { name: /^\d{4}-\d{2}-\d{2}$/ })
    .first()
    .click();
  await expect(
    page.getByRole("img", {
      name: /Sequential feature contribution waterfall/,
    }),
  ).toBeVisible({ timeout: 30_000 });
  await page.getByRole("button", { name: "Measure counterfactual" }).click();
  await expect(page.getByText(/Measured probability:/)).toBeVisible({
    timeout: 30_000,
  });
  await tabs.getByRole("link", { name: "Explainability", exact: true }).click();
  await expect(
    page.getByRole("img", { name: /Feature sensitivity/ }),
  ).toBeVisible();
  await tabs.getByRole("link", { name: "Replay", exact: true }).click();
  await page.getByRole("button", { name: "Generate replay" }).click();
  await expect(
    page.getByRole("heading", { name: "Current blind decision" }),
  ).toBeVisible({ timeout: 120_000 });
  await page.getByRole("button", { name: "Reveal next session" }).click();
  await expect(page.getByRole("img", { name: /Marked equity/ })).toBeVisible();
  await page.getByRole("button", { name: "Measured facts" }).click();
  await expect(page.getByText(/fixed threshold policy selected/)).toBeVisible();
  const png = page.waitForEvent("download");
  await page
    .getByRole("button", { name: /Export Market candles.* as PNG/ })
    .click();
  expect((await png).suggestedFilename()).toMatch(/\.png$/);
  const svg = page.waitForEvent("download");
  await page
    .getByRole("button", { name: /Export Marked equity.* as SVG/ })
    .click();
  expect((await svg).suggestedFilename()).toMatch(/\.svg$/);
  await page.evaluate(() => window.scrollTo(0, 0));
  const screenshot = testInfo.outputPath("laboratory.png");
  await page.screenshot({ path: screenshot, fullPage: true });
  await testInfo.attach("Laboratory replay", {
    path: screenshot,
    contentType: "image/png",
  });
  await expect(page).toHaveURL(/cursor=1/);
  await page.reload();
  await expect(page.getByText(/Revealed decisions: 1\//)).toBeVisible();
  await page.getByRole("button", { name: "Play", exact: true }).click();
  await expect(page.getByText(/Revealed decisions: [2-9]\//)).toBeVisible();
  await page.getByRole("button", { name: "Pause", exact: true }).click();
  await tabs.getByRole("link", { name: /Presentation/ }).click();
  await expect(
    page.getByRole("heading", { name: "Research question", exact: true }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Results", exact: true }).click();
  await expect(page.getByLabel("Principal scientific finding")).toBeVisible();
  await page.screenshot({
    path: testInfo.outputPath("presentation.png"),
    fullPage: true,
  });
  await page.getByRole("button", { name: "Exit presentation" }).click();
  await expect(
    page.getByRole("heading", { name: "Scientific evaluation" }),
  ).toBeVisible();
  await page.setViewportSize({ width: 390, height: 844 });
  await page.getByRole("button", { name: "Switch to dark theme" }).click();
  await page.screenshot({
    path: testInfo.outputPath("laboratory-mobile.png"),
    fullPage: true,
  });
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth),
  ).toBeLessThanOrEqual(390);
});
