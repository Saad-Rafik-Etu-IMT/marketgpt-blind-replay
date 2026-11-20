import { expect, test } from "@playwright/test";
import { readFile } from "node:fs/promises";

test("a saved study shares one run across training, errors, blind replay and portfolio", async ({
  page,
}, testInfo) => {
  test.setTimeout(180000);
  await page.goto("/studies");
  const launcher = page.getByText("New validation study", { exact: true });
  await launcher.waitFor();
  if (!(await page.getByLabel("Epochs").isVisible())) await launcher.click();
  await page.getByLabel("Epochs").fill("1");
  const started = page.waitForResponse(
    (r) => r.url().endsWith("/api/studies") && r.request().method() === "POST",
  );
  await page.getByRole("button", { name: "Start training" }).click();
  const created = await (await started).json();
  await expect(page).toHaveURL(new RegExp(`study=${created.id}`));
  await expect(
    page.getByRole("heading", { name: "Model and ablation comparison" }),
  ).toBeVisible({ timeout: 140000 });
  await expect(
    page.getByRole("img", { name: "MLP training and validation loss" }),
  ).toBeVisible();
  await page.reload();
  await page.getByRole("button", { name: "Evaluation", exact: true }).click();
  await expect(
    page.getByText("Confusion matrix", { exact: true }),
  ).toBeVisible();
  await expect(
    page.getByRole("img", { name: "Probability reliability diagram" }),
  ).toBeVisible();
  await page.getByLabel("Inspect prediction").selectOption({ index: 1 });
  await expect(
    page.getByText("Occlusion sensitivity", { exact: true }),
  ).toBeVisible();
  await expect(
    page.getByRole("img", {
      name: "Occlusion sensitivity by eight-return block",
    }),
  ).toBeVisible();
  await page
    .getByRole("button", { name: "Explain from measured facts" })
    .click();
  await expect(
    page.getByText(/estimated upward probability was/i),
  ).toBeVisible();
  await page.getByRole("button", { name: "Blind replay", exact: true }).click();
  await page.getByRole("button", { name: "Draw blind case" }).click();
  await page.getByRole("button", { name: "Vote up", exact: true }).click();
  await expect(page.getByRole("heading", { name: /Revealed/ })).toBeVisible();
  await page.getByRole("button", { name: "Portfolio", exact: true }).click();
  await expect(
    page.getByRole("img", { name: "Marked equity" }).first(),
  ).toBeVisible();
  await page.getByRole("button", { name: "Data", exact: true }).click();
  await expect(
    page.getByRole("img", {
      name: "Chronological log returns through validation",
    }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Explanations", exact: true }).click();
  await expect(
    page.getByText("No participant responses collected for this run."),
  ).toBeVisible();
  await page.getByLabel("Import and replace responses (JSON)").setInputFiles({
    name: "test-responses.json",
    mimeType: "application/json",
    buffer: Buffer.from(
      JSON.stringify({
        responses: [
          {
            participant: "Test01",
            case: 1,
            format: "facts",
            source: "Test fact sheet",
            probability_percent: 50,
            block: 1,
          },
          {
            participant: "Test01",
            case: 2,
            format: "llm",
            source: "Test template",
            fallback: true,
            probability_percent: 50,
            block: 1,
          },
        ],
      }),
    ),
  });
  await expect(
    page.getByRole("img", { name: "Paired explanation comprehension scores" }),
  ).toBeVisible();
  await page.getByRole("link", { name: "Watch AI competition" }).click();
  await expect(
    page.getByRole("heading", { name: "AI competition replay" }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Change study", exact: true }).click();
  await expect(page.getByLabel("Saved study")).toBeVisible();
  await page.getByRole("button", { name: "Done", exact: true }).click();
  await expect(page.getByLabel("Saved study")).not.toBeVisible();
  await page.getByRole("button", { name: "Restart", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: /Session 0 \// }),
  ).toBeVisible();
  await expect(
    page.getByRole("img", { name: "Portfolio equity comparison" }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Inspect decision" }).click();
  await expect(
    page.getByText("The next session has not yet been revealed.", {
      exact: false,
    }),
  ).toBeVisible();
  await expect(page.getByLabel("Evidence", { exact: true })).toHaveCount(0);
  await page.getByRole("button", { name: "Advance one session" }).click();
  await expect(
    page.getByRole("heading", { name: /Session 1 \// }),
  ).toBeVisible();
  await page.getByLabel("Evidence", { exact: true }).selectOption("outcome");
  await page
    .getByRole("button", { name: "Explain from measured facts" })
    .click();
  await expect(page.getByText(/Portfolio value changed by/)).toBeVisible();
  await page.screenshot({
    path: testInfo.outputPath("competition-decision.png"),
    fullPage: true,
  });
  await page
    .getByRole("navigation", { name: "Competition views" })
    .getByRole("button", { name: "Performance" })
    .click();
  await expect(
    page.getByRole("img", { name: "Drawdown comparison" }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", { name: /Session 1 \// }),
  ).toBeVisible();
  await page
    .getByRole("navigation", { name: "Competition views" })
    .getByRole("button", { name: "Overview" })
    .click();
  const expand = page.getByRole("button", {
    name: "Expand Portfolio equity comparison",
  });
  await expand.click();
  await expect(
    page.getByRole("dialog", { name: "Portfolio equity comparison" }),
  ).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(
    page.getByRole("dialog", { name: "Portfolio equity comparison" }),
  ).not.toBeVisible();
  await expect(expand).toBeFocused();
  const download = page.waitForEvent("download");
  await page
    .getByRole("button", { name: "Export Portfolio equity comparison as SVG" })
    .click();
  const exported = await download;
  expect(exported.suggestedFilename()).toBe("portfolio-equity-comparison.svg");
  const xml = await readFile((await exported.path())!, "utf8");
  expect(
    await page.evaluate(
      (value) =>
        new DOMParser().parseFromString(value, "image/svg+xml").documentElement
          .localName,
      xml,
    ),
  ).toBe("svg");
  await page.getByRole("slider", { name: "Seek replay session" }).fill("64");
  await expect(
    page.getByRole("heading", { name: /Session 64 \// }),
  ).toBeVisible();
  for (const width of [1440, 390]) {
    await page.setViewportSize({ width, height: 900 });
    expect(
      await page.evaluate(() => document.documentElement.scrollWidth),
    ).toBeLessThanOrEqual(width);
    await page.screenshot({
      path: testInfo.outputPath(`competition-${width}.png`),
      fullPage: true,
    });
  }
  await page.goto("/replay");
  await expect(page).toHaveURL(/study=/);
  await expect(
    page.getByRole("heading", { name: /Session 0 \// }),
  ).toBeVisible();
});
