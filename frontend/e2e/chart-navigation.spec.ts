import { expect, test } from "@playwright/test";

test("wheel zoom and drag synchronize aligned chart windows", async ({
  page,
}) => {
  await page.goto("/laboratory?tab=Data");
  const market = page.getByRole("img", {
    name: "Aligned underlying market history",
  });
  const feature = page.getByRole("img", {
    name: /rsi_14.*development history/,
  });
  await expect(market).toBeVisible();
  await expect(feature).toBeVisible();
  const surface = market.locator(".chart-surface"),
    other = feature.locator(".chart-surface");
  await surface.scrollIntoViewIfNeeded();
  const before = await surface.getAttribute("data-range-start");
  const box = await surface.boundingBox();
  if (!box) throw Error("Chart plot surface missing");
  const x = box.x + box.width / 2,
    y = box.y + box.height / 2;
  await page.mouse.move(x, y);
  await page.mouse.wheel(0, -250);
  await expect
    .poll(() => surface.getAttribute("data-range-start"))
    .not.toBe(before);
  const zoomed = await surface.getAttribute("data-range-start");
  await expect(other).toHaveAttribute("data-range-start", zoomed!);
  await page.mouse.move(x, y);
  await page.mouse.down();
  await page.mouse.move(x + box.width * 0.12, y, { steps: 4 });
  await page.mouse.up();
  await expect
    .poll(() => surface.getAttribute("data-range-start"))
    .not.toBe(zoomed);
  const yBefore = await surface.getAttribute("data-y-start");
  await page.keyboard.down("Shift");
  await page.mouse.wheel(0, -200);
  await page.keyboard.up("Shift");
  await expect
    .poll(() => surface.getAttribute("data-y-start"))
    .not.toBe(yBefore);
  await page
    .getByRole("button", { name: "Fit / reset", exact: true })
    .first()
    .click();
  await expect(surface).toHaveAttribute("data-range-start", before!);
  await expect(other).toHaveAttribute("data-range-start", before!);
});
