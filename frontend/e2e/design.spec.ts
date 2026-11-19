import { expect, test } from "@playwright/test";

for (const [device, width] of [["desktop", 1440], ["phone", 390]] as const) {
  for (const theme of ["light", "dark"] as const) {
    test(`cockpit review · ${device} · ${theme}`, async ({ page }, testInfo) => {
      await page.setViewportSize({ width, height: 900 });
      await page.emulateMedia({ colorScheme: theme });
      await page.goto("/experiments");
      await expect(page.locator("html")).toHaveAttribute("data-theme", theme);
      const capture = async (screen: string) => {
        expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(width);
        await page.screenshot({ path: testInfo.outputPath(`${screen}-${device}-${theme}.png`) });
      };

      await expect(page.getByRole("region", { name: "Explore a blind replay" })).toBeVisible();
      await capture("experiments");
      await page.getByRole("link", { name: "Open blind comparison" }).click();
      await page.getByRole("button", { name: "Open trained synthetic demo" }).click();
      const chart = page.getByRole("img", { name: /Relative adjusted closes/ });
      await chart.scrollIntoViewIfNeeded();
      await expect(chart).toBeVisible();
      await capture("comparison");
    });
  }
}
