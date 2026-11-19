import { expect, test } from "@playwright/test";

test("blind comparison reveals and resets each case and clears rejected imports", async ({page}) => {
  await page.goto("/comparison");
  await page.getByRole("button", {name:"Open trained synthetic demo"}).click();
  const outcome = page.getByRole("heading", {name:"Next session",exact:true});
  await expect(outcome).toHaveCount(0);
  await page.getByRole("button", {name:"Up",exact:true}).click();
  await page.getByRole("button", {name:"Reveal next session"}).click();
  await expect(outcome).toBeVisible();
  await page.getByRole("button", {name:"Next case"}).click();
  await expect(outcome).toHaveCount(0);
  await page.getByText("Review full validation results", {exact:true}).click();
  await expect(page.getByRole("heading",{name:"Calibration"})).toBeVisible();
  await page.getByLabel("Open validation JSON").setInputFiles({name:"bad.json",mimeType:"application/json",buffer:Buffer.from("{}")});
  await expect(page.getByRole("alert")).toBeVisible();
  await expect(page.getByText("Synthetic market · trained MLP")).toHaveCount(0);
});
