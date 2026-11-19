import { expect, test } from "@playwright/test";

test("a saved study shares one run across training, errors, blind replay and portfolio", async ({page})=>{
  test.setTimeout(180000);
  await page.goto("/studies");
  const launcher=page.getByText("New validation study",{exact:true});
  await launcher.waitFor();
  if(!await page.getByLabel("Epochs").isVisible()) await launcher.click();
  await page.getByLabel("Epochs").fill("1");
  await page.getByRole("button",{name:"Start training"}).click();
  await expect(page.getByRole("heading",{name:"Model and ablation comparison"})).toBeVisible({timeout:140000});
  await page.reload();
  await page.getByRole("button",{name:"Evaluation",exact:true}).click();
  await expect(page.getByText("Confusion matrix")).toBeVisible();
  await page.getByLabel("Inspect prediction").selectOption({index:1});
  await expect(page.getByText("Occlusion sensitivity",{exact:true})).toBeVisible();
  await page.getByRole("button",{name:"Explain from measured facts"}).click();
  await expect(page.getByText(/estimated upward probability was/i)).toBeVisible();
  await page.getByRole("button",{name:"Blind replay",exact:true}).click();
  await page.getByRole("button",{name:"Draw blind case"}).click();
  await page.getByRole("button",{name:"Vote up",exact:true}).click();
  await expect(page.getByRole("heading",{name:/Revealed/})).toBeVisible();
  await page.getByRole("button",{name:"Portfolio",exact:true}).click();
  await expect(page.getByRole("img",{name:"Marked equity"}).first()).toBeVisible();
});
