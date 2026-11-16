import { expect, test } from "@playwright/test";

const backend = "http://localhost:8100";

test("synthetic replay: upload, start, order, step, portfolio, complete", async ({
  page, request,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  const docs = await request.get(`${backend}/docs`);
  expect(docs.status()).toBe(200);
  expect(await docs.text()).toContain("swagger-ui");

  const instrument = `SYNTH-${Date.now()}`;
  const csv = [
    "date,open,high,low,close,volume",
    ...Array.from({ length: 8 }, (_, i) =>
      `2023-01-${String(i + 1).padStart(2, "0")},${100 + i},${105 + i},${98 + i},${102 + i},1000`,
    ),
  ].join("\n");
  const uploaded = await request.post(`${backend}/api/market/upload`, {
    params: { instrument },
    multipart: {
      file: { name: "synthetic.csv", mimeType: "text/csv", buffer: Buffer.from(csv) },
    },
  });
  expect(uploaded.status()).toBe(201);
  expect((await uploaded.json()).loaded_count).toBe(8);
  expect((await request.get(`${backend}/api/market/bars/${instrument}`)).status()).toBe(404);

  await page.goto("/replay");
  await expect(page.getByLabel("Instrument", { exact: true })).toBeVisible();
  await page.getByLabel("Instrument", { exact: true }).selectOption(instrument);
  await page.getByLabel("Fee per unit", { exact: true }).fill("0");
  await page.getByLabel("Synthetic spread", { exact: true }).fill("0");
  const created = page.waitForResponse((response) =>
    response.url().endsWith("/api/replay/sessions") &&
    response.request().method() === "POST",
  );
  await page.getByRole("button", { name: "Start session", exact: true }).click();
  const sessionResponse = await created;
  expect(sessionResponse.status()).toBe(201);
  const sid: string = (await sessionResponse.json()).session_id;
  const sessionUrl = `${backend}/api/replay/sessions/${sid}`;
  expect((await (await request.get(`${sessionUrl}/visible-bars`)).json()).length).toBe(1);

  await expect(page.locator("canvas").first()).toBeVisible();
  await page.getByPlaceholder("rationale (required)…").fill("Synthetic execution check");
  await page.getByRole("button", { name: "Submit", exact: true }).click();
  await expect(page.getByText("pending", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Step forward 1 bar", exact: true }).click();
  await expect(page.getByText("filled", { exact: true })).toBeVisible();
  const visible = await (await request.get(`${sessionUrl}/visible-bars`)).json();
  expect(visible).toHaveLength(2);
  expect(visible[1].date).toBe("2023-01-02");
  const snapshot = await (await request.get(`${backend}/api/portfolio/${sid}`)).json();
  expect(snapshot.cash).toBe(99899);
  expect(snapshot.equity).toBe(100002);
  expect(snapshot.positions[0].quantity).toBe(1);
  const risk = await (await request.get(`${backend}/api/risk/${sid}`)).json();
  expect(risk.max_drawdown_pct).toBe(0);
  expect(risk.max_position_exposure_pct).toBeGreaterThan(0);

  await page.getByPlaceholder("rationale (required)…").fill("Pending at completion");
  await page.getByRole("button", { name: "Submit", exact: true }).click();
  await expect(page.getByText("pending", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Complete", exact: true }).click();
  await page.getByRole("dialog").getByRole("button", { name: "Complete session", exact: true }).click();
  await expect(page.getByText("completed", { exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: "Step forward 1 bar", exact: true })).toBeDisabled();
  const orders = await (await request.get(`${backend}/api/orders?session_id=${sid}`)).json();
  expect(orders.map((order: { status: string }) => order.status).sort()).toEqual(["cancelled", "filled"]);
  const finished = await (await request.get(`${backend}/api/portfolio/${sid}`)).json();
  expect(finished.positions[0].quantity).toBe(1);
  expect(errors).toEqual([]);
});
