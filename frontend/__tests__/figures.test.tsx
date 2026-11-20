import { afterEach, expect, it } from "vitest";
import "@testing-library/jest-dom/vitest";
import { cleanup, render, screen, fireEvent } from "@testing-library/react";
import {
  ModelFigures,
  ReliabilityPlot,
  SensitivityPlot,
} from "@/components/research/ResearchFigures";

afterEach(cleanup);

const run = (model: string, variant: string, seed: number, f1: number) => ({
  key: `${model}-${variant}-${seed}`,
  model,
  variant,
  seed,
  f1,
  accuracy: f1,
  curves: [
    { step: 1, trainingLoss: 0.6, validationLoss: 0.5 },
    { step: 2, trainingLoss: 0.4, validationLoss: 0.5 },
  ],
});

it("pairs only matching seeds and marks the first minimum validation epoch", () => {
  render(
    <ModelFigures
      runs={[
        run("Transformer", "reference", 0, 0.6),
        run("Transformer", "no-position", 0, 0.5),
        run("Transformer", "reference", 1, 0.4),
        run("MLP", "reference", 0, 0.45),
      ]}
      seed={0}
      onSeed={() => {}}
    />,
  );
  expect(
    screen.getByRole("img", { name: "Paired Transformer ablation" }),
  ).toBeVisible();
  expect(
    screen.getByText(
      /One paired seed · mean reference minus ablation 0.100 · 1 seed/,
    ),
  ).toBeVisible();
  expect(screen.getAllByText(/Retained checkpoint: epoch 1/)).toHaveLength(2);
});

it("bins probabilities at endpoints and links the selected observation", () => {
  let inspected = -1;
  render(
    <ReliabilityPlot
      samples={[
        { index: 1, p: 0, up: false },
        { index: 2, p: 1, up: true },
        { index: 3, p: 0.81, up: false },
      ]}
      inspect={(i) => {
        inspected = i;
      }}
    />,
  );
  expect(screen.getByRole("button", { name: "80–100% · 2" })).toBeVisible();
  fireEvent.click(screen.getByRole("button", { name: "80–100% · 2" }));
  fireEvent.change(screen.getByLabelText("Inspect case in bin"), {
    target: { value: "3" },
  });
  expect(inspected).toBe(3);
});

it("converts original-minus-masked sensitivity to signed percentage points", () => {
  render(<SensitivityPlot values={[0.08, -0.03, 0, 0, 0, 0, 0, 0]} />);
  expect(screen.getByText("-8.0", { exact: true })).toBeVisible();
  expect(screen.getByText("+3.0", { exact: true })).toBeVisible();
});
