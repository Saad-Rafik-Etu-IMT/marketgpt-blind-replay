import { afterEach, describe, expect, it, vi } from "vitest";
import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { parseResearchBundle, importResearchFile, MAX_ARTIFACT_BYTES } from "@/lib/research/contract";
import { demoBundle } from "@/lib/research/demo";
import { ResearchProvider, useResearch } from "@/components/research/ResearchProvider";
import { Inspector, Evaluation } from "@/components/research/ResearchPages";
import { TrainingCurve } from "@/components/research/TrainingCurve";

let search = "";
vi.mock("next/navigation", () => ({ useSearchParams: () => new URLSearchParams(search) }));
afterEach(() => { cleanup(); search = ""; });
const copy = () => structuredClone(demoBundle);
function fileWith(text: () => Promise<string>, size = 10): File {
  const file = new File([], "bundle.json");
  Object.defineProperties(file, { text: { value: text }, size: { value: size } });
  return file;
}

describe("proposed research contract", () => {
  it("accepts the connected example and nullable metrics", () => {
    const result = parseResearchBundle(copy());
    expect(result.samples[0].runId).toBe(result.runs[0].id);
    expect(result.reports[0].failures[0].sampleId).toBe(result.samples[0].id);
    expect(result.runs[0].curves[2].validationLoss).toBeNull();
  });
  it.each([
    ["unsupported version", (v: Record<string, unknown>) => { v.schemaVersion = 2; }, /schemaVersion: 1/],
    ["excess runs", (v: Record<string, unknown>) => { v.runs = Array(21).fill(demoBundle.runs[0]); }, /at most 20/],
  ])("rejects %s", (_, mutate, message) => {
    const v = { ...copy() };
    mutate(v);
    expect(() => parseResearchBundle(v)).toThrow(message);
  });
  it.each([NaN, Infinity, -0.1, 1.1, 0.5])("rejects invalid probability %s", value => {
    const v = copy(); v.samples[0].probabilities.long = value;
    expect(() => parseResearchBundle(v)).toThrow(/probabilities/);
  });
  it("rejects non-finite inputs, duplicate IDs and oversized nested collections", () => {
    const v = copy(); v.samples[0].inputs[0].value = Infinity;
    expect(() => parseResearchBundle(v)).toThrow(/finite/);
    const duplicate = copy(); duplicate.runs.push(duplicate.runs[0]);
    expect(() => parseResearchBundle(duplicate)).toThrow(/duplicate/);
    const large = copy(); large.samples[0].inputs = Array(513).fill(large.samples[0].inputs[0]);
    expect(() => parseResearchBundle(large)).toThrow(/at most 512/);
  });
  it("rejects unknown and cross-run references", () => {
    const v = copy(); v.samples[0].runId = "missing";
    expect(() => parseResearchBundle(v)).toThrow(/unknown runId/);
    const cross = copy(); cross.runs.push({ ...cross.runs[0], id: "other-run" }); cross.reports[0].runId = "other-run";
    expect(() => parseResearchBundle(cross)).toThrow(/different run/);
    const fact = copy(); fact.samples[0].explanation!.references = ["unknown"];
    expect(() => parseResearchBundle(fact)).toThrow(/unknown fact/);
  });
  it("rejects future inputs, future citations and outcomes beyond cutoff", () => {
    const v = copy(); v.samples[0].inputs[0].bar = 65;
    expect(() => parseResearchBundle(v)).toThrow(/future bars/);
    const fact = copy(); fact.samples[0].explanation!.facts[0].bar = 65;
    expect(() => parseResearchBundle(fact)).toThrow(/visible input/);
    const report = copy(); report.reports[0].cutoffBar = 64;
    expect(() => parseResearchBundle(report)).toThrow(/cutoff/);
  });
  it("checks size before reading and rejects malformed JSON", async () => {
    const read = vi.fn(async () => "{}");
    await expect(importResearchFile(fileWith(read, MAX_ARTIFACT_BYTES + 1))).rejects.toThrow(/2 MiB/);
    expect(read).not.toHaveBeenCalled();
    await expect(importResearchFile(fileWith(async () => "{"))).rejects.toThrow(/Invalid JSON/);
  });
});

function Controls({ file }: { file?: File }) {
  const { bundle, error, useDemo, useReal, importFile } = useResearch();
  return <><button onClick={useDemo}>Demo</button><button onClick={useReal}>Real</button>
    <button onClick={() => file && void importFile(file)}>Import</button>
    <output>{bundle?.runs[0]?.id ?? "No bundle"}</output>{error && <p role="alert">{error}</p>}</>;
}
describe("research source and sample isolation", () => {
  it("defaults to no bundle and never substitutes demo after import failure", async () => {
    render(<ResearchProvider><Controls file={fileWith(async () => "{}")} /></ResearchProvider>);
    expect(screen.getByText("No bundle")).toBeTruthy();
    fireEvent.click(screen.getByText("Demo"));
    expect(screen.getByText("demo-run-01")).toBeTruthy();
    fireEvent.click(screen.getByText("Import"));
    await screen.findByRole("alert");
    expect(screen.getByText("No bundle")).toBeTruthy();
  });
  it("ignores an old import when mode changes", async () => {
    let resolve!: (value: string) => void;
    const read = new Promise<string>(done => { resolve = done; });
    render(<ResearchProvider><Controls file={fileWith(() => read)} /></ResearchProvider>);
    fireEvent.click(screen.getByText("Import"));
    fireEvent.click(screen.getByText("Real"));
    await act(async () => { resolve(JSON.stringify(demoBundle)); await read; });
    expect(screen.getByText("No bundle")).toBeTruthy();
    expect(screen.queryByText("demo-run-01")).toBeNull();
  });
  it("does not show an explanation under an invalid sample or mismatched run", () => {
    search = "?run=other&sample=demo-sample-064";
    render(<ResearchProvider><Controls /><Inspector /></ResearchProvider>);
    fireEvent.click(screen.getByText("Demo"));
    expect(screen.getByRole("alert").textContent).toMatch(/incompatible/);
    expect(screen.queryByText("Synthetic explanation example")).toBeNull();
  });
  it("renders missing explanation honestly and imported markup as text", async () => {
    const v = copy(); v.samples[0].explanation = null; v.runs[0].model = "<img src=x onerror=alert(1)>";
    render(<ResearchProvider><Controls file={fileWith(async () => JSON.stringify(v))} /><Inspector /></ResearchProvider>);
    fireEvent.click(screen.getByText("Import"));
    await screen.findByText(/Explanation unavailable/);
    expect(screen.queryByRole("img")).toBeNull();
    expect(screen.getByText(/<img src=x/)).toBeTruthy();
  });
  it("keeps failure links associated with the saved sample and missing metrics unavailable", () => {
    render(<ResearchProvider><Controls /><Evaluation /></ResearchProvider>);
    fireEvent.click(screen.getByText("Demo"));
    expect(screen.getByRole("link", { name: /demo-sample-064/ }).getAttribute("href")).toBe("/inspector?run=demo-run-01&sample=demo-sample-064");
    expect(screen.getByText("Unavailable")).toBeTruthy();
  });
  it("gives missing losses an accessible unavailable value", () => {
    render(<TrainingCurve run={demoBundle.runs[0]} />);
    expect(screen.getByText("Unavailable")).toBeTruthy();
    expect(screen.getByRole("img").getAttribute("aria-label")).toMatch(/Training|training/);
  });
});
