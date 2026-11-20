export type StudySummary = {
  id: string;
  state: string;
  message: string;
  done: number;
  total: number;
  source?: string | null;
  models?: string[];
  epochs?: number | null;
  hasTrace?: boolean;
};

export function studyLabel(study: StudySummary) {
  const source =
    study.source === "synthetic" ? "Synthetic" : study.source || "Study";
  const models = study.models?.length ? study.models.join(" + ") : "";
  return [
    source,
    models,
    study.state === "complete" ? "Ready" : study.state,
    study.id.slice(0, 8),
  ]
    .filter(Boolean)
    .join(" · ");
}
