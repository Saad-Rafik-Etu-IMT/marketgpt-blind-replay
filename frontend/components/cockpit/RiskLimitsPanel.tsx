"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";

import { Alert, Button, Field, Input, Panel, Skeleton, useToast } from "@/components/ui";
import { getRiskLimits, updateRiskLimits } from "@/lib/api";
import { qk } from "@/lib/queryKeys";
import type { RiskLimitsRead, RiskMetricsRead } from "@/lib/types";

interface Props {
  sessionId: string | null;
  metrics: RiskMetricsRead | null;
}

/**
 * Edit the session's risk limits and surface any live breach.
 *
 * The breach banner compares the current metrics against the configured
 * limits so a drawdown or exposure violation is impossible to miss — the
 * backend already auto-pauses on a drawdown breach, this makes the *why*
 * explicit and lets the user retune limits mid-session.
 */
export function RiskLimitsPanel({ sessionId, metrics }: Props) {
  const qc = useQueryClient();
  const toast = useToast();

  const { data: limits, isLoading } = useQuery({
    queryKey: sessionId ? ["riskLimits", sessionId] : ["riskLimits", "none"],
    queryFn: () => getRiskLimits(sessionId as string),
    enabled: !!sessionId,
  });

  const [form, setForm] = useState<RiskLimitsRead | null>(null);
  useEffect(() => {
    if (limits) setForm(limits);
  }, [limits]);

  const save = useMutation({
    mutationFn: (body: RiskLimitsRead) =>
      updateRiskLimits(sessionId as string, body),
    onSuccess: (updated) => {
      qc.setQueryData(["riskLimits", sessionId], updated);
      qc.invalidateQueries({ queryKey: qk.riskMetrics(sessionId as string) });
      toast.success("Risk limits updated.");
    },
    onError: (e) =>
      toast.error(e instanceof Error ? e.message : "Failed to update limits"),
  });

  const breaches = collectBreaches(metrics, limits ?? null);

  return (
    <Panel title="Risk limits">
      {breaches.length > 0 && (
        <Alert tone="danger" title="Risk limit breached" className="mb-3">
          <ul className="list-inside list-disc">
            {breaches.map((b) => (
              <li key={b}>{b}</li>
            ))}
          </ul>
        </Alert>
      )}

      {isLoading || !form ? (
        <Skeleton className="h-32 w-full" />
      ) : (
        <div className="grid grid-cols-2 gap-3 text-sm">
          <Field label="Max drawdown %">
            <Input
              type="number"
              value={form.max_drawdown_pct}
              onChange={(e) =>
                setForm({ ...form, max_drawdown_pct: Number(e.target.value) })
              }
            />
          </Field>
          <Field label="Max risk/trade %">
            <Input
              type="number"
              value={form.max_risk_per_trade_pct}
              onChange={(e) =>
                setForm({
                  ...form,
                  max_risk_per_trade_pct: Number(e.target.value),
                })
              }
            />
          </Field>
          <Field label="Max open positions">
            <Input
              type="number"
              value={form.max_open_positions}
              onChange={(e) =>
                setForm({ ...form, max_open_positions: Number(e.target.value) })
              }
            />
          </Field>
          <Field label="Max exposure %">
            <Input
              type="number"
              value={form.max_exposure_pct}
              onChange={(e) =>
                setForm({ ...form, max_exposure_pct: Number(e.target.value) })
              }
            />
          </Field>
          <div className="col-span-2">
            <Button
              variant="primary"
              size="sm"
              disabled={save.isPending}
              onClick={() => form && save.mutate(form)}
            >
              {save.isPending ? "Saving…" : "Save limits"}
            </Button>
          </div>
        </div>
      )}
    </Panel>
  );
}

function collectBreaches(
  metrics: RiskMetricsRead | null,
  limits: RiskLimitsRead | null,
): string[] {
  if (!metrics || !limits) return [];
  const out: string[] = [];
  if (metrics.current_drawdown_pct >= limits.max_drawdown_pct) {
    out.push(
      `Drawdown ${metrics.current_drawdown_pct.toFixed(1)}% ≥ limit ${limits.max_drawdown_pct.toFixed(1)}%`,
    );
  }
  if (metrics.max_position_exposure_pct >= limits.max_exposure_pct) {
    out.push(
      `Exposure ${metrics.max_position_exposure_pct.toFixed(1)}% ≥ limit ${limits.max_exposure_pct.toFixed(1)}%`,
    );
  }
  return out;
}
