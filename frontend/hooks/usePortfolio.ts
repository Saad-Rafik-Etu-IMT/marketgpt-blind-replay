"use client";

import { useQuery } from "@tanstack/react-query";

import { getEquityCurve, getPortfolio, getRiskMetrics } from "@/lib/api";
import { qk } from "@/lib/queryKeys";

const NONE = "none";

export function usePortfolio(sessionId: string | null) {
  const enabled = !!sessionId;

  const snapshotQuery = useQuery({
    queryKey: sessionId ? qk.portfolio(sessionId) : qk.portfolio(NONE),
    queryFn: () => getPortfolio(sessionId as string),
    enabled,
  });

  const curveQuery = useQuery({
    queryKey: sessionId ? qk.equityCurve(sessionId) : qk.equityCurve(NONE),
    queryFn: () => getEquityCurve(sessionId as string),
    enabled,
  });

  const metricsQuery = useQuery({
    queryKey: sessionId ? qk.riskMetrics(sessionId) : qk.riskMetrics(NONE),
    queryFn: () => getRiskMetrics(sessionId as string),
    enabled,
  });

  return {
    snapshot: snapshotQuery.data ?? null,
    curve: curveQuery.data ?? [],
    metrics: metricsQuery.data ?? null,
  };
}
