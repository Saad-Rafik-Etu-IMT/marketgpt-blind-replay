"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";

import {
  completeSession,
  createSession,
  getSession,
  getVisibleBars,
  jumpToBar,
  stepSession,
} from "@/lib/api";
import { POLL_INTERVAL_MS } from "@/lib/constants";
import { qk } from "@/lib/queryKeys";
import type { CreateSessionRequest } from "@/lib/types";

const NONE = "none";

/**
 * Owns the replay session lifecycle via React Query.
 *
 * The session and its visible bars are queries; create/step/jump/complete
 * are mutations that write the fresh session back into the cache and
 * invalidate every derived query (bars, portfolio, orders). That
 * invalidation replaces the hand-rolled ``*AndRefresh`` glue the cockpit
 * used to carry.
 */
export function useReplaySession() {
  const qc = useQueryClient();
  const [sessionId, setSessionId] = useState<string | null>(null);
  useEffect(() => {
    try { setSessionId(localStorage.getItem("mrtl.activeSession")); } catch { /* storage may be disabled */ }
  }, []);
  const [startError, setStartError] = useState<string | null>(null);

  const sessionQuery = useQuery({
    queryKey: sessionId ? qk.session(sessionId) : qk.session(NONE),
    queryFn: () => getSession(sessionId as string),
    enabled: !!sessionId,
    refetchInterval: POLL_INTERVAL_MS,
  });

  const barsQuery = useQuery({
    queryKey: sessionId ? qk.bars(sessionId) : qk.bars(NONE),
    queryFn: () => getVisibleBars(sessionId as string),
    enabled: !!sessionId,
  });

  const invalidateDerived = (id: string) => {
    for (const key of [
      qk.bars(id),
      qk.portfolio(id),
      qk.equityCurve(id),
      qk.riskMetrics(id),
      qk.orders(id),
    ]) {
      qc.invalidateQueries({ queryKey: key });
    }
  };

  const startMutation = useMutation({
    mutationFn: (request: CreateSessionRequest) => createSession(request),
    onSuccess: (session) => {
      setStartError(null);
      setSessionId(session.session_id);
      try { localStorage.setItem("mrtl.activeSession", session.session_id); } catch { /* optional preference */ }
      qc.setQueryData(qk.session(session.session_id), session);
    },
    onError: (err) =>
      setStartError(
        err instanceof Error ? err.message : "failed to start session",
      ),
  });

  const stepMutation = useMutation({
    mutationFn: (n: number) => stepSession(sessionId as string, n),
    onSuccess: (session) => {
      qc.setQueryData(qk.session(session.session_id), session);
      invalidateDerived(session.session_id);
    },
  });

  const jumpMutation = useMutation({
    mutationFn: (targetIndex: number) =>
      jumpToBar(sessionId as string, targetIndex),
    onSuccess: (session) => {
      qc.setQueryData(qk.session(session.session_id), session);
      invalidateDerived(session.session_id);
    },
  });

  const completeMutation = useMutation({
    mutationFn: () => completeSession(sessionId as string),
    onSuccess: (session) => {
      qc.setQueryData(qk.session(session.session_id), session);
      invalidateDerived(session.session_id);
    },
  });

  return {
    session: sessionQuery.data ?? null,
    bars: barsQuery.data ?? [],
    loading: startMutation.isPending,
    error: startError ?? (stepMutation.error instanceof Error ? stepMutation.error.message : null),
    reset: () => { setSessionId(null); try { localStorage.removeItem("mrtl.activeSession"); } catch {} },
    // The polling session query failing means the backend is unreachable.
    connectionLost: !!sessionId && sessionQuery.isError,
    start: async (request: CreateSessionRequest) => {
      await startMutation.mutateAsync(request).catch(() => undefined);
    },
    step: async (n = 1) => {
      await stepMutation.mutateAsync(n).catch(() => undefined);
    },
    jump: async (targetIndex: number) => {
      await jumpMutation.mutateAsync(targetIndex);
    },
    complete: async () => {
      await completeMutation.mutateAsync();
    },
  };
}

