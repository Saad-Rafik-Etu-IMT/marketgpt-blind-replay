/**
 * Centralised React Query keys.
 *
 * All hooks share these builders so a mutation in one hook can invalidate
 * the queries owned by another (e.g. stepping the replay invalidates the
 * portfolio and orders queries). Keys are prefix-structured.
 */
export const qk = {
  sessions: () => ["sessions"] as const,
  session: (id: string) => ["session", id] as const,
  bars: (id: string) => ["bars", id] as const,
  orders: (id: string) => ["orders", id] as const,
  portfolio: (id: string) => ["portfolio", id] as const,
  equityCurve: (id: string) => ["equityCurve", id] as const,
  riskMetrics: (id: string) => ["riskMetrics", id] as const,
};
