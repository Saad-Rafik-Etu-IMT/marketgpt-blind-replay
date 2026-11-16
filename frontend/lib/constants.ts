export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

// Background safety poll. The cockpit refetches portfolio/session state
// eagerly after each user action (step/jump/order/complete), so this slow
// poll only exists to catch backend-driven changes (e.g. auto-pause on a
// drawdown breach). Kept well above 1s to avoid request pile-up.
export const POLL_INTERVAL_MS = 2500;
export const DEFAULT_INITIAL_CASH = 100_000;
