"use client";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Suspense, useState } from "react";
import { ResearchProvider } from "@/components/research/ResearchProvider";
import { AppShell } from "@/components/research/AppShell";

import { ToastProvider } from "@/components/ui";
import { POLL_INTERVAL_MS } from "@/lib/constants";

/**
 * App-wide React Query provider.
 *
 * A single QueryClient is created once per browser session (via lazy
 * useState init so it survives re-renders but is never shared across
 * requests). Sensible defaults for this local-only app: no refetch on
 * window focus, one retry, and a short stale time so event-driven
 * invalidations (order fills, steps) win over background refetches.
 */
export function Providers({ children }: { children: React.ReactNode }) {
  const [client] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {
            refetchOnWindowFocus: false,
            retry: 1,
            staleTime: POLL_INTERVAL_MS,
          },
        },
      }),
  );

  return (
    <QueryClientProvider client={client}>
      <ToastProvider>
        <ResearchProvider>
          <Suspense fallback={<p role="status">Loading research cockpit…</p>}>
            <AppShell>{children}</AppShell>
          </Suspense>
        </ResearchProvider>
      </ToastProvider>
    </QueryClientProvider>
  );
}
