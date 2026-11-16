"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { cancelOrder, listOrders, submitOrder } from "@/lib/api";
import { qk } from "@/lib/queryKeys";
import type { CreateOrderRequest } from "@/lib/types";

const NONE = "none";

export function useOrders(sessionId: string | null) {
  const qc = useQueryClient();

  const ordersQuery = useQuery({
    queryKey: sessionId ? qk.orders(sessionId) : qk.orders(NONE),
    queryFn: () => listOrders(sessionId as string),
    enabled: !!sessionId,
  });

  // A fill changes cash/positions/equity too, so invalidate the portfolio
  // side alongside the order list.
  const invalidate = () => {
    if (!sessionId) return;
    for (const key of [
      qk.orders(sessionId),
      qk.portfolio(sessionId),
      qk.equityCurve(sessionId),
      qk.riskMetrics(sessionId),
    ]) {
      qc.invalidateQueries({ queryKey: key });
    }
  };

  const submitMutation = useMutation({
    mutationFn: (request: CreateOrderRequest) => submitOrder(request),
    onSuccess: invalidate,
  });

  const cancelMutation = useMutation({
    mutationFn: (orderId: string) => cancelOrder(orderId),
    onSuccess: invalidate,
  });

  return {
    orders: ordersQuery.data ?? [],
    error:
      ordersQuery.error instanceof Error ? ordersQuery.error.message : null,
    submit: (request: CreateOrderRequest) =>
      submitMutation.mutateAsync(request),
    cancel: async (orderId: string) => {
      await cancelMutation.mutateAsync(orderId);
    },
  };
}
