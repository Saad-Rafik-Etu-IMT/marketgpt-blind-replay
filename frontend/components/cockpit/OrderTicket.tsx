"use client";

import { useState } from "react";

import type {
  CreateOrderRequest,
  OrderSide,
  OrderType,
  ReplaySessionRead,
} from "@/lib/types";
import { useToast } from "@/components/ui";


interface Props {
  session: ReplaySessionRead | null;
  onSubmit: (req: CreateOrderRequest) => Promise<unknown>;
}

export function OrderTicket({ session, onSubmit }: Props) {
  const [side, setSide] = useState<OrderSide>("buy");
  const [orderType, setOrderType] = useState<OrderType>("market");
  const [quantity, setQuantity] = useState(1);
  const [limitPrice, setLimitPrice] = useState<number | "">("");
  const [stopPrice, setStopPrice] = useState<number | "">("");
  const [stopLoss, setStopLoss] = useState<number | "">("");
  const [takeProfit, setTakeProfit] = useState<number | "">("");
  const [rationale, setRationale] = useState("");
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const toast = useToast();

  const disabled = !session || session.status === "completed" || pending;

  const handleSubmit = async () => {
    if (!session) return;
    setError(null);
    if (rationale.trim().length === 0) {
      setError("rationale is mandatory");
      return;
    }
    setPending(true);
    try {
      await onSubmit({
        session_id: session.session_id,
        side,
        order_type: orderType,
        quantity,
        limit_price:
          orderType === "limit" && limitPrice !== "" ? limitPrice : null,
        stop_price:
          orderType === "stop" && stopPrice !== "" ? stopPrice : null,
        stop_loss: stopLoss === "" ? null : stopLoss,
        take_profit: takeProfit === "" ? null : takeProfit,
        rationale,
        instrument_kind: "underlying",
      });
      setRationale("");
      toast.success(
        `${side === "buy" ? "Buy" : "Sell"} ${quantity} underlying order submitted.`,
      );
    } catch (err) {
      const message = err instanceof Error ? err.message : "submit failed";
      setError(message);
      toast.error(message);
    } finally {
      setPending(false);
    }
  };

  return (
    <div className="flex flex-col gap-2 text-sm bg-slate-900 p-3 rounded border border-slate-800">
      <div className="flex items-center gap-2">
        <div className="font-semibold text-slate-200">Order ticket</div>
      </div>

      <div className="flex flex-wrap gap-2">
        <select
          aria-label="Order side"
          value={side}
          onChange={(e) => setSide(e.target.value as OrderSide)}
          className="px-2 py-1 bg-slate-950 rounded border border-slate-700"
        >
          <option value="buy">Buy</option>
          <option value="sell">Sell</option>
        </select>
        <select
          aria-label="Order type"
          value={orderType}
          onChange={(e) => setOrderType(e.target.value as OrderType)}
          className="px-2 py-1 bg-slate-950 rounded border border-slate-700 disabled:opacity-40"
        >
          <option value="market">Market</option>
          <option value="limit">Limit</option>
          <option value="stop">Stop</option>
        </select>
        <input
          type="number"
          aria-label="Order quantity"
          min={1}
          step={1}
          value={quantity}
          onChange={(e) => setQuantity(Math.max(1, Number(e.target.value)))}
          className="w-20 px-2 py-1 bg-slate-950 rounded border border-slate-700"
        />
      </div>

      {orderType === "limit" && (
        <PriceInput label="Limit" value={limitPrice} onChange={setLimitPrice} />
      )}
      {orderType === "stop" && (
        <PriceInput label="Stop" value={stopPrice} onChange={setStopPrice} />
      )}

          <PriceInput
            label="Stop loss (opt)"
            value={stopLoss}
            onChange={setStopLoss}
          />
          <PriceInput
            label="Take profit (opt)"
            value={takeProfit}
            onChange={setTakeProfit}
          />

      <details className="text-xs text-slate-400"><summary className="cursor-pointer">Execution, risk and costs</summary><p className="mt-2">Market orders queue for the next opening. Limit orders may not fill. Stops may gap beyond their level. Buy enters long; sell closes a long or opens a short. Quantity × price × multiplier is exposure, not risk. Without a stop, a cash long risks the full outlay; short/futures entries require a stop. Closing orders remain possible during a risk pause.</p><p className="mt-2">This session: multiplier {session?.multiplier ?? 1}, fee {session?.fee_per_unit ?? 0} USD per unit per side, synthetic spread {session?.spread_bps ?? 0} bps.</p></details>
      <textarea
        aria-label="Pre-trade rationale"
        rows={3}
        placeholder="rationale (required)…"
        value={rationale}
        onChange={(e) => setRationale(e.target.value)}
        className="px-2 py-1 bg-slate-950 rounded border border-slate-700"
      />

      {error && <div className="text-red-400">{error}</div>}

      <button
        type="button"
        disabled={disabled}
        onClick={handleSubmit}
        className="px-3 py-1 bg-emerald-700 rounded disabled:opacity-40"
      >
        {pending ? "Submitting…" : "Submit"}
      </button>
    </div>
  );
}

interface PriceProps {
  label: string;
  value: number | "";
  onChange: (v: number | "") => void;
}

function PriceInput({ label, value, onChange }: PriceProps) {
  return (
    <label className="flex items-center gap-2 text-slate-300">
      <span className="w-32 text-xs">{label}</span>
      <input
        type="number"
        step="0.01"
        value={value}
        onChange={(e) =>
          onChange(e.target.value === "" ? "" : Number(e.target.value))
        }
        className="flex-1 px-2 py-1 bg-slate-950 rounded border border-slate-700"
      />
    </label>
  );
}
