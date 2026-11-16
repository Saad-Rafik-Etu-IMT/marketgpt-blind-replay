"use client";

import type { PortfolioSnapshotRead } from "@/lib/types";

interface Props {
  snapshot: PortfolioSnapshotRead | null;
}

function money(value: number) {
  return value.toLocaleString(undefined, {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 2,
  });
}

export function PortfolioPanel({ snapshot }: Props) {
  if (!snapshot) {
    return <div className="text-slate-400 text-sm">No portfolio data yet.</div>;
  }
  return (
    <div className="text-sm bg-slate-900 p-3 rounded border border-slate-800">
      <div className="font-semibold text-slate-200 mb-2">Portfolio</div>
      <div className="grid grid-cols-2 gap-y-1">
        <span className="text-slate-400">Cash</span>
        <span className="font-mono text-right">{money(snapshot.cash)}</span>
        <span className="text-slate-400">Equity</span>
        <span className="font-mono text-right">{money(snapshot.equity)}</span>
        <span className="text-slate-400">Realised P&amp;L (gross)</span>
        <span className="font-mono text-right">
          {money(snapshot.total_realized_pnl)}
        </span>
        <span className="text-slate-400">Commissions paid</span><span className="font-mono text-right">{money(snapshot.total_fees ?? 0)}</span>
        <span className="text-slate-400">Net P&amp;L (incl. open)</span><span className="font-mono text-right">{money(snapshot.total_realized_pnl + snapshot.total_unrealized_pnl - (snapshot.total_fees ?? 0))}</span>
        <span className="text-slate-400">Unrealised P&amp;L</span>
        <span className="font-mono text-right">
          {money(snapshot.total_unrealized_pnl)}
        </span>
      </div>

      {snapshot.positions.length > 0 && (
        <div className="mt-3">
          <div className="text-slate-400 mb-1">Positions</div>
          <table className="w-full text-xs">
            <thead className="text-slate-500">
              <tr>
                <th className="text-left">Inst</th>
                <th className="text-left">Side</th>
                <th className="text-right">Qty</th>
                <th className="text-right">Avg</th>
                <th className="text-right">uPnL</th>
              </tr>
            </thead>
            <tbody>
              {snapshot.positions.map((p) => (
                <tr key={p.instrument}>
                  <td>{p.instrument}</td>
                  <td>{p.side}</td>
                  <td className="text-right font-mono">{p.quantity}</td>
                  <td className="text-right font-mono">
                    {p.avg_entry_price.toFixed(2)}
                  </td>
                  <td className="text-right font-mono">
                    {p.unrealized_pnl.toFixed(2)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

