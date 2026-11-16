"use client";

import type { RiskMetricsRead } from "@/lib/types";


interface Props {
  metrics: RiskMetricsRead | null;
}

function num(n: number | null, decimals = 2) {
  return n === null ? "—" : n.toFixed(decimals);
}

export function RiskPanel({ metrics }: Props) {
  if (!metrics) {
    return <div className="text-slate-400 text-sm">No metrics yet.</div>;
  }

  return (
    <div className="text-sm bg-slate-900 p-3 rounded border border-slate-800">
      <div className="flex items-center gap-2 mb-2">
        <div className="font-semibold text-slate-200">Risk</div>
      </div>
      <div className="grid grid-cols-2 gap-y-1">
        <span className="text-slate-400">Drawdown (now)</span>
        <span
          className="font-mono text-right"
        >
          {num(metrics.current_drawdown_pct)}%
        </span>
        <span className="text-slate-400">Drawdown (historical max)</span>
        <span className="font-mono text-right">{num(metrics.max_drawdown_pct)}%</span>
        <span className="text-slate-400">Sharpe</span>
        <span className="font-mono text-right">{num(metrics.sharpe_ratio)}</span>
        <span className="text-slate-400">Win rate</span>
        <span className="font-mono text-right">
          {metrics.win_rate === null
            ? "—"
            : `${(metrics.win_rate * 100).toFixed(1)}%`}
        </span>
        <span className="text-slate-400">Expectancy</span>
        <span className="font-mono text-right">{num(metrics.expectancy)} USD</span>
        <span className="text-slate-400">Exposure</span>
        <span className="font-mono text-right">
          {num(metrics.max_position_exposure_pct)}%
        </span>
      </div>
      <p className="mt-2 text-xs text-slate-400">Historical drawdown is an observation, not the configured loss limit. Win rate and expectancy describe realized exits before commissions; they do not measure skill. Account equity includes costs and open positions.</p>
    </div>
  );
}
