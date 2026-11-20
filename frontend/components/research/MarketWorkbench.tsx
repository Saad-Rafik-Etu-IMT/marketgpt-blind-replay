"use client";
import { useState } from "react";
import {
  LinePlot,
  plotColors,
  type Series,
  type Marker,
} from "./ScientificPlot";
import { type Replay } from "@/lib/research/laboratory";
import { humanLabel } from "@/lib/research/product-context";

export function MarketWorkbench({
  replay,
  end,
  onSelect,
  annotations = [],
  hiddenBar,
}: {
  replay: Replay;
  end: number;
  onSelect: (bar: number) => void;
  annotations?: { bar: number; note: string }[];
  hiddenBar?: number;
}) {
  const [overlays, setOverlays] = useState<string[]>(["SMA 20"]);
  const [drawing, setDrawing] = useState(false);
  const [anchor, setAnchor] = useState<number | null>(null);
  const [lines, setLines] = useState<[number, number][]>([]);
  const bars = replay.market.slice(0, end + 1);
  const close = bars.map((b) => b.close);
  const moving = (n: number, kind: string) =>
    close.flatMap((v, i) => {
      if (i < n - 1) return [];
      const slice = close.slice(i - n + 1, i + 1),
        mean = slice.reduce((a, b) => a + b, 0) / n;
      const sigma = Math.sqrt(
        slice.reduce((a, b) => a + (b - mean) ** 2, 0) / n,
      );
      return [
        {
          x: i,
          y:
            kind === "upper"
              ? mean + 2 * sigma
              : kind === "lower"
                ? mean - 2 * sigma
                : mean,
        },
      ];
    });
  let ema = close[0] ?? 0;
  const series: Series[] = [];
  for (const [i, [a, b]] of lines.entries())
    if (
      a <= end &&
      b <= end &&
      close[a] !== undefined &&
      close[b] !== undefined
    )
      series.push({
        name: `Annotation line ${i + 1}`,
        points: [
          { x: a, y: close[a] },
          { x: b, y: close[b] },
        ].sort((a, b) => a.x - b.x),
        color: "#9d65b8",
        dashed: true,
      });
  if (overlays.includes("SMA 20"))
    series.push({
      name: "SMA 20",
      points: moving(20, "mean"),
      color: plotColors[1],
    });
  if (overlays.includes("EMA 20"))
    series.push({
      name: "EMA 20",
      points: close.map((v, x) => {
        ema = (v * 2) / 21 + (ema * 19) / 21;
        return { x, y: ema };
      }),
      color: plotColors[2],
    });
  if (overlays.includes("Bollinger"))
    series.push(
      { name: "Bollinger upper", points: moving(20, "upper"), dashed: true },
      { name: "Bollinger lower", points: moving(20, "lower"), dashed: true },
    );
  const candles = bars.flatMap((b, x) =>
    b.open !== undefined && b.high !== undefined && b.low !== undefined
      ? [{ x, open: b.open, high: b.high, low: b.low, close: b.close }]
      : [],
  );
  if (candles.length !== bars.length)
    series.unshift({ name: "Close", points: close.map((y, x) => ({ x, y })) });
  const markers: Marker[] = (replay.players[0]?.events ?? [])
    .filter(
      (e) =>
        e.decision.decision_bar_index <= end &&
        e.decision.decision_bar_index !== hiddenBar &&
        bars[e.decision.decision_bar_index],
    )
    .map((e) => ({
      x: e.decision.decision_bar_index,
      y: bars[e.decision.decision_bar_index].close,
      label: humanLabel(e.decision.action),
      color:
        e.decision.order_side === "buy"
          ? "#148b82"
          : e.decision.order_side === "sell"
            ? "#b95061"
            : "#a1a1aa",
      direction: e.decision.order_side ?? undefined,
    }));
  markers.push(
    ...annotations
      .filter((a) => a.bar <= end && bars[a.bar])
      .map((a) => ({
        x: a.bar,
        y: bars[a.bar].close,
        label: a.note,
        color: "#9d65b8",
      })),
  );
  const domain: [number, number] = [0, Math.max(1, end)];
  const date = (x: number) => bars[Math.round(x)]?.date ?? "";
  const rsi = close.flatMap((v, i) => {
    if (i < 14) return [];
    let gain = 0,
      loss = 0;
    for (let j = i - 13; j <= i; j++) {
      const d = close[j] - close[j - 1];
      gain += Math.max(d, 0);
      loss += Math.max(-d, 0);
    }
    return [{ x: i, y: gain + loss === 0 ? 50 : (100 * gain) / (gain + loss) }];
  });
  const volatility = close.flatMap((v, i) => {
    if (i < 20) return [];
    const r = close
      .slice(i - 20, i + 1)
      .slice(1)
      .map((c, j) => Math.log(c / close[i - 20 + j]));
    const m = r.reduce((a, b) => a + b, 0) / r.length;
    return [
      {
        x: i,
        y:
          Math.sqrt(r.reduce((a, b) => a + (b - m) ** 2, 0) / r.length) *
          Math.sqrt(252),
      },
    ];
  });
  const regimeBands = volatility.map(({ x, y }) => {
    const change = close[x] / close[x - 20] - 1;
    const label =
      y > 0.3
        ? "High volatility"
        : change > 0.02
          ? "Rising"
          : change < -0.02
            ? "Falling"
            : "Sideways";
    return {
      from: x - 0.5,
      to: x + 0.5,
      label,
      color:
        label === "High volatility"
          ? "#d07822"
          : label === "Rising"
            ? "#148b82"
            : label === "Falling"
              ? "#b95061"
              : "#7c8797",
    };
  });
  return (
    <div className="market-workbench">
      <div className="bench-actions">
        {[
          "SMA 20",
          "EMA 20",
          "Bollinger",
          "RSI 14",
          "Volume",
          "Volatility",
          "Market regimes",
        ].map((name) => (
          <label key={name}>
            <input
              type="checkbox"
              checked={overlays.includes(name)}
              onChange={() =>
                setOverlays((v) =>
                  v.includes(name) ? v.filter((x) => x !== name) : [...v, name],
                )
              }
            />{" "}
            {name}
          </label>
        ))}
        <button
          className="bench-button"
          aria-pressed={drawing}
          onClick={() => {
            setDrawing(!drawing);
            setAnchor(null);
          }}
        >
          {drawing ? "Cancel drawing" : "Draw trend line"}
        </button>
        {!!lines.length && (
          <button className="bench-button" onClick={() => setLines([])}>
            Clear trend lines
          </button>
        )}
      </div>
      {drawing && (
        <p className="muted">
          {anchor === null
            ? "Choose the first revealed session on the chart."
            : "Choose the second revealed session."}{" "}
          Lines connect actual closing prices.
        </p>
      )}
      <LinePlot
        title="Market candles · latest revealed sessions"
        xLabel="Session · click to inspect a revealed decision"
        yLabel="Price"
        sync
        xDomain={domain}
        formatX={date}
        onSelect={(v) => {
          if (!drawing) {
            onSelect(v);
            return;
          }
          const bar = Math.min(end, Math.max(0, Math.round(v)));
          if (anchor === null) {
            setAnchor(bar);
            return;
          }
          if (anchor !== bar)
            setLines(
              (old) => [...old, [anchor, bar]].slice(-20) as [number, number][],
            );
          setAnchor(null);
          setDrawing(false);
        }}
        candles={candles.length === bars.length ? candles : []}
        bands={overlays.includes("Market regimes") ? regimeBands : []}
        series={series}
        markers={markers}
      />
      {overlays.includes("Market regimes") && (
        <p className="muted">
          Descriptive, causal overlay (not the evaluation regime classifier):
          20-session price change above +2% rising, below −2% falling, otherwise
          sideways; annualized volatility above 30% overrides as high
          volatility.
        </p>
      )}
      {overlays.includes("RSI 14") && (
        <LinePlot
          title="RSI · simple 14-session"
          sync
          xDomain={domain}
          formatX={date}
          xLabel="Session"
          yDomain={[0, 100]}
          series={[{ name: "RSI", points: rsi }]}
        />
      )}
      {overlays.includes("Volume") && (
        <LinePlot
          title="Recorded volume"
          sync
          xDomain={domain}
          formatX={date}
          xLabel="Session"
          series={[
            {
              name: "Volume",
              points: bars.flatMap((b, x) =>
                b.volume === undefined ? [] : [{ x, y: b.volume }],
              ),
            },
          ]}
        />
      )}
      {overlays.includes("Volatility") && (
        <LinePlot
          title="Annualized 20-session log-return volatility"
          sync
          xDomain={domain}
          formatX={date}
          xLabel="Session"
          series={[{ name: "Volatility", points: volatility }]}
        />
      )}
    </div>
  );
}
