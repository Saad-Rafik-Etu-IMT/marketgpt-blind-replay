"use client";

import { useEffect, useRef, useState, memo } from "react";

import {
  anchoredVwap,
  atr,
  bollinger,
  ema,
  latestAtr,
  sma,
} from "@/lib/indicators";
import type { OHLCVBar, PositionRead } from "@/lib/types";

import { ChartDrawingLayer, type DrawingTool } from "./ChartDrawingLayer";

// Comprehensive replay chart. Everything overlaid here is computed from the
// `bars` prop (the visible bars only), so all indicators are anti-lookahead by
// construction. lightweight-charts v4 API surface used:
//   createChart, addCandlestickSeries, addHistogramSeries, addLineSeries,
//   series.setData, series.setMarkers, series.createPriceLine,
//   series.priceToCoordinate, timeScale().timeToCoordinate,
//   subscribeCrosshairMove.

export interface IndicatorToggles {
  sma20: boolean;
  sma50: boolean;
  ema12: boolean;
  bollinger: boolean;
  vwap: boolean;
  volume: boolean;
  atrBand: boolean; // risk-sizing "1R" band around price
}

interface Props {
  bars: OHLCVBar[];
  position?: PositionRead | null;
  stopLoss?: number | null;
  takeProfit?: number | null;
  indicators?: Partial<IndicatorToggles>;
  activeTool?: DrawingTool;
  onToolConsumed?: () => void;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  onChartReady?: (chart: any) => void; // for syncing the RSI pane
  width?: number;
  height?: number;
}

interface CrosshairData {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
}

const DEFAULT_INDICATORS: IndicatorToggles = {
  sma20: true,
  sma50: false,
  ema12: false,
  bollinger: false,
  vwap: false,
  volume: true,
  atrBand: false,
};

function CandlestickReplayChartImpl({
  bars,
  position = null,
  stopLoss = null,
  takeProfit = null,
  indicators,
  activeTool = "none",
  onToolConsumed,
  onChartReady,
  width = 800,
  height = 400,
}: Props) {
  const ind = { ...DEFAULT_INDICATORS, ...indicators };
  const containerRef = useRef<HTMLDivElement>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const chartRef = useRef<any>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const candleRef = useRef<any>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const overlayRef = useRef<Record<string, any>>({});
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const priceLinesRef = useRef<any[]>([]);
  const [crosshair, setCrosshair] = useState<CrosshairData | null>(null);
  const [chartLive, setChartLive] = useState(false);

  // Build the chart + all series once.
  useEffect(() => {
    if (!containerRef.current) return;
    let cancelled = false;

    void (async () => {
      const lwc = await import("lightweight-charts");
      if (cancelled || !containerRef.current) return;
      const chart = lwc.createChart(containerRef.current, {
        width,
        height,
        layout: { background: { color: "#0f172a" }, textColor: "#cbd5e1" },
        grid: {
          vertLines: { color: "#1e293b" },
          horzLines: { color: "#1e293b" },
        },
        timeScale: { timeVisible: false, borderColor: "#1e293b" },
        rightPriceScale: { borderColor: "#1e293b" },
        crosshair: { mode: lwc.CrosshairMode.Normal },
      });
      const candle = chart.addCandlestickSeries({
        upColor: "#16a34a",
        downColor: "#dc2626",
        wickUpColor: "#16a34a",
        wickDownColor: "#dc2626",
        borderVisible: false,
      });

      const volume = chart.addHistogramSeries({
        priceFormat: { type: "volume" },
        priceScaleId: "vol",
        color: "#334155",
      });
      chart.priceScale("vol").applyOptions({
        scaleMargins: { top: 0.82, bottom: 0 },
      });

      const mkLine = (color: string, w = 1, dashed = false) =>
        chart.addLineSeries({
          color,
          lineWidth: w as 1 | 2 | 3 | 4,
          lineStyle: dashed ? lwc.LineStyle.Dashed : lwc.LineStyle.Solid,
          priceLineVisible: false,
          lastValueVisible: false,
        });

      overlayRef.current = {
        volume,
        sma20: mkLine("#60a5fa", 1),
        sma50: mkLine("#a78bfa", 1),
        ema12: mkLine("#fbbf24", 1),
        bbU: mkLine("#475569", 1, true),
        bbM: mkLine("#64748b", 1),
        bbL: mkLine("#475569", 1, true),
        vwap: mkLine("#22d3ee", 1, true),
        atrU: mkLine("#f87171", 1, true),
        atrL: mkLine("#34d399", 1, true),
      };

      chart.subscribeCrosshairMove((param: unknown) => {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const p = param as any;
        if (!p?.time || !p.seriesData) {
          setCrosshair(null);
          return;
        }
        const d = p.seriesData.get(candle);
        if (d && typeof d.open === "number") {
          setCrosshair({
            date: String(p.time),
            open: d.open,
            high: d.high,
            low: d.low,
            close: d.close,
          });
        }
      });

      chartRef.current = chart;
      candleRef.current = candle;
      setChartLive(true);
      onChartReady?.(chart);
    })();

    return () => {
      cancelled = true;
      if (chartRef.current) {
        chartRef.current.remove();
        chartRef.current = null;
        candleRef.current = null;
        overlayRef.current = {};
        priceLinesRef.current = [];
        setChartLive(false);
      }
    };
  }, [width, height, onChartReady]);

  // Candles and volume are limited to the visible bars.
  useEffect(() => {
    if (!candleRef.current) return;
    candleRef.current.setData(
      bars.map((b) => ({
        time: b.date,
        open: b.open,
        high: b.high,
        low: b.low,
        close: b.close,
      })),
    );
    if (overlayRef.current.volume) {
      overlayRef.current.volume.setData(
        bars.map((b) => ({
          time: b.date,
          value: b.volume,
          color: b.close >= b.open ? "rgba(22,163,74,0.4)" : "rgba(220,38,38,0.4)",
        })),
      );
    }
  }, [bars, chartLive]);

  // Overlay indicators — recomputed from visible bars on every change.
  useEffect(() => {
    const o = overlayRef.current;
    if (!o.sma20) return;

    o.sma20.setData(ind.sma20 ? sma(bars, 20) : []);
    o.sma50.setData(ind.sma50 ? sma(bars, 50) : []);
    o.ema12.setData(ind.ema12 ? ema(bars, 12) : []);
    o.vwap.setData(ind.vwap ? anchoredVwap(bars) : []);

    if (ind.bollinger) {
      const bb = bollinger(bars, 20, 2);
      o.bbU.setData(bb.upper);
      o.bbM.setData(bb.middle);
      o.bbL.setData(bb.lower);
    } else {
      o.bbU.setData([]);
      o.bbM.setData([]);
      o.bbL.setData([]);
    }

    // ATR band = "1R" around the latest close (risk-sizing visual).
    if (ind.atrBand) {
      const series = atr(bars, 14);
      const atrByTime = new Map(series.map((p) => [p.time, p.value]));
      const upper: { time: string; value: number }[] = [];
      const lower: { time: string; value: number }[] = [];
      for (const b of bars) {
        const a = atrByTime.get(b.date);
        if (a !== undefined) {
          upper.push({ time: b.date, value: b.close + a });
          lower.push({ time: b.date, value: b.close - a });
        }
      }
      o.atrU.setData(upper);
      o.atrL.setData(lower);
    } else {
      o.atrU.setData([]);
      o.atrL.setData([]);
    }
  }, [bars, ind.sma20, ind.sma50, ind.ema12, ind.bollinger, ind.vwap, ind.atrBand, chartLive]);

  // Position price lines: entry, stop, take-profit, break-even.
  useEffect(() => {
    const candle = candleRef.current;
    if (!candle) return;
    void (async () => {
      const lwc = await import("lightweight-charts");
      // Clear previous lines.
      for (const pl of priceLinesRef.current) {
        try {
          candle.removePriceLine(pl);
        } catch {
          /* already gone */
        }
      }
      priceLinesRef.current = [];

      const add = (price: number, color: string, title: string) => {
        const pl = candle.createPriceLine({
          price,
          color,
          lineWidth: 1,
          lineStyle: lwc.LineStyle.Dashed,
          axisLabelVisible: true,
          title,
        });
        priceLinesRef.current.push(pl);
      };

      if (position && position.side !== "flat" && position.quantity > 0) {
        add(position.avg_entry_price, "#34d399", "Entry");
      }
      if (stopLoss != null) add(stopLoss, "#f87171", "Stop");
      if (takeProfit != null) add(takeProfit, "#22d3ee", "Target");
    })();
  }, [position, stopLoss, takeProfit, chartLive]);

  // Risk-sizing readout (dollar risk = qty * |entry - stop|).
  const atrVal = latestAtr(bars, 14);
  const dollarRisk =
    position && position.side !== "flat" && stopLoss != null
      ? position.quantity * Math.abs(position.avg_entry_price - stopLoss)
      : null;

  return (
    <div className="relative" style={{ width }}>
      <div ref={containerRef} className="rounded border border-slate-800" />

      {/* Drawing overlay */}
      {chartLive && (
        <ChartDrawingLayer
          chart={chartRef.current}
          series={candleRef.current}
          width={width}
          height={height}
          activeTool={activeTool}
          onToolConsumed={onToolConsumed}
        />
      )}

      {/* OHLCV crosshair tooltip + bar counter */}
      <div className="absolute top-2 left-2 z-10 pointer-events-none flex flex-col gap-1">
        <div className="text-xs font-mono bg-slate-900/80 rounded px-2 py-1 border border-slate-700">
          <span className="text-slate-500">Bar </span>
          <span className="text-slate-300">{bars.length}</span>
          {atrVal !== null && (
            <span className="text-slate-500 ml-3">
              ATR14 <span className="text-amber-400">{atrVal.toFixed(2)}</span>
            </span>
          )}
        </div>
        {crosshair && (
          <div className="text-xs font-mono bg-slate-900/80 rounded px-2 py-1 border border-slate-700 text-slate-400">
            O {crosshair.open.toFixed(2)} H {crosshair.high.toFixed(2)} L{" "}
            {crosshair.low.toFixed(2)} C{" "}
            <span className={crosshair.close >= crosshair.open ? "text-emerald-400" : "text-red-400"}>
              {crosshair.close.toFixed(2)}
            </span>
          </div>
        )}
      </div>

      {/* Risk-sizing badge */}
      {dollarRisk !== null && (
        <div className="absolute bottom-2 right-2 z-10 pointer-events-none text-xs font-mono bg-slate-900/80 rounded px-2 py-1 border border-red-900 text-red-300">
          Risque ${dollarRisk.toFixed(0)}
        </div>
      )}

    </div>
  );
}

// Memoized so the chart only re-renders when its own props change, not on
// every background-poll-driven re-render of the cockpit.
export const CandlestickReplayChart = memo(CandlestickReplayChartImpl);
