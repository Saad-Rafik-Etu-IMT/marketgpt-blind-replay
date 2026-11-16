"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { MouseEvent as ReactMouseEvent } from "react";

// SVG overlay for chart drawing tools. Sits on top of the lightweight-charts
// canvas. Drawings are stored in (time, price) coordinates and re-projected to
// pixels on every pan/zoom via the chart's coordinate-conversion API, so they
// stay pinned to the data as the user navigates.
//
// Supported tools:
//   - horizontal: a support/resistance line at a fixed price (one click)
//   - trend:      a diagonal line between two (time, price) points (two clicks)
//   - rectangle:  a zone between two (time, price) corners (two clicks)

export type DrawingTool = "none" | "horizontal" | "trend" | "rectangle";

interface DataPoint {
  time: string;
  price: number;
}

interface Drawing {
  id: string;
  tool: Exclude<DrawingTool, "none">;
  points: DataPoint[];
  color: string;
}

interface Props {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  chart: any; // lightweight-charts IChartApi
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  series: any; // candlestick series, for price<->coordinate conversion
  width: number;
  height: number;
  activeTool: DrawingTool;
  onToolConsumed?: () => void;
}

const TOOL_COLORS: Record<string, string> = {
  horizontal: "#f59e0b",
  trend: "#60a5fa",
  rectangle: "#34d399",
};

export function ChartDrawingLayer({
  chart,
  series,
  width,
  height,
  activeTool,
  onToolConsumed,
}: Props) {
  const [drawings, setDrawings] = useState<Drawing[]>([]);
  const [pending, setPending] = useState<DataPoint[]>([]);
  const [, forceRender] = useState(0);
  const svgRef = useRef<SVGSVGElement>(null);

  // Re-project on every visible-range change so drawings track pan/zoom.
  useEffect(() => {
    if (!chart) return;
    const ts = chart.timeScale();
    const handler = () => forceRender((n) => n + 1);
    ts.subscribeVisibleLogicalRangeChange(handler);
    chart.subscribeCrosshairMove(handler);
    return () => {
      ts.unsubscribeVisibleLogicalRangeChange(handler);
      chart.unsubscribeCrosshairMove(handler);
    };
  }, [chart]);

  const toPixel = useCallback(
    (p: DataPoint): { x: number; y: number } | null => {
      if (!chart || !series) return null;
      const x = chart.timeScale().timeToCoordinate(p.time);
      const y = series.priceToCoordinate(p.price);
      if (x === null || y === null) return null;
      return { x, y };
    },
    [chart, series],
  );

  const fromPixel = useCallback(
    (x: number, y: number): DataPoint | null => {
      if (!chart || !series) return null;
      const time = chart.timeScale().coordinateToTime(x);
      const price = series.coordinateToPrice(y);
      if (time === null || price === null) return null;
      return { time: String(time), price };
    },
    [chart, series],
  );

  const handleClick = useCallback(
    (ev: ReactMouseEvent<SVGSVGElement>) => {
      if (activeTool === "none") return;
      const rect = svgRef.current?.getBoundingClientRect();
      if (!rect) return;
      const x = ev.clientX - rect.left;
      const y = ev.clientY - rect.top;
      const dp = fromPixel(x, y);
      if (!dp) return;

      if (activeTool === "horizontal") {
        setDrawings((d) => [
          ...d,
          {
            id: `h-${Date.now()}`,
            tool: "horizontal",
            points: [dp],
            color: TOOL_COLORS.horizontal,
          },
        ]);
        onToolConsumed?.();
      } else {
        // trend / rectangle need two points
        const next = [...pending, dp];
        if (next.length === 2) {
          setDrawings((d) => [
            ...d,
            {
              id: `${activeTool}-${Date.now()}`,
              tool: activeTool,
              points: next,
              color: TOOL_COLORS[activeTool],
            },
          ]);
          setPending([]);
          onToolConsumed?.();
        } else {
          setPending(next);
        }
      }
    },
    [activeTool, fromPixel, pending, onToolConsumed],
  );

  const removeDrawing = useCallback((id: string) => {
    setDrawings((d) => d.filter((x) => x.id !== id));
  }, []);

  const interactive = activeTool !== "none";

  return (
    <svg
      ref={svgRef}
      width={width}
      height={height}
      onClick={handleClick}
      style={{
        position: "absolute",
        top: 0,
        left: 0,
        pointerEvents: interactive ? "auto" : "none",
        cursor: interactive ? "crosshair" : "default",
      }}
    >
      {drawings.map((d) => {
        if (d.tool === "horizontal") {
          const p = toPixel(d.points[0]);
          if (!p) return null;
          return (
            <g key={d.id}>
              <line
                x1={0}
                x2={width}
                y1={p.y}
                y2={p.y}
                stroke={d.color}
                strokeWidth={1}
                strokeDasharray="4 3"
              />
              <text x={width - 64} y={p.y - 4} fill={d.color} fontSize={10} fontFamily="monospace">
                {d.points[0].price.toFixed(2)}
              </text>
              <circle
                cx={width - 10}
                cy={p.y}
                r={5}
                fill="#0f172a"
                stroke={d.color}
                style={{ pointerEvents: "auto", cursor: "pointer" }}
                onClick={(e) => {
                  e.stopPropagation();
                  removeDrawing(d.id);
                }}
              />
            </g>
          );
        }
        if (d.tool === "trend") {
          const a = toPixel(d.points[0]);
          const b = toPixel(d.points[1]);
          if (!a || !b) return null;
          return (
            <g key={d.id}>
              <line x1={a.x} y1={a.y} x2={b.x} y2={b.y} stroke={d.color} strokeWidth={1.5} />
              <circle cx={a.x} cy={a.y} r={3} fill={d.color} />
              <circle
                cx={b.x}
                cy={b.y}
                r={5}
                fill="#0f172a"
                stroke={d.color}
                style={{ pointerEvents: "auto", cursor: "pointer" }}
                onClick={(e) => {
                  e.stopPropagation();
                  removeDrawing(d.id);
                }}
              />
            </g>
          );
        }
        // rectangle
        const a = toPixel(d.points[0]);
        const b = toPixel(d.points[1]);
        if (!a || !b) return null;
        const x = Math.min(a.x, b.x);
        const y = Math.min(a.y, b.y);
        const w = Math.abs(b.x - a.x);
        const h = Math.abs(b.y - a.y);
        return (
          <g key={d.id}>
            <rect
              x={x}
              y={y}
              width={w}
              height={h}
              fill={d.color}
              fillOpacity={0.08}
              stroke={d.color}
              strokeWidth={1}
              strokeDasharray="4 3"
            />
            <circle
              cx={x + w}
              cy={y}
              r={5}
              fill="#0f172a"
              stroke={d.color}
              style={{ pointerEvents: "auto", cursor: "pointer" }}
              onClick={(e) => {
                e.stopPropagation();
                removeDrawing(d.id);
              }}
            />
          </g>
        );
      })}

      {/* Pending first point preview */}
      {pending.map((p, i) => {
        const px = toPixel(p);
        if (!px) return null;
        return <circle key={`pending-${i}`} cx={px.x} cy={px.y} r={4} fill="#94a3b8" />;
      })}
    </svg>
  );
}
