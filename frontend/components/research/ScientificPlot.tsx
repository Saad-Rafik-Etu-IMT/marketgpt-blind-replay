"use client";

import { useRef } from "react";

export const plotColors = [
  "#148b82",
  "#d07822",
  "#527bd1",
  "#9d65b8",
  "#698a37",
  "#c95680",
];
export type Point = { x: number; y: number };
export type Series = {
  name: string;
  points: Point[];
  color?: string;
  dashed?: boolean;
  dots?: boolean;
  scatter?: boolean;
};
export type Marker = Point & {
  label: string;
  color: string;
  direction?: "buy" | "sell";
};

export function PlotFrame({
  title,
  children,
  height = 236,
}: {
  title: string;
  children: React.ReactNode;
  height?: number;
}) {
  const ref = useRef<SVGSVGElement>(null);
  function download() {
    if (!ref.current) return;
    const clone = ref.current.cloneNode(true) as SVGSVGElement;
    clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
    clone.setAttribute("width", "1400");
    clone.setAttribute("height", String(height * 2));
    // Export on paper-friendly white, independent of the active theme.
    clone.style.setProperty("--fg", "#142e32");
    clone.style.setProperty("--muted", "#52696b");
    clone.style.setProperty("--line", "#ccdad8");
    clone.style.setProperty("--panel", "#ffffff");
    clone.style.setProperty("font-family", "Arial, sans-serif");
    const url = URL.createObjectURL(
      new Blob([new XMLSerializer().serializeToString(clone)], {
        type: "image/svg+xml;charset=utf-8",
      }),
    );
    const link = document.createElement("a");
    link.href = url;
    link.download = title.toLowerCase().replace(/[^a-z0-9]+/g, "-") + ".svg";
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  return (
    <figure className="science-figure">
      <figcaption>
        <strong>{title}</strong>
        <button
          className="bench-button plot-export"
          onClick={download}
          aria-label={`Export ${title} as SVG`}
        >
          Export SVG
        </button>
      </figcaption>
      <div
        className="science-scroll"
        tabIndex={0}
        role="region"
        aria-label={`${title}, scroll to view full chart on narrow screens`}
      >
        <svg
          ref={ref}
          className="science-svg"
          viewBox={`0 0 700 ${height}`}
          role="img"
          aria-label={title}
        >
          <rect width="700" height={height} fill="var(--panel)" />
          <text
            x="350"
            y="15"
            fill="var(--fg)"
            fontSize="11"
            textAnchor="middle"
          >
            {title}
          </text>
          {children}
        </svg>
      </div>
    </figure>
  );
}

export function LinePlot({
  title,
  series,
  markers = [],
  xDomain,
  yDomain,
  xLabel,
  yLabel,
  legend = true,
  formatY = (v) => v.toLocaleString("en-US", { maximumFractionDigits: 3 }),
  formatX = (v) => v.toLocaleString("en-US", { maximumFractionDigits: 1 }),
}: {
  title: string;
  series: Series[];
  markers?: Marker[];
  xDomain?: [number, number];
  yDomain?: [number, number];
  xLabel: string;
  yLabel?: string;
  legend?: boolean;
  formatY?: (v: number) => string;
  formatX?: (v: number) => string;
}) {
  const all = [...series.flatMap((s) => s.points), ...markers];
  if (!all.length)
    return <p className="muted">{title}: no observations available.</p>;
  const [xmin, xmax] = xDomain ?? [
    Math.min(...all.map((p) => p.x)),
    Math.max(...all.map((p) => p.x)),
  ];
  const [ymin, ymax] = yDomain ?? [
    Math.min(...all.map((p) => p.y)),
    Math.max(...all.map((p) => p.y)),
  ];
  const x = (v: number) =>
    xmin === xmax ? 78 : 78 + (592 * (v - xmin)) / (xmax - xmin);
  const y = (v: number) =>
    ymin === ymax ? 104 : 176 - (144 * (v - ymin)) / (ymax - ymin);
  const palette = (s: Series, i: number) =>
    s.color ?? plotColors[i % plotColors.length];
  const legendHeight = 226 + (legend ? Math.ceil(series.length / 3) * 18 : 0);
  return (
    <PlotFrame title={title} height={legendHeight}>
      {(ymin === ymax ? [ymin] : [ymin, (ymin + ymax) / 2, ymax]).map((v) => (
        <g key={v}>
          <line x1="78" x2="670" y1={y(v)} y2={y(v)} stroke="var(--line)" />
          <text
            x="70"
            y={y(v) + 4}
            fill="var(--muted)"
            textAnchor="end"
            fontSize="11"
          >
            {formatY(v)}
          </text>
        </g>
      ))}
      {yLabel && (
        <text x="78" y="27" fill="var(--muted)" fontSize="11">
          {yLabel}
        </text>
      )}
      {(xmin === xmax ? [xmin] : [xmin, xmax]).map((v) => (
        <text
          key={v}
          x={x(v)}
          y="195"
          fill="var(--muted)"
          fontSize="11"
          textAnchor={v === xmin ? "start" : "end"}
        >
          {formatX(v)}
        </text>
      ))}
      <text
        x="374"
        y="213"
        fill="var(--muted)"
        textAnchor="middle"
        fontSize="11"
      >
        {xLabel}
      </text>
      {series.map((s, i) => (
        <g key={s.name}>
          {!s.scatter && (
            <polyline
              points={s.points.map((p) => `${x(p.x)},${y(p.y)}`).join(" ")}
              fill="none"
              stroke={palette(s, i)}
              strokeWidth="2.2"
              strokeDasharray={s.dashed ? "6 4" : undefined}
              strokeLinejoin="round"
            />
          )}
          {(s.dots || s.scatter || s.points.length === 1) &&
            s.points.map((p, j) => (
              <circle
                key={j}
                cx={x(p.x)}
                cy={y(p.y)}
                r="4"
                fill={palette(s, i)}
              >
                <title>
                  {s.name}: {formatX(p.x)}, {formatY(p.y)}
                </title>
              </circle>
            ))}
          {legend && (
            <g
              transform={`translate(${78 + (i % 3) * 200},${231 + Math.floor(i / 3) * 18})`}
            >
              <line
                x2="18"
                y1="-3"
                y2="-3"
                stroke={palette(s, i)}
                strokeWidth="2.5"
                strokeDasharray={s.dashed ? "6 4" : undefined}
              />
              <text x="23" fill="var(--fg)" fontSize="11">
                {s.name}
              </text>
            </g>
          )}
        </g>
      ))}
      {markers.map((p, i) =>
        p.direction ? (
          <path
            key={i}
            d={`M ${x(p.x)} ${y(p.y) + (p.direction === "buy" ? -6 : 6)} l -5 ${p.direction === "buy" ? 10 : -10} h 10 Z`}
            stroke="var(--panel)"
            fill={p.color}
          >
            <title>
              {p.label}: {formatX(p.x)}, {formatY(p.y)}
            </title>
          </path>
        ) : (
          <circle
            key={i}
            cx={x(p.x)}
            cy={y(p.y)}
            r="4.5"
            stroke="var(--panel)"
            strokeWidth="1"
            fill={p.color}
          >
            <title>
              {p.label}: {formatX(p.x)}, {formatY(p.y)}
            </title>
          </circle>
        ),
      )}
    </PlotFrame>
  );
}

export const points = (values: number[]): Point[] =>
  values.map((y, x) => ({ x, y }));
