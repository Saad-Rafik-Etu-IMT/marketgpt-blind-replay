"use client";

import { useRef, useState } from "react";

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
  controls,
}: {
  title: string;
  children: React.ReactNode;
  height?: number;
  controls?: React.ReactNode;
}) {
  const ref = useRef<SVGSVGElement>(null);
  const dialog = useRef<HTMLDialogElement>(null);
  const [expanded, setExpanded] = useState(false);
  const [exportError, setExportError] = useState("");
  function download(format: "svg" | "png" = "svg") {
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
    function save(href: string, extension: string) {
      const link = document.createElement("a");
      link.href = href;
      link.download =
        title.toLowerCase().replace(/[^a-z0-9]+/g, "-") + "." + extension;
      link.click();
      setTimeout(() => URL.revokeObjectURL(href), 1000);
    }
    setExportError("");
    if (format === "svg") {
      save(url, "svg");
      return;
    }
    const img = new Image();
    img.onload = () => {
      try {
        const canvas = document.createElement("canvas");
        canvas.width = 1400;
        canvas.height = height * 2;
        const ctx = canvas.getContext("2d");
        if (!ctx) throw new Error("Canvas unavailable");
        ctx.drawImage(img, 0, 0);
        canvas.toBlob((blob) => {
          if (blob) save(URL.createObjectURL(blob), "png");
          else setExportError("PNG export unavailable.");
        }, "image/png");
      } catch {
        setExportError("PNG export unavailable.");
      } finally {
        URL.revokeObjectURL(url);
      }
    };
    img.onerror = () => {
      URL.revokeObjectURL(url);
      setExportError("PNG export unavailable.");
    };
    img.src = url;
  }
  const chart = (isExpanded = false) => (
    <svg
      ref={isExpanded ? undefined : ref}
      className="science-svg"
      viewBox={`0 0 700 ${height}`}
      role="img"
      aria-label={title}
    >
      <rect width="700" height={height} fill="var(--panel)" />
      <text
        className="plot-paper-title"
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
  );
  return (
    <figure className="science-figure">
      <figcaption>
        <strong>{title}</strong>
        <div className="plot-tools">
          <button
            className="bench-button plot-export"
            aria-label={`Expand ${title}`}
            onClick={() => {
              setExpanded(true);
              dialog.current?.showModal();
            }}
            title="Expand chart"
          >
            <svg viewBox="0 0 20 20" width="15" height="15" aria-hidden="true">
              <path
                d="M3 8V3h5m4 0h5v5m0 4v5h-5m-4 0H3v-5"
                stroke="currentColor"
                fill="none"
                strokeWidth="1.5"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </button>
          <button
            className="bench-button plot-export"
            onClick={() => download()}
            aria-label={`Export ${title} as SVG`}
          >
            SVG
          </button>
          <button
            className="bench-button plot-export"
            onClick={() => download("png")}
            aria-label={`Export ${title} as PNG`}
          >
            PNG
          </button>
        </div>
      </figcaption>
      {controls && <div className="plot-controls">{controls}</div>}
      {exportError && (
        <p role="alert" className="muted">
          {exportError}
        </p>
      )}
      <div
        className="science-scroll"
        tabIndex={0}
        role="region"
        aria-label={`${title}, scroll to view full chart on narrow screens`}
      >
        {chart()}
      </div>
      <dialog
        ref={dialog}
        className="plot-dialog"
        aria-label={title}
        onClose={() => setExpanded(false)}
      >
        <div className="plot-dialog-header">
          <h2>{title}</h2>
          <button
            className="bench-button plot-export"
            onClick={() => download()}
            aria-label={`Download ${title} as SVG`}
          >
            Export SVG
          </button>
          <button
            className="bench-button"
            onClick={() => dialog.current?.close()}
            aria-label="Close expanded chart"
          >
            Close
          </button>
        </div>
        {expanded && <div className="science-scroll">{chart(true)}</div>}
      </dialog>
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
  const [hidden, setHidden] = useState<string[]>([]);
  const [zoom, setZoom] = useState("full");
  const [hover, setHover] = useState<number | null>(null);
  const all = [...series.flatMap((s) => s.points), ...markers].filter(
    (p) => Number.isFinite(p.x) && Number.isFinite(p.y),
  );
  if (!all.length)
    return <p className="muted">{title}: no observations available.</p>;
  const [originalMin, xmax] = xDomain ?? [
    Math.min(...all.map((p) => p.x)),
    Math.max(...all.map((p) => p.x)),
  ];
  const xmin =
    zoom === "full" ? originalMin : xmax - (xmax - originalMin) * Number(zoom);
  const visible = series.map((s) => ({
    ...s,
    points: hidden.includes(s.name)
      ? []
      : s.points.filter(
          (p) => p.x >= xmin && p.x <= xmax && Number.isFinite(p.y),
        ),
  }));
  const visibleMarkers = markers.filter(
    (p) => p.x >= xmin && p.x <= xmax && Number.isFinite(p.y),
  );
  const visiblePoints = [
    ...visible.flatMap((s) => s.points),
    ...visibleMarkers,
  ];
  const [ymin, ymax] = yDomain ?? [
    Math.min(...(visiblePoints.length ? visiblePoints : all).map((p) => p.y)),
    Math.max(...(visiblePoints.length ? visiblePoints : all).map((p) => p.y)),
  ];
  const x = (v: number) =>
    xmin === xmax ? 78 : 78 + (592 * (v - xmin)) / (xmax - xmin);
  const y = (v: number) =>
    ymin === ymax ? 104 : 176 - (144 * (v - ymin)) / (ymax - ymin);
  const palette = (s: Series, i: number) =>
    s.color ?? plotColors[i % plotColors.length];
  const legendHeight = 226 + (legend ? Math.ceil(series.length / 3) * 18 : 0);
  return (
    <PlotFrame
      title={title}
      height={legendHeight}
      controls={
        <>
          {legend &&
            series.length > 1 &&
            series.map((s, i) => (
              <button
                key={s.name}
                className="bench-button plot-toggle"
                aria-pressed={!hidden.includes(s.name)}
                onClick={() =>
                  setHidden((old) =>
                    old.includes(s.name)
                      ? old.filter((v) => v !== s.name)
                      : [...old, s.name],
                  )
                }
              >
                <span style={{ background: palette(s, i) }} />
                {s.name}
              </button>
            ))}
          <label className="plot-zoom">
            Window{" "}
            <select
              aria-label={`Window for ${title}`}
              value={zoom}
              onChange={(e) => {
                setZoom(e.target.value);
                setHover(null);
              }}
            >
              <option value="full">All</option>
              <option value="0.5">Last half</option>
              <option value="0.25">Last quarter</option>
            </select>
          </label>
          {hover !== null && (
            <span className="plot-hover">
              {" "}
              {formatX(hover)} ·{" "}
              {visible
                .flatMap((s) => {
                  const nearest = s.points.reduce<Point | undefined>(
                    (a, b) =>
                      !a || Math.abs(b.x - hover) < Math.abs(a.x - hover)
                        ? b
                        : a,
                    undefined,
                  );
                  return nearest ? [s.name + ": " + formatY(nearest.y)] : [];
                })
                .join(" · ")}
            </span>
          )}
        </>
      }
    >
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
      {visible.map((s, i) => (
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
      {visibleMarkers.map((p, i) =>
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
      <rect
        x="78"
        y="32"
        width="592"
        height="144"
        fill="transparent"
        pointerEvents="all"
        onPointerMove={(e) => {
          const bounds = e.currentTarget.getBoundingClientRect();
          setHover(
            xmin +
              (xmax - xmin) *
                Math.max(
                  0,
                  Math.min(1, (e.clientX - bounds.left) / bounds.width),
                ),
          );
        }}
        onPointerLeave={() => setHover(null)}
      />
      {hover !== null && (
        <line
          x1={x(hover)}
          x2={x(hover)}
          y1="32"
          y2="176"
          stroke="var(--muted)"
          strokeDasharray="3 3"
          pointerEvents="none"
        />
      )}
    </PlotFrame>
  );
}

export const points = (values: number[]): Point[] =>
  values.map((y, x) => ({ x, y }));
