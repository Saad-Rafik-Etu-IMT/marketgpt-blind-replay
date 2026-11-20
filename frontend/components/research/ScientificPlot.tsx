"use client";

import {
  createContext,
  useContext,
  useEffect,
  useId,
  useRef,
  useState,
} from "react";
import {
  bounds,
  clampRange,
  decimate,
  nearest,
  panRange,
  zoomRange,
  type Range,
} from "@/lib/research/chart-navigation";

const Sync = createContext<{
  range: Range | null;
  setRange: (r: Range | null) => void;
  hover: number | null;
  setHover: (n: number | null) => void;
} | null>(null);
export function ChartSync({ children }: { children: React.ReactNode }) {
  const [range, setRange] = useState<Range | null>(null);
  const [hover, setHover] = useState<number | null>(null);
  return (
    <Sync.Provider value={{ range, setRange, hover, setHover }}>
      {children}
    </Sync.Provider>
  );
}
function ChartSurface({
  onZoom,
  onPan,
  onHover,
  onPick,
  range,
  priceRange,
}: {
  onZoom: (x: number, y: number, factor: number, vertical: boolean) => void;
  onPan: (dx: number, dy: number, vertical: boolean) => void;
  onHover: (x: number | null, y?: number) => void;
  onPick?: (x: number) => void;
  range: Range;
  priceRange: Range;
}) {
  const ref = useRef<SVGRectElement>(null);
  const drag = useRef<{
    x: number;
    y: number;
    distance: number;
    vertical: boolean;
  } | null>(null);
  const cb = useRef(onZoom);
  useEffect(() => {
    cb.current = onZoom;
  }, [onZoom]);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const wheel = (e: WheelEvent) => {
      e.preventDefault();
      const b = el.getBoundingClientRect();
      cb.current(
        (e.clientX - b.left) / b.width,
        (e.clientY - b.top) / b.height,
        Math.exp(Math.max(-3, Math.min(3, e.deltaY * 0.002))),
        e.shiftKey,
      );
    };
    el.addEventListener("wheel", wheel, { passive: false });
    return () => el.removeEventListener("wheel", wheel);
  }, []);
  return (
    <rect
      ref={ref}
      data-range-start={range[0]}
      data-range-end={range[1]}
      data-y-start={priceRange[0]}
      data-y-end={priceRange[1]}
      x="78"
      y="32"
      width="592"
      height="144"
      fill="transparent"
      pointerEvents="all"
      className="chart-surface"
      onPointerDown={(e) => {
        if (e.button !== 0) return;
        e.currentTarget.setPointerCapture(e.pointerId);
        drag.current = {
          x: e.clientX,
          y: e.clientY,
          distance: 0,
          vertical: e.shiftKey,
        };
      }}
      onPointerMove={(e) => {
        const b = e.currentTarget.getBoundingClientRect();
        const x = Math.max(0, Math.min(1, (e.clientX - b.left) / b.width)),
          y = Math.max(0, Math.min(1, (e.clientY - b.top) / b.height));
        if (drag.current) {
          const d = drag.current,
            dx = e.clientX - d.x,
            dy = e.clientY - d.y;
          d.distance += Math.abs(dx) + Math.abs(dy);
          onPan(dx / b.width, dy / b.height, d.vertical);
          d.x = e.clientX;
          d.y = e.clientY;
        }
        onHover(x, y);
      }}
      onPointerUp={(e) => {
        if (drag.current && drag.current.distance < 5) {
          const b = e.currentTarget.getBoundingClientRect();
          onPick?.((e.clientX - b.left) / b.width);
        }
        drag.current = null;
      }}
      onPointerCancel={() => {
        drag.current = null;
        onHover(null);
      }}
      onPointerLeave={() => {
        if (!drag.current) onHover(null);
      }}
    />
  );
}

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
  sync = false,
  onSelect,
  candles = [],
  bands = [],
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
  sync?: boolean;
  onSelect?: (x: number) => void;
  candles?: {
    x: number;
    open: number;
    high: number;
    low: number;
    close: number;
  }[];
  bands?: { from: number; to: number; color: string; label: string }[];
}) {
  const shared = useContext(Sync);
  const group = sync ? shared : null;
  const clip = useId().replaceAll(":", "");
  const [hidden, setHidden] = useState<string[]>([]);
  const [zoom, setZoom] = useState("full");
  const [localHover, setLocalHover] = useState<number | null>(null);
  const [hoverY, setHoverY] = useState<number | null>(null);
  const [localRange, setLocalRange] = useState<Range | null>(null);
  const [manualY, setManualY] = useState<Range | null>(null);
  const hover = group ? group.hover : localHover;
  const setHover = group ? group.setHover : setLocalHover;
  const setRange = group ? group.setRange : setLocalRange;
  const all = [
    ...series.flatMap((s) => s.points),
    ...markers,
    ...candles.flatMap((c) => [
      { x: c.x, y: c.low },
      { x: c.x, y: c.high },
    ]),
  ].filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y));
  if (!all.length)
    return <p className="muted">{title}: no observations available.</p>;
  const domain =
    xDomain && xDomain[0] < xDomain[1] ? xDomain : bounds(all.map((p) => p.x));
  const current = group ? group.range : localRange;
  const [xmin, xmax] = current
    ? clampRange(current, domain)
    : group || zoom === "full"
      ? domain
      : [domain[1] - (domain[1] - domain[0]) * Number(zoom), domain[1]];
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
    ...candles
      .filter((c) => c.x >= xmin && c.x <= xmax)
      .flatMap((c) => [
        { x: c.x, y: c.low },
        { x: c.x, y: c.high },
      ]),
  ];
  const [ymin, ymax] =
    manualY ??
    yDomain ??
    bounds((visiblePoints.length ? visiblePoints : all).map((p) => p.y));
  const range: Range = [xmin, xmax],
    priceRange: Range = [ymin, ymax];
  const zoomX = (factor: number, anchor = (xmin + xmax) / 2) =>
    setRange(zoomRange(range, anchor, factor, domain));
  const activeCandle = hover === null ? undefined : nearest(candles, hover);
  const visibleCandles = candles.filter((c) => c.x >= xmin && c.x <= xmax);
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
                setRange(
                  e.target.value === "full"
                    ? null
                    : [
                        domain[1] -
                          (domain[1] - domain[0]) * Number(e.target.value),
                        domain[1],
                      ],
                );
              }}
            >
              <option value="full">All</option>
              <option value="0.5">Last half</option>
              <option value="0.25">Last quarter</option>
            </select>
          </label>
          <button
            className="bench-button"
            aria-label={`Zoom in ${title}`}
            onClick={() => zoomX(0.7)}
          >
            +
          </button>
          <button
            className="bench-button"
            aria-label={`Zoom out ${title}`}
            onClick={() => zoomX(1.4)}
          >
            −
          </button>
          <button
            className="bench-button"
            onClick={() =>
              setRange(panRange(range, -(xmax - xmin) * 0.25, domain))
            }
          >
            Earlier
          </button>
          <button
            className="bench-button"
            onClick={() =>
              setRange(panRange(range, (xmax - xmin) * 0.25, domain))
            }
          >
            Later
          </button>
          <button
            className="bench-button"
            onClick={() => {
              setRange(null);
              setZoom("full");
              setManualY(null);
              setHover(null);
            }}
          >
            Fit / reset
          </button>
          <button
            className="bench-button"
            aria-pressed={manualY === null}
            onClick={() => setManualY(manualY ? null : priceRange)}
          >
            {manualY ? "Manual Y" : "Auto Y"}
          </button>
          <span className="muted chart-help">
            Wheel zoom · drag time · Shift-wheel / Shift-drag price
          </span>
          <span className="plot-hover">
            {hover !== null ? (
              <>
                {" "}
                {formatX(hover)} ·{" "}
                {hoverY !== null ? `Cursor Y: ${formatY(hoverY)} · ` : ""}
                {activeCandle &&
                  `O ${formatY(activeCandle.open)} · H ${formatY(activeCandle.high)} · L ${formatY(activeCandle.low)} · C ${formatY(activeCandle.close)} · `}
                {visibleMarkers
                  .filter((m) => Math.abs(m.x - hover) < 0.6)
                  .map((m) => m.label)
                  .join(" · ")}
                {bands.find((b) => hover >= b.from && hover <= b.to)?.label &&
                  ` · ${bands.find((b) => hover >= b.from && hover <= b.to)?.label} · `}
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
              </>
            ) : (
              "Move the pointer over the plot to inspect values."
            )}
          </span>
        </>
      }
    >
      <defs>
        <clipPath id={clip}>
          <rect x="78" y="32" width="592" height="144" />
        </clipPath>
      </defs>
      <g clipPath={`url(#${clip})`}>
        {bands
          .filter((b) => b.to >= xmin && b.from <= xmax)
          .map((b, i) => (
            <rect
              key={i}
              x={x(b.from)}
              y="32"
              width={Math.max(1, x(b.to) - x(b.from))}
              height="144"
              fill={b.color}
              opacity=".10"
            />
          ))}
      </g>
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
      {Array.from({ length: 5 }, (_, i) => xmin + ((xmax - xmin) * i) / 4).map(
        (v) => (
          <text
            key={v}
            x={x(v)}
            y="195"
            fill="var(--muted)"
            fontSize="11"
            textAnchor={v === xmin ? "start" : v === xmax ? "end" : "middle"}
          >
            {formatX(v)}
          </text>
        ),
      )}
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
              clipPath={`url(#${clip})`}
              points={decimate(s.points)
                .map((p) => `${x(p.x)},${y(p.y)}`)
                .join(" ")}
              fill="none"
              stroke={palette(s, i)}
              strokeWidth="2.2"
              strokeDasharray={s.dashed ? "6 4" : undefined}
              strokeLinejoin="round"
            />
          )}
          {(s.dots || s.scatter || s.points.length === 1) &&
            decimate(s.points).map((p, j) => (
              <circle
                key={j}
                cx={x(p.x)}
                cy={y(p.y)}
                r="4"
                fill={palette(s, i)}
                clipPath={`url(#${clip})`}
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
      <g clipPath={`url(#${clip})`}>
        {visibleCandles.length <= 1800 ? (
          visibleCandles.map((c) => {
            const color = c.close >= c.open ? "#148b82" : "#b95061";
            const w = Math.max(
              0.6,
              Math.min(9, (592 / (xmax - xmin || 1)) * 0.65),
            );
            return (
              <g key={c.x}>
                <line
                  x1={x(c.x)}
                  x2={x(c.x)}
                  y1={y(c.high)}
                  y2={y(c.low)}
                  stroke={color}
                />
                <rect
                  x={x(c.x) - w / 2}
                  width={w}
                  y={Math.min(y(c.open), y(c.close))}
                  height={Math.max(1, Math.abs(y(c.open) - y(c.close)))}
                  fill={color}
                />
              </g>
            );
          })
        ) : (
          <polyline
            points={decimate(
              visibleCandles.map((c) => ({ x: c.x, y: c.close })),
            )
              .map((c) => `${x(c.x)},${y(c.y)}`)
              .join(" ")}
            fill="none"
            stroke={plotColors[0]}
          />
        )}
      </g>
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
      <ChartSurface
        range={range}
        priceRange={priceRange}
        onZoom={(fx, fy, factor, vertical) =>
          vertical
            ? setManualY(
                zoomRange(priceRange, ymax - fy * (ymax - ymin), factor),
              )
            : zoomX(factor, xmin + fx * (xmax - xmin))
        }
        onPan={(dx, dy, vertical) => {
          if (vertical) setManualY(panRange(priceRange, dy * (ymax - ymin)));
          else {
            setRange(panRange(range, -dx * (xmax - xmin), domain));
            if (manualY) setManualY(panRange(priceRange, dy * (ymax - ymin)));
          }
        }}
        onHover={(fx, fy) => {
          setHover(fx === null ? null : xmin + fx * (xmax - xmin));
          setHoverY(fy === undefined ? null : ymax - fy * (ymax - ymin));
        }}
        onPick={(fx) => onSelect?.(xmin + fx * (xmax - xmin))}
      />
      {hoverY !== null && hover !== null && (
        <line
          x1="78"
          x2="670"
          y1={y(hoverY)}
          y2={y(hoverY)}
          stroke="var(--muted)"
          strokeDasharray="3 3"
          pointerEvents="none"
        />
      )}
      {hover !== null && hover >= xmin && hover <= xmax && (
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
