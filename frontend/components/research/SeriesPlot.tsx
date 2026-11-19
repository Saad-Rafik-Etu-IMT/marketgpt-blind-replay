type PlotProps = {
  values: number[];
  label: string;
  percent?: boolean;
  startLabel?: string;
  endLabel?: string;
};

export function SeriesPlot({ values, label, percent, startLabel, endLabel }: PlotProps) {
  if (!values.length) return null;

  const min = Math.min(...values);
  const max = Math.max(...values);
  const format = (v: number) => percent ? (100 * v).toFixed(1) + "%" : v.toLocaleString("en-US", { maximumFractionDigits: 4 });
  const y = (v: number) => max === min ? 50 : 96 - 92 * (v - min) / (max - min);
  const points = values.map((value, i) => {
    const x = values.length === 1 ? 50 : 100 * i / (values.length - 1);
    return x + "," + y(value);
  }).join(" ");

  return <figure className="series-plot">
    <figcaption><strong>{label}</strong><span>{format(values[0])} → {format(values[values.length - 1])}</span></figcaption>
    <div className="plot-body">
      <div className="plot-scale">
        <span>{format(max)}</span>
        {min !== max && <span>{format(min)}</span>}
      </div>
      <svg role="img" aria-label={label} viewBox="0 0 100 100" preserveAspectRatio="none">
        <title>{label}. Minimum {format(min)}, maximum {format(max)}.</title>
        {[4, 50, 96].map(height => <line key={height} x1="0" x2="100" y1={height} y2={height} vectorEffect="non-scaling-stroke" />)}
        {values.length > 1 && <polygon points={'0,100 ' + points + ' 100,100'} />}
        <polyline points={points} vectorEffect="non-scaling-stroke" />
        {values.length === 1 && <line x1="49" x2="51" y1="50" y2="50" className="single-point" vectorEffect="non-scaling-stroke" />}
      </svg>
    </div>
    <div className="plot-bounds"><span>{startLabel || "First"}</span><span>{endLabel || "Latest"}</span></div>
  </figure>;
}
