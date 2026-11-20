export type Range = [number, number];
export function bounds(values: Iterable<number>): Range {
  let lo = Infinity,
    hi = -Infinity;
  for (const v of values)
    if (Number.isFinite(v)) {
      lo = Math.min(lo, v);
      hi = Math.max(hi, v);
    }
  return lo === Infinity ? [0, 1] : lo === hi ? [lo - 0.5, hi + 0.5] : [lo, hi];
}
export function clampRange(range: Range, domain: Range): Range {
  const span = Math.min(
    domain[1] - domain[0],
    Math.max((domain[1] - domain[0]) / 10000, range[1] - range[0]),
  );
  const left = Math.max(domain[0], Math.min(domain[1] - span, range[0]));
  return [left, left + span];
}
export function zoomRange(
  range: Range,
  anchor: number,
  factor: number,
  domain?: Range,
): Range {
  const next: Range = [
    anchor + (range[0] - anchor) * factor,
    anchor + (range[1] - anchor) * factor,
  ];
  if (
    !next.every(Number.isFinite) ||
    next[1] <= next[0] ||
    next[1] - next[0] < Number.EPSILON * Math.max(1, Math.abs(anchor)) * 16
  )
    return range;
  return domain ? clampRange(next, domain) : next;
}
export function panRange(range: Range, delta: number, domain?: Range): Range {
  const next: Range = [range[0] + delta, range[1] + delta];
  if (!next.every(Number.isFinite) || next[1] <= next[0]) return range;
  return domain ? clampRange(next, domain) : next;
}
// Retain extrema and endpoints in each bucket; hover always reads the original data.
export function decimate<T extends { x: number; y: number }>(
  points: T[],
  limit = 1800,
): T[] {
  if (points.length <= limit) return points;
  const stride = Math.ceil(points.length / Math.floor(limit / 4)),
    out: T[] = [];
  for (let i = 0; i < points.length; i += stride) {
    const chunk = points.slice(i, i + stride);
    let lo = 0,
      hi = 0;
    for (let j = 1; j < chunk.length; j++) {
      if (chunk[j].y < chunk[lo].y) lo = j;
      if (chunk[j].y > chunk[hi].y) hi = j;
    }
    for (const j of [...new Set([0, lo, hi, chunk.length - 1])].sort(
      (a, b) => a - b,
    ))
      out.push(chunk[j]);
  }
  return out;
}
export function nearest<T extends { x: number }>(
  points: T[],
  x: number,
): T | undefined {
  return points.reduce<T | undefined>(
    (a, b) => (!a || Math.abs(b.x - x) < Math.abs(a.x - x) ? b : a),
    undefined,
  );
}
