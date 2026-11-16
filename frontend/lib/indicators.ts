// Pure technical-indicator math, computed entirely from the bars passed in.
//
// ANTI-LOOKAHEAD GUARANTEE: every function consumes only the `bars` array it
// is given. The chart only ever passes the *visible* bars (0..current), so no
// indicator can incorporate a future value. There is no global state and no
// I/O — these are deterministic transforms, unit-tested in indicators math.
//
// All public functions return `IndicatorPoint[]` ({ time, value }), already
// filtered to defined points, ready to hand to a lightweight-charts line or
// histogram series via setData().

import type { OHLCVBar } from "./types";

export interface IndicatorPoint {
  time: string; // YYYY-MM-DD, matches lightweight-charts time
  value: number;
}

export interface BollingerBands {
  upper: IndicatorPoint[];
  middle: IndicatorPoint[];
  lower: IndicatorPoint[];
}

export interface MacdResult {
  macd: IndicatorPoint[];
  signal: IndicatorPoint[];
  histogram: IndicatorPoint[];
}

// ──────────────────────────────────────────────────────────────────────────
// Internal helpers operating on raw number arrays (null = undefined at index)
// ──────────────────────────────────────────────────────────────────────────

function smaValues(values: number[], period: number): (number | null)[] {
  const out: (number | null)[] = new Array(values.length).fill(null);
  if (period < 1) return out;
  let sum = 0;
  for (let i = 0; i < values.length; i++) {
    sum += values[i];
    if (i >= period) sum -= values[i - period];
    if (i >= period - 1) out[i] = sum / period;
  }
  return out;
}

function emaValues(values: number[], period: number): (number | null)[] {
  const out: (number | null)[] = new Array(values.length).fill(null);
  if (period < 1 || values.length < period) return out;
  const k = 2 / (period + 1);
  // Seed with the SMA of the first `period` values (standard convention).
  let seed = 0;
  for (let i = 0; i < period; i++) seed += values[i];
  let prev = seed / period;
  out[period - 1] = prev;
  for (let i = period; i < values.length; i++) {
    prev = values[i] * k + prev * (1 - k);
    out[i] = prev;
  }
  return out;
}

function stddevValues(values: number[], period: number): (number | null)[] {
  // Population standard deviation over a rolling window (Bollinger convention).
  const out: (number | null)[] = new Array(values.length).fill(null);
  if (period < 1) return out;
  for (let i = period - 1; i < values.length; i++) {
    let mean = 0;
    for (let j = i - period + 1; j <= i; j++) mean += values[j];
    mean /= period;
    let variance = 0;
    for (let j = i - period + 1; j <= i; j++) {
      const d = values[j] - mean;
      variance += d * d;
    }
    out[i] = Math.sqrt(variance / period);
  }
  return out;
}

// ──────────────────────────────────────────────────────────────────────────
// Public indicators
// ──────────────────────────────────────────────────────────────────────────

export function sma(bars: OHLCVBar[], period: number): IndicatorPoint[] {
  const vals = smaValues(bars.map((b) => b.close), period);
  return toPoints(bars, vals);
}

export function ema(bars: OHLCVBar[], period: number): IndicatorPoint[] {
  const vals = emaValues(bars.map((b) => b.close), period);
  return toPoints(bars, vals);
}

export function bollinger(
  bars: OHLCVBar[],
  period = 20,
  mult = 2,
): BollingerBands {
  const closes = bars.map((b) => b.close);
  const mid = smaValues(closes, period);
  const sd = stddevValues(closes, period);
  const upper: IndicatorPoint[] = [];
  const middle: IndicatorPoint[] = [];
  const lower: IndicatorPoint[] = [];
  for (let i = 0; i < bars.length; i++) {
    const m = mid[i];
    const s = sd[i];
    if (m === null || s === null) continue;
    middle.push({ time: bars[i].date, value: m });
    upper.push({ time: bars[i].date, value: m + mult * s });
    lower.push({ time: bars[i].date, value: m - mult * s });
  }
  return { upper, middle, lower };
}

export function atr(bars: OHLCVBar[], period = 14): IndicatorPoint[] {
  // Wilder's ATR. TR = max(H-L, |H-prevClose|, |L-prevClose|).
  const out: IndicatorPoint[] = [];
  if (bars.length < period + 1) return out;
  const tr: number[] = [];
  for (let i = 0; i < bars.length; i++) {
    if (i === 0) {
      tr.push(bars[i].high - bars[i].low);
    } else {
      const prevClose = bars[i - 1].close;
      tr.push(
        Math.max(
          bars[i].high - bars[i].low,
          Math.abs(bars[i].high - prevClose),
          Math.abs(bars[i].low - prevClose),
        ),
      );
    }
  }
  // First ATR = simple average of the first `period` true ranges (indices 1..period).
  let prevAtr = 0;
  for (let i = 1; i <= period; i++) prevAtr += tr[i];
  prevAtr /= period;
  out.push({ time: bars[period].date, value: prevAtr });
  for (let i = period + 1; i < bars.length; i++) {
    prevAtr = (prevAtr * (period - 1) + tr[i]) / period;
    out.push({ time: bars[i].date, value: prevAtr });
  }
  return out;
}

export function rsi(bars: OHLCVBar[], period = 14): IndicatorPoint[] {
  // Wilder's RSI.
  const out: IndicatorPoint[] = [];
  if (bars.length < period + 1) return out;
  const closes = bars.map((b) => b.close);

  let gainSum = 0;
  let lossSum = 0;
  for (let i = 1; i <= period; i++) {
    const change = closes[i] - closes[i - 1];
    if (change >= 0) gainSum += change;
    else lossSum -= change;
  }
  let avgGain = gainSum / period;
  let avgLoss = lossSum / period;
  out.push({ time: bars[period].date, value: rsiFrom(avgGain, avgLoss) });

  for (let i = period + 1; i < bars.length; i++) {
    const change = closes[i] - closes[i - 1];
    const gain = change >= 0 ? change : 0;
    const loss = change < 0 ? -change : 0;
    avgGain = (avgGain * (period - 1) + gain) / period;
    avgLoss = (avgLoss * (period - 1) + loss) / period;
    out.push({ time: bars[i].date, value: rsiFrom(avgGain, avgLoss) });
  }
  return out;
}

function rsiFrom(avgGain: number, avgLoss: number): number {
  if (avgLoss === 0) return 100;
  const rs = avgGain / avgLoss;
  return 100 - 100 / (1 + rs);
}

export function macd(
  bars: OHLCVBar[],
  fast = 12,
  slow = 26,
  signal = 9,
): MacdResult {
  const closes = bars.map((b) => b.close);
  const fastE = emaValues(closes, fast);
  const slowE = emaValues(closes, slow);

  // MACD line is defined only where both EMAs exist (i.e. from slow-1).
  const macdLineFull: (number | null)[] = new Array(bars.length).fill(null);
  const macdCompact: number[] = [];
  const macdIndices: number[] = [];
  for (let i = 0; i < bars.length; i++) {
    if (fastE[i] !== null && slowE[i] !== null) {
      const v = (fastE[i] as number) - (slowE[i] as number);
      macdLineFull[i] = v;
      macdCompact.push(v);
      macdIndices.push(i);
    }
  }

  // Signal = EMA(signal) of the compact MACD line.
  const signalCompact = emaValues(macdCompact, signal);

  const macdPts: IndicatorPoint[] = [];
  const signalPts: IndicatorPoint[] = [];
  const histPts: IndicatorPoint[] = [];
  for (let k = 0; k < macdIndices.length; k++) {
    const barIdx = macdIndices[k];
    const time = bars[barIdx].date;
    const m = macdCompact[k];
    macdPts.push({ time, value: m });
    const s = signalCompact[k];
    if (s !== null) {
      signalPts.push({ time, value: s });
      histPts.push({ time, value: m - s });
    }
  }
  return { macd: macdPts, signal: signalPts, histogram: histPts };
}

export function anchoredVwap(bars: OHLCVBar[]): IndicatorPoint[] {
  // Cumulative VWAP anchored at the first visible bar. For daily bars this is
  // a session-to-date volume-weighted average of the typical price.
  const out: IndicatorPoint[] = [];
  let cumPV = 0;
  let cumV = 0;
  for (const b of bars) {
    const typical = (b.high + b.low + b.close) / 3;
    cumPV += typical * b.volume;
    cumV += b.volume;
    if (cumV > 0) out.push({ time: b.date, value: cumPV / cumV });
  }
  return out;
}

// Latest ATR value (used for the risk-sizing overlay = "1R" band).
export function latestAtr(bars: OHLCVBar[], period = 14): number | null {
  const series = atr(bars, period);
  return series.length ? series[series.length - 1].value : null;
}

// ──────────────────────────────────────────────────────────────────────────

function toPoints(bars: OHLCVBar[], values: (number | null)[]): IndicatorPoint[] {
  const out: IndicatorPoint[] = [];
  for (let i = 0; i < bars.length; i++) {
    if (values[i] !== null) out.push({ time: bars[i].date, value: values[i] as number });
  }
  return out;
}
