// Mirrors the retained backend Pydantic schemas.

export type Iso8601 = string;
export type IsoDate = string;

export interface OHLCVBar {
  date: IsoDate;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  adjusted_close: number | null;
}

export interface InstrumentSummary {
  multiplier?: number | null;
  instrument: string;
  name: string;
  bar_count: number;
  first_date: IsoDate | null;
  last_date: IsoDate | null;
}

export interface UploadResponse {
  instrument: string;
  loaded_count: number;
  total_rows: number;
  errors: RowError[];
}

export interface RowError {
  row_number: number;
  field: string | null;
  message: string;
}

export type SessionStatus = "active" | "paused" | "completed";

export interface ReplaySessionRead {
  session_id: string;
  simulation_version?: number;
  instrument: string;
  start_date: IsoDate;
  end_date: IsoDate;
  current_bar_index: number;
  status: SessionStatus;
  initial_cash: number;
  created_at: Iso8601;
  total_bars: number;
  multiplier?: number;
  fee_per_unit?: number;
  spread_bps?: number;
}

export interface CreateSessionRequest {
  instrument: string;
  start_date: IsoDate;
  end_date: IsoDate;
  initial_cash?: number;
  multiplier?: number;
  fee_per_unit?: number;
  spread_bps?: number;
}

export type OrderSide = "buy" | "sell";
export type OrderType = "market" | "limit" | "stop";
export type OrderStatus =
  | "pending"
  | "filled"
  | "partially_filled"
  | "cancelled"
  | "rejected";
export type InstrumentKind = "underlying";

export interface CreateOrderRequest {
  session_id: string;
  side: OrderSide;
  order_type: OrderType;
  quantity: number;
  limit_price?: number | null;
  stop_price?: number | null;
  stop_loss?: number | null;
  take_profit?: number | null;
  rationale: string;
  instrument_kind?: InstrumentKind;
}

export interface OrderRead {
  order_id: string;
  session_id: string;
  side: OrderSide;
  order_type: OrderType;
  quantity: number;
  limit_price: number | null;
  stop_price: number | null;
  stop_loss: number | null;
  take_profit: number | null;
  status: OrderStatus;
  submitted_at_bar: number;
  submitted_at: Iso8601;
  rationale: string;
  parent_order_id: string | null;
  oco_pair_id: string | null;
  instrument_kind?: InstrumentKind;
}

export type PositionSide = "long" | "short" | "flat";

export interface PositionRead {
  instrument: string;
  side: PositionSide;
  quantity: number;
  avg_entry_price: number;
  unrealized_pnl: number;
  realized_pnl: number;
  multiplier?: number;
  instrument_kind?: InstrumentKind;
}

export interface PortfolioSnapshotRead {
  total_fees?: number;
  session_id: string;
  bar_index: number;
  cash: number;
  equity: number;
  positions: PositionRead[];
  total_realized_pnl: number;
  total_unrealized_pnl: number;
  drawdown_pct: number;
}

export interface RiskMetricsRead {
  max_drawdown_pct: number;
  current_drawdown_pct: number;
  sharpe_ratio: number | null;
  win_rate: number | null;
  avg_win: number | null;
  avg_loss: number | null;
  expectancy: number | null;
  risk_per_trade_pct: number;
  max_position_exposure_pct: number;
}

export interface RiskLimitsRead {
  max_drawdown_pct: number;
  max_risk_per_trade_pct: number;
  max_open_positions: number;
  max_exposure_pct: number;
}

export interface RiskLimitsUpdate {
  max_drawdown_pct?: number;
  max_risk_per_trade_pct?: number;
  max_open_positions?: number;
  max_exposure_pct?: number;
}

export interface ApiError {
  error_code: string;
  detail: string;
  timestamp?: Iso8601;
}
