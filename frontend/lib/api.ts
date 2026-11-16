// Typed thin wrapper around fetch().
import { API_BASE_URL } from "./constants";
import type {
  ApiError,
  CreateOrderRequest,
  CreateSessionRequest,
  InstrumentSummary,
  OHLCVBar,
  OrderRead,
  PortfolioSnapshotRead,
  ReplaySessionRead,
  RiskLimitsRead,
  RiskLimitsUpdate,
  RiskMetricsRead,
  UploadResponse,
} from "./types";

export class ApiException extends Error {
  readonly status: number;
  readonly errorCode: string;
  readonly raw: unknown;

  constructor(message: string, status: number, errorCode: string, raw: unknown) {
    super(message);
    this.status = status;
    this.errorCode = errorCode;
    this.raw = raw;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
    ...init,
  });
  if (!response.ok) {
    let parsed: ApiError | { detail?: unknown } = { detail: response.statusText };
    try {
      parsed = await response.json();
    } catch {
      // The body was not JSON; retain the HTTP error.
    }
    const errorCode =
      typeof (parsed as ApiError).error_code === "string"
        ? (parsed as ApiError).error_code
        : "UNKNOWN";
    const detail =
      typeof (parsed as ApiError).detail === "string"
        ? (parsed as ApiError).detail
        : `HTTP ${response.status}`;
    throw new ApiException(detail, response.status, errorCode, parsed);
  }
  if (response.status === 204) {
    return undefined as T;
  }
  return (await response.json()) as T;
}

export const health = () =>
  request<{ status: string; version: string }>("/api/health");

export const listInstruments = () =>
  request<InstrumentSummary[]>("/api/market/instruments");

export const uploadCsv = async (
  instrument: string,
  file: File,
): Promise<UploadResponse> => {
  const form = new FormData();
  form.append("file", file);
  const response = await fetch(
    `${API_BASE_URL}/api/market/upload?instrument=${encodeURIComponent(instrument)}`,
    { method: "POST", body: form },
  );
  if (!response.ok) {
    const body = (await response.json()) as ApiError;
    throw new ApiException(body.detail, response.status, body.error_code, body);
  }
  return (await response.json()) as UploadResponse;
};

export const createSession = (body: CreateSessionRequest) =>
  request<ReplaySessionRead>("/api/replay/sessions", {
    method: "POST",
    body: JSON.stringify(body),
  });
export const listSessions = () =>
  request<ReplaySessionRead[]>("/api/replay/sessions");
export const getSession = (id: string) =>
  request<ReplaySessionRead>(`/api/replay/sessions/${id}`);
export const stepSession = (id: string, n = 1) =>
  request<ReplaySessionRead>(`/api/replay/sessions/${id}/step`, {
    method: "POST",
    body: JSON.stringify({ n }),
  });
export const jumpToBar = (id: string, target_index: number) =>
  request<ReplaySessionRead>(`/api/replay/sessions/${id}/jump`, {
    method: "POST",
    body: JSON.stringify({ target_index }),
  });
export const getVisibleBars = (id: string) =>
  request<OHLCVBar[]>(`/api/replay/sessions/${id}/visible-bars`);
export const completeSession = (id: string) =>
  request<ReplaySessionRead>(`/api/replay/sessions/${id}/complete`, {
    method: "POST",
  });

export const submitOrder = (body: CreateOrderRequest) =>
  request<OrderRead>("/api/orders", {
    method: "POST",
    body: JSON.stringify(body),
  });
export const listOrders = (sessionId: string) =>
  request<OrderRead[]>(`/api/orders?session_id=${encodeURIComponent(sessionId)}`);
export const cancelOrder = (orderId: string) =>
  request<OrderRead>(`/api/orders/${orderId}`, { method: "DELETE" });
export const getPortfolio = (sessionId: string) =>
  request<PortfolioSnapshotRead>(`/api/portfolio/${sessionId}`);
export const getEquityCurve = (sessionId: string) =>
  request<PortfolioSnapshotRead[]>(`/api/portfolio/${sessionId}/history`);
export const getRiskMetrics = (sessionId: string) =>
  request<RiskMetricsRead>(`/api/risk/${sessionId}`);
export const getRiskLimits = (sessionId: string) =>
  request<RiskLimitsRead>(`/api/risk/${sessionId}/limits`);
export const updateRiskLimits = (sessionId: string, body: RiskLimitsUpdate) =>
  request<RiskLimitsRead>(`/api/risk/${sessionId}/limits`, {
    method: "PUT",
    body: JSON.stringify(body),
  });
