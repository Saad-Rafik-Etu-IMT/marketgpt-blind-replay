import { API_BASE_URL } from "@/lib/constants";

export const pct = (x: number) => (100 * x).toFixed(1) + "%";
export async function api<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(
    (path.startsWith("/jury/")
      ? (process.env.NEXT_PUBLIC_JURY_API_BASE_URL || "http://localhost:8001") +
        "/api"
      : API_BASE_URL + "/api/studies") + path,
    body === undefined
      ? {}
      : {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        },
  );
  if (!res.ok) {
    const msg = await res.json().catch(() => ({}));
    throw Error(
      typeof msg.detail === "string"
        ? msg.detail
        : "Request failed (" + res.status + ")",
    );
  }
  return res.json();
}
