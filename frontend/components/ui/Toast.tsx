"use client";

import { createContext, useCallback, useContext, useMemo, useState } from "react";

import { cn } from "@/lib/cn";

type ToastTone = "success" | "error" | "info";

interface Toast {
  id: number;
  tone: ToastTone;
  message: string;
}

interface ToastApi {
  toast: (message: string, tone?: ToastTone) => void;
  success: (message: string) => void;
  error: (message: string) => void;
}

const ToastContext = createContext<ToastApi | null>(null);

const TONE_STYLES: Record<ToastTone, string> = {
  success: "border-emerald-700 bg-emerald-950 text-emerald-100",
  error: "border-red-700 bg-red-950 text-red-100",
  info: "border-slate-700 bg-slate-900 text-slate-100",
};

const TONE_ICON: Record<ToastTone, string> = {
  success: "✓",
  error: "✕",
  info: "ℹ",
};

const AUTO_DISMISS_MS = 4000;

/**
 * App-level toast host. Wrap the cockpit so any component can call
 * `useToast()` to surface transient success/error feedback (order filled,
 * order cancelled, submit failed) without prop-drilling.
 */
export function ToastProvider({ children }: { children: React.ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);

  const remove = useCallback((id: number) => {
    setToasts((list) => list.filter((t) => t.id !== id));
  }, []);

  const push = useCallback(
    (message: string, tone: ToastTone) => {
      const id = Date.now() + Math.random();
      setToasts((list) => [...list, { id, tone, message }]);
      setTimeout(() => remove(id), AUTO_DISMISS_MS);
    },
    [remove],
  );

  const api = useMemo<ToastApi>(
    () => ({
      toast: (message, tone = "info") => push(message, tone),
      success: (message) => push(message, "success"),
      error: (message) => push(message, "error"),
    }),
    [push],
  );

  return (
    <ToastContext.Provider value={api}>
      {children}
      <div
        className="pointer-events-none fixed bottom-4 right-4 z-50 flex w-80 max-w-[90vw] flex-col gap-2"
        aria-live="polite"
        aria-atomic="false"
      >
        {toasts.map((t) => (
          <div
            key={t.id}
            role={t.tone === "error" ? "alert" : "status"}
            className={cn(
              "pointer-events-auto flex items-start gap-2 rounded border px-3 py-2 text-sm shadow-lg",
              TONE_STYLES[t.tone],
            )}
          >
            <span aria-hidden className="mt-0.5 font-bold">
              {TONE_ICON[t.tone]}
            </span>
            <span className="flex-1">{t.message}</span>
            <button
              type="button"
              onClick={() => remove(t.id)}
              aria-label="Dismiss notification"
              className="text-slate-400 hover:text-slate-200"
            >
              ✕
            </button>
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  );
}

export function useToast(): ToastApi {
  const ctx = useContext(ToastContext);
  if (!ctx) {
    throw new Error("useToast must be used within a ToastProvider");
  }
  return ctx;
}
