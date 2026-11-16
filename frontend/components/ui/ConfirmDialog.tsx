"use client";

import { useEffect, useRef } from "react";

import { Button } from "./Button";

/**
 * A lightweight modal confirmation dialog. Uses the native <dialog> element
 * for built-in focus trapping, Escape-to-close, and backdrop, so
 * destructive/irreversible actions (Complete session, cancel order) require
 * an explicit confirm.
 */
export function ConfirmDialog({
  open,
  title,
  description,
  confirmLabel = "Confirm",
  cancelLabel = "Cancel",
  destructive = false,
  onConfirm,
  onCancel,
}: {
  open: boolean;
  title: string;
  description?: React.ReactNode;
  confirmLabel?: string;
  cancelLabel?: string;
  destructive?: boolean;
  onConfirm: () => void;
  onCancel: () => void;
}) {
  const ref = useRef<HTMLDialogElement>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    if (open && !el.open) el.showModal();
    if (!open && el.open) el.close();
  }, [open]);

  return (
    <dialog
      ref={ref}
      onCancel={(e) => {
        e.preventDefault();
        onCancel();
      }}
      className="rounded-card border border-slate-700 bg-surface p-0 text-slate-100 backdrop:bg-black/60"
      aria-labelledby="confirm-title"
    >
      <div className="w-80 max-w-[90vw] p-4">
        <h2 id="confirm-title" className="text-sm font-semibold text-slate-100">
          {title}
        </h2>
        {description && (
          <p className="mt-2 text-sm text-slate-400">{description}</p>
        )}
        <div className="mt-4 flex justify-end gap-2">
          <Button variant="ghost" size="sm" onClick={onCancel}>
            {cancelLabel}
          </Button>
          <Button
            variant={destructive ? "danger" : "primary"}
            size="sm"
            onClick={onConfirm}
          >
            {confirmLabel}
          </Button>
        </div>
      </div>
    </dialog>
  );
}
