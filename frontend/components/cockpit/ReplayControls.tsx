"use client";

import { useState } from "react";

import { Badge, Button, ConfirmDialog, Input } from "@/components/ui";
import type { ReplaySessionRead } from "@/lib/types";

interface Props {
  session: ReplaySessionRead | null;
  onStep: (n: number) => Promise<void> | void;
  onJump: (target: number) => Promise<void> | void;
  onComplete: () => Promise<void> | void;
}

export function ReplayControls({ session, onStep, onComplete }: Props) {
  const [stepSize, setStepSize] = useState(1);
  const [confirmComplete, setConfirmComplete] = useState(false);

  const disabled = !session || session.status === "completed";
  const statusTone =
    session?.status === "active"
      ? "success"
      : session?.status === "paused"
        ? "warning"
        : "neutral";

  return (
    <div className="flex flex-wrap items-center gap-2 text-sm">
      <Button
        variant="primary"
        disabled={disabled}
        onClick={() => onStep(stepSize)}
        aria-label={`Step forward ${stepSize} bar${stepSize === 1 ? "" : "s"}`}
      >
        ▶ Step
      </Button>
      <Input
        type="number"
        min={1}
        value={stepSize}
        aria-label="Number of bars to step"
        onChange={(e) => setStepSize(Math.max(1, Number(e.target.value)))}
        className="w-16"
      />

      <span aria-hidden className="text-slate-600">
        |
      </span>
      <Button
        disabled={!session || session.status === "completed"}
        onClick={() => setConfirmComplete(true)}
      >
        Complete
      </Button>

      <span className="ml-auto flex items-center gap-2 text-slate-300">
        <span>
          bar{" "}
          <span className="font-mono text-slate-100">
            {session?.current_bar_index ?? 0}
          </span>
          {" / "}
          <span className="font-mono text-slate-100">
            {(session?.total_bars ?? 1) - 1}
          </span>
        </span>
        <Badge tone={statusTone}>{session?.status ?? "—"}</Badge>
      </span>

      <ConfirmDialog
        open={confirmComplete}
        title="Complete this session?"
        description="Completing cancels pending orders. Open positions stay marked, NOT liquidated: equity contains unrealised P&L. Close positions and advance execution first if you want a fully realised result."
        confirmLabel="Complete session"
        onConfirm={() => {
          setConfirmComplete(false);
          void onComplete();
        }}
        onCancel={() => setConfirmComplete(false)}
      />
    </div>
  );
}

