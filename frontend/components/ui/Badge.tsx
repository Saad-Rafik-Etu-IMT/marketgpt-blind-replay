import { cn } from "@/lib/cn";

type Tone = "neutral" | "success" | "warning" | "danger" | "info";

const TONES: Record<Tone, string> = {
  neutral: "bg-slate-800 text-slate-300 border-slate-700",
  success: "bg-emerald-950 text-emerald-300 border-emerald-800",
  warning: "bg-amber-950 text-amber-300 border-amber-800",
  danger: "bg-red-950 text-red-300 border-red-800",
  info: "bg-sky-950 text-sky-300 border-sky-800",
};

/**
 * A small status pill. Callers should pair colour with text so status is
 * never conveyed by colour alone (accessibility).
 */
export function Badge({
  tone = "neutral",
  children,
  className,
}: {
  tone?: Tone;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded border px-1.5 py-0.5 text-xs font-medium",
        TONES[tone],
        className,
      )}
    >
      {children}
    </span>
  );
}
