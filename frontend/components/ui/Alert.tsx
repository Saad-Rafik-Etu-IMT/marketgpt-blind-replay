import { cn } from "@/lib/cn";

type Tone = "info" | "success" | "warning" | "danger";

const TONES: Record<Tone, string> = {
  info: "border-sky-800 bg-sky-950 text-sky-200",
  success: "border-emerald-800 bg-emerald-950 text-emerald-200",
  warning: "border-amber-800 bg-amber-950 text-amber-200",
  danger: "border-red-700 bg-red-950 text-red-200",
};

const ICONS: Record<Tone, string> = {
  info: "ℹ",
  success: "✓",
  warning: "⚠",
  danger: "✕",
};

/**
 * Inline alert banner. Uses `role="alert"` for assertive tones so screen
 * readers announce connection/error states, and pairs an icon with text so
 * meaning is not colour-only.
 */
export function Alert({
  tone = "info",
  title,
  children,
  className,
}: {
  tone?: Tone;
  title?: string;
  children?: React.ReactNode;
  className?: string;
}) {
  const assertive = tone === "danger" || tone === "warning";
  return (
    <div
      role={assertive ? "alert" : "status"}
      className={cn(
        "flex items-start gap-2 rounded border px-3 py-2 text-sm",
        TONES[tone],
        className,
      )}
    >
      <span aria-hidden className="mt-0.5 font-bold">
        {ICONS[tone]}
      </span>
      <div>
        {title && <div className="font-semibold">{title}</div>}
        {children && <div>{children}</div>}
      </div>
    </div>
  );
}
