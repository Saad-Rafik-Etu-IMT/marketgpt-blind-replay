import { cn } from "@/lib/cn";

/**
 * A shimmering placeholder block shown while data loads, so panels reserve
 * space instead of popping in. Decorative — hidden from screen readers.
 */
export function Skeleton({ className }: { className?: string }) {
  return (
    <div
      aria-hidden
      className={cn(
        "animate-pulse rounded bg-slate-800/70",
        className,
      )}
    />
  );
}
