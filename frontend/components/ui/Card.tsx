import { cn } from "@/lib/cn";

/**
 * Panel — the standard cockpit surface: a rounded, bordered card with an
 * optional titled header (and header actions slot). Replaces the ad-hoc
 * `bg-slate-900 p-3 rounded border border-slate-800` blocks scattered
 * across the panels.
 */
export function Panel({
  title,
  actions,
  children,
  className,
  bodyClassName,
  as: Tag = "section",
}: {
  title?: React.ReactNode;
  actions?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
  bodyClassName?: string;
  as?: "section" | "div" | "aside";
}) {
  return (
    <Tag
      className={cn(
        "rounded-card border border-slate-800 bg-surface",
        className,
      )}
    >
      {(title || actions) && (
        <header className="flex items-center justify-between gap-2 border-b border-slate-800 px-3 py-2">
          {title ? (
            <h2 className="text-sm font-semibold text-slate-200">{title}</h2>
          ) : (
            <span />
          )}
          {actions}
        </header>
      )}
      <div className={cn("p-3", bodyClassName)}>{children}</div>
    </Tag>
  );
}

export function Card({
  children,
  className,
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "rounded-card border border-slate-800 bg-surface p-3",
        className,
      )}
    >
      {children}
    </div>
  );
}
