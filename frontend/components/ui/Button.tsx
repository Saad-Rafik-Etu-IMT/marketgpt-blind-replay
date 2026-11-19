import { forwardRef } from "react";

import { cn } from "@/lib/cn";

type Variant = "primary" | "secondary" | "danger" | "ghost";
type Size = "sm" | "md" | "icon";

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  size?: Size;
}

const VARIANTS: Record<Variant, string> = {
  primary: "bg-primary hover:bg-primary-hover text-surface",
  secondary: "bg-surface-raised hover:bg-slate-700 text-slate-100",
  danger: "bg-danger hover:bg-danger-hover text-white",
  ghost: "bg-transparent hover:bg-surface-raised text-slate-300",
};

const SIZES: Record<Size, string> = {
  sm: "px-2 py-1 text-xs",
  md: "px-3 py-1.5 text-sm",
  icon: "p-1.5 text-sm leading-none",
};

/**
 * The single button primitive for the cockpit. Handles variants, sizes, a
 * consistent disabled treatment, and inherits the global focus-visible
 * ring. Icon-only buttons must pass an `aria-label` (enforced by callers).
 */
export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ variant = "secondary", size = "md", className, type, ...props }, ref) => (
    <button
      ref={ref}
      type={type ?? "button"}
      className={cn(
        "inline-flex items-center justify-center gap-1.5 rounded font-medium",
        "transition-colors disabled:cursor-not-allowed disabled:opacity-40",
        VARIANTS[variant],
        SIZES[size],
        className,
      )}
      {...props}
    />
  ),
);

Button.displayName = "Button";
