import { forwardRef } from "react";

import { cn } from "@/lib/cn";

export const Select = forwardRef<
  HTMLSelectElement,
  React.SelectHTMLAttributes<HTMLSelectElement>
>(({ className, children, ...props }, ref) => (
  <select
    ref={ref}
    className={cn(
      "min-w-0 rounded border border-slate-700 bg-surface-muted px-2 py-1.5",
      "text-sm text-slate-100 disabled:cursor-not-allowed disabled:opacity-40",
      className,
    )}
    {...props}
  >
    {children}
  </select>
));
Select.displayName = "Select";
