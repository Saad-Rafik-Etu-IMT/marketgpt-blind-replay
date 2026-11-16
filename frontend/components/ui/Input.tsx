import { forwardRef } from "react";

import { cn } from "@/lib/cn";

const FIELD =
  "w-full min-w-0 rounded border border-slate-700 bg-surface-muted px-2 py-1.5 " +
  "text-sm text-slate-100 placeholder:text-slate-500 " +
  "disabled:cursor-not-allowed disabled:opacity-40";

export const Input = forwardRef<
  HTMLInputElement,
  React.InputHTMLAttributes<HTMLInputElement>
>(({ className, ...props }, ref) => (
  <input ref={ref} className={cn(FIELD, className)} {...props} />
));
Input.displayName = "Input";

export const Textarea = forwardRef<
  HTMLTextAreaElement,
  React.TextareaHTMLAttributes<HTMLTextAreaElement>
>(({ className, ...props }, ref) => (
  <textarea ref={ref} className={cn(FIELD, className)} {...props} />
));
Textarea.displayName = "Textarea";

/**
 * A labelled field wrapper: a stacked label + control, so forms stay
 * consistent and the label is programmatically associated for a11y.
 */
export function Field({
  label,
  htmlFor,
  children,
  className,
}: {
  label: string;
  htmlFor?: string;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <label className={cn("flex flex-col gap-1 text-sm", className)} htmlFor={htmlFor}>
      <span className="text-slate-400">{label}</span>
      {children}
    </label>
  );
}
