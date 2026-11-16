/**
 * Minimal classnames combiner. Filters falsy values and joins with a
 * space — enough for conditional Tailwind classes without pulling in a
 * dependency. Later strings win by convention (call sites append overrides
 * last).
 */
export type ClassValue = string | number | false | null | undefined;

export function cn(...values: ClassValue[]): string {
  return values.filter(Boolean).join(" ");
}
