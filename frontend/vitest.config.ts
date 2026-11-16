import { defineConfig } from "vitest/config";
import path from "node:path";

export default defineConfig({
  // Use the automatic JSX runtime so test files don't need to import React
  // explicitly (matches the Next.js app runtime).
  esbuild: { jsx: "automatic" },
  test: {
    environment: "jsdom",
    globals: false,
    // Unit tests use *.test.*; Playwright E2E specs use *.spec.* under e2e/.
    include: ["**/*.test.{ts,tsx}"],
    exclude: ["node_modules", "e2e", ".next"],
  },
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "."),
    },
  },
});
