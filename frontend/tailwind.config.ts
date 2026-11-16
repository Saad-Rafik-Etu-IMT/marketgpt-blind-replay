import type { Config } from "tailwindcss";

/**
 * Design tokens. The cockpit is a dark, dense trading UI; these semantic
 * aliases sit on top of Tailwind's slate/emerald palette so components can
 * reference intent (`surface`, `border`, `primary`) rather than raw shades.
 */
const config: Config = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        surface: {
          DEFAULT: "#0f172a",
          muted: "#020617",
          raised: "#1e293b",
        },
        primary: {
          DEFAULT: "#047857",
          hover: "#059669",
        },
        danger: {
          DEFAULT: "#b91c1c",
          hover: "#dc2626",
        },
      },
      borderRadius: {
        card: "0.5rem",
      },
    },
  },
  plugins: [],
};

export default config;
