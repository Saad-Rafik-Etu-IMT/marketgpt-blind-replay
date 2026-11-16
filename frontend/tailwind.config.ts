import type { Config } from "tailwindcss";

/**
 * Semantic tokens shared by research and the retained replay components.
 */
const config: Config = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        slate: {
          50: "var(--s50)", 100: "var(--s100)", 200: "var(--s200)",
          300: "var(--s300)", 400: "var(--s400)", 500: "var(--s500)",
          600: "var(--s600)", 700: "var(--s700)", 800: "var(--s800)",
          900: "var(--s900)", 950: "var(--s950)",
        },
        surface: {
          DEFAULT: "var(--panel)",
          muted: "var(--bg)",
          raised: "var(--raised)",
        },
        primary: {
          DEFAULT: "var(--accent)",
          hover: "var(--accent-hover)",
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
