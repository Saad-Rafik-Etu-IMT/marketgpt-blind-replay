"use client";

import Link from "next/link";
import { usePathname, useSearchParams } from "next/navigation";
import { useEffect, useState } from "react";
import { useResearch } from "./ResearchProvider";

const pages = [
  {
    path: "/studies",
    title: "Research studies",
    purpose: "Train models and inspect saved validation experiments.",
  },
  {
    path: "/comparison",
    title: "Blind comparison",
    purpose: "Compare decisions on a shared validation period.",
  },
  {
    path: "/experiments",
    title: "Experiments",
    purpose: "Understand the run before interpreting the result.",
  },
  {
    path: "/inspector",
    title: "Inspector",
    purpose: "Trace one prediction back to its inputs and evidence.",
  },
  {
    path: "/evaluation",
    title: "Evaluation",
    purpose: "Separate model errors from explanation quality.",
  },
  {
    path: "/replay",
    title: "Replay",
    purpose: "Watch competing models trade on a shared market clock and inspect the engine.",
  },
];

const navIcons: Record<string, string> = {
  "/studies": "M8 3h8m-6 0v6l-6 9a2 2 0 0 0 2 3h12a2 2 0 0 0 2-3l-6-9V3M7 15h10",
  "/comparison": "M3 6h5l8 12h5m-3-3 3 3-3 3M3 18h5l3-5m3-3 2-4h5m-3-3 3 3-3 3",
  "/experiments": "M4 4h16v16H4zM4 9h16M9 9v11",
  "/inspector": "M16 16l5 5M18 10a8 8 0 1 1-16 0 8 8 0 0 1 16 0M5 11h3l2-4 2 6 2-3h1",
  "/evaluation": "M3 3v17h18M7 16v-4m5 4V6m5 10V9",
  "/replay": "M4 4v16m4-16 12 8-12 8z",
};

export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const params = useSearchParams();
  const linkedViews: Record<string, string> = {
    "/experiments": "Experiment",
    "/inspector": "Inspector",
    "/evaluation": "Evaluation",
    "/replay": "Blind replay",
    "/comparison": "Portfolio",
    "/studies": "Experiment",
  };
  function destination(path: string) {
    if (!params.get("study")) return path;
    const q = new URLSearchParams(params.toString());
    if (path === "/replay") return "/replay?" + q.toString();
    q.set("tab", linkedViews[path]);
    return "/studies?" + q.toString();
  }
  const page = pages.find((p) => p.path === pathname) ?? pages[2];
  const replay = pathname === "/replay";
  const { bundle, mode, useDemo, useReal, importFile, loading, error } =
    useResearch();
  const [theme, setTheme] = useState("light");
  const [themeError, setThemeError] = useState<string | null>(null);
  const [ready, setReady] = useState(false);
  useEffect(() => {
    setReady(true);
    try {
      const saved = localStorage.getItem("blind-replay.theme");
      const next =
        saved === "light" || saved === "dark"
          ? saved
          : matchMedia("(prefers-color-scheme: dark)").matches
            ? "dark"
            : "light";
      setTheme(next);
      document.documentElement.dataset.theme = next;
    } catch {
      setThemeError(
        "Theme preference could not be loaded. Theme changes will still work for this visit.",
      );
    }
  }, []);
  function toggleTheme() {
    const next = theme === "dark" ? "light" : "dark";
    setTheme(next);
    document.documentElement.dataset.theme = next;
    try {
      localStorage.setItem("blind-replay.theme", next);
    } catch {
      setThemeError("Theme preference could not be saved.");
    }
  }
  const comparison = pathname === "/comparison" || pathname === "/studies";
  const synthetic = !comparison && !replay && bundle?.origin === "synthetic";
  if (pathname === "/jury")
    return (
      <main className="bench-main jury-main">
        <h1>Jury session</h1>
        {children}
      </main>
    );
  return (
    <div className="bench-shell">
      <a href="#main-content" className="skip-link">
        Skip to content
      </a>
      <aside className="bench-sidebar">
        <Link href="/experiments" className="bench-brand">
          <span className="bench-mark" aria-hidden="true"><i /><i /><i /></span>
          <span className="brand-name">Blind Replay<small>Research bench</small></span>
        </Link>
        <nav aria-label="Main navigation">
          {pages.map((p) => (
            <Link
              key={p.path}
              href={destination(p.path)}
              aria-current={pathname === p.path ? "page" : undefined}
            >
              <svg viewBox="0 0 24 24" className="nav-icon" aria-hidden="true">
                <path d={navIcons[p.path]} />
              </svg>
              {p.title}
            </Link>
          ))}
        </nav>
        <p className="sidebar-note">
          Educational simulation
          <br />
          No live trading or investment advice.
        </p>
        <button
          disabled={!ready}
          className="bench-button"
          onClick={toggleTheme}
          aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
        >
          {theme === "dark" ? "Light" : "Dark"} theme
        </button>
      </aside>
      <div className="bench-workspace">
        <header className="bench-header">
          <div>

            <h1>{page.title}</h1>
            <p className="muted">{page.purpose}</p>
          </div>
          <span className="bench-badge source-status" role="status">
            {comparison
              ? "Validation lab"
              : replay
                ? "Recorded engine · simulation"
                : synthetic
                  ? "Synthetic Demo"
                  : bundle
                    ? "Real · Imported artifacts"
                    : "No Data · Real mode"}
          </span>
        </header>
        {!comparison && !replay && (
          <section aria-label="Research data source" className="source-bar">
            <div className="bench-actions">
              <button
                disabled={!ready}
                className="bench-button"
                onClick={useReal}
                aria-pressed={mode === "real"}
              >
                Real mode
              </button>
              <button
                disabled={!ready}
                className="bench-button"
                onClick={useDemo}
                aria-pressed={mode === "demo"}
              >
                Load synthetic demo
              </button>
              <label className="import-control">
                Import research JSON
                <input
                  disabled={!ready}
                  aria-label="Import research JSON"
                  type="file"
                  accept=".json,application/json"
                  onChange={(e) => {
                    const file = e.target.files?.[0];
                    if (file) void importFile(file);
                    e.target.value = "";
                  }}
                />
              </label>
            </div>
            <details className="source-help">
              <summary>About imported data</summary>
              <p>Imports stay in this browser tab until reload. Limit: 2 MiB. Real mode clears the bundle. Results and origin are supplied by the exporter.</p>
            </details>
          </section>
        )}
        {synthetic && (
          <p role="status" className="bench-notice">
            Synthetic demo data — not experimental results. Curves, predictions
            and explanation are illustrative; no model was trained or called.
          </p>
        )}
        {themeError && (
          <p role="alert" className="bench-notice">
            {themeError}
          </p>
        )}
        {!comparison && !replay && error && (
          <p
            role="alert"
            aria-label="Artifact import error"
            className="bench-error"
          >
            {error}
          </p>
        )}
        <main id="main-content" tabIndex={-1} className="bench-main">
          {!comparison && !replay && loading ? (
            <p role="status" className="bench-panel">
              Validating artifact…
            </p>
          ) : (
            children
          )}
        </main>

      </div>
    </div>
  );
}
