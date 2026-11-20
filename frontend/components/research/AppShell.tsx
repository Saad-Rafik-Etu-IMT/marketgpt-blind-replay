"use client";

import Link from "next/link";
import { usePathname, useSearchParams } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import { useResearch } from "./ResearchProvider";
import { laboratories, labHref } from "@/lib/research/product-context";

const pages = [
  {
    path: "/laboratory",
    title: "Research laboratory",
    purpose:
      "Automated campaigns, measured evidence, and registered-model replay.",
  },
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
    purpose: "Follow models on the same market clock.",
  },
];

const navIcons: Record<string, string> = {
  "/laboratory":
    "M8 3h8m-6 0v6l-6 9a2 2 0 0 0 2 3h12a2 2 0 0 0 2-3l-6-9V3M7 15h10",
  "/studies":
    "M8 3h8m-6 0v6l-6 9a2 2 0 0 0 2 3h12a2 2 0 0 0 2-3l-6-9V3M7 15h10",
  "/comparison": "M3 6h5l8 12h5m-3-3 3 3-3 3M3 18h5l3-5m3-3 2-4h5m-3-3 3 3-3 3",
  "/experiments": "M4 4h16v16H4zM4 9h16M9 9v11",
  "/inspector":
    "M16 16l5 5M18 10a8 8 0 1 1-16 0 8 8 0 0 1 16 0M5 11h3l2-4 2 6 2-3h1",
  "/evaluation": "M3 3v17h18M7 16v-4m5 4V6m5 10V9",
  "/replay": "M4 4v16m4-16 12 8-12 8z",
};

export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const params = useSearchParams();
  function destination(path: string) {
    if (!params.get("study")) return path;
    const q = new URLSearchParams(params.toString());
    if (path === "/replay") return "/replay?" + q.toString();
    if (path === "/studies") return "/studies?" + q.toString();
    return path;
  }
  const page = pages.find((p) => p.path === pathname) ?? pages[0];
  const replay = pathname === "/replay";
  const laboratory = pathname === "/laboratory";
  const present = laboratory && params.get("mode") === "present";
  const activeLab =
    laboratories.find((t) => t === params.get("tab")) ?? "Overview";
  const { bundle, mode, useDemo, useReal, importFile, loading, error } =
    useResearch();
  const [theme, setTheme] = useState("light");
  const [themeError, setThemeError] = useState<string | null>(null);
  const [ready, setReady] = useState(false);
  const [collapsed, setCollapsed] = useState(false);
  const [stacked, setStacked] = useState(false);
  const [split, setSplit] = useState(50);
  const [command, setCommand] = useState("");
  const commandDialog = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    try {
      setCollapsed(
        localStorage.getItem("blind-replay.sidebar") === "collapsed",
      );
      setStacked(localStorage.getItem("blind-replay.layout") === "stacked");
      const saved = Number(localStorage.getItem("blind-replay.split") ?? 50);
      if (Number.isFinite(saved) && saved >= 30 && saved <= 70) setSplit(saved);
    } catch {}
  }, []);
  useEffect(() => {
    if (!laboratory || present) return;
    function key(e: KeyboardEvent) {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setCommand("");
        if (!commandDialog.current?.open) commandDialog.current?.showModal();
      }
    }
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  }, [laboratory, present]);
  function preference(key: string, value: string) {
    try {
      localStorage.setItem("blind-replay." + key, value);
    } catch {}
  }
  const navigation = useRef<HTMLElement>(null);
  useEffect(() => {
    const nav = navigation.current;
    const active = nav?.querySelector<HTMLElement>('[aria-current="page"]');
    if (!nav || !active || nav.scrollWidth <= nav.clientWidth) return;
    const n = nav.getBoundingClientRect(),
      a = active.getBoundingClientRect();
    if (a.left < n.left) nav.scrollLeft -= n.left - a.left + 4;
    else if (a.right > n.right) nav.scrollLeft += a.right - n.right + 4;
  }, [pathname, activeLab]);
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
  const comparison =
    pathname === "/comparison" ||
    pathname === "/studies" ||
    pathname === "/laboratory";
  const synthetic = !comparison && !replay && bundle?.origin === "synthetic";
  if (pathname === "/jury")
    return (
      <main className="bench-main jury-main">
        <h1>Jury session</h1>
        {children}
      </main>
    );
  return (
    <div
      className={`bench-shell ${laboratory ? "product-shell" : ""} ${present ? "presentation-shell" : ""} ${collapsed && !present ? "sidebar-collapsed" : ""} ${stacked ? "workspace-stacked" : ""}`}
      style={
        {
          "--workspace-split": split / 100 + "fr",
          "--workspace-other": 1 - split / 100 + "fr",
        } as React.CSSProperties
      }
    >
      <a href="#main-content" className="skip-link">
        Skip to content
      </a>
      <aside className="bench-sidebar">
        <Link
          href={
            laboratory
              ? labHref(params, { tab: "Overview", mode: null, step: null })
              : "/laboratory"
          }
          className="bench-brand"
        >
          <span className="bench-mark" aria-hidden="true">
            <i />
            <i />
            <i />
          </span>
          <span className="brand-name">
            Blind Replay<small>Research bench</small>
          </span>
        </Link>
        {laboratory ? (
          <>
            <nav
              className="product-nav"
              aria-label="Laboratories"
              ref={navigation}
            >
              <span className="nav-label">Research workspace</span>
              {laboratories.map((tab, i) => (
                <Link
                  key={tab}
                  href={labHref(params, { tab, mode: null, step: null })}
                  aria-current={
                    tab === activeLab && !present ? "page" : undefined
                  }
                >
                  <span className="product-nav-number" aria-hidden="true">
                    {String(i + 1).padStart(2, "0")}
                  </span>
                  {tab}
                </Link>
              ))}
              <Link
                className="presentation-link"
                href={labHref(params, { mode: "present", step: 0 })}
                aria-current={present ? "page" : undefined}
              >
                ▶ Presentation
              </Link>
            </nav>
            <details className="legacy-nav">
              <summary>Legacy and imported tools</summary>
              <nav aria-label="Legacy tools">
                {pages
                  .filter((p) => p.path !== "/laboratory")
                  .map((p) => (
                    <Link key={p.path} href={destination(p.path)}>
                      {p.title}
                    </Link>
                  ))}
              </nav>
            </details>
          </>
        ) : (
          <nav aria-label="Main navigation" ref={navigation}>
            {[
              {
                title: "Workspace",
                paths: ["/laboratory", "/studies", "/replay", "/comparison"],
              },
              {
                title: "Imported reports",
                paths: ["/experiments", "/inspector", "/evaluation"],
              },
            ].map((group) => (
              <div className="nav-group" key={group.title}>
                <span className="nav-label">{group.title}</span>
                {group.paths
                  .map((path) => pages.find((p) => p.path === path)!)
                  .map((p) => (
                    <Link
                      key={p.path}
                      href={destination(p.path)}
                      aria-current={pathname === p.path ? "page" : undefined}
                    >
                      <svg
                        viewBox="0 0 24 24"
                        className="nav-icon"
                        aria-hidden="true"
                      >
                        <path d={navIcons[p.path]} />
                      </svg>
                      {p.title}
                    </Link>
                  ))}
              </div>
            ))}
          </nav>
        )}
        <p className="sidebar-note">Educational simulation · virtual capital</p>
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
            <h1>
              {laboratory && activeLab !== "Overview"
                ? `${activeLab} laboratory`
                : page.title}
            </h1>
            <p className="muted">
              {laboratory
                ? "Blind Replay Bench · research workspace"
                : page.purpose}
            </p>
          </div>
          <span className="bench-badge source-status" role="status">
            {comparison
              ? "Validation lab"
              : replay
                ? "Simulation"
                : synthetic
                  ? "Synthetic Demo"
                  : bundle
                    ? "Real · Imported artifacts"
                    : "No Data · Real mode"}
          </span>
        </header>
        {laboratory && !present && (
          <div className="workspace-tools bench-actions">
            <button
              className="bench-button"
              onClick={() => {
                setCommand("");
                commandDialog.current?.showModal();
              }}
            >
              Commands · Ctrl K
            </button>
            <button
              className="bench-button"
              aria-pressed={collapsed}
              onClick={() => {
                setCollapsed(!collapsed);
                preference("sidebar", !collapsed ? "collapsed" : "expanded");
              }}
            >
              {" "}
              {collapsed ? "Expand sidebar" : "Collapse sidebar"}
            </button>
            <button
              className="bench-button"
              aria-pressed={stacked}
              onClick={() => {
                setStacked(!stacked);
                preference("layout", !stacked ? "stacked" : "side-by-side");
              }}
            >
              {stacked ? "Stacked layout" : "Side-by-side layout"}
            </button>
            {!stacked && (
              <label className="workspace-split">
                Panel split{" "}
                <input
                  aria-label="Panel split"
                  type="range"
                  min={30}
                  max={70}
                  value={split}
                  onChange={(e) => {
                    setSplit(Number(e.target.value));
                    preference("split", e.target.value);
                  }}
                />
              </label>
            )}
            <button
              className="bench-button"
              onClick={async () => {
                try {
                  await navigator.clipboard.writeText(window.location.href);
                } catch {}
              }}
            >
              Copy workspace link
            </button>
          </div>
        )}
        <dialog
          ref={commandDialog}
          className="command-dialog"
          aria-label="Research command palette"
        >
          <h2>Go to laboratory</h2>
          <input
            autoFocus
            placeholder="Search workspace…"
            aria-label="Search commands"
            value={command}
            onChange={(e) => setCommand(e.target.value)}
          />
          <nav>
            {laboratories
              .filter((t) => t.toLowerCase().includes(command.toLowerCase()))
              .map((t) => (
                <Link
                  key={t}
                  href={labHref(params, { tab: t, mode: null, step: null })}
                  onClick={() => commandDialog.current?.close()}
                >
                  {t}
                </Link>
              ))}
            {"presentation".includes(command.toLowerCase()) && (
              <Link
                href={labHref(params, { mode: "present", step: 0 })}
                onClick={() => commandDialog.current?.close()}
              >
                Presentation
              </Link>
            )}
          </nav>
          <button
            className="bench-button"
            onClick={() => commandDialog.current?.close()}
          >
            Close
          </button>
        </dialog>
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
              <p>
                Imports stay in this browser tab until reload. Limit: 2 MiB.
                Real mode clears the bundle. Results and origin are supplied by
                the exporter.
              </p>
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
