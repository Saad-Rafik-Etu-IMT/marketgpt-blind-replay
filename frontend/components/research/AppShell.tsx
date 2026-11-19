"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { useResearch } from "./ResearchProvider";

const pages = [
  { path: "/comparison", title: "Blind comparison", purpose: "Compare decisions on a shared validation period.", number: "↗" },
  { path: "/experiments", title: "Experiments", purpose: "Understand the run before interpreting the result.", number: "01" },
  { path: "/inspector", title: "Inspector", purpose: "Trace one prediction back to its inputs and evidence.", number: "02" },
  { path: "/evaluation", title: "Evaluation", purpose: "Separate model errors from explanation quality.", number: "03" },
  { path: "/replay", title: "Replay", purpose: "The execution environment, not the research result.", number: "04" },
];

export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const page = pages.find(p => p.path === pathname) ?? pages[1];
  const replay = pathname === "/replay";
  const { bundle, mode, useDemo, useReal, importFile, loading, error } = useResearch();
  const [theme, setTheme] = useState("light");
  const [themeError, setThemeError] = useState<string | null>(null);
  const [ready, setReady] = useState(false);
  useEffect(() => {
    setReady(true);
    try {
      const saved = localStorage.getItem("blind-replay.theme");
      const next = saved === "light" || saved === "dark" ? saved : matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
      setTheme(next);
      document.documentElement.dataset.theme = next;
    } catch {
      setThemeError("Theme preference could not be loaded. Theme changes will still work for this visit.");
    }
  }, []);
  function toggleTheme() {
    const next = theme === "dark" ? "light" : "dark";
    setTheme(next);
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("blind-replay.theme", next); }
    catch { setThemeError("Theme preference could not be saved."); }
  }
  const comparison = pathname === "/comparison";
  const synthetic = !comparison && !replay && bundle?.origin === "synthetic";
  return (
    <div className="bench-shell">
      <a href="#main-content" className="skip-link">Skip to content</a>
      <aside className="bench-sidebar">
        <Link href="/experiments" className="bench-brand">Blind Replay<span>BENCH / AI RESEARCH</span></Link>
        <nav aria-label="Main navigation">
          {pages.map(p => <Link key={p.path} href={p.path} aria-current={pathname === p.path ? "page" : undefined}><span aria-hidden="true">{p.number}</span>{p.title}</Link>)}
        </nav>
        <p className="sidebar-note">Educational simulation<br />No live trading or investment advice.</p>
        <button disabled={!ready} className="bench-button" onClick={toggleTheme} aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}>{theme === "dark" ? "Light" : "Dark"} theme</button>
      </aside>
      <div className="bench-workspace">
        <header className="bench-header">
          <div><p className="eyebrow">AI Research Cockpit</p><h1>{page.title}</h1><p className="muted">{page.purpose}</p></div>
          <span className="bench-badge" role="status">{comparison ? "Validation lab" : replay ? "Real · Replay API" : synthetic ? "Synthetic Demo" : bundle ? "Real · Imported artifacts" : "No Data · Real mode"}</span>
        </header>
        {!comparison && !replay && <section aria-label="Research data source" className="source-bar">
          <div className="bench-actions">
            <button disabled={!ready} className="bench-button" onClick={useReal} aria-pressed={mode === "real"}>Real mode</button>
            <button disabled={!ready} className="bench-button" onClick={useDemo} aria-pressed={mode === "demo"}>Load synthetic demo</button>
            <label className="import-control">Import research JSON<input disabled={!ready} aria-label="Import research JSON" type="file" accept=".json,application/json" onChange={e => {
              const file = e.target.files?.[0];
              if (file) void importFile(file);
              e.target.value = "";
            }} /></label>
          </div>
          <p className="muted">Imports stay in browser memory, disappear on reload, and are never uploaded. Limit: 2 MiB. Real mode clears the loaded bundle. Imported origin and results are exporter declarations, not independently verified.</p>
        </section>}
        {synthetic && <p role="status" className="bench-notice">Synthetic demo data — not experimental results. Curves, predictions and explanation are illustrative; no model was trained or called.</p>}
        {themeError && <p role="alert" className="bench-notice">{themeError}</p>}
        {!comparison && !replay && error && <p role="alert" aria-label="Artifact import error" className="bench-error">{error}</p>}
        <main id="main-content" tabIndex={-1} className="bench-main">
          {!comparison && !replay && loading ? <p role="status" className="bench-panel">Validating artifact…</p> : children}
        </main>
        <footer className="bench-footer">Evidence first. Uncertainty visible. Research artifacts are separate from replay observations.</footer>
      </div>
    </div>
  );
}
