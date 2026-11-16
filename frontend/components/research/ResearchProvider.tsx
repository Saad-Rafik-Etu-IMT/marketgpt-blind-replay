"use client";

import { createContext, useContext, useRef, useState } from "react";
import { importResearchFile, type ResearchBundle } from "@/lib/research/contract";
import { demoBundle } from "@/lib/research/demo";

interface ResearchContextValue {
  bundle: ResearchBundle | null;
  mode: "real" | "demo";
  loading: boolean;
  error: string | null;
  useDemo: () => void;
  useReal: () => void;
  importFile: (file: File) => Promise<void>;
}
const ResearchContext = createContext<ResearchContextValue | null>(null);

export function ResearchProvider({ children }: { children: React.ReactNode }) {
  const [bundle, setBundle] = useState<ResearchBundle | null>(null);
  const [mode, setMode] = useState<"real" | "demo">("real");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const generation = useRef(0);
  function reset(next: "real" | "demo") {
    generation.current++;
    setLoading(false);
    setError(null);
    setMode(next);
    setBundle(next === "demo" ? demoBundle : null);
  }
  async function importFile(file: File) {
    const request = ++generation.current;
    setLoading(true);
    setError(null);
    // Clear previous evidence immediately, including any synthetic example.
    setBundle(null);
    setMode("real");
    try {
      const result = await importResearchFile(file);
      if (request === generation.current) setBundle(result);
    } catch (e) {
      if (request === generation.current) setError(e instanceof Error ? e.message : "Artifact import failed.");
    } finally {
      if (request === generation.current) setLoading(false);
    }
  }
  return <ResearchContext.Provider value={{ bundle, mode, loading, error, useDemo: () => reset("demo"), useReal: () => reset("real"), importFile }}>{children}</ResearchContext.Provider>;
}

export function useResearch() {
  const value = useContext(ResearchContext);
  if (!value) throw new Error("ResearchProvider is required.");
  return value;
}
