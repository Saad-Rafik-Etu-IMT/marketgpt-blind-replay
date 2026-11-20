"use client";

import { useState } from "react";
import { MarketReplayCockpit } from "@/components/cockpit/MarketReplayCockpit";
import { CompetitionReplay } from "@/components/research/CompetitionReplay";

export default function ReplayPage() {
  const [manual, setManual] = useState(false);
  return (
    <div className="research-stack replay-page">
      <nav className="study-nav" aria-label="Replay mode">
        <button
          className="bench-button"
          aria-pressed={!manual}
          onClick={() => setManual(false)}
        >
          AI competition
        </button>
        <button
          className="bench-button"
          aria-pressed={manual}
          onClick={() => setManual(true)}
        >
          Manual simulation
        </button>
      </nav>
      {!manual ? (
        <CompetitionReplay />
      ) : (
        <>
          <p className="bench-notice">
            Educational OHLC simulation: next-opening execution, synthetic
            spread and fees. Reference prices lie inside OHLC; executed prices
            may lie outside it after spread. No real order book, margin or
            contract rolls. This foundation is not yet anonymized.
          </p>
          <MarketReplayCockpit />
        </>
      )}
    </div>
  );
}
