import { MarketReplayCockpit } from "@/components/cockpit/MarketReplayCockpit";

export default function ReplayPage() {
  return (
    <>
      <p className="bench-notice">
        Educational OHLC simulation: next-opening execution, synthetic spread
        and fees. Reference prices lie inside OHLC; executed prices may lie
        outside it after spread. No real order book, margin or contract rolls.
        This foundation is not yet anonymized.
      </p>
      <MarketReplayCockpit />
    </>
  );
}
