import { MarketReplayCockpit } from "@/components/cockpit/MarketReplayCockpit";

export default function ReplayPage() {
  return (
    <>
      <p className="px-4 pt-4 text-sm text-amber-200">
        Educational OHLC simulation: next-opening execution, synthetic spread
        and fees. Reference prices lie inside OHLC; executed prices may lie
        outside it after spread. No real order book, margin or contract rolls.
        This foundation is not yet anonymized.
      </p>
      <MarketReplayCockpit />
    </>
  );
}
