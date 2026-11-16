import { describe, expect, it } from "vitest";
import { render } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

import { PortfolioPanel } from "@/components/cockpit/PortfolioPanel";
import { RiskPanel } from "@/components/cockpit/RiskPanel";
import { TradeBlotter } from "@/components/cockpit/TradeBlotter";
import { RiskLimitsPanel } from "@/components/cockpit/RiskLimitsPanel";
import { ToastProvider } from "@/components/ui";

function withProviders(ui: React.ReactElement) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return (
    <QueryClientProvider client={client}>
      <ToastProvider>{ui}</ToastProvider>
    </QueryClientProvider>
  );
}

describe("cockpit components — smoke", () => {
  it("PortfolioPanel renders with no snapshot", () => {
    const { container } = render(<PortfolioPanel snapshot={null} />);
    expect(container.textContent).toMatch(/No portfolio data/);
  });

  it("RiskPanel renders with no metrics", () => {
    const { container } = render(<RiskPanel metrics={null} />);
    expect(container.textContent).toMatch(/No metrics/);
  });

  it("TradeBlotter renders with empty orders", () => {
    const { container } = render(
      <ToastProvider>
        <TradeBlotter orders={[]} onCancel={async () => {}} />
      </ToastProvider>,
    );
    expect(container.textContent).toMatch(/No orders yet/);
  });

  it("RiskLimitsPanel renders its title with no session", () => {
    const { container } = render(
      withProviders(<RiskLimitsPanel sessionId={null} metrics={null} />),
    );
    expect(container.textContent).toMatch(/Risk limits/);
  });

});
