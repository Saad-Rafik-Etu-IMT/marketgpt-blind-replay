"""Human-readable display names for known instruments.

The cockpit used to carry this map in the browser; serving it from the API
keeps a single source of truth and lets the instruments endpoint return a
ready-to-render ``name`` alongside each symbol. Unknown symbols fall back
to the raw ticker.
"""

from __future__ import annotations

INSTRUMENT_NAMES: dict[str, str] = {
    # ETF / spot proxies
    "SPY": "S&P 500 ETF",
    "QQQ": "Nasdaq-100 ETF",
    "IWM": "Russell 2000 ETF",
    "TLT": "20yr Treasury ETF",
    "GLD": "Gold ETF",
    "USO": "WTI Crude ETF",
    "UNG": "Natural Gas ETF",
    "XLE": "Energy Sector ETF",
    "XLF": "Financials Sector ETF",
    "XLK": "Tech Sector ETF",
    "VXX": "VIX Futures ETN",
    "UUP": "US Dollar ETF",
    # Equity index futures
    "ES=F": "E-mini S&P 500 Future",
    "MES=F": "Micro E-mini S&P 500 Future",
    "NQ=F": "E-mini Nasdaq-100 Future",
    "MNQ=F": "Micro Nasdaq-100 Future",
    "YM=F": "E-mini Dow Future",
    "MYM=F": "Micro Dow Future",
    "RTY=F": "E-mini Russell 2000 Future",
    "M2K=F": "Micro Russell 2000 Future",
    # Energy futures
    "CL=F": "WTI Crude Oil Future",
    "BZ=F": "Brent Crude Oil Future",
    "NG=F": "Henry Hub Natural Gas Future",
    "RB=F": "RBOB Gasoline Future",
    "HO=F": "Heating Oil Future",
    "QM=F": "Mini WTI Crude Future",
    "MCL=F": "Micro WTI Crude Future",
    # Metals futures
    "GC=F": "Gold Future",
    "MGC=F": "Micro Gold Future",
    "SI=F": "Silver Future",
    "HG=F": "Copper Future",
    "PL=F": "Platinum Future",
    "PA=F": "Palladium Future",
    # Rates futures
    "ZT=F": "2-Year Treasury Note Future",
    "ZF=F": "5-Year Treasury Note Future",
    "ZN=F": "10-Year Treasury Note Future",
    "ZB=F": "30-Year Treasury Bond Future",
    "UB=F": "Ultra Bond Future",
    "ZQ=F": "30-Day Fed Funds Future",
    # FX futures
    "6E=F": "EUR/USD Future",
    "6J=F": "JPY/USD Future",
    "6B=F": "GBP/USD Future",
    "6A=F": "AUD/USD Future",
    "6C=F": "CAD/USD Future",
    "6S=F": "CHF/USD Future",
    "DX-Y.NYB": "US Dollar Index",
    # Agriculture futures
    "ZC=F": "Corn Future",
    "ZS=F": "Soybean Future",
    "ZW=F": "Wheat Future",
    "ZL=F": "Soybean Oil Future",
    "ZM=F": "Soybean Meal Future",
    "KC=F": "Coffee Future",
    "CC=F": "Cocoa Future",
    "SB=F": "Sugar Future",
    "CT=F": "Cotton Future",
    "LE=F": "Live Cattle Future",
    "HE=F": "Lean Hogs Future",
    # Crypto
    "BTC-USD": "Bitcoin (spot)",
    "ETH-USD": "Ethereum (spot)",
    "BTC=F": "CME Bitcoin Future",
    "ETH=F": "CME Ether Future",
}


def display_name(symbol: str) -> str:
    """Return a friendly name for ``symbol``, falling back to the ticker."""
    return INSTRUMENT_NAMES.get(symbol, symbol)


# USD value of a full quoted price point, not a tick. ZN is quoted in points.
CONTRACT_MULTIPLIERS = {
    "SPY": 1.0,
    "QQQ": 1.0,
    "ES=F": 50.0,
    "NQ=F": 20.0,
    "CL=F": 1000.0,
    "NG=F": 10000.0,
    "GC=F": 100.0,
    "ZN=F": 1000.0,
    "6E=F": 125000.0,
}


def contract_multiplier(symbol: str) -> float | None:
    return CONTRACT_MULTIPLIERS.get(symbol)
