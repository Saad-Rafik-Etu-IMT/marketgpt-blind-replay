"""Pydantic schemas for market data.

``OHLCVBar`` is the atomic unit of market data and the row-level
validator: constructing one enforces every OHLC integrity invariant, so
no downstream module ever has to re-check intrabar price ordering.

Phase 1, Module 1.
"""

from datetime import date

from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self


class OHLCVBar(BaseModel):
    """A single OHLCV candle — the atomic unit of market data.

    Construction enforces:
      * ``open``, ``close``, ``volume`` are non-negative (field constraints);
      * ``high >= max(open, close)``;
      * ``low  <= min(open, close)``.

    Note: per the domain spec (§4.1), ``high`` and ``low`` are constrained
    only *relative* to ``open``/``close`` — they are not independently
    floored at 0.
    """

    date: date
    open: float = Field(ge=0, description="Opening price; >= 0.")
    high: float = Field(description="Session high; >= max(open, close).")
    low: float = Field(description="Session low; <= min(open, close).")
    close: float = Field(ge=0, description="Closing price; >= 0.")
    volume: int = Field(ge=0, description="Traded volume; >= 0.")
    adjusted_close: float | None = Field(
        default=None,
        description="Split/dividend-adjusted close, if available.",
    )

    @model_validator(mode="after")
    def validate_ohlcv_integrity(self) -> Self:
        """Enforce intrabar price ordering: low <= open/close <= high.

        Raises ``ValueError`` (wrapped by Pydantic into a
        ``ValidationError``) with the offending values, so a rejected
        CSV row is diagnosable without re-deriving the cause.
        """
        max_open_close = max(self.open, self.close)
        if self.high < max_open_close:
            raise ValueError(
                f"high ({self.high}) must be >= max(open, close) ({max_open_close})"
            )

        min_open_close = min(self.open, self.close)
        if self.low > min_open_close:
            raise ValueError(
                f"low ({self.low}) must be <= min(open, close) ({min_open_close})"
            )

        return self


class RowError(BaseModel):
    """A single validation failure located within a CSV.

    ``row_number`` is the 1-based CSV line number — the header is line 1,
    so the first data row is line 2. A value of ``0`` denotes a
    file-level failure not tied to any specific row.
    """

    row_number: int = Field(
        ge=0, description="1-based CSV line; 0 = file-level failure."
    )
    field: str | None = Field(
        default=None, description="Offending field name, if applicable."
    )
    message: str = Field(description="Human-readable explanation of the failure.")


class ValidationReport(BaseModel):
    """The outcome of validating a CSV of OHLCV bars.

    ``bars`` holds every fully valid bar in file order. When ``is_valid``
    is False the caller must not persist them.
    """

    is_valid: bool = Field(description="True only when there are zero errors.")
    total_rows: int = Field(ge=0, description="Data rows seen in the file.")
    valid_rows: int = Field(ge=0, description="Rows that passed every check.")
    errors: list[RowError] = Field(
        default_factory=list, description="All failures, in file order."
    )
    bars: list[OHLCVBar] = Field(
        default_factory=list, description="Valid bars, in file order."
    )


class UploadResponse(BaseModel):
    """Outcome of POST /api/market/upload."""

    instrument: str
    loaded_count: int = Field(ge=0)
    total_rows: int = Field(ge=0)
    errors: list[RowError] = Field(default_factory=list)


class InstrumentSummary(BaseModel):
    """One row of GET /api/market/instruments."""

    instrument: str
    name: str = ""
    multiplier: float | None = None
    bar_count: int = Field(ge=0)
    first_date: date | None = None
    last_date: date | None = None


