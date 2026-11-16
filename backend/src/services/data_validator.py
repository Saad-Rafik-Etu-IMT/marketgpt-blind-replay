"""CSV market-data validation.

Validates an entire CSV of OHLCV bars row by row, collecting *typed*
per-row errors instead of failing on the first bad row, and applying
file-level checks: required columns are present, and dates are strictly
ascending with no duplicates.

Each row is validated by constructing an ``OHLCVBar`` — so OHLC integrity
is enforced by the schema (Module 1), not re-implemented here.

Phase 1, Module 2.
"""

import io
from datetime import date
from pathlib import Path

import pandas as pd
from pydantic import ValidationError

from src.schemas.market import OHLCVBar, RowError, ValidationReport

# Columns that must be present AND populated in every row.
# adjusted_close is intentionally excluded — OHLCVBar permits it to be None.
REQUIRED_COLUMNS: tuple[str, ...] = (
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
)


def validate_csv_content(content: str) -> ValidationReport:
    """Validate raw CSV text and return a typed report.

    This is the entry point for the upload endpoint, which receives file
    content rather than a path.
    """
    try:
        # dtype=str: keep every present cell as a string and let Pydantic
        # perform all coercion. Missing cells remain NaN (caught below).
        frame = pd.read_csv(io.StringIO(content), dtype=str)
    except pd.errors.EmptyDataError:
        return _file_level_failure("file is empty or has no header row")
    except pd.errors.ParserError as exc:
        return _file_level_failure(f"CSV could not be parsed: {exc}")
    return _validate_frame(frame)


def validate_csv_file(path: str | Path) -> ValidationReport:
    """Validate a CSV file on disk and return a typed report."""
    try:
        content = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return _file_level_failure(f"file not found: {path}")
    return validate_csv_content(content)


def _validate_frame(frame: pd.DataFrame) -> ValidationReport:
    """Validate an already-parsed, string-typed DataFrame."""
    missing_columns = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing_columns:
        return _file_level_failure(
            f"missing required column(s): {', '.join(missing_columns)}"
        )

    total_rows = len(frame)
    if total_rows == 0:
        return _file_level_failure("file contains no data rows")

    errors: list[RowError] = []
    bars: list[OHLCVBar] = []
    seen_dates: set[date] = set()
    previous_date: date | None = None

    for position, (_, row) in enumerate(frame.iterrows()):
        line_number = position + 2  # CSV line: +1 for header, +1 to 1-index
        errors_before = len(errors)

        missing_fields = [f for f in REQUIRED_COLUMNS if pd.isna(row[f])]
        if missing_fields:
            errors.extend(
                RowError(
                    row_number=line_number,
                    field=field,
                    message=f"missing value for '{field}'",
                )
                for field in missing_fields
            )
            continue

        try:
            bar = _build_bar(row)
        except ValidationError as exc:
            errors.extend(_row_errors_from_pydantic(exc, line_number))
            continue

        _check_date_sequence(bar, line_number, previous_date, seen_dates, errors)
        seen_dates.add(bar.date)
        previous_date = bar.date

        # Append the bar only if THIS row produced no error of any kind.
        if len(errors) == errors_before:
            bars.append(bar)

    return ValidationReport(
        is_valid=not errors,
        total_rows=total_rows,
        valid_rows=len(bars),
        errors=errors,
        bars=bars,
    )


def _build_bar(row: pd.Series) -> OHLCVBar:
    """Construct an OHLCVBar from a string-typed CSV row.

    Values are passed as strings; Pydantic performs type coercion and
    raises ``ValidationError`` on bad types or OHLC integrity violations.
    """
    return OHLCVBar(
        date=str(row["date"]),
        open=str(row["open"]),
        high=str(row["high"]),
        low=str(row["low"]),
        close=str(row["close"]),
        volume=str(row["volume"]),
        adjusted_close=_optional_str(row.get("adjusted_close")),
    )


def _optional_str(value: object) -> str | None:
    """Return ``str(value)``, or None if the value is absent or NaN."""
    if value is None or pd.isna(value):
        return None
    return str(value)


def _check_date_sequence(
    bar: OHLCVBar,
    line_number: int,
    previous_date: date | None,
    seen_dates: set[date],
    errors: list[RowError],
) -> None:
    """Append an error if the bar's date duplicates or precedes the prior row."""
    if bar.date in seen_dates:
        errors.append(
            RowError(
                row_number=line_number,
                field="date",
                message=f"duplicate date: {bar.date.isoformat()}",
            )
        )
    elif previous_date is not None and bar.date <= previous_date:
        errors.append(
            RowError(
                row_number=line_number,
                field="date",
                message=(
                    f"date {bar.date.isoformat()} is not strictly after "
                    f"the previous row's date {previous_date.isoformat()}"
                ),
            )
        )


def _row_errors_from_pydantic(exc: ValidationError, line_number: int) -> list[RowError]:
    """Convert a Pydantic ValidationError into one RowError per failure."""
    converted: list[RowError] = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"])
        converted.append(
            RowError(
                row_number=line_number,
                field=location or None,
                message=error["msg"],
            )
        )
    return converted


def _file_level_failure(message: str) -> ValidationReport:
    """Build a report for a failure not tied to a specific data row."""
    return ValidationReport(
        is_valid=False,
        total_rows=0,
        valid_rows=0,
        errors=[RowError(row_number=0, field=None, message=message)],
        bars=[],
    )
