"""F1 command: ``python -m src.data.download`` (run from ``backend``).

Downloads the configured tickers into the git-ignored ``data/raw/<source>/``
folder with a ``provenance.json``. Refuses to run until the source's terms are
recorded as approved in ``config/data_sources.json``.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from src.config import get_settings
from src.data.pipeline import (
    DEFAULT_DATA_SOURCES_PATH,
    DataPipelineError,
    load_data_sources_config,
    run_download,
)
from src.data.sources import SOURCES, PriceSource
from src.data.split import DEFAULT_SPLIT_CONFIG_PATH, SplitError, load_split_config


def main(
    argv: Sequence[str] | None = None,
    source_factory: Callable[[str], PriceSource] | None = None,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--config", type=Path, default=DEFAULT_DATA_SOURCES_PATH)
    parser.add_argument("--split-config", type=Path, default=DEFAULT_SPLIT_CONFIG_PATH)
    parser.add_argument("--out", type=Path, default=None, help="default: data/raw/<source>")
    args = parser.parse_args(argv)

    try:
        config = load_data_sources_config(args.config)
        split_config = load_split_config(args.split_config)
        source = (source_factory or (lambda name: SOURCES[name]()))(config.source)
        out_dir = args.out or get_settings().data_dir / "raw" / config.source
        provenance = run_download(config, source, out_dir, split_config)
    except (DataPipelineError, SplitError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"{'ticker':<7}{'rows':>7}{'dropped':>9}  {'first':<11}{'last':<11}{'train':>7}{'val':>6}{'test':>6}")
    for t in provenance.tickers:
        bars = t.split_bars
        print(
            f"{t.ticker:<7}{t.rows:>7}{len(t.dropped_rows):>9}  {t.first_date!s:<11}{t.last_date!s:<11}"
            f"{bars['train']:>7}{bars['validation']:>6}{bars['test']:>6}"
        )
    print(f"wrote {len(provenance.tickers)} files and provenance to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
