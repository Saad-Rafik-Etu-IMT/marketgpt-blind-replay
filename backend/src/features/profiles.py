"""Versioned feature profiles: the named feature sets experiments compare.

Every available profile starts from the permanent baseline
(``returns_window_64``), is blind-safe and contains no declared redundancy
(``validate_profiles``). Group profiles add one feature group at a time, so the
feature-group ablation (tranche T7) compares each against ``returns_only_v1``.

Selection rule inside a group, decided before any model is trained: keep one
representative of each declared redundancy, and also drop a member of a pair
whose training-period Spearman correlation is in the 0.90 to 0.95 band when
both measure the same concept (docs/features.md, "Redundancy").
"""

from __future__ import annotations

from dataclasses import dataclass

from src.features.catalog import BASELINE_FEATURES, DEFAULT_REGISTRY
from src.features.registry import FeatureRegistry


class FeatureProfileError(ValueError):
    """A profile is unknown, unavailable or breaks the profile rules."""


@dataclass(frozen=True)
class FeatureProfile:
    id: str
    version: str
    description: str
    features: tuple[str, ...]
    available: bool = True
    unavailable_reason: str = ""

    def describe(
        self, registry: FeatureRegistry = DEFAULT_REGISTRY
    ) -> dict[str, object]:
        out: dict[str, object] = {
            "id": self.id,
            "version": self.version,
            "description": self.description,
            "features": list(self.features),
            "available": self.available,
            "unavailable_reason": self.unavailable_reason,
        }
        if self.available:
            out["column_count"] = len(registry.columns(self.features))
            out["lookback"] = registry.lookback(self.features)
            out["fingerprint"] = registry.fingerprint(self.features)
        return out


TREND = (
    "dist_sma_10",
    "dist_sma_50",
    "dist_sma_200",
    "rsi_14",
    "adx_14",
    "macd_hist",
    "stoch_k_14",
    "momentum_accel_10",
)
VOLATILITY = (
    "realized_vol_20",
    "realized_vol_60",
    "vol_ratio_5_20",
    "vol_percentile_252",
    "bb_width_20",
    "downside_semivol_20",
    "max_drawdown_60",
    "drawdown_252",
)
PRICE_STRUCTURE = (
    "hl_range_pct",
    "clv",
    "upper_wick_ratio",
    "lower_wick_ratio",
    "dist_high_20",
    "dist_low_20",
    "dist_high_60",
    "dist_low_60",
)
VOLUME = (
    "volume_z_20",
    "volume_change_5",
    "obv_flow_20",
    "price_volume_corr_20",
    "volume_trend_20",
)
STATISTICS = (
    "autocorr_1_60",
    "skew_60",
    "kurtosis_60",
    "up_fraction_20",
    "efficiency_ratio_20",
)
CORE = (
    "dist_sma_50",
    "dist_sma_200",
    "rsi_14",
    "adx_14",
    "realized_vol_20",
    "vol_ratio_5_20",
    "drawdown_252",
    "hl_range_pct",
    "clv",
    "volume_z_20",
    "obv_flow_20",
    "autocorr_1_60",
)
RETURN_SUMMARIES = (
    "cum_return_5",
    "cum_return_20",
    "cum_return_60",
    "overnight_return",
)


def _with_baseline(*features: str) -> tuple[str, ...]:
    return BASELINE_FEATURES + tuple(dict.fromkeys(features))


PROFILES: dict[str, FeatureProfile] = {
    p.id: p
    for p in (
        FeatureProfile(
            "returns_only_v1",
            "1.0.0",
            "Permanent baseline: the 64 most recent daily log returns.",
            BASELINE_FEATURES,
        ),
        FeatureProfile(
            "returns_plus_trend",
            "1.0.0",
            "Baseline + trend and momentum (B).",
            _with_baseline(*TREND),
        ),
        FeatureProfile(
            "returns_plus_volatility",
            "1.0.0",
            "Baseline + realized volatility and risk (C).",
            _with_baseline(*VOLATILITY),
        ),
        FeatureProfile(
            "returns_plus_price_structure",
            "1.0.0",
            "Baseline + bar structure and range position (D).",
            _with_baseline(*PRICE_STRUCTURE),
        ),
        FeatureProfile(
            "returns_plus_volume",
            "1.0.0",
            "Baseline + relative volume (E).",
            _with_baseline(*VOLUME),
        ),
        FeatureProfile(
            "returns_plus_statistics",
            "1.0.0",
            "Baseline + return statistics (F).",
            _with_baseline(*STATISTICS),
        ),
        FeatureProfile(
            "returns_plus_market_context",
            "1.0.0",
            "Baseline + VIX, sector and rates context (H).",
            (),
            available=False,
            unavailable_reason="Group H external sources arrive in tranche T8.",
        ),
        FeatureProfile(
            "core_features",
            "1.0.0",
            "Baseline + one or two representatives per group (12 features).",
            _with_baseline(*CORE),
        ),
        FeatureProfile(
            "extended_features",
            "1.0.0",
            "Baseline + every group profile and return summaries, without redundancy.",
            _with_baseline(
                *RETURN_SUMMARIES,
                *TREND,
                *VOLATILITY,
                *PRICE_STRUCTURE,
                *VOLUME,
                *STATISTICS,
            ),
        ),
    )
}


def get_profile(profile_id: str) -> FeatureProfile:
    try:
        profile = PROFILES[profile_id]
    except KeyError:
        raise FeatureProfileError(f"unknown feature profile {profile_id!r}") from None
    if not profile.available:
        raise FeatureProfileError(
            f"{profile_id} is unavailable: {profile.unavailable_reason}"
        )
    return profile


def validate_profiles(
    profiles: dict[str, FeatureProfile] = PROFILES,
    registry: FeatureRegistry = DEFAULT_REGISTRY,
) -> None:
    """Every available profile: known features, baseline first, blind-safe, no redundancy."""
    for profile in profiles.values():
        if not profile.available:
            if profile.features or not profile.unavailable_reason:
                raise FeatureProfileError(
                    f"{profile.id}: unavailable profiles list no features"
                )
            continue
        if profile.features[: len(BASELINE_FEATURES)] != BASELINE_FEATURES:
            raise FeatureProfileError(f"{profile.id}: must start with the baseline")
        resolved = registry.resolve(profile.features)
        unsafe = [s.id for s in resolved if not s.blind_safe]
        if unsafe:
            raise FeatureProfileError(f"{profile.id}: not blind-safe: {unsafe}")
        findings = registry.redundancies(profile.features)
        if findings:
            raise FeatureProfileError(
                f"{profile.id}: redundant combination: {findings}"
            )
