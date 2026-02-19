"""
=============================================================================
test_validator.py — Unit tests for validator.py
=============================================================================
Project:    Synthetic Sleep Environment Dataset Generator
Course:     TECHIN 513: Signal Processing & Machine Learning
Team:       Team 7 — Rushav Dash & Lisa Li
University: University of Washington
Date:       2026-02-19
=============================================================================
Tests use a small, hand-crafted synthetic DataFrame to verify that each
validation check fires correctly (both PASS and FAIL paths) without
requiring the real Kaggle datasets.

Run with: pytest tests/test_validator.py -v

Authors: Rushav Dash, Lisa Li
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from src.validator import Validator, ValidationReport


# ============================================================
# FIXTURES: MINIMAL SYNTHETIC DATASET
# ============================================================

def _make_synthetic_df(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """
    Create a minimal synthetic DataFrame that satisfies all sanity checks.

    Designed so that all Tier 3 checks PASS.
    """
    rng = np.random.default_rng(seed)

    seasons       = (["winter", "spring", "summer", "fall"] * ((n // 4) + 1))[:n]
    age_groups    = (["young", "middle", "senior"] * ((n // 3) + 1))[:n]
    sensitivities = (["low", "normal", "high"] * ((n // 3) + 1))[:n]

    # Temperature: summer warmer; optimal fraction varies
    season_temp = {"winter": 19.0, "spring": 20.5, "summer": 22.0, "fall": 20.0}
    temp_means = np.array([season_temp[s] + rng.normal(0, 0.5) for s in seasons])

    # temp_optimal_fraction: use a reasonable spread
    opt_fracs = np.clip(rng.beta(5, 2, n), 0, 1)

    # Light events: mostly low, some high
    light_events = rng.poisson(2.5, n)
    disruption   = rng.exponential(500, n)

    # Sleep efficiency: correlated with optimal fraction, penalised by light events
    base_eff   = 0.72 + 0.15 * opt_fracs - 0.02 * light_events + rng.normal(0, 0.04, n)
    sleep_eff  = np.clip(base_eff, 0.50, 0.99)

    # Awakenings: negatively correlated with efficiency
    awakenings = np.clip(np.round(4 - 3 * sleep_eff + rng.normal(0, 0.5, n)), 0, 12).astype(int)

    # Deep sleep: negatively correlated with awakenings
    deep_pct   = np.clip(22 - 1.5 * awakenings + rng.normal(0, 3, n), 5, 40)
    rem_pct    = np.clip(rng.normal(22, 4, n), 5, 40)
    # Normalise so they sum to 100
    light_pct  = 100 - rem_pct - deep_pct
    # Handle negative light_pct
    need_fix   = light_pct < 5
    if need_fix.any():
        rem_pct[need_fix]   = 20
        deep_pct[need_fix]  = 20
        light_pct[need_fix] = 60

    # Humidity
    hum_means = rng.normal(50, 8, n)
    hum_oor   = rng.integers(0, 60, n)

    # Sound
    sound_means = rng.normal(38, 5, n)
    sound_above = rng.integers(0, 30, n)

    df = pd.DataFrame({
        "session_index":              range(n),
        "season":                     seasons,
        "age_group":                  age_groups,
        "sensitivity":                sensitivities,
        "temp_mean":                  temp_means,
        "temp_std":                   rng.uniform(0.3, 1.2, n),
        "temp_min":                   temp_means - rng.uniform(1, 2, n),
        "temp_max":                   temp_means + rng.uniform(1, 2, n),
        "temp_range":                 rng.uniform(2, 4, n),
        "temp_above_21_minutes":      rng.integers(0, 120, n),
        "temp_below_18_minutes":      rng.integers(0, 30, n),
        "temp_optimal_fraction":      opt_fracs,
        "temp_mean_rate_change":      rng.uniform(0.01, 0.1, n),
        "temp_max_rate_change":       rng.uniform(0.1, 0.5, n),
        "light_mean":                 rng.uniform(0, 15, n),
        "light_std":                  rng.uniform(0, 10, n),
        "light_max":                  rng.uniform(5, 150, n),
        "light_peak_lux":             rng.uniform(5, 150, n),
        "light_event_count":          light_events,
        "light_total_exposure_minutes": rng.integers(0, 60, n),
        "light_disruption_score":     disruption,
        "sound_mean_db":              sound_means,
        "sound_std_db":               rng.uniform(1, 5, n),
        "sound_max_db":               sound_means + rng.uniform(5, 20, n),
        "sound_above_55db_minutes":   sound_above,
        "sound_event_count":          rng.integers(0, 8, n),
        "sound_leq_db":               sound_means,
        "humidity_mean":              hum_means,
        "humidity_std":               rng.uniform(0.5, 3, n),
        "humidity_min":               hum_means - 10,
        "humidity_max":               hum_means + 10,
        "humidity_out_of_range_minutes": hum_oor,
        "humidity_comfort_fraction":  rng.uniform(0.6, 1.0, n),
        "sleep_efficiency":           sleep_eff,
        "awakenings":                 awakenings,
        "rem_pct":                    rem_pct,
        "deep_pct":                   deep_pct,
        "light_pct":                  light_pct,
    })
    return df


def _make_failing_df(n: int = 200, seed: int = 7) -> pd.DataFrame:
    """
    Create a DataFrame where Tier 3 sanity checks should FAIL.

    - Good temp → low efficiency (FAIL check 1)
    - Many light events → high efficiency (FAIL check 2)
    """
    rng = np.random.default_rng(seed)

    seasons    = (["winter", "spring", "summer", "fall"] * 50)[:n]
    age_groups = (["young", "middle", "senior"] * 67)[:n]
    sensitivities = (["low", "normal", "high"] * 67)[:n]

    opt_fracs    = np.clip(rng.beta(8, 1, n), 0, 1)    # mostly high (>0.8)
    light_events = rng.integers(5, 10, n)                # always > 4

    # INVERTED: good temp → LOW efficiency (should fail check 1)
    sleep_eff = np.clip(rng.uniform(0.50, 0.65, n), 0.50, 0.99)   # below 0.78
    awakenings = rng.integers(2, 8, n)

    deep_pct  = np.clip(rng.normal(18, 3, n), 5, 40)
    rem_pct   = np.clip(rng.normal(22, 4, n), 5, 40)
    light_pct = 100 - rem_pct - deep_pct

    df = pd.DataFrame({
        "session_index":            range(n),
        "season":                   seasons,
        "age_group":                age_groups,
        "sensitivity":              sensitivities,
        "temp_mean":                rng.normal(20, 1, n),
        "temp_optimal_fraction":    opt_fracs,
        "light_event_count":        light_events,
        "light_disruption_score":   rng.exponential(1000, n),
        "light_mean":               rng.uniform(10, 40, n),
        "light_max":                rng.uniform(50, 200, n),
        "sleep_efficiency":         sleep_eff,
        "awakenings":               awakenings,
        "deep_pct":                 deep_pct,
        "rem_pct":                  rem_pct,
        "light_pct":                light_pct,
    })
    return df


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def good_validator():
    """Validator with a well-behaved synthetic dataset."""
    df = _make_synthetic_df(n=400, seed=42)
    return Validator(df, real_occupancy_df=None, real_sleep_df=None)


@pytest.fixture
def bad_validator():
    """Validator with a dataset designed to fail Tier 3 checks."""
    df = _make_failing_df(n=200, seed=7)
    return Validator(df, real_occupancy_df=None, real_sleep_df=None)


# ============================================================
# TESTS: ValidationReport
# ============================================================

class TestValidationReport:

    def test_summary_returns_string(self, good_validator):
        good_validator.run_all()
        summary = good_validator.report.summary()
        assert isinstance(summary, str)
        assert "VALIDATION REPORT" in summary

    def test_overall_score_range(self, good_validator):
        good_validator.run_all()
        assert 0.0 <= good_validator.report.overall_score <= 100.0

    def test_passed_checks_leq_total(self, good_validator):
        good_validator.run_all()
        assert good_validator.report.passed_checks <= good_validator.report.total_checks


# ============================================================
# TESTS: Tier 1 — Statistical Tests (no real data)
# ============================================================

class TestTier1:

    def test_no_crash_without_real_data(self, good_validator):
        """Should not raise when real datasets are None."""
        result = good_validator.tier1_statistical()
        assert isinstance(result, dict)

    def test_returns_warning_when_no_real_data(self, good_validator):
        result = good_validator.tier1_statistical()
        # Either returns warning key or returns empty dict — both acceptable
        # (real data was not provided)
        assert "_warning" in result or len(result) == 0

    def test_with_mock_real_data(self):
        """With real occupancy data provided, should run KS-test."""
        rng = np.random.default_rng(0)
        # Create a mock real occupancy DataFrame
        real_occ = pd.DataFrame({
            "Temperature": rng.normal(20.5, 1.5, 500),
            "Light": rng.exponential(5.0, 500),
        })
        syn_df = _make_synthetic_df(n=300, seed=42)
        validator = Validator(syn_df, real_occupancy_df=real_occ)
        result = validator.tier1_statistical()
        assert "Temperature mean (KS-test)" in result
        assert "p_value" in result["Temperature mean (KS-test)"]
        assert "pass" in result["Temperature mean (KS-test)"]


# ============================================================
# TESTS: Tier 3 — Sanity Checks
# ============================================================

class TestTier3:

    def test_check_count_positive(self, good_validator):
        """Should run at least 3 sanity checks."""
        checks = good_validator.tier3_sanity_checks()
        assert len(checks) >= 3

    def test_each_check_has_required_keys(self, good_validator):
        checks = good_validator.tier3_sanity_checks()
        for check in checks:
            assert "name" in check
            assert "pass" in check
            assert "actual_value" in check
            assert "threshold" in check

    def test_stage_sum_check_passes(self, good_validator):
        """Sleep stages should sum to 100 → this check should PASS."""
        checks = good_validator.tier3_sanity_checks()
        sum_check = [c for c in checks if "sum to 100" in c["name"]]
        if sum_check:
            assert sum_check[0]["pass"] is True, (
                f"Stage sum check failed: deviation = {sum_check[0]['actual_value']:.4f}"
            )

    def test_deep_sleep_awakenings_correlation(self, good_validator):
        """Good dataset should show negative deep–awakening correlation."""
        checks = good_validator.tier3_sanity_checks()
        corr_check = [c for c in checks if "correlation" in c["name"].lower()]
        if corr_check:
            # The actual value is the Pearson r — should be negative
            assert corr_check[0]["actual_value"] < 0, (
                "Deep sleep vs. awakenings correlation should be negative"
            )

    def test_failing_dataset_has_failures(self, bad_validator):
        """The adversarially constructed dataset should fail at least one check."""
        checks = bad_validator.tier3_sanity_checks()
        failed = [c for c in checks if not c["pass"]]
        assert len(failed) >= 1, (
            "Expected at least one failing check on the adversarial dataset"
        )

    def test_seasonal_temp_check(self):
        """Summer must be warmer than winter to pass the seasonal check."""
        rng = np.random.default_rng(42)
        n = 200
        seasons = ["winter"] * 100 + ["summer"] * 100
        temp_means = (
            list(rng.normal(18.5, 0.5, 100))    # winter: cold
            + list(rng.normal(23.0, 0.5, 100))  # summer: warm
        )
        # Add other required columns
        df = pd.DataFrame({
            "season":              seasons,
            "age_group":           (["young"] * 200),
            "sensitivity":         (["normal"] * 200),
            "temp_mean":           temp_means,
            "temp_optimal_fraction": rng.uniform(0.5, 0.9, n),
            "light_event_count":   rng.integers(0, 5, n),
            "light_disruption_score": rng.exponential(300, n),
            "sleep_efficiency":    rng.uniform(0.60, 0.95, n),
            "awakenings":          rng.integers(0, 5, n),
            "deep_pct":            rng.uniform(10, 35, n),
            "rem_pct":             rng.uniform(15, 30, n),
            "light_pct":           rng.uniform(30, 60, n),
        })
        validator = Validator(df)
        checks = validator.tier3_sanity_checks()
        season_check = [c for c in checks if "summer" in c["name"].lower()]
        if season_check:
            assert season_check[0]["pass"] is True, (
                f"Seasonal temp check failed: diff = {season_check[0]['actual_value']:.2f}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
