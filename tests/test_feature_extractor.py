"""
=============================================================================
test_feature_extractor.py — Unit tests for feature_extractor.py
=============================================================================
Project:    Synthetic Sleep Environment Dataset Generator
Course:     TECHIN 513: Signal Processing & Machine Learning
Team:       Team 7 — Rushav Dash & Lisa Li
University: University of Washington
Date:       2026-02-19
=============================================================================
Tests verify:
  - All expected feature keys are returned
  - Feature values are finite (no NaN, no inf)
  - Feature values are within expected ranges
  - Edge cases: flat signals, all-zero signals

Run with: pytest tests/test_feature_extractor.py -v

Authors: Rushav Dash, Lisa Li
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from src.feature_extractor import (
    extract_temperature_features,
    extract_light_features,
    extract_sound_features,
    extract_humidity_features,
    extract_all_features,
)


# ============================================================
# HELPERS
# ============================================================

def _make_flat(value: float, n: int = 96) -> np.ndarray:
    """Create a constant signal of length n."""
    return np.full(n, value, dtype=float)


def _make_sine(amplitude: float, n: int = 96) -> np.ndarray:
    """Create a sinusoidal signal."""
    t = np.linspace(0, 2 * np.pi, n)
    return 20.0 + amplitude * np.sin(t)


def _assert_finite(d: dict, prefix: str = "") -> None:
    """Assert all values in dict are finite floats or ints."""
    for k, v in d.items():
        assert np.isfinite(v), f"{prefix}Feature '{k}' is not finite: {v}"


# ============================================================
# TESTS: TEMPERATURE FEATURES
# ============================================================

class TestTemperatureFeatures:

    EXPECTED_KEYS = [
        "temp_mean", "temp_std", "temp_min", "temp_max", "temp_range",
        "temp_above_21_minutes", "temp_below_18_minutes",
        "temp_optimal_fraction", "temp_mean_rate_change", "temp_max_rate_change",
    ]

    def test_all_keys_present(self):
        t = _make_sine(1.0) + 19.5   # oscillates 18.5–20.5°C → in optimal zone
        feats = extract_temperature_features(t)
        for key in self.EXPECTED_KEYS:
            assert key in feats, f"Missing key: '{key}'"

    def test_values_finite(self):
        t = _make_sine(2.0) + 20.0
        feats = extract_temperature_features(t)
        _assert_finite(feats, prefix="temperature ")

    def test_flat_signal_zero_std(self):
        t = _make_flat(20.0)
        feats = extract_temperature_features(t)
        assert feats["temp_std"] == pytest.approx(0.0, abs=1e-9)
        assert feats["temp_range"] == pytest.approx(0.0, abs=1e-9)
        assert feats["temp_mean_rate_change"] == pytest.approx(0.0, abs=1e-9)

    def test_optimal_fraction_all_optimal(self):
        """Temperature constantly in [18, 21] → optimal_fraction = 1.0."""
        t = _make_flat(19.5)
        feats = extract_temperature_features(t)
        assert feats["temp_optimal_fraction"] == pytest.approx(1.0, abs=1e-6)
        assert feats["temp_above_21_minutes"] == 0
        assert feats["temp_below_18_minutes"] == 0

    def test_optimal_fraction_all_hot(self):
        """Temperature always above 21 → optimal_fraction = 0.0."""
        t = _make_flat(25.0)
        feats = extract_temperature_features(t)
        assert feats["temp_optimal_fraction"] == pytest.approx(0.0, abs=1e-6)
        assert feats["temp_above_21_minutes"] == 96 * 5  # all 96 samples × 5 min

    def test_minutes_scale_correctly(self):
        """Half the signal above 21°C → 240 minutes (48 samples × 5 min)."""
        t = np.concatenate([_make_flat(22.0, 48), _make_flat(19.0, 48)])
        feats = extract_temperature_features(t, dt_minutes=5)
        assert feats["temp_above_21_minutes"] == 48 * 5   # 240 minutes


# ============================================================
# TESTS: LIGHT FEATURES
# ============================================================

class TestLightFeatures:

    EXPECTED_KEYS = [
        "light_mean", "light_std", "light_max", "light_peak_lux",
        "light_event_count", "light_total_exposure_minutes",
        "light_disruption_score",
    ]

    def test_all_keys_present(self):
        light = np.zeros(96)
        feats = extract_light_features(light)
        for key in self.EXPECTED_KEYS:
            assert key in feats, f"Missing key: '{key}'"

    def test_all_dark_no_events(self):
        """All-dark signal → zero events, zero exposure."""
        light = np.zeros(96)
        feats = extract_light_features(light)
        assert feats["light_event_count"] == 0
        assert feats["light_total_exposure_minutes"] == 0
        assert feats["light_disruption_score"] == pytest.approx(0.0, abs=1e-6)

    def test_single_event_detected(self):
        """One 5-minute pulse of 50 lux → 1 event."""
        light = np.zeros(96)
        light[40:41] = 50.0   # one sample above threshold
        feats = extract_light_features(light, event_threshold_lux=10.0)
        assert feats["light_event_count"] >= 1

    def test_non_negative_values(self):
        """All feature values must be ≥ 0."""
        light = np.random.default_rng(0).uniform(0, 100, 96)
        feats = extract_light_features(light)
        for k, v in feats.items():
            assert v >= 0, f"Feature '{k}' is negative: {v}"

    def test_values_finite(self):
        light = np.random.default_rng(1).uniform(0, 80, 96)
        feats = extract_light_features(light)
        _assert_finite(feats, prefix="light ")


# ============================================================
# TESTS: SOUND FEATURES
# ============================================================

class TestSoundFeatures:

    EXPECTED_KEYS = [
        "sound_mean_db", "sound_std_db", "sound_max_db",
        "sound_above_55db_minutes", "sound_event_count", "sound_leq_db",
    ]

    def test_all_keys_present(self):
        sound = _make_flat(35.0)
        feats = extract_sound_features(sound)
        for key in self.EXPECTED_KEYS:
            assert key in feats, f"Missing key: '{key}'"

    def test_quiet_signal_no_events(self):
        sound = _make_flat(35.0)   # well below 55 dB threshold
        feats = extract_sound_features(sound)
        assert feats["sound_above_55db_minutes"] == 0
        assert feats["sound_event_count"] == 0

    def test_leq_reasonable(self):
        """Leq of 40 dB flat signal should be approximately 40 dB."""
        sound = _make_flat(40.0)
        feats = extract_sound_features(sound)
        assert feats["sound_leq_db"] == pytest.approx(40.0, abs=0.5)

    def test_values_finite(self):
        sound = np.random.default_rng(2).uniform(25, 75, 96)
        feats = extract_sound_features(sound)
        _assert_finite(feats, prefix="sound ")


# ============================================================
# TESTS: HUMIDITY FEATURES
# ============================================================

class TestHumidityFeatures:

    EXPECTED_KEYS = [
        "humidity_mean", "humidity_std", "humidity_min", "humidity_max",
        "humidity_out_of_range_minutes", "humidity_comfort_fraction",
    ]

    def test_all_keys_present(self):
        hum = _make_flat(50.0)
        feats = extract_humidity_features(hum)
        for key in self.EXPECTED_KEYS:
            assert key in feats, f"Missing key: '{key}'"

    def test_comfort_zone_all_in_range(self):
        hum = _make_flat(45.0)   # in [30, 60]
        feats = extract_humidity_features(hum)
        assert feats["humidity_out_of_range_minutes"] == 0
        assert feats["humidity_comfort_fraction"] == pytest.approx(1.0, abs=1e-6)

    def test_all_out_of_range(self):
        hum = _make_flat(5.0)   # below 30 → always out of range
        feats = extract_humidity_features(hum)
        assert feats["humidity_comfort_fraction"] == pytest.approx(0.0, abs=1e-6)

    def test_values_finite(self):
        hum = np.random.default_rng(3).uniform(20, 80, 96)
        feats = extract_humidity_features(hum)
        _assert_finite(feats, prefix="humidity ")


# ============================================================
# TESTS: EXTRACT ALL FEATURES (TOP-LEVEL)
# ============================================================

class TestExtractAllFeatures:

    def _make_signals(self):
        rng = np.random.default_rng(42)
        return {
            "t":           np.arange(0, 480, 5, dtype=float),
            "temperature": rng.normal(20.0, 1.0, 96),
            "light":       np.abs(rng.normal(5.0, 10.0, 96)),
            "sound":       rng.normal(38.0, 5.0, 96) + 25.0,
            "humidity":    rng.normal(50.0, 5.0, 96),
        }

    def test_returns_dict(self):
        signals = self._make_signals()
        feats = extract_all_features(signals)
        assert isinstance(feats, dict)

    def test_all_features_finite(self):
        signals = self._make_signals()
        feats = extract_all_features(signals)
        _assert_finite(feats, prefix="all_features ")

    def test_optional_signals_none(self):
        """Should not crash when optional signals are None."""
        signals = {
            "t":           np.arange(0, 480, 5, dtype=float),
            "temperature": np.full(96, 20.0),
            "light":       np.full(96, 2.0),
            "sound":       None,
            "humidity":    None,
        }
        feats = extract_all_features(signals)
        assert "temp_mean" in feats
        assert "light_mean" in feats
        # Sound and humidity keys should not be present when signals are None
        assert "sound_mean_db" not in feats
        assert "humidity_mean" not in feats

    def test_minimum_feature_count(self):
        """Should produce at least 10 features from temperature + light alone."""
        signals = {
            "t":           np.arange(0, 480, 5, dtype=float),
            "temperature": np.full(96, 20.0),
            "light":       np.full(96, 2.0),
            "sound":       None,
            "humidity":    None,
        }
        feats = extract_all_features(signals)
        assert len(feats) >= 10, f"Too few features returned: {len(feats)}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
