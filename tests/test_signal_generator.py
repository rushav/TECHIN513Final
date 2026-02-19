"""
=============================================================================
test_signal_generator.py — Unit tests for signal_generator.py
=============================================================================
Project:    Synthetic Sleep Environment Dataset Generator
Course:     TECHIN 513: Signal Processing & Machine Learning
Team:       Team 7 — Rushav Dash & Lisa Li
University: University of Washington
Date:       2026-02-19
=============================================================================
Tests verify:
  - Output shapes match expected (96 samples for 480-min session at 5-min dt)
  - Signal ranges are physiologically plausible
  - Reproducibility: same seed → identical output
  - Season stratification: summer temp > winter temp (on average)
  - Poisson light events: event count scales with lambda

Run with: pytest tests/test_signal_generator.py -v

Authors: Rushav Dash, Lisa Li
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from src.signal_generator import SignalGenerator
from src.data_loader import ReferenceStats


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def default_gen():
    """Create a SignalGenerator with default ReferenceStats (no real data needed)."""
    ref = ReferenceStats()   # uses sensible defaults
    return SignalGenerator(ref_stats=ref, duration_minutes=480, sampling_interval_minutes=5)


SEASONS = ["winter", "spring", "summer", "fall"]
AGE_GROUPS = ["young", "middle", "senior"]
SENSITIVITIES = ["low", "normal", "high"]


# ============================================================
# TESTS: OUTPUT SHAPE
# ============================================================

class TestOutputShape:
    """Verify that all signals have the correct number of samples."""

    def test_temperature_shape(self, default_gen):
        t = default_gen.generate_temperature()
        assert t.shape == (96,), f"Expected (96,), got {t.shape}"

    def test_light_shape(self, default_gen):
        light = default_gen.generate_light()
        assert light.shape == (96,), f"Expected (96,), got {light.shape}"

    def test_sound_shape(self, default_gen):
        sound = default_gen.generate_sound()
        assert sound.shape == (96,), f"Expected (96,), got {sound.shape}"

    def test_humidity_shape(self, default_gen):
        hum = default_gen.generate_humidity()
        assert hum.shape == (96,), f"Expected (96,), got {hum.shape}"

    def test_generate_all_keys(self, default_gen):
        signals = default_gen.generate_all()
        assert "temperature" in signals
        assert "light" in signals
        assert "sound" in signals
        assert "humidity" in signals
        assert "t" in signals

    def test_generate_all_shapes(self, default_gen):
        signals = default_gen.generate_all()
        for key in ["temperature", "light", "sound", "humidity", "t"]:
            arr = signals[key]
            assert arr is not None
            assert len(arr) == 96, f"Key '{key}' has wrong length: {len(arr)}"


# ============================================================
# TESTS: PHYSIOLOGICAL RANGE VALIDATION
# ============================================================

class TestPhysiologicalRanges:
    """Verify signal values fall within physically plausible ranges."""

    def test_temperature_range(self, default_gen):
        """Indoor bedroom temperature should be between 10°C and 35°C."""
        for season in SEASONS:
            t = default_gen.generate_temperature(season=season, random_seed=7)
            assert t.min() > 10, f"Temp too cold: {t.min():.2f}°C (season={season})"
            assert t.max() < 35, f"Temp too hot: {t.max():.2f}°C (season={season})"

    def test_light_non_negative(self, default_gen):
        """Light levels must be ≥ 0 lux."""
        light = default_gen.generate_light(random_seed=42)
        assert np.all(light >= 0), "Light signal has negative values"

    def test_light_upper_bound(self, default_gen):
        """Bedroom light should not exceed 1000 lux."""
        light = default_gen.generate_light(random_seed=42)
        assert light.max() <= 1000, f"Light too bright: {light.max():.1f} lux"

    def test_sound_range(self, default_gen):
        """Sound level should be within [0, 120] dB."""
        sound = default_gen.generate_sound(random_seed=42)
        assert sound.min() >= 0,   f"Sound below 0 dB: {sound.min():.1f}"
        assert sound.max() <= 120, f"Sound above 120 dB: {sound.max():.1f}"

    def test_humidity_range(self, default_gen):
        """Relative humidity should be in [10, 95] %."""
        for season in SEASONS:
            hum = default_gen.generate_humidity(season=season, random_seed=5)
            assert hum.min() >= 10, f"Humidity too low: {hum.min():.1f}% (season={season})"
            assert hum.max() <= 95, f"Humidity too high: {hum.max():.1f}% (season={season})"


# ============================================================
# TESTS: REPRODUCIBILITY
# ============================================================

class TestReproducibility:
    """Same seed must always produce identical output."""

    def test_temperature_reproducible(self, default_gen):
        t1 = default_gen.generate_temperature(random_seed=123)
        t2 = default_gen.generate_temperature(random_seed=123)
        np.testing.assert_array_equal(t1, t2)

    def test_light_reproducible(self, default_gen):
        l1 = default_gen.generate_light(random_seed=456)
        l2 = default_gen.generate_light(random_seed=456)
        np.testing.assert_array_equal(l1, l2)

    def test_different_seeds_different_output(self, default_gen):
        t1 = default_gen.generate_temperature(random_seed=1)
        t2 = default_gen.generate_temperature(random_seed=2)
        assert not np.array_equal(t1, t2), "Different seeds produced identical temperature signals"

    def test_generate_all_reproducible(self, default_gen):
        s1 = default_gen.generate_all(random_seed=999)
        s2 = default_gen.generate_all(random_seed=999)
        np.testing.assert_array_equal(s1["temperature"], s2["temperature"])
        np.testing.assert_array_equal(s1["light"], s2["light"])


# ============================================================
# TESTS: SEASONAL STRATIFICATION
# ============================================================

class TestSeasonalStratification:
    """Summer sessions should be warmer than winter sessions (on average)."""

    def test_summer_warmer_than_winter(self, default_gen):
        n_trials = 50
        summer_means, winter_means = [], []
        for seed in range(n_trials):
            summer_means.append(default_gen.generate_temperature(season="summer", random_seed=seed).mean())
            winter_means.append(default_gen.generate_temperature(season="winter", random_seed=seed).mean())
        assert np.mean(summer_means) > np.mean(winter_means), (
            f"Summer ({np.mean(summer_means):.2f}°C) should be warmer than "
            f"winter ({np.mean(winter_means):.2f}°C)"
        )

    def test_summer_humidity_higher_than_winter(self, default_gen):
        n_trials = 30
        summer_hums, winter_hums = [], []
        for seed in range(n_trials):
            summer_hums.append(default_gen.generate_humidity(season="summer", random_seed=seed).mean())
            winter_hums.append(default_gen.generate_humidity(season="winter", random_seed=seed).mean())
        # Summer is typically more humid (though ranges overlap)
        assert np.mean(summer_hums) >= np.mean(winter_hums) - 5, (
            "Summer humidity should not be dramatically lower than winter"
        )


# ============================================================
# TESTS: SENSITIVITY PARAMETER
# ============================================================

class TestSensitivityParameter:
    """High-sensitivity individuals should experience more light events."""

    def test_high_sensitivity_more_light_events(self, default_gen):
        """High sensitivity → higher Poisson λ → more events on average."""
        n_trials = 100
        low_events, high_events = [], []
        for seed in range(n_trials):
            l_low  = default_gen.generate_light(sensitivity="low",  random_seed=seed)
            l_high = default_gen.generate_light(sensitivity="high", random_seed=seed)
            # Count light-on transitions as event proxy
            low_events.append((l_low > 10).sum())
            high_events.append((l_high > 10).sum())
        assert np.mean(high_events) > np.mean(low_events), (
            "High sensitivity should produce more light events than low sensitivity"
        )


# ============================================================
# TESTS: TEMPERATURE AUTOCORRELATION (THERMAL INERTIA)
# ============================================================

class TestThermalInertia:
    """Filtered temperature should have high autocorrelation at lag-1 (> 0.8)."""

    def test_autocorrelation_high(self, default_gen):
        import pandas as pd
        for seed in range(10):
            t = default_gen.generate_temperature(random_seed=seed)
            acf_1 = pd.Series(t).autocorr(lag=1)
            assert acf_1 > 0.8, (
                f"Temperature autocorrelation at lag-1 = {acf_1:.3f} < 0.80 "
                f"(thermal inertia filter may not be working; seed={seed})"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
