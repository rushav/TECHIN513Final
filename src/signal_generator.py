"""
=============================================================================
signal_generator.py — Synthetic environmental time-series generation
=============================================================================
Project:    Synthetic Sleep Environment Dataset Generator
Course:     TECHIN 513: Signal Processing & Machine Learning
Team:       Team 7 — Rushav Dash & Lisa Li
University: University of Washington
Date:       2026-02-19
=============================================================================
Core signal-processing module.  For each 8-hour sleep session we generate
four environmental time-series at a 5-minute sampling interval (96 points):

  A) Indoor Temperature (°C)   — spectral synthesis with 3 additive components
  B) Light Level (lux)          — Poisson-process event model
  C) Sound Level (dB SPL)       — pink noise baseline + Poisson events
  D) Relative Humidity (%)      — sinusoidal + mild noise

Each signal is calibrated against real IoT statistics extracted by
data_loader.py.  All random operations accept a `random_seed` argument
for full reproducibility.

Signal-processing references:
  - Butterworth filter: scipy.signal.butter / sosfiltfilt
  - Pink noise via inverse FFT: https://en.wikipedia.org/wiki/Colors_of_noise
  - Poisson process: np.random.Generator.poisson

Authors: Rushav Dash, Lisa Li
"""

import warnings
from typing import Optional

import numpy as np
from scipy import signal as sp_signal

from src.data_loader import ReferenceStats


# ============================================================
# SECTION 1: SEASON BASELINE TABLES
# Temperature and humidity baselines vary by season.
# Values are medians derived from ASHRAE comfort guidelines.
# ============================================================

# (min_base_temp, max_base_temp) in °C for each season
SEASON_TEMP_RANGES = {
    "winter": (17.0, 20.0),
    "spring": (19.0, 22.0),
    "summer": (21.0, 25.0),
    "fall":   (18.0, 22.0),
}

# (min_humidity, max_humidity) % for each season
SEASON_HUMIDITY_RANGES = {
    "winter": (25.0, 45.0),
    "spring": (40.0, 60.0),
    "summer": (45.0, 65.0),
    "fall":   (35.0, 55.0),
}

# Sensitivity multipliers: adjusts event frequency and signal amplitudes
# for individuals who are more or less sensitive to environmental disturbances.
SENSITIVITY_MULTIPLIERS = {
    "low":    0.6,
    "normal": 1.0,
    "high":   1.5,
}

# Age-group adjustments: seniors sleep lighter (more awakenings, lower efficiency)
AGE_TEMP_BIAS = {
    "young":  0.0,    # no bias
    "middle": 0.5,    # slightly warmer preference
    "senior": 1.0,    # significantly warmer preference (cold sensitivity)
}


# ============================================================
# SECTION 2: LOW-LEVEL DSP PRIMITIVES
# ============================================================

def _generate_pink_noise(n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """
    Generate normalised pink (1/f) noise using the inverse-FFT method.

    The spectrum is shaped so that power ∝ 1/f, then transformed back to
    the time domain.  The result is normalised to zero mean, unit variance.

    Parameters
    ----------
    n_samples : int
        Number of time-domain samples to produce.
    rng : np.random.Generator
        Seeded random number generator (for reproducibility).

    Returns
    -------
    np.ndarray of shape (n_samples,)
        Pink-noise signal with mean≈0, std≈1.

    Notes
    -----
    Authors: Rushav Dash, Lisa Li
    Unlike white noise (flat spectrum), 1/f noise has more power at low
    frequencies, mimicking many natural processes such as temperature drift
    and ambient sound.
    """
    # Build the one-sided frequency array for an N-point real FFT
    n_freqs = n_samples // 2 + 1
    freqs = np.arange(1, n_freqs + 1, dtype=float)  # start at 1 to avoid DC singularity

    # Random complex spectral amplitudes with 1/f magnitude profile
    amplitudes = 1.0 / np.sqrt(freqs)                        # 1/f shaping
    phases = rng.uniform(0, 2 * np.pi, size=n_freqs)         # random phases
    spectrum = amplitudes * (np.cos(phases) + 1j * np.sin(phases))

    # Enforce Hermitian symmetry so the IFFT yields a real signal
    time_domain = np.fft.irfft(spectrum, n=n_samples)

    # Normalise to zero mean, unit variance
    time_domain -= time_domain.mean()
    std = time_domain.std()
    if std > 1e-10:
        time_domain /= std

    return time_domain


def _sawtooth_hvac(t: np.ndarray, period_min: float, amplitude: float) -> np.ndarray:
    """
    Generate a sawtooth wave modelling one HVAC heating/cooling cycle.

    A heating cycle climbs linearly then drops suddenly (reverse sawtooth
    is used: 1 → 0, so the signal rises then snaps back — like a thermostat
    heating and then switching off).

    Parameters
    ----------
    t : np.ndarray
        Time axis in minutes.
    period_min : float
        Duration of one HVAC cycle in minutes (typically 30–60 min).
    amplitude : float
        Peak-to-peak temperature variation in °C (typically 0.5–1.5°C).

    Returns
    -------
    np.ndarray
        HVAC component values at each time step.

    Authors: Rushav Dash, Lisa Li
    """
    # scipy.signal.sawtooth(phase, width=1) produces /| shape (rising then snap)
    # We use width=0 to get a falling sawtooth (|\) — models cooling after heating
    phase = 2 * np.pi * t / period_min
    wave = sp_signal.sawtooth(phase, width=0)   # range [-1, 1]
    # Shift to [0, 1] range, then scale by amplitude
    return amplitude * (wave + 1) / 2.0


def _apply_butterworth_lpf(
    data: np.ndarray,
    cutoff_freq: float,
    fs: float = 1.0 / 5.0,   # default: 1 sample per 5 minutes
    order: int = 4,
) -> np.ndarray:
    """
    Apply a zero-phase Butterworth low-pass filter (IIR, forward-backward).

    Zero-phase filtering with sosfiltfilt ensures no group delay — the
    filtered output is time-aligned with the input.

    Parameters
    ----------
    data : np.ndarray
        Input signal (1-D).
    cutoff_freq : float
        Cutoff frequency in the same units as fs (cycles per minute by default).
    fs : float
        Sampling frequency in cycles per minute.
        Default = 1/5 min⁻¹ (one sample every 5 minutes).
    order : int
        Filter order.  Order 4 gives −80 dB/decade rolloff.

    Returns
    -------
    np.ndarray
        Filtered signal, same length as input.

    Authors: Rushav Dash, Lisa Li
    """
    # Normalise cutoff to Nyquist (required by scipy.signal.butter)
    nyq = 0.5 * fs
    normalised_cutoff = cutoff_freq / nyq

    # Clamp to (0, 1) to avoid scipy raising an error
    normalised_cutoff = float(np.clip(normalised_cutoff, 1e-4, 0.999))

    # Design the filter as second-order sections (numerically stable)
    sos = sp_signal.butter(order, normalised_cutoff, btype="low", output="sos")

    # sosfiltfilt applies the filter twice (forward + backward) → zero phase
    return sp_signal.sosfiltfilt(sos, data)


def _gaussian_smooth_pulse(
    pulse: np.ndarray, sigma_samples: float
) -> np.ndarray:
    """
    Convolve a rectangular pulse with a Gaussian window to soften edges.

    This prevents physically impossible instantaneous step changes in light
    or sound levels.

    Parameters
    ----------
    pulse : np.ndarray
        Binary or amplitude mask (1-D).
    sigma_samples : float
        Standard deviation of the Gaussian in samples.

    Returns
    -------
    np.ndarray
        Smoothed pulse, same length as input.

    Authors: Rushav Dash, Lisa Li
    """
    # Build a Gaussian kernel of width 6σ (captures >99.7% of area)
    kernel_size = max(3, int(6 * sigma_samples) | 1)  # force odd length
    half = kernel_size // 2
    x = np.arange(-half, half + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / sigma_samples) ** 2)
    kernel /= kernel.sum()  # normalise to unit sum

    # Full convolution with 'same' mode preserves signal length
    return np.convolve(pulse, kernel, mode="same")


# ============================================================
# SECTION 3: SIGNAL GENERATOR CLASS
# ============================================================

class SignalGenerator:
    """
    Generate realistic synthetic environmental time-series for one sleep session.

    All signals span `duration_minutes` (default 480 = 8 hours) at
    `sampling_interval_minutes` (default 5 min), producing 96 samples.

    The generator is calibrated with ReferenceStats extracted from real
    IoT recordings by DataLoader.

    Usage
    -----
    >>> gen = SignalGenerator(ref_stats)
    >>> signals = gen.generate_all(season="winter", age_group="middle",
    ...                            sensitivity="normal", random_seed=42)

    Authors: Rushav Dash, Lisa Li
    """

    def __init__(
        self,
        ref_stats: Optional[ReferenceStats] = None,
        duration_minutes: int = 480,
        sampling_interval_minutes: int = 5,
    ):
        """
        Initialise the generator with calibration statistics.

        Parameters
        ----------
        ref_stats : ReferenceStats or None
            Calibration statistics from DataLoader.  Uses sensible defaults
            if None (useful for unit tests without real data).
        duration_minutes : int
            Length of one sleep session (default 480 = 8 hours).
        sampling_interval_minutes : int
            Temporal resolution of output signals (default 5 min).

        Authors: Rushav Dash, Lisa Li
        """
        self.ref = ref_stats if ref_stats is not None else ReferenceStats()
        self.duration = duration_minutes
        self.dt = sampling_interval_minutes
        # Number of samples per signal
        self.n = duration_minutes // sampling_interval_minutes
        # Time axis in minutes: 0, 5, 10, …, 475 for an 8-h session
        self.t = np.arange(0, duration_minutes, sampling_interval_minutes, dtype=float)

    # ----------------------------------------------------------
    # 3a. Temperature signal
    # ----------------------------------------------------------

    def generate_temperature(
        self,
        season: str = "fall",
        age_group: str = "middle",
        sensitivity: str = "normal",
        base_temp: Optional[float] = None,
        random_seed: int = 42,
    ) -> np.ndarray:
        """
        Generate a synthetic indoor temperature time-series using spectral synthesis.

        The signal is the sum of three physically motivated components:

          temperature(t) = T_base + T_circadian(t) + T_hvac(t) + T_noise(t)

        A Butterworth low-pass filter is applied last to enforce thermal
        inertia (temperatures cannot change faster than the filter allows).

        Parameters
        ----------
        season : str
            One of ['winter', 'spring', 'summer', 'fall'].
        age_group : str
            One of ['young', 'middle', 'senior'].  Seniors prefer warmer rooms.
        sensitivity : str
            One of ['low', 'normal', 'high'].  Affects noise amplitude.
        base_temp : float or None
            Override the baseline temperature in °C.  If None, sampled from
            the season-specific range calibrated against real data.
        random_seed : int
            NumPy random seed for reproducibility.

        Returns
        -------
        np.ndarray of shape (self.n,)
            Temperature in °C at each 5-minute interval.

        Notes
        -----
        Authors: Rushav Dash, Lisa Li
        The pink noise is scaled to std ≈ 0.15°C before filtering.
        Post-filter the total std is typically 0.5–1.2°C depending on season
        and HVAC amplitude.
        """
        rng = np.random.default_rng(random_seed)
        sens_mult = SENSITIVITY_MULTIPLIERS.get(sensitivity, 1.0)

        # ---- Component 1: Baseline temperature ----
        if base_temp is None:
            t_min, t_max = SEASON_TEMP_RANGES.get(season, (19.0, 22.0))
            # Add age-based bias (seniors prefer warmer)
            age_bias = AGE_TEMP_BIAS.get(age_group, 0.0)
            # Sample from a narrow normal centred in the seasonal range
            t_centre = (t_min + t_max) / 2.0 + age_bias
            t_sigma = (t_max - t_min) / 4.0   # σ ≈ half-range / 2
            base_temp = float(rng.normal(t_centre, t_sigma))
            # Clamp to the season's range ± 1°C buffer
            base_temp = float(np.clip(base_temp, t_min - 1.0, t_max + 1.0))

        T_base = base_temp

        # ============================================================
        # SECTION 3a-i: CIRCADIAN DRIFT COMPONENT
        # Models the natural room cool-down during sleep (body heat loss
        # and overnight ambient drop).  Period = full night = 480 min.
        # ============================================================
        A_circ = rng.uniform(0.4, 1.0)   # amplitude in °C
        phi_circ = rng.uniform(0, 2 * np.pi)  # random phase offset
        T_circadian = A_circ * np.sin(2 * np.pi * self.t / self.duration + phi_circ)

        # ============================================================
        # SECTION 3a-ii: HVAC SAWTOOTH COMPONENT
        # Models the periodic heating/cooling cycles of residential HVAC.
        # Period and amplitude are calibrated from the smart-home dataset
        # (or defaults if that dataset was unavailable).
        # ============================================================
        hvac_period = rng.uniform(30.0, 70.0)   # minutes per cycle
        hvac_amp = rng.uniform(0.3, 1.2) * sens_mult   # °C
        T_hvac = _sawtooth_hvac(self.t, hvac_period, hvac_amp)
        # Centre the HVAC wave around zero (remove its mean)
        T_hvac -= T_hvac.mean()

        # ============================================================
        # SECTION 3a-iii: PINK NOISE COMPONENT
        # Adds natural stochastic fluctuations (air currents, micro-events).
        # Scaled to std ≈ 0.10–0.20°C before filtering.
        # ============================================================
        noise_std = rng.uniform(0.08, 0.20) * sens_mult
        T_noise = _generate_pink_noise(self.n, rng) * noise_std

        # ---- Combine all components ----
        temperature = T_base + T_circadian + T_hvac + T_noise

        # ============================================================
        # SECTION 3a-iv: BUTTERWORTH LOW-PASS FILTER
        # Thermal inertia means temperatures cannot change abruptly.
        # Cutoff at 1/30 cycles/min (one cycle per 30 minutes) smooths
        # out sub-30-minute transients while preserving HVAC cycles.
        # fs = 1 sample / 5 min → Nyquist = 0.1 cycles/min
        # ============================================================
        fs = 1.0 / self.dt          # sampling frequency (cycles per minute)
        cutoff = 1.0 / 30.0         # low-pass cutoff (cycles per minute)

        # Ensure cutoff < Nyquist; warn if dataset parameters force issues
        if cutoff >= 0.5 * fs:
            cutoff = 0.4 * fs
            warnings.warn(
                "Temperature LPF cutoff adjusted to stay below Nyquist."
            )

        temperature_filtered = _apply_butterworth_lpf(
            temperature, cutoff_freq=cutoff, fs=fs, order=4
        )

        return temperature_filtered

    # ----------------------------------------------------------
    # 3b. Light signal
    # ----------------------------------------------------------

    def generate_light(
        self,
        season: str = "fall",
        age_group: str = "middle",
        sensitivity: str = "normal",
        random_seed: int = 42,
    ) -> np.ndarray:
        """
        Generate a synthetic bedroom light-level time-series in lux.

        Model:
          light(t) = L_background(t) + L_events(t)

        Background is near-zero (streetlight/moonlight seeping through curtains).
        Discrete events (phone checks, bathroom trips, partner waking) follow a
        Poisson process with Exponentially distributed durations.

        Parameters
        ----------
        season : str
            Season affects background light (longer summer nights have more
            ambient light from earlier sunrises).
        age_group : str
            Seniors tend to wake more → higher Poisson λ.
        sensitivity : str
            High-sensitivity individuals notice and are disrupted by more events.
        random_seed : int
            For reproducibility.

        Returns
        -------
        np.ndarray of shape (self.n,)
            Light level in lux at each 5-minute interval.

        Authors: Rushav Dash, Lisa Li
        """
        rng = np.random.default_rng(random_seed)
        sens_mult = SENSITIVITY_MULTIPLIERS.get(sensitivity, 1.0)

        # ============================================================
        # SECTION 3b-i: BACKGROUND (AMBIENT) LIGHT
        # Very slow sinusoidal model of moonlight/streetlight infiltration.
        # Amplitude scaled by season (summer: more ambient, winter: darker).
        # ============================================================
        season_bg_amp = {"winter": 1.5, "spring": 2.5, "summer": 4.0, "fall": 2.0}
        bg_amp = season_bg_amp.get(season, 2.5)
        phi_bg = rng.uniform(0, 2 * np.pi)
        L_background = (
            self.ref.light_night_mean
            + bg_amp * np.sin(2 * np.pi * self.t / self.duration + phi_bg)
        )
        # Clip background to non-negative lux values
        L_background = np.clip(L_background, 0.0, None)

        # ============================================================
        # SECTION 3b-ii: DISCRETE LIGHT EVENTS (POISSON PROCESS)
        # Each event represents a real-world disruption:
        #   - checking phone  →  low amplitude (~20–50 lux), short duration
        #   - bathroom visit  →  high amplitude (~80–150 lux), medium duration
        #   - partner wakes   →  medium amplitude, random duration
        #
        # λ (events per night) is scaled by sensitivity and age group.
        # Seniors generally have more fragmented sleep (higher λ).
        # ============================================================
        age_lambda_scale = {"young": 0.7, "middle": 1.0, "senior": 1.6}
        lambda_base = self.ref.light_event_lambda * sens_mult
        lambda_adj = lambda_base * age_lambda_scale.get(age_group, 1.0)

        # Draw number of events this night from a Poisson distribution
        n_events = rng.poisson(lambda_adj)

        # Initialise the light-events array (in lux)
        L_events = np.zeros(self.n)

        for _ in range(n_events):
            # Randomly choose event start time (uniformly across the night)
            start_min = rng.uniform(0, self.duration)
            start_idx = int(start_min / self.dt)

            # Duration drawn from Exponential distribution (mean = 8 min)
            duration_min = rng.exponential(self.ref.light_event_duration_mean)
            duration_samples = max(1, int(duration_min / self.dt))

            # Amplitude (lux): two modes — dim (phone) or bright (lamp)
            if rng.random() < 0.6:
                amplitude = rng.uniform(10, 60)    # phone / dim
            else:
                amplitude = rng.uniform(60, 150)   # lamp / bathroom

            # Place rectangular pulse, clipping to array bounds
            end_idx = min(self.n, start_idx + duration_samples)
            if start_idx < self.n:
                L_events[start_idx:end_idx] += amplitude

        # Smooth event edges with a 2-minute Gaussian to remove abrupt steps
        sigma_samples = 2.0 / self.dt   # 2 min → samples
        if sigma_samples >= 0.5:
            L_events = _gaussian_smooth_pulse(L_events, sigma_samples)

        # Combine background + events; enforce non-negative
        light = np.clip(L_background + L_events, 0.0, None)

        return light

    # ----------------------------------------------------------
    # 3c. Sound signal
    # ----------------------------------------------------------

    def generate_sound(
        self,
        age_group: str = "middle",
        sensitivity: str = "normal",
        random_seed: int = 42,
    ) -> np.ndarray:
        """
        Generate a synthetic bedroom sound-level time-series in dB SPL.

        Model:
          sound(t) = S_background(t) + S_events(t)

        Background is pink noise at 30–45 dB (typical quiet bedroom).
        Events model snoring, traffic, etc. via a Poisson process.

        Parameters
        ----------
        age_group : str
            Older adults tend to snore more → potential higher event amplitude.
        sensitivity : str
            High sensitivity → perception of more disruptive events.
        random_seed : int
            For reproducibility.

        Returns
        -------
        np.ndarray of shape (self.n,)
            Sound level in dB at each 5-minute interval.

        Authors: Rushav Dash, Lisa Li
        """
        rng = np.random.default_rng(random_seed)
        sens_mult = SENSITIVITY_MULTIPLIERS.get(sensitivity, 1.0)

        # ============================================================
        # SECTION 3c-i: BACKGROUND SOUND (PINK NOISE)
        # Quiet bedroom ambient: 30–45 dB SPL.
        # Scaled pink noise gives realistic temporal variation.
        # ============================================================
        bg_mean = rng.uniform(30.0, 42.0)
        bg_std = rng.uniform(1.0, 3.5)
        S_background = bg_mean + _generate_pink_noise(self.n, rng) * bg_std

        # ============================================================
        # SECTION 3c-ii: DISCRETE SOUND EVENTS
        # Snoring, traffic bursts, a phone ringing, etc.
        # Events drawn from Poisson(λ), each with:
        #   - Amplitude: N(65, 10) dB with exponential temporal decay
        #   - Duration:  Exp(mean=3 min)
        # ============================================================
        age_lambda_sound = {"young": 2.0, "middle": 4.0, "senior": 5.0}
        lambda_base = age_lambda_sound.get(age_group, 4.0) * sens_mult
        n_events = rng.poisson(lambda_base)

        S_events = np.zeros(self.n)

        for _ in range(n_events):
            start_idx = int(rng.uniform(0, self.n - 1))
            amp = rng.normal(65, 10)                          # dB
            decay_samples = max(1, int(rng.exponential(3.0) / self.dt))  # ~3 min mean

            # Exponential decay envelope for each event
            idx_end = min(self.n, start_idx + decay_samples * 4)
            event_len = idx_end - start_idx
            decay = amp * np.exp(-np.arange(event_len) / max(1, decay_samples))
            S_events[start_idx:idx_end] = np.maximum(
                S_events[start_idx:idx_end], decay
            )

        # dB levels: take the louder of background or event at each sample
        sound = np.maximum(S_background, S_background + S_events)

        # Physical bounds: 0–120 dB
        sound = np.clip(sound, 0.0, 120.0)

        return sound

    # ----------------------------------------------------------
    # 3d. Humidity signal
    # ----------------------------------------------------------

    def generate_humidity(
        self,
        season: str = "fall",
        random_seed: int = 42,
    ) -> np.ndarray:
        """
        Generate a synthetic relative humidity time-series in %.

        Simple sinusoidal model with mild Gaussian noise — humidity changes
        slowly overnight and is primarily determined by outdoor climate and
        HVAC settings.

        Parameters
        ----------
        season : str
            Sets the baseline humidity range.
        random_seed : int
            For reproducibility.

        Returns
        -------
        np.ndarray of shape (self.n,)
            Relative humidity in % at each 5-minute interval.

        Authors: Rushav Dash, Lisa Li
        """
        rng = np.random.default_rng(random_seed)

        # ---- Baseline humidity drawn from seasonal range ----
        h_min, h_max = SEASON_HUMIDITY_RANGES.get(season, (35.0, 60.0))
        H_base = rng.uniform(h_min, h_max)

        # ---- Sinusoidal variation over the night ----
        A_h = rng.uniform(1.0, 3.5)                # amplitude in %
        phi_h = rng.uniform(0, 2 * np.pi)
        H_sin = A_h * np.sin(2 * np.pi * self.t / self.duration + phi_h)

        # ---- Gaussian white noise (slowly varying: LPF applied) ----
        noise_raw = rng.normal(0, 1.0, size=self.n)
        # Apply a simple moving average (3-sample) to slow the noise down
        noise_smooth = np.convolve(noise_raw, np.ones(3) / 3, mode="same")

        humidity = H_base + H_sin + noise_smooth

        # Clamp to physically valid range [10, 95] %
        humidity = np.clip(humidity, 10.0, 95.0)

        return humidity

    # ----------------------------------------------------------
    # 3e. Orchestration: generate all signals for one session
    # ----------------------------------------------------------

    def generate_all(
        self,
        season: str = "fall",
        age_group: str = "middle",
        sensitivity: str = "normal",
        random_seed: int = 42,
        include_sound: bool = True,
        include_humidity: bool = True,
    ) -> dict:
        """
        Generate all environmental signals for a single sleep session.

        Uses derived seeds for each signal type so the session seed
        fully determines all signals while keeping signals independent.

        Parameters
        ----------
        season : str
        age_group : str
        sensitivity : str
        random_seed : int
        include_sound : bool
            Whether to generate the optional sound signal.
        include_humidity : bool
            Whether to generate the optional humidity signal.

        Returns
        -------
        dict with keys:
            't'           — time axis (np.ndarray, minutes)
            'temperature' — temperature signal (np.ndarray, °C)
            'light'       — light signal (np.ndarray, lux)
            'sound'       — sound signal (np.ndarray, dB) or None
            'humidity'    — humidity signal (np.ndarray, %) or None

        Authors: Rushav Dash, Lisa Li
        """
        # Derive independent seeds per signal type so changing one doesn't
        # alter the others — important for ablation studies.
        seed_temp  = random_seed
        seed_light = random_seed + 1_000_000
        seed_sound = random_seed + 2_000_000
        seed_humid = random_seed + 3_000_000

        temperature = self.generate_temperature(
            season=season,
            age_group=age_group,
            sensitivity=sensitivity,
            random_seed=seed_temp,
        )

        light = self.generate_light(
            season=season,
            age_group=age_group,
            sensitivity=sensitivity,
            random_seed=seed_light,
        )

        sound = None
        if include_sound:
            sound = self.generate_sound(
                age_group=age_group,
                sensitivity=sensitivity,
                random_seed=seed_sound,
            )

        humidity = None
        if include_humidity:
            humidity = self.generate_humidity(
                season=season,
                random_seed=seed_humid,
            )

        return {
            "t": self.t.copy(),
            "temperature": temperature,
            "light": light,
            "sound": sound,
            "humidity": humidity,
        }
