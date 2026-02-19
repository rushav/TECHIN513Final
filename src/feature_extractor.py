"""
=============================================================================
feature_extractor.py — Feature extraction from synthetic environmental signals
=============================================================================
Project:    Synthetic Sleep Environment Dataset Generator
Course:     TECHIN 513: Signal Processing & Machine Learning
Team:       Team 7 — Rushav Dash & Lisa Li
University: University of Washington
Date:       2026-02-19
=============================================================================
Converts raw time-series arrays produced by SignalGenerator into a flat
dictionary of scalar features that can be fed into the ML pipeline.

Each function accepts a 1-D numpy array (one signal's time-series) and
returns a dict of {feature_name: scalar_value} pairs.  The top-level
`extract_all_features()` function combines all signal features into one
flat dict and is the primary interface used by dataset_generator.py.

Feature philosophy:
  - Features capture sleep-relevant aspects of each signal (e.g., time
    spent outside the thermal comfort zone, number of light events).
  - Features are derived from well-established sleep science literature
    (e.g., Okamoto-Mizuno & Mizuno 2012 for temperature; Zeitzer et al.
    2000 for light effects on circadian disruption).

Authors: Rushav Dash, Lisa Li
"""

import numpy as np


# ============================================================
# SECTION 1: TEMPERATURE FEATURE EXTRACTION
# ============================================================

def extract_temperature_features(
    temperature: np.ndarray,
    dt_minutes: int = 5,
) -> dict:
    """
    Derive sleep-relevant scalar features from a temperature time-series.

    Clinical context: The optimal sleep temperature range is ~18–21°C
    (65–70°F) according to sleep medicine guidelines.  Deviations from
    this zone are associated with reduced sleep efficiency and lower
    slow-wave sleep percentage.

    Parameters
    ----------
    temperature : np.ndarray
        Temperature values in °C (one value per time step).
    dt_minutes : int
        Temporal resolution of the signal in minutes (default 5 min).

    Returns
    -------
    dict
        Keys and units described below.

    Authors: Rushav Dash, Lisa Li
    """
    n = len(temperature)

    # ---- Basic descriptive stats ----
    temp_mean = float(np.mean(temperature))
    temp_std  = float(np.std(temperature))
    temp_min  = float(np.min(temperature))
    temp_max  = float(np.max(temperature))
    temp_range = temp_max - temp_min

    # ---- Time outside optimal zone ----
    # Each sample corresponds to dt_minutes of real time
    above_21 = int(np.sum(temperature > 21.0)) * dt_minutes  # minutes
    below_18 = int(np.sum(temperature < 18.0)) * dt_minutes  # minutes

    # Fraction of total night in the comfort zone [18, 21] °C
    in_optimal = np.sum((temperature >= 18.0) & (temperature <= 21.0))
    temp_optimal_fraction = float(in_optimal / n) if n > 0 else 0.0

    # ---- Rate of change (absolute gradient) ----
    # Captures thermal instability (e.g., aggressive HVAC cycling)
    if n > 1:
        diffs = np.abs(np.diff(temperature))
        temp_mean_rate_change = float(np.mean(diffs))   # °C per 5 min
        temp_max_rate_change  = float(np.max(diffs))
    else:
        temp_mean_rate_change = 0.0
        temp_max_rate_change  = 0.0

    return {
        "temp_mean":               temp_mean,
        "temp_std":                temp_std,
        "temp_min":                temp_min,
        "temp_max":                temp_max,
        "temp_range":              temp_range,
        "temp_above_21_minutes":   above_21,
        "temp_below_18_minutes":   below_18,
        "temp_optimal_fraction":   temp_optimal_fraction,
        "temp_mean_rate_change":   temp_mean_rate_change,
        "temp_max_rate_change":    temp_max_rate_change,
    }


# ============================================================
# SECTION 2: LIGHT FEATURE EXTRACTION
# ============================================================

def extract_light_features(
    light: np.ndarray,
    dt_minutes: int = 5,
    event_threshold_lux: float = 10.0,
) -> dict:
    """
    Derive sleep-relevant scalar features from a light-level time-series.

    Clinical context: Even brief light exposure at night can suppress
    melatonin secretion and delay sleep onset (Zeitzer et al. 2000).
    The features here quantify both intensity and duration of exposure.

    Parameters
    ----------
    light : np.ndarray
        Light values in lux.
    dt_minutes : int
        Temporal resolution in minutes.
    event_threshold_lux : float
        Lux threshold above which an event is considered "on" (default 10).

    Returns
    -------
    dict

    Authors: Rushav Dash, Lisa Li
    """
    # ---- Basic descriptive stats ----
    light_mean = float(np.mean(light))
    light_std  = float(np.std(light))
    light_max  = float(np.max(light))

    # ---- Total exposure ----
    # Number of time steps with lux > threshold, converted to minutes
    above_threshold = light > event_threshold_lux
    light_total_exposure_minutes = int(np.sum(above_threshold)) * dt_minutes

    # ---- Event counting ----
    # Count transitions from "off" to "on" (rising edges = event starts)
    edges = np.diff(above_threshold.astype(int), prepend=0)
    light_event_count = int(np.sum(edges == 1))

    # ---- Disruption score ----
    # Weighted sum: amplitude × duration for each event block.
    # This single number captures the combined melatonin-suppression impact.
    disruption_score = 0.0
    in_event = False
    event_start = 0
    for i, val in enumerate(above_threshold):
        if val and not in_event:
            in_event = True
            event_start = i
        elif not val and in_event:
            in_event = False
            # Event ended at index i-1
            event_duration = (i - event_start) * dt_minutes       # minutes
            event_amplitude = float(np.mean(light[event_start:i]))  # avg lux
            disruption_score += event_amplitude * event_duration
    # Handle event running to the end of the array
    if in_event:
        event_duration = (len(light) - event_start) * dt_minutes
        event_amplitude = float(np.mean(light[event_start:]))
        disruption_score += event_amplitude * event_duration

    # ---- Peak lux ----
    # Maximum instantaneous light level (could be single sample spike)
    light_peak_lux = light_max

    return {
        "light_mean":                    light_mean,
        "light_std":                     light_std,
        "light_max":                     light_max,
        "light_peak_lux":                light_peak_lux,
        "light_event_count":             light_event_count,
        "light_total_exposure_minutes":  light_total_exposure_minutes,
        "light_disruption_score":        disruption_score,
    }


# ============================================================
# SECTION 3: SOUND FEATURE EXTRACTION
# ============================================================

def extract_sound_features(
    sound: np.ndarray,
    dt_minutes: int = 5,
    disturbance_threshold_db: float = 55.0,
) -> dict:
    """
    Derive sleep-relevant scalar features from a sound-level time-series.

    Clinical context: The WHO recommends nighttime sound levels below
    40 dB Lnight outdoors; indoor exceedances above ~55 dB are associated
    with arousal events and sleep fragmentation (WHO 2009).

    Parameters
    ----------
    sound : np.ndarray
        Sound level values in dB SPL.
    dt_minutes : int
        Temporal resolution in minutes.
    disturbance_threshold_db : float
        dB level above which sound is considered a potential arousal trigger.

    Returns
    -------
    dict

    Authors: Rushav Dash, Lisa Li
    """
    # ---- Basic descriptive stats ----
    sound_mean_db = float(np.mean(sound))
    sound_std_db  = float(np.std(sound))
    sound_max_db  = float(np.max(sound))

    # ---- Time above disturbance threshold ----
    above_thresh = sound > disturbance_threshold_db
    sound_above_55db_minutes = int(np.sum(above_thresh)) * dt_minutes

    # ---- Event counting ----
    # Rising edges above threshold = distinct disturbance events
    edges = np.diff(above_thresh.astype(int), prepend=0)
    sound_event_count = int(np.sum(edges == 1))

    # ---- Energy-equivalent level (Leq) ----
    # Convert dB to linear energy, average, convert back
    # This is the standard acoustics metric for sustained exposure.
    linear_energy = 10 ** (sound / 10.0)
    leq = float(10.0 * np.log10(np.mean(linear_energy) + 1e-12))

    return {
        "sound_mean_db":             sound_mean_db,
        "sound_std_db":              sound_std_db,
        "sound_max_db":              sound_max_db,
        "sound_above_55db_minutes":  sound_above_55db_minutes,
        "sound_event_count":         sound_event_count,
        "sound_leq_db":              leq,
    }


# ============================================================
# SECTION 4: HUMIDITY FEATURE EXTRACTION
# ============================================================

def extract_humidity_features(
    humidity: np.ndarray,
    dt_minutes: int = 5,
    comfort_low: float = 30.0,
    comfort_high: float = 60.0,
) -> dict:
    """
    Derive sleep-relevant scalar features from a humidity time-series.

    Clinical context: Relative humidity outside 30–60% is associated with
    increased nasal resistance, dry throat, and sleep-disordered breathing
    (Nuckton et al. 2006).

    Parameters
    ----------
    humidity : np.ndarray
        Relative humidity values in %.
    dt_minutes : int
        Temporal resolution in minutes.
    comfort_low : float
        Lower bound of comfortable humidity range (%).
    comfort_high : float
        Upper bound of comfortable humidity range (%).

    Returns
    -------
    dict

    Authors: Rushav Dash, Lisa Li
    """
    # ---- Basic descriptive stats ----
    humidity_mean = float(np.mean(humidity))
    humidity_std  = float(np.std(humidity))
    humidity_min  = float(np.min(humidity))
    humidity_max  = float(np.max(humidity))

    # ---- Time outside comfort range ----
    out_of_range = (humidity < comfort_low) | (humidity > comfort_high)
    humidity_out_of_range_minutes = int(np.sum(out_of_range)) * dt_minutes

    # ---- Comfort fraction ----
    in_comfort = (~out_of_range).sum()
    humidity_comfort_fraction = float(in_comfort / len(humidity)) if len(humidity) > 0 else 0.0

    return {
        "humidity_mean":                  humidity_mean,
        "humidity_std":                   humidity_std,
        "humidity_min":                   humidity_min,
        "humidity_max":                   humidity_max,
        "humidity_out_of_range_minutes":  humidity_out_of_range_minutes,
        "humidity_comfort_fraction":      humidity_comfort_fraction,
    }


# ============================================================
# SECTION 5: TOP-LEVEL FEATURE EXTRACTION ORCHESTRATOR
# ============================================================

def extract_all_features(
    signals: dict,
    dt_minutes: int = 5,
) -> dict:
    """
    Extract features from all signals produced by SignalGenerator.generate_all().

    This is the primary interface called by dataset_generator.py.  It
    combines temperature, light, sound, and humidity features into one
    flat dictionary ready for pandas DataFrame construction.

    Parameters
    ----------
    signals : dict
        Dictionary returned by SignalGenerator.generate_all().
        Expected keys: 't', 'temperature', 'light', 'sound' (optional),
        'humidity' (optional).
    dt_minutes : int
        Signal temporal resolution in minutes.

    Returns
    -------
    dict
        Flat mapping of feature_name → scalar_value for all available signals.

    Authors: Rushav Dash, Lisa Li
    """
    features = {}

    # ---- Temperature features (always present) ----
    if "temperature" in signals and signals["temperature"] is not None:
        features.update(
            extract_temperature_features(signals["temperature"], dt_minutes=dt_minutes)
        )

    # ---- Light features (always present) ----
    if "light" in signals and signals["light"] is not None:
        features.update(
            extract_light_features(signals["light"], dt_minutes=dt_minutes)
        )

    # ---- Sound features (optional) ----
    if "sound" in signals and signals["sound"] is not None:
        features.update(
            extract_sound_features(signals["sound"], dt_minutes=dt_minutes)
        )

    # ---- Humidity features (optional) ----
    if "humidity" in signals and signals["humidity"] is not None:
        features.update(
            extract_humidity_features(signals["humidity"], dt_minutes=dt_minutes)
        )

    return features
