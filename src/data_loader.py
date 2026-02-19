"""
=============================================================================
data_loader.py — Dataset acquisition and reference statistics extraction
=============================================================================
Project:    Synthetic Sleep Environment Dataset Generator
Course:     TECHIN 513: Signal Processing & Machine Learning
Team:       Team 7 — Rushav Dash & Lisa Li
University: University of Washington
Date:       2026-02-19
=============================================================================
This module handles two responsibilities:

1. **Dataset acquisition** – Downloads the three source Kaggle datasets using
   the `kagglehub` library (which caches downloads in ~/.cache/kagglehub/).
   Falls back gracefully when optional datasets are unavailable.

2. **Reference statistics extraction** – Parses the raw CSVs to derive the
   calibration parameters (means, stds, autocorrelations, FFT peaks, etc.)
   that parameterise the signal generators in signal_generator.py.

Authors: Rushav Dash, Lisa Li
"""

import os
import glob
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import signal, stats


# ============================================================
# SECTION 1: REFERENCE STATISTICS DATACLASS
# A single container that carries all calibration values
# extracted from the real-world datasets.
# ============================================================

@dataclass
class ReferenceStats:
    """
    Calibration statistics extracted from real IoT / sleep datasets.

    These values are consumed by SignalGenerator to produce synthetic
    time-series that are statistically plausible when compared to
    real sensor recordings.

    Attributes
    ----------
    Authors: Rushav Dash, Lisa Li
    """

    # --- Temperature stats (from Room Occupancy dataset) ---
    temp_mean: float = 21.0
    temp_std: float = 1.5
    temp_min: float = 17.0
    temp_max: float = 26.0
    temp_autocorr_lag1: float = 0.97   # AR(1) coefficient
    temp_dominant_freq_cpm: float = 1 / 480.0  # cycles per minute (circadian)

    # --- Light stats (from Room Occupancy dataset) ---
    light_night_mean: float = 3.0      # lux during darkness
    light_night_std: float = 5.0
    light_event_lambda: float = 2.0    # mean events per night (Poisson)
    light_event_duration_mean: float = 8.0   # minutes (Exponential)

    # --- Sleep quality stats (from Sleep Efficiency dataset) ---
    sleep_efficiency_mean: float = 0.79
    sleep_efficiency_std: float = 0.10
    awakenings_mean: float = 1.8
    rem_pct_mean: float = 22.0
    deep_pct_mean: float = 18.0
    light_pct_mean: float = 60.0

    # --- Feature correlation summary (for model training guidance) ---
    top_correlated_features: list = field(default_factory=list)

    # --- Optional: HVAC cycle period extracted from smart-home data ---
    hvac_period_min: float = 45.0      # minutes per heating/cooling cycle


# ============================================================
# SECTION 2: MAIN DATA LOADER CLASS
# ============================================================

class DataLoader:
    """
    Download and parse three Kaggle datasets; extract calibration statistics.

    Usage
    -----
    >>> loader = DataLoader()
    >>> loader.download_all()
    >>> stats = loader.extract_reference_stats()

    Authors: Rushav Dash, Lisa Li
    """

    # Kaggle dataset identifiers
    SLEEP_EFFICIENCY_DATASET = "equilibriumm/sleep-efficiency"
    ROOM_OCCUPANCY_DATASET = "kukuroo3/room-occupancy-detection-data-iot-sensor"
    SMART_HOME_DATASET = "taranvee/smart-home-dataset-with-weather-information"

    def __init__(self, verbose: bool = True):
        """
        Initialise the DataLoader.

        Parameters
        ----------
        verbose : bool
            If True, print progress messages to stdout.

        Authors: Rushav Dash, Lisa Li
        """
        self.verbose = verbose
        # Paths set after download
        self._sleep_csv: Optional[str] = None
        self._occupancy_csv: Optional[str] = None
        self._smart_home_csv: Optional[str] = None

        # Loaded DataFrames (None until loaded)
        self._df_sleep: Optional[pd.DataFrame] = None
        self._df_occupancy: Optional[pd.DataFrame] = None
        self._df_smart: Optional[pd.DataFrame] = None

    # ----------------------------------------------------------
    # 2a. Dataset download helpers
    # ----------------------------------------------------------

    def _log(self, msg: str) -> None:
        """Print message if verbose mode is on."""
        if self.verbose:
            print(f"[DataLoader] {msg}")

    def _find_csv(self, directory: str, candidates: list[str]) -> Optional[str]:
        """
        Search a directory for any of the candidate filenames (case-insensitive).

        Parameters
        ----------
        directory : str
            Root directory returned by kagglehub.
        candidates : list[str]
            Filenames to look for, in order of preference.

        Returns
        -------
        str or None
            Absolute path to the first matching file, or None if not found.

        Authors: Rushav Dash, Lisa Li
        """
        # Walk the full directory tree (kagglehub may nest files in subdirs)
        all_csvs = glob.glob(os.path.join(directory, "**", "*.csv"), recursive=True)
        name_map = {os.path.basename(p).lower(): p for p in all_csvs}

        for candidate in candidates:
            if candidate.lower() in name_map:
                return name_map[candidate.lower()]

        # If no exact match, return the first CSV we find (best-effort)
        if all_csvs:
            warnings.warn(
                f"None of {candidates} found in {directory}. "
                f"Using first CSV: {all_csvs[0]}"
            )
            return all_csvs[0]

        return None

    def download_all(self) -> None:
        """
        Download all three Kaggle datasets via kagglehub.

        kagglehub caches downloads at ~/.cache/kagglehub/ so subsequent
        calls are near-instant.  The smart-home dataset is optional; a
        warning is issued if it cannot be retrieved.

        Raises
        ------
        RuntimeError
            If the primary Sleep Efficiency or Room Occupancy datasets
            cannot be downloaded or located.

        Authors: Rushav Dash, Lisa Li
        """
        try:
            import kagglehub  # imported here so the rest of the module works without it
        except ImportError as exc:
            raise ImportError(
                "kagglehub is required to download datasets. "
                "Install it with: pip install kagglehub"
            ) from exc

        # --- Dataset 1: Sleep Efficiency (PRIMARY) ---
        self._log("Downloading Sleep Efficiency dataset…")
        try:
            sleep_dir = kagglehub.dataset_download(self.SLEEP_EFFICIENCY_DATASET)
            self._sleep_csv = self._find_csv(
                sleep_dir, ["Sleep_Efficiency.csv", "sleep_efficiency.csv"]
            )
            if self._sleep_csv is None:
                raise FileNotFoundError(f"No CSV found in {sleep_dir}")
            self._log(f"  Sleep Efficiency CSV: {self._sleep_csv}")
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download/locate Sleep Efficiency dataset "
                f"({self.SLEEP_EFFICIENCY_DATASET}). "
                f"Error: {exc}"
            ) from exc

        # --- Dataset 2: Room Occupancy (PRIMARY for IoT calibration) ---
        self._log("Downloading Room Occupancy dataset…")
        try:
            occ_dir = kagglehub.dataset_download(self.ROOM_OCCUPANCY_DATASET)
            self._occupancy_csv = self._find_csv(
                occ_dir,
                [
                    "room_occupancy.csv",
                    "Occupancy_Estimation.csv",
                    "occupancy.csv",
                    "datatraining.txt",
                ],
            )
            if self._occupancy_csv is None:
                raise FileNotFoundError(f"No CSV found in {occ_dir}")
            self._log(f"  Room Occupancy CSV: {self._occupancy_csv}")
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download/locate Room Occupancy dataset "
                f"({self.ROOM_OCCUPANCY_DATASET}). "
                f"Error: {exc}"
            ) from exc

        # --- Dataset 3: Smart Home (OPTIONAL) ---
        self._log("Downloading Smart Home dataset (optional)…")
        try:
            smart_dir = kagglehub.dataset_download(self.SMART_HOME_DATASET)
            self._smart_home_csv = self._find_csv(
                smart_dir,
                [
                    "smart_home_dataset.csv",
                    "HomeC.csv",
                    "smarthome.csv",
                ],
            )
            if self._smart_home_csv:
                self._log(f"  Smart Home CSV: {self._smart_home_csv}")
            else:
                warnings.warn(
                    "Smart Home CSV not found; HVAC parameters will use defaults."
                )
        except Exception as exc:
            warnings.warn(
                f"Could not download Smart Home dataset ({exc}). "
                "Continuing without it — HVAC defaults will be used."
            )

    # ----------------------------------------------------------
    # 2b. DataFrame loaders
    # ----------------------------------------------------------

    def load_sleep_efficiency(self) -> pd.DataFrame:
        """
        Load and lightly clean the Sleep Efficiency CSV.

        Cleaning steps:
        - Strip whitespace from column names
        - Parse Bedtime and Wakeup time as datetime
        - Derive 'Season' from Bedtime month if available

        Returns
        -------
        pd.DataFrame

        Authors: Rushav Dash, Lisa Li
        """
        if self._sleep_csv is None:
            raise RuntimeError("Call download_all() before load_sleep_efficiency().")

        df = pd.read_csv(self._sleep_csv)
        df.columns = df.columns.str.strip()

        # Parse datetime columns if present
        for col in ["Bedtime", "Wakeup time"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        # Derive season from bedtime month
        if "Bedtime" in df.columns and df["Bedtime"].notna().any():
            month = df["Bedtime"].dt.month
            df["Season"] = month.map(self._month_to_season)
        else:
            # Assign seasons uniformly if no date info
            seasons = ["winter", "spring", "summer", "fall"]
            df["Season"] = [seasons[i % 4] for i in range(len(df))]

        self._df_sleep = df
        self._log(f"Loaded Sleep Efficiency: {df.shape[0]} rows × {df.shape[1]} cols")
        return df

    def load_room_occupancy(self) -> pd.DataFrame:
        """
        Load the Room Occupancy IoT sensor CSV.

        Attempts to parse a timestamp column and filter to nighttime hours
        (22:00–06:00) to derive sleep-relevant calibration statistics.

        Returns
        -------
        pd.DataFrame

        Authors: Rushav Dash, Lisa Li
        """
        if self._occupancy_csv is None:
            raise RuntimeError("Call download_all() before load_room_occupancy().")

        df = pd.read_csv(self._occupancy_csv)
        df.columns = df.columns.str.strip()

        # Normalise common column name variants
        col_renames = {
            "Temp": "Temperature",
            "temp": "Temperature",
            "temperature": "Temperature",
            "light": "Light",
            "lux": "Light",
            "humidity": "Humidity",
            "co2": "CO2",
        }
        df.rename(columns={k: v for k, v in col_renames.items() if k in df.columns}, inplace=True)

        # Parse timestamp if present
        ts_candidates = ["date", "Date", "timestamp", "Timestamp", "time", "Time", "datetime"]
        for col in ts_candidates:
            if col in df.columns:
                df["datetime"] = pd.to_datetime(df[col], errors="coerce")
                break

        self._df_occupancy = df
        self._log(f"Loaded Room Occupancy: {df.shape[0]} rows × {df.shape[1]} cols")
        return df

    def load_smart_home(self) -> Optional[pd.DataFrame]:
        """
        Load the Smart Home dataset (optional).

        Returns
        -------
        pd.DataFrame or None
            None if the dataset was not downloaded or fails to load.

        Authors: Rushav Dash, Lisa Li
        """
        if self._smart_home_csv is None:
            warnings.warn("Smart Home CSV not available — skipping.")
            return None

        try:
            df = pd.read_csv(self._smart_home_csv, low_memory=False)
            df.columns = df.columns.str.strip()
            self._df_smart = df
            self._log(f"Loaded Smart Home: {df.shape[0]} rows × {df.shape[1]} cols")
            return df
        except Exception as exc:
            warnings.warn(f"Failed to load Smart Home CSV: {exc}")
            return None

    # ----------------------------------------------------------
    # 2c. Reference statistics extraction
    # ----------------------------------------------------------

    def extract_reference_stats(self) -> ReferenceStats:
        """
        Extract calibration statistics from loaded DataFrames.

        Must call load_sleep_efficiency() and load_room_occupancy() first
        (or download_all() which calls them internally).

        Returns
        -------
        ReferenceStats
            Populated dataclass used by SignalGenerator.

        Authors: Rushav Dash, Lisa Li
        """
        stats_obj = ReferenceStats()  # start with sensible defaults

        # ---- Extract from Room Occupancy (temperature & light) ----
        if self._df_occupancy is not None:
            df_occ = self._df_occupancy
            stats_obj = self._extract_temp_light_stats(df_occ, stats_obj)

        # ---- Extract from Smart Home (HVAC periods) ----
        if self._df_smart is not None:
            stats_obj = self._extract_hvac_stats(self._df_smart, stats_obj)

        # ---- Extract from Sleep Efficiency (sleep quality) ----
        if self._df_sleep is None:
            self.load_sleep_efficiency()
        if self._df_sleep is not None:
            stats_obj = self._extract_sleep_stats(self._df_sleep, stats_obj)

        return stats_obj

    def _extract_temp_light_stats(
        self, df: pd.DataFrame, stats_obj: ReferenceStats
    ) -> ReferenceStats:
        """
        Populate temperature and light fields in ReferenceStats from IoT data.

        Filters to nighttime rows if a datetime column is available, otherwise
        uses the full dataset (real room occupancy measurements are already
        mostly indoor so the distribution is relevant regardless).

        Parameters
        ----------
        df : pd.DataFrame
        stats_obj : ReferenceStats (mutated in-place and returned)

        Authors: Rushav Dash, Lisa Li
        """
        # Attempt to isolate nighttime readings (22:00 – 06:00)
        if "datetime" in df.columns and df["datetime"].notna().any():
            hour = df["datetime"].dt.hour
            night_mask = (hour >= 22) | (hour < 6)
            df_night = df[night_mask] if night_mask.sum() > 50 else df
        else:
            df_night = df

        # --- Temperature ---
        if "Temperature" in df_night.columns:
            temp = df_night["Temperature"].dropna()
            # Cap extreme outliers using IQR fence
            q1, q3 = temp.quantile(0.05), temp.quantile(0.95)
            temp = temp[(temp >= q1) & (temp <= q3)]

            stats_obj.temp_mean = float(temp.mean())
            stats_obj.temp_std = float(temp.std())
            stats_obj.temp_min = float(temp.min())
            stats_obj.temp_max = float(temp.max())

            # Autocorrelation at lag-1 (AR coefficient approximation)
            if len(temp) > 2:
                acf_val = float(pd.Series(temp.values).autocorr(lag=1))
                stats_obj.temp_autocorr_lag1 = max(0.8, min(0.999, acf_val))

            # FFT: find dominant frequency (cycles per minute)
            if len(temp) > 10:
                fft_vals = np.abs(np.fft.rfft(temp.values - temp.mean()))
                freqs = np.fft.rfftfreq(len(temp))  # fraction of sampling rate
                dominant_idx = np.argmax(fft_vals[1:]) + 1  # skip DC
                stats_obj.temp_dominant_freq_cpm = float(freqs[dominant_idx])

        # --- Light ---
        if "Light" in df_night.columns:
            light = df_night["Light"].dropna()
            # Keep only low-light subset (night: lux < 100)
            light_night = light[light < 100]
            if len(light_night) > 10:
                stats_obj.light_night_mean = float(light_night.mean())
                stats_obj.light_night_std = float(max(0.5, light_night.std()))
            # Estimate Poisson λ for light-on events
            light_events = light[light > 10]
            if len(light_events) > 0:
                # Rough estimate: if 8-hour night has ~480 samples at 1-min,
                # count distinct "on" bursts
                is_on = (light > 10).astype(int)
                transitions = is_on.diff().fillna(0)
                event_count = int((transitions == 1).sum())
                stats_obj.light_event_lambda = float(
                    np.clip(event_count / max(1, len(df_night) / 480), 0.5, 8.0)
                )

        return stats_obj

    def _extract_hvac_stats(
        self, df: pd.DataFrame, stats_obj: ReferenceStats
    ) -> ReferenceStats:
        """
        Estimate HVAC cycle period from smart-home thermostat data.

        Looks for a temperature column, computes the power spectrum, and
        picks the dominant period in the 20–120 minute range.

        Authors: Rushav Dash, Lisa Li
        """
        temp_cols = [c for c in df.columns if "temp" in c.lower()]
        if not temp_cols:
            return stats_obj

        temp = df[temp_cols[0]].dropna().values
        if len(temp) < 100:
            return stats_obj

        fft_vals = np.abs(np.fft.rfft(temp - temp.mean()))
        # Assume 1-minute sampling; convert freq index to period in minutes
        freqs = np.fft.rfftfreq(len(temp))  # cycles per sample
        with np.errstate(divide='ignore', invalid='ignore'):
            periods_min = np.where(freqs > 0, 1.0 / freqs, np.inf)

        # Focus on HVAC-plausible periods: 20–120 min
        mask = (periods_min >= 20) & (periods_min <= 120)
        if mask.any():
            best_idx = np.argmax(fft_vals[mask])
            stats_obj.hvac_period_min = float(periods_min[mask][best_idx])

        return stats_obj

    def _extract_sleep_stats(
        self, df: pd.DataFrame, stats_obj: ReferenceStats
    ) -> ReferenceStats:
        """
        Extract sleep quality summary statistics from Sleep Efficiency dataset.

        Also identifies which features are most correlated with Sleep efficiency
        to guide ML feature mapping decisions.

        Authors: Rushav Dash, Lisa Li
        """
        numeric_df = df.select_dtypes(include=[np.number])

        if "Sleep efficiency" in numeric_df.columns:
            se = df["Sleep efficiency"].dropna()
            stats_obj.sleep_efficiency_mean = float(se.mean())
            stats_obj.sleep_efficiency_std = float(se.std())

            # Top correlates (absolute Pearson r)
            corr = numeric_df.corrwith(df["Sleep efficiency"]).abs().sort_values(ascending=False)
            stats_obj.top_correlated_features = corr.index.tolist()[:10]

        if "Awakenings" in df.columns:
            stats_obj.awakenings_mean = float(df["Awakenings"].dropna().mean())

        if "REM sleep percentage" in df.columns:
            stats_obj.rem_pct_mean = float(df["REM sleep percentage"].dropna().mean())

        if "Deep sleep percentage" in df.columns:
            stats_obj.deep_pct_mean = float(df["Deep sleep percentage"].dropna().mean())

        if "Light sleep percentage" in df.columns:
            stats_obj.light_pct_mean = float(df["Light sleep percentage"].dropna().mean())

        return stats_obj

    # ----------------------------------------------------------
    # Utility helpers
    # ----------------------------------------------------------

    @staticmethod
    def _month_to_season(month: int) -> str:
        """Map a calendar month (1-12) to a season string."""
        if month in [12, 1, 2]:
            return "winter"
        elif month in [3, 4, 5]:
            return "spring"
        elif month in [6, 7, 8]:
            return "summer"
        else:
            return "fall"

    def get_sleep_df(self) -> pd.DataFrame:
        """Return the loaded Sleep Efficiency DataFrame (loads if needed)."""
        if self._df_sleep is None:
            self.load_sleep_efficiency()
        return self._df_sleep

    def get_occupancy_df(self) -> pd.DataFrame:
        """Return the loaded Room Occupancy DataFrame (loads if needed)."""
        if self._df_occupancy is None:
            self.load_room_occupancy()
        return self._df_occupancy
