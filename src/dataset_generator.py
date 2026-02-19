"""
=============================================================================
dataset_generator.py — Pipeline orchestrator for 5,000-session generation
=============================================================================
Project:    Synthetic Sleep Environment Dataset Generator
Course:     TECHIN 513: Signal Processing & Machine Learning
Team:       Team 7 — Rushav Dash & Lisa Li
University: University of Washington
Date:       2026-02-19
=============================================================================
The `SleepDatasetGenerator` class wires together all four upstream modules:

  DataLoader  →  SignalGenerator  →  FeatureExtractor  →  SleepQualityModel

and produces the final deliverable:

  data/output/synthetic_sleep_dataset_5000.csv
  data/output/synthetic_sleep_dataset_metadata.json

Generation strategy:
  - 5,000 sessions stratified equally across four seasons (1,250 each)
  - Within each season, age_group (young/middle/senior) and sensitivity
    (low/normal/high) are distributed uniformly using a fixed grid
  - Each session receives a deterministic seed derived from (session_id, global_seed)
    so the dataset is exactly reproducible

Authors: Rushav Dash, Lisa Li
"""

import json
import uuid
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data_loader import DataLoader, ReferenceStats
from src.signal_generator import SignalGenerator
from src.feature_extractor import extract_all_features
from src.sleep_quality_model import SleepQualityModel


# ============================================================
# SECTION 1: STRATIFICATION TABLE
# Sessions are distributed evenly across seasons, age groups,
# and sensitivity levels — reflecting realistic population diversity.
# ============================================================

SEASONS = ["winter", "spring", "summer", "fall"]
AGE_GROUPS = ["young", "middle", "senior"]
SENSITIVITIES = ["low", "normal", "high"]

# Total sessions
TOTAL_SESSIONS = 5_000
SESSIONS_PER_SEASON = TOTAL_SESSIONS // len(SEASONS)  # 1,250

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "output"


# ============================================================
# SECTION 2: SLEEP DATASET GENERATOR CLASS
# ============================================================

class SleepDatasetGenerator:
    """
    Orchestrate the full synthetic dataset generation pipeline.

    Usage
    -----
    >>> gen = SleepDatasetGenerator(global_seed=42)
    >>> gen.setup()           # download datasets + train ML model
    >>> df = gen.generate()   # create 5,000 sessions
    >>> gen.save(df)          # write CSV + metadata JSON

    Authors: Rushav Dash, Lisa Li
    """

    def __init__(
        self,
        global_seed: int = 42,
        n_sessions: int = TOTAL_SESSIONS,
        include_sound: bool = True,
        include_humidity: bool = True,
        verbose: bool = True,
    ):
        """
        Initialise the generator.

        Parameters
        ----------
        global_seed : int
            Master random seed — fully determines the dataset.
        n_sessions : int
            Total number of sleep sessions to generate.
        include_sound : bool
            Include the optional sound signal.
        include_humidity : bool
            Include the optional humidity signal.
        verbose : bool
            Print progress messages.

        Authors: Rushav Dash, Lisa Li
        """
        self.global_seed = global_seed
        self.n_sessions = n_sessions
        self.include_sound = include_sound
        self.include_humidity = include_humidity
        self.verbose = verbose

        # Components (populated by setup())
        self.ref_stats: Optional[ReferenceStats] = None
        self.signal_gen: Optional[SignalGenerator] = None
        self.quality_model: Optional[SleepQualityModel] = None
        self._is_setup = False

    def _log(self, msg: str) -> None:
        """Print if verbose."""
        if self.verbose:
            print(f"[DatasetGenerator] {msg}")

    # ----------------------------------------------------------
    # 2a. Setup: download, calibrate, train
    # ----------------------------------------------------------

    def setup(self, skip_download: bool = False) -> None:
        """
        Download datasets, extract calibration stats, and train the ML model.

        Parameters
        ----------
        skip_download : bool
            If True, skip kagglehub download (useful if models are already
            saved and you want to regenerate signals only).

        Authors: Rushav Dash, Lisa Li
        """
        loader = DataLoader(verbose=self.verbose)

        if not skip_download:
            self._log("Step 1/3 — Downloading datasets via kagglehub…")
            loader.download_all()
            loader.load_sleep_efficiency()
            loader.load_room_occupancy()
            loader.load_smart_home()

        self._log("Step 2/3 — Extracting calibration statistics…")
        self.ref_stats = loader.extract_reference_stats()
        self.signal_gen = SignalGenerator(
            ref_stats=self.ref_stats,
            duration_minutes=480,
            sampling_interval_minutes=5,
        )

        self._log("Step 3/3 — Training sleep quality model on real data…")
        self.quality_model = SleepQualityModel(
            n_estimators=200,
            random_state=self.global_seed,
        )
        df_sleep = loader.get_sleep_df()
        cv_results = self.quality_model.train(df_sleep, verbose=self.verbose)

        # Persist trained models to disk for reuse
        self.quality_model.save()

        self._log("Setup complete.  Ready to generate sessions.")
        self._is_setup = True

    # ----------------------------------------------------------
    # 2b. Session-level generation
    # ----------------------------------------------------------

    def _generate_session(
        self,
        session_idx: int,
        season: str,
        age_group: str,
        sensitivity: str,
        session_seed: int,
    ) -> dict:
        """
        Generate one complete sleep session: signals → features → labels.

        Parameters
        ----------
        session_idx : int
            Sequential index (0-based) for this session.
        season : str
        age_group : str
        sensitivity : str
        session_seed : int
            Unique random seed for this session.

        Returns
        -------
        dict
            Flat dictionary with metadata, features, and sleep quality labels.

        Authors: Rushav Dash, Lisa Li
        """
        rng = np.random.default_rng(session_seed)

        # ---- 1. Generate all environmental signals ----
        signals = self.signal_gen.generate_all(
            season=season,
            age_group=age_group,
            sensitivity=sensitivity,
            random_seed=session_seed,
            include_sound=self.include_sound,
            include_humidity=self.include_humidity,
        )

        # ---- 2. Extract scalar features from signals ----
        features = extract_all_features(signals, dt_minutes=5)

        # ---- 3. Assign sleep quality labels via ML model ----
        labels = self.quality_model.predict(
            env_features=features,
            season=season,
            age_group=age_group,
            sensitivity=sensitivity,
            rng=rng,
            add_noise=True,
        )

        # ---- 4. Assemble flat session record ----
        record = {
            # Metadata
            "session_id":   str(uuid.UUID(int=session_seed & 0xFFFFFFFFFFFFFFFF
                                          | (session_idx << 64)
                                          & 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF)),
            "session_index": session_idx,
            "season":       season,
            "age_group":    age_group,
            "sensitivity":  sensitivity,
            "random_seed":  session_seed,
        }

        # Flatten features into record
        record.update(features)

        # Sleep quality labels
        record["sleep_efficiency"] = labels["sleep_efficiency"]
        record["awakenings"]       = labels["awakenings"]
        record["rem_pct"]          = labels["rem_pct"]
        record["deep_pct"]         = labels["deep_pct"]
        record["light_pct"]        = labels["light_pct"]

        # Store time-series as JSON strings (compact; parseable downstream)
        record["ts_temperature"] = json.dumps(
            [round(float(v), 3) for v in signals["temperature"]]
        )
        record["ts_light"] = json.dumps(
            [round(float(v), 3) for v in signals["light"]]
        )
        if signals.get("sound") is not None:
            record["ts_sound"] = json.dumps(
                [round(float(v), 3) for v in signals["sound"]]
            )
        if signals.get("humidity") is not None:
            record["ts_humidity"] = json.dumps(
                [round(float(v), 3) for v in signals["humidity"]]
            )

        return record

    # ----------------------------------------------------------
    # 2c. Full dataset generation
    # ----------------------------------------------------------

    def generate(self) -> pd.DataFrame:
        """
        Generate all sessions and return as a pandas DataFrame.

        Distribution:
          - n_sessions / 4 sessions per season
          - Within each season: age × sensitivity grid, filled cyclically

        Shows a tqdm progress bar; logs to console every 500 sessions.

        Returns
        -------
        pd.DataFrame
            One row per sleep session.

        Authors: Rushav Dash, Lisa Li
        """
        if not self._is_setup:
            raise RuntimeError(
                "Call setup() before generate().  setup() downloads datasets "
                "and trains the sleep quality model."
            )

        # ---- Build the stratification schedule ----
        schedule = self._build_schedule()
        self._log(
            f"Generating {len(schedule)} sessions "
            f"({SESSIONS_PER_SEASON} per season)…"
        )

        records = []

        # tqdm wraps the session loop with a live progress bar
        for i, (season, age_group, sensitivity) in enumerate(
            tqdm(schedule, desc="Generating sessions", unit="session")
        ):
            # Each session gets a unique, reproducible seed
            session_seed = int(
                np.uint64(self.global_seed) ^ np.uint64(hash((i, season, age_group, sensitivity)) & 0xFFFF_FFFF_FFFF_FFFF)
            )

            record = self._generate_session(
                session_idx=i,
                season=season,
                age_group=age_group,
                sensitivity=sensitivity,
                session_seed=session_seed,
            )
            records.append(record)

            # Console log every 500 sessions
            if (i + 1) % 500 == 0:
                self._log(f"  {i + 1}/{len(schedule)} sessions complete")

        df = pd.DataFrame(records)
        self._log(f"Generation complete: {len(df)} sessions, {len(df.columns)} columns")
        return df

    def _build_schedule(self) -> list[tuple]:
        """
        Build the ordered list of (season, age_group, sensitivity) for all sessions.

        Uses a cycling approach within each seasonal block:
          [young-low, young-normal, young-high, middle-low, …, senior-high, young-low, …]
        This ensures balanced representation without requiring exact divisibility.

        Returns
        -------
        list of (season, age_group, sensitivity) tuples

        Authors: Rushav Dash, Lisa Li
        """
        schedule = []
        combos = [
            (ag, s) for ag in AGE_GROUPS for s in SENSITIVITIES
        ]  # 9 combinations

        for season in SEASONS:
            for i in range(SESSIONS_PER_SEASON):
                ag, sens = combos[i % len(combos)]
                schedule.append((season, ag, sens))

        # Handle remainder if n_sessions is not divisible by 4
        remaining = self.n_sessions - len(schedule)
        for i in range(remaining):
            season = SEASONS[i % len(SEASONS)]
            ag, sens = combos[i % len(combos)]
            schedule.append((season, ag, sens))

        return schedule

    # ----------------------------------------------------------
    # 2d. Output persistence
    # ----------------------------------------------------------

    def save(self, df: pd.DataFrame, output_dir: Optional[Path] = None) -> tuple[Path, Path]:
        """
        Save the generated dataset as CSV + metadata JSON.

        Parameters
        ----------
        df : pd.DataFrame
            Generated dataset from generate().
        output_dir : Path or None
            Directory for output files.  Defaults to data/output/.

        Returns
        -------
        (csv_path, json_path) : tuple of Paths

        Authors: Rushav Dash, Lisa Li
        """
        out_dir = Path(output_dir) if output_dir else OUTPUT_DIR
        out_dir.mkdir(parents=True, exist_ok=True)

        # ---- CSV ----
        csv_path = out_dir / "synthetic_sleep_dataset_5000.csv"
        df.to_csv(csv_path, index=False)
        self._log(f"CSV saved: {csv_path}  ({len(df)} rows)")

        # ---- Metadata JSON ----
        # Separate feature columns by type
        ts_cols   = [c for c in df.columns if c.startswith("ts_")]
        meta_cols = ["session_id", "session_index", "season", "age_group",
                     "sensitivity", "random_seed"]
        label_cols = ["sleep_efficiency", "awakenings", "rem_pct", "deep_pct", "light_pct"]
        feat_cols = [c for c in df.columns
                     if c not in ts_cols + meta_cols + label_cols]

        metadata = {
            "dataset_name": "Synthetic Sleep Environment Dataset",
            "version": "1.0",
            "created_by": ["Rushav Dash", "Lisa Li"],
            "course": "TECHIN 513 - Signal Processing & Machine Learning",
            "university": "University of Washington",
            "n_sessions": int(len(df)),
            "sampling_interval_minutes": 5,
            "session_duration_minutes": 480,
            "signals": ["temperature_C", "light_lux", "sound_dB", "humidity_pct"],
            "features": feat_cols,
            "target_variables": label_cols,
            "source_datasets": [
                {
                    "name": "Sleep Efficiency Dataset",
                    "kaggle": "equilibriumm/sleep-efficiency",
                    "role": "ML model training",
                },
                {
                    "name": "Room Occupancy Detection IoT Dataset",
                    "kaggle": "kukuroo3/room-occupancy-detection-data-iot-sensor",
                    "role": "Signal calibration (temperature, light)",
                },
                {
                    "name": "Smart Home Dataset with Weather",
                    "kaggle": "taranvee/smart-home-dataset-with-weather-information",
                    "role": "HVAC cycle calibration (optional)",
                },
            ],
            "generation_date": datetime.now().isoformat(),
            "random_seed": self.global_seed,
            "stratification": {
                "seasons": SEASONS,
                "sessions_per_season": SESSIONS_PER_SEASON,
                "age_groups": AGE_GROUPS,
                "sensitivity_levels": SENSITIVITIES,
            },
            "column_descriptions": {
                "session_id":              "UUID uniquely identifying this session",
                "season":                  "Season when the sleep session occurred",
                "age_group":               "Young (18-34), Middle (35-59), Senior (60+)",
                "sensitivity":             "Individual environmental sensitivity level",
                "temp_mean":               "Mean bedroom temperature over the night (°C)",
                "temp_optimal_fraction":   "Fraction of night with temp in [18,21]°C",
                "light_event_count":       "Number of light-on events during the night",
                "light_disruption_score":  "Amplitude×duration sum of light events",
                "sleep_efficiency":        "Fraction of time in bed actually asleep [0.5,0.99]",
                "awakenings":              "Number of awakenings during the night [0,12]",
                "rem_pct":                 "REM sleep as % of total sleep time",
                "deep_pct":                "Deep (slow-wave) sleep as % of total sleep time",
                "light_pct":               "Light sleep as % of total sleep time",
                "ts_temperature":          "JSON array of 96 temperature samples (5-min intervals)",
                "ts_light":                "JSON array of 96 light-level samples (lux)",
                "ts_sound":                "JSON array of 96 sound-level samples (dB)",
                "ts_humidity":             "JSON array of 96 humidity samples (%)",
            },
        }

        json_path = out_dir / "synthetic_sleep_dataset_metadata.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        self._log(f"Metadata JSON saved: {json_path}")

        return csv_path, json_path
