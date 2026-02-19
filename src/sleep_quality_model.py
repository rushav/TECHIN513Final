"""
=============================================================================
sleep_quality_model.py — ML model training and sleep quality label assignment
=============================================================================
Project:    Synthetic Sleep Environment Dataset Generator
Course:     TECHIN 513: Signal Processing & Machine Learning
Team:       Team 7 — Rushav Dash & Lisa Li
University: University of Washington
Date:       2026-02-19
=============================================================================
This module trains a collection of Random Forest Regressors on the real
Sleep Efficiency dataset and uses them to assign physiologically plausible
sleep quality labels to each synthetic session.

Four target variables are predicted:
  1. Sleep efficiency        (primary, range [0.50, 0.99])
  2. Awakenings              (secondary, range [0, 12], integer)
  3. REM sleep percentage    (secondary, sums with deep + light = 100)
  4. Deep sleep percentage   (secondary)

The mapping from environmental features (extracted by feature_extractor.py)
to the model's training-space features is carefully documented so it can
be critically reviewed and extended in future work.

Authors: Rushav Dash, Lisa Li
"""

import os
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score


# ============================================================
# SECTION 1: CONSTANTS
# ============================================================

TARGET_VARIABLES = [
    "Sleep efficiency",
    "Awakenings",
    "REM sleep percentage",
    "Deep sleep percentage",
]

# Valid output ranges used to clip predictions.
# Note: the Sleep Efficiency training dataset labels "Deep sleep percentage"
# in the range 18–75 % (it includes N2/N3 combined as "deep"), so the
# clip is set to match that distribution.  After normalization the three
# stage columns always sum exactly to 100.
TARGET_CLIPS = {
    "Sleep efficiency":      (0.50, 0.99),
    "Awakenings":            (0, 12),
    "REM sleep percentage":  (5, 40),
    "Deep sleep percentage": (5, 80),
}

# Residual noise stds added after prediction (derived from real-data residuals
# in 03_ml_model_training.ipynb; these are starting defaults before fitting)
DEFAULT_RESIDUAL_STDS = {
    "Sleep efficiency":      0.08,
    "Awakenings":            1.2,
    "REM sleep percentage":  5.0,
    "Deep sleep percentage": 4.5,
}

MODEL_DIR = Path(__file__).parent.parent / "data" / "processed" / "models"


# ============================================================
# SECTION 2: FEATURE ENGINEERING HELPERS
# ============================================================

def _encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical columns in the Sleep Efficiency DataFrame.

    Converts binary string columns (Gender, Smoking status) to 0/1
    integers.  Drops columns that cannot be used directly.

    Parameters
    ----------
    df : pd.DataFrame
        The raw Sleep Efficiency DataFrame.

    Returns
    -------
    pd.DataFrame
        A copy with categoricals encoded and datetimes dropped.

    Authors: Rushav Dash, Lisa Li
    """
    df = df.copy()

    # Binary encoding for Gender (Male=1, Female=0)
    if "Gender" in df.columns:
        df["Gender"] = (
            df["Gender"].str.strip().str.lower().map({"male": 1, "female": 0})
        )

    # Binary encoding for Smoking status (Yes=1, No=0)
    if "Smoking status" in df.columns:
        df["Smoking status"] = (
            df["Smoking status"].str.strip().str.lower().map({"yes": 1, "no": 0})
        )

    # Drop datetime and ID columns — not useful as numeric predictors
    drop_cols = [c for c in df.columns if "time" in c.lower() or "id" in c.lower()]
    # Also drop Season if it's categorical string; will be handled later
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    return df


def _engineer_proxy_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create proxy environmental features from lifestyle columns.

    The Sleep Efficiency dataset contains lifestyle predictors (caffeine,
    alcohol, exercise) that serve as proxies for how the person's bedroom
    environment affects them.  We derive compound features that bridge the
    lifestyle domain to the environmental domain.

    Proxy mappings (documented for transparency):
      - Caffeine consumption (mg) → arousal proxy → models "sensitivity high"
      - Alcohol consumption (oz)  → sleep fragmentation proxy
      - Exercise frequency/week   → baseline sleep quality proxy
      - Age                       → age_group proxy

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame with additional proxy columns.

    Authors: Rushav Dash, Lisa Li
    """
    df = df.copy()

    # -- Arousal index: high caffeine + smoking = higher arousal --
    caffeine = df.get("Caffeine consumption", pd.Series(0, index=df.index)).fillna(0)
    smoking  = df.get("Smoking status", pd.Series(0, index=df.index)).fillna(0)
    df["arousal_index"] = caffeine / 200.0 + smoking * 0.5   # normalised 0–1.5

    # -- Fragmentation proxy: alcohol disrupts sleep architecture --
    alcohol = df.get("Alcohol consumption", pd.Series(0, index=df.index)).fillna(0)
    df["fragmentation_proxy"] = alcohol / 5.0   # normalised 0–1

    # -- Fitness proxy: exercise frequency → baseline efficiency boost --
    exercise = df.get("Exercise frequency", pd.Series(3, index=df.index)).fillna(3)
    df["fitness_proxy"] = exercise / 7.0   # normalised 0–1

    # -- Age group encoding --
    if "Age" in df.columns:
        df["age_numeric"] = df["Age"].fillna(35)
        df["age_young"]   = (df["age_numeric"] < 35).astype(int)
        df["age_middle"]  = ((df["age_numeric"] >= 35) & (df["age_numeric"] < 60)).astype(int)
        df["age_senior"]  = (df["age_numeric"] >= 60).astype(int)

    return df


# ============================================================
# SECTION 3: MAIN MODEL CLASS
# ============================================================

class SleepQualityModel:
    """
    Ensemble of Random Forest Regressors for predicting sleep quality.

    One Random Forest is trained per target variable.  After training,
    the model can predict labels for synthetic sessions by mapping
    environmental features to the training input space.

    Usage
    -----
    >>> model = SleepQualityModel()
    >>> model.train(sleep_efficiency_df)
    >>> labels = model.predict(env_features_dict, season="winter")
    >>> model.save()

    Authors: Rushav Dash, Lisa Li
    """

    def __init__(
        self,
        n_estimators: int = 200,
        random_state: int = 42,
    ):
        """
        Initialise the model collection.

        Parameters
        ----------
        n_estimators : int
            Number of trees per Random Forest.
        random_state : int
            Global random seed.

        Authors: Rushav Dash, Lisa Li
        """
        self.n_estimators = n_estimators
        self.random_state = random_state

        # One RF per target variable
        self.models: dict[str, RandomForestRegressor] = {}
        self.feature_names: list[str] = []   # columns used during training
        self.residual_stds: dict[str, float] = dict(DEFAULT_RESIDUAL_STDS)
        self._is_trained = False

    # ----------------------------------------------------------
    # 3a. Training
    # ----------------------------------------------------------

    def train(
        self,
        df: pd.DataFrame,
        cv_folds: int = 5,
        verbose: bool = True,
    ) -> dict:
        """
        Train one Random Forest Regressor per target variable.

        Pipeline:
          1. Drop rows where any target variable is NaN.
          2. Encode categoricals and engineer proxy features.
          3. Build feature matrix X (all numeric columns not in targets).
          4. For each target: fit RF, run 5-fold CV, compute RMSE & R².
          5. Estimate residual std from out-of-bag predictions.
          6. Store trained models and feature column list.

        Parameters
        ----------
        df : pd.DataFrame
            The cleaned Sleep Efficiency DataFrame from DataLoader.
        cv_folds : int
            Number of cross-validation folds.
        verbose : bool
            Print CV metrics to stdout.

        Returns
        -------
        dict
            {target_name: {'rmse_cv': float, 'r2_cv': float}}

        Authors: Rushav Dash, Lisa Li
        """
        # ---- Data preparation ----
        df = _encode_categoricals(df)
        df = _engineer_proxy_features(df)

        # Remove rows missing ANY target variable
        present_targets = [t for t in TARGET_VARIABLES if t in df.columns]
        df = df.dropna(subset=present_targets).copy()

        # Identify feature columns: all numeric columns not in targets and not ID
        non_feature_cols = set(present_targets) | {"ID", "id"}
        potential_features = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c not in non_feature_cols
        ]

        # Fill remaining NaNs in features with column median
        X = df[potential_features].copy()
        for col in X.columns:
            X[col] = X[col].fillna(X[col].median())

        self.feature_names = list(X.columns)

        cv_results = {}

        # ---- Train one model per target ----
        for target in present_targets:
            y = df[target].values

            rf = RandomForestRegressor(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                n_jobs=-1,            # use all CPU cores
                oob_score=True,       # enables out-of-bag R² estimate
            )

            # 5-fold cross-validation for unbiased performance estimate
            cv_scores = cross_validate(
                rf, X, y,
                cv=cv_folds,
                scoring=["neg_root_mean_squared_error", "r2"],
                return_train_score=False,
            )

            rmse_cv = float(-np.mean(cv_scores["test_neg_root_mean_squared_error"]))
            r2_cv   = float(np.mean(cv_scores["test_r2"]))

            if verbose:
                print(
                    f"  [{target}] CV RMSE = {rmse_cv:.4f}, "
                    f"CV R² = {r2_cv:.4f}"
                )

            # Fit the final model on ALL available data (no held-out split)
            rf.fit(X, y)
            self.models[target] = rf

            # Estimate residual std from OOB predictions
            y_oob = rf.oob_prediction_
            residuals = y - y_oob
            self.residual_stds[target] = float(np.std(residuals))

            cv_results[target] = {"rmse_cv": rmse_cv, "r2_cv": r2_cv}

        self._is_trained = True
        return cv_results

    # ----------------------------------------------------------
    # 3b. Prediction for synthetic sessions
    # ----------------------------------------------------------

    def predict(
        self,
        env_features: dict,
        season: str = "fall",
        age_group: str = "middle",
        sensitivity: str = "normal",
        rng: Optional[np.random.Generator] = None,
        add_noise: bool = True,
    ) -> dict:
        """
        Assign sleep quality labels to one synthetic session.

        Maps environmental features extracted by feature_extractor.py to
        the training feature space, runs the Random Forests, adds
        physiologically realistic residual noise, and clips outputs to
        valid ranges.

        Parameters
        ----------
        env_features : dict
            Output of feature_extractor.extract_all_features().
        season : str
        age_group : str
        sensitivity : str
        rng : np.random.Generator or None
            Random generator for residual noise.  If None, no noise is added.
        add_noise : bool
            Whether to add residual noise to predictions.

        Returns
        -------
        dict with keys:
            'sleep_efficiency', 'awakenings', 'rem_pct',
            'deep_pct', 'light_pct'

        Authors: Rushav Dash, Lisa Li
        """
        if not self._is_trained:
            raise RuntimeError(
                "Model has not been trained.  Call train() or load() first."
            )

        # ---- Build input feature vector ----
        x_dict = self._map_env_to_training_space(env_features, season, age_group, sensitivity)

        # Align with training feature order; fill missing with 0
        x_row = np.array([x_dict.get(f, 0.0) for f in self.feature_names]).reshape(1, -1)

        # ---- Predict all targets ----
        raw_preds = {}
        for target, model in self.models.items():
            pred = float(model.predict(x_row)[0])

            # Add physiologically realistic noise (residual uncertainty)
            if add_noise and rng is not None:
                noise = rng.normal(0, self.residual_stds.get(target, 0.05))
                pred += noise

            raw_preds[target] = pred

        # ---- Clip to valid physiological ranges ----
        se   = float(np.clip(raw_preds.get("Sleep efficiency", 0.75),    0.50, 0.99))
        awk  = int(np.clip(round(raw_preds.get("Awakenings", 2)),         0,    12))
        rem  = float(np.clip(raw_preds.get("REM sleep percentage", 22),   5,    40))
        deep = float(np.clip(raw_preds.get("Deep sleep percentage", 18),  5,    80))

        # ---- Normalise sleep-stage percentages to sum to 100 ----
        # Light sleep = 100 - REM - Deep; ensure everything is positive
        light = max(5.0, 100.0 - rem - deep)
        total = rem + deep + light
        if total > 0:
            rem   = round(rem   / total * 100, 1)
            deep  = round(deep  / total * 100, 1)
            light = round(100.0 - rem - deep,  1)

        return {
            "sleep_efficiency": round(se, 4),
            "awakenings":       awk,
            "rem_pct":          rem,
            "deep_pct":         deep,
            "light_pct":        light,
        }

    def _map_env_to_training_space(
        self,
        env: dict,
        season: str,
        age_group: str,
        sensitivity: str,
    ) -> dict:
        """
        Map environmental features to the training feature space.

        The Sleep Efficiency dataset does not contain direct environmental
        measurements, so we use evidence-based mappings:

          env feature          → training proxy
          ─────────────────────────────────────────────────────────────
          temp_optimal_fraction → higher → lower arousal_index (less disruption)
          light_disruption_score → higher → higher fragmentation_proxy
          sound_above_55db_minutes → higher → higher fragmentation_proxy
          humidity_out_of_range_minutes → mild arousal_index increase
          age_group             → age_young/middle/senior flags
          sensitivity           → arousal_index scaling

        Parameters
        ----------
        env : dict
            Extracted environmental features.
        season, age_group, sensitivity : str

        Returns
        -------
        dict mapping training feature names → values

        Authors: Rushav Dash, Lisa Li
        """
        # Sensitivity scaling factor
        sens_map = {"low": 0.5, "normal": 1.0, "high": 1.8}
        sens = sens_map.get(sensitivity, 1.0)

        # Arousal index: penalise bad temperature + light + sound
        temp_penalty  = 1.0 - env.get("temp_optimal_fraction", 0.7)
        light_pen     = min(1.0, env.get("light_disruption_score", 0.0) / 2000.0)
        sound_pen     = min(1.0, env.get("sound_above_55db_minutes", 0.0) / 120.0)
        humidity_pen  = min(0.3, env.get("humidity_out_of_range_minutes", 0.0) / 480.0)

        arousal_index = sens * (temp_penalty * 0.4 + light_pen * 0.35 + sound_pen * 0.2 + humidity_pen * 0.05)

        # Fragmentation proxy: driven primarily by light events and sound
        fragmentation = (
            env.get("light_event_count", 0) * 0.08
            + light_pen * 0.4
            + sound_pen * 0.4
        ) * sens

        # Fitness proxy: neutral (we don't know exercise for synthetic sessions)
        fitness_proxy = 0.5

        # Age encoding
        age_young  = 1 if age_group == "young"  else 0
        age_middle = 1 if age_group == "middle" else 0
        age_senior = 1 if age_group == "senior" else 0
        age_numeric = {"young": 26, "middle": 45, "senior": 68}.get(age_group, 45)

        # Season → caffeine / alcohol proxy (no direct link; use neutral values)
        caffeine_proxy = 0.0   # unknown for synthetic sessions
        alcohol_proxy  = 0.0

        return {
            "arousal_index":      arousal_index,
            "fragmentation_proxy": fragmentation,
            "fitness_proxy":      fitness_proxy,
            "age_numeric":        age_numeric,
            "age_young":          age_young,
            "age_middle":         age_middle,
            "age_senior":         age_senior,
            "Gender":             0.5,         # unknown → midpoint
            "Smoking status":     0.0,         # assume non-smoker (mode)
            "Caffeine consumption": caffeine_proxy,
            "Alcohol consumption":  alcohol_proxy,
            "Exercise frequency":   fitness_proxy * 7,
            "Sleep duration":       8.0,        # we always generate 8-hour sessions
        }

    # ----------------------------------------------------------
    # 3c. Model persistence
    # ----------------------------------------------------------

    def save(self, directory: Optional[Path] = None) -> None:
        """
        Persist all trained Random Forest models and metadata to disk.

        Saves one .joblib file per target variable plus a metadata dict.

        Parameters
        ----------
        directory : Path or None
            Save directory.  Defaults to data/processed/models/.

        Authors: Rushav Dash, Lisa Li
        """
        save_dir = Path(directory) if directory else MODEL_DIR
        save_dir.mkdir(parents=True, exist_ok=True)

        for target, model in self.models.items():
            fname = target.lower().replace(" ", "_") + ".joblib"
            joblib.dump(model, save_dir / fname)

        # Save metadata (feature names, residual stds)
        meta = {
            "feature_names":   self.feature_names,
            "residual_stds":   self.residual_stds,
            "n_estimators":    self.n_estimators,
            "random_state":    self.random_state,
        }
        joblib.dump(meta, save_dir / "model_metadata.joblib")
        print(f"[SleepQualityModel] Models saved to {save_dir}")

    def load(self, directory: Optional[Path] = None) -> None:
        """
        Load previously saved models from disk.

        Parameters
        ----------
        directory : Path or None
            Directory containing .joblib files.  Defaults to data/processed/models/.

        Authors: Rushav Dash, Lisa Li
        """
        load_dir = Path(directory) if directory else MODEL_DIR

        meta_path = load_dir / "model_metadata.joblib"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Model metadata not found at {meta_path}. "
                "Train the model first by running 03_ml_model_training.ipynb."
            )

        meta = joblib.load(meta_path)
        self.feature_names = meta["feature_names"]
        self.residual_stds = meta["residual_stds"]

        # Load individual target models
        for target in TARGET_VARIABLES:
            fname = target.lower().replace(" ", "_") + ".joblib"
            fpath = load_dir / fname
            if fpath.exists():
                self.models[target] = joblib.load(fpath)

        self._is_trained = True
        print(f"[SleepQualityModel] Loaded {len(self.models)} models from {load_dir}")
