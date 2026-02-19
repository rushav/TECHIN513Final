# Synthetic Sleep Environment Dataset Generator

> **TECHIN 513: Signal Processing & Machine Learning — Team 7**
> Rushav Dash & Lisa Li — University of Washington

---

## 1. Project Overview

No public dataset directly links bedroom environmental conditions (temperature, light, sound, humidity) to polysomnographic sleep quality metrics. This project closes that gap by generating a realistic, statistically validated **synthetic dataset of 5,000 sleep sessions** using:

- **Signal processing** (spectral synthesis, Butterworth filtering, Poisson-process event models) to produce realistic 8-hour environmental time-series
- **Machine learning** (Random Forest trained on the real Sleep Efficiency dataset) to assign physiologically plausible sleep quality labels
- **Three-tier validation** (KS-tests vs. real IoT data, ML cross-dataset evaluation, sleep science sanity checks)

The output is a single, ready-to-use CSV (`synthetic_sleep_dataset_5000.csv`) plus a companion metadata JSON. Researchers can load it and immediately study the relationship between bedroom environment and sleep quality.

---

## 2. Team

| Name | Role | Institution |
|------|------|-------------|
| **Rushav Dash** | Signal processing, pipeline architecture | University of Washington |
| **Lisa Li** | ML model, validation, notebooks | University of Washington |

**Course:** TECHIN 513 — Signal Processing & Machine Learning
**Team:** 7

---

## 3. Dataset Download Instructions

This project downloads the Kaggle datasets automatically via `kagglehub`. You must have a Kaggle account and API credentials configured.

### 3a. Set up Kaggle credentials

```bash
pip install kagglehub
```

Then either:
- Place `~/.kaggle/kaggle.json` (download from https://www.kaggle.com/settings → API → "Create New Token")
- Or set environment variables: `KAGGLE_USERNAME` and `KAGGLE_KEY`

### 3b. Datasets used

| Dataset | Kaggle Slug | Role |
|---------|-------------|------|
| Sleep Efficiency Dataset | `equilibriumm/sleep-efficiency` | ML model training |
| Room Occupancy Detection IoT | `kukuroo3/room-occupancy-detection-data-iot-sensor` | Signal calibration |
| Smart Home Dataset with Weather | `taranvee/smart-home-dataset-with-weather-information` | HVAC calibration (optional) |

Downloads happen automatically when you run `DataLoader.download_all()` or call `SleepDatasetGenerator.setup()`. Files are cached in `~/.cache/kagglehub/` — re-runs are near-instant.

### 3c. Manual fallback

If kagglehub is unavailable, you can place CSVs manually:

```
data/raw/Sleep_Efficiency.csv          # from equilibriumm/sleep-efficiency
data/raw/room_occupancy.csv            # from kukuroo3/room-occupancy-detection-data-iot-sensor
data/raw/smart_home_dataset.csv        # from taranvee/smart-home-dataset-with-weather-information (optional)
```

Then load with: `pd.read_csv("data/raw/Sleep_Efficiency.csv")` and pass the DataFrame directly to `SleepQualityModel.train()`.

---

## 4. Installation

```bash
# Clone the project and navigate into it
cd sleep_dataset_generator

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the src package in editable mode so notebooks can import it
pip install -e .
```

---

## 5. Quick Start

Run the full pipeline in 3 steps:

```bash
# Step 1: Explore real data (EDA)
jupyter notebook notebooks/01_data_exploration.ipynb

# Step 2: Train models and generate dataset
jupyter notebook notebooks/04_dataset_generation.ipynb

# Step 3: Validate the generated dataset
jupyter notebook notebooks/05_validation.ipynb
```

Or run the pipeline entirely in Python:

```python
from src.dataset_generator import SleepDatasetGenerator

gen = SleepDatasetGenerator(global_seed=42, n_sessions=5000)
gen.setup()          # download datasets + train ML model
df = gen.generate()  # generate 5,000 sessions (takes ~5-10 min)
gen.save(df)         # write CSV + metadata JSON to data/output/
```

---

## 6. Notebook Guide

| Notebook | Description |
|----------|-------------|
| `01_data_exploration.ipynb` | EDA of all three real Kaggle datasets: distributions, correlations, FFT of temperature, autocorrelation plots |
| `02_signal_generation.ipynb` | Deep dive into signal generation: component-by-component walkthrough, frequency-domain analysis, multi-session diversity plots |
| `03_ml_model_training.ipynb` | Random Forest training on Sleep Efficiency data: CV scores, feature importance, residual analysis, model serialisation |
| `04_dataset_generation.ipynb` | Full pipeline execution: generate all 5,000 sessions, preview output, diversity checks |
| `05_validation.ipynb` | All three validation tiers: KS-tests, ML cross-dataset evaluation, sleep science sanity checks |
| `06_demo_usage.ipynb` | Researcher-facing demo: predictive modelling, seasonal analysis, optimal sleep clustering, case studies |

---

## 7. Output Dataset Schema

**File:** `data/output/synthetic_sleep_dataset_5000.csv`

### Session Metadata

| Column | Type | Description |
|--------|------|-------------|
| `session_id` | str | UUID uniquely identifying this session |
| `session_index` | int | Sequential index (0–4999) |
| `season` | str | winter / spring / summer / fall |
| `age_group` | str | young (18-34) / middle (35-59) / senior (60+) |
| `sensitivity` | str | low / normal / high environmental sensitivity |
| `random_seed` | int | Seed used for this session (fully reproducible) |

### Temperature Features

| Column | Units | Description |
|--------|-------|-------------|
| `temp_mean` | °C | Mean temperature over 8-hour session |
| `temp_std` | °C | Standard deviation of temperature |
| `temp_min` / `temp_max` | °C | Minimum / maximum temperature |
| `temp_range` | °C | Peak-to-peak temperature variation |
| `temp_above_21_minutes` | min | Minutes above 21°C (too warm) |
| `temp_below_18_minutes` | min | Minutes below 18°C (too cold) |
| `temp_optimal_fraction` | [0,1] | Fraction of night in optimal zone [18,21]°C |
| `temp_mean_rate_change` | °C/5min | Mean absolute rate of temperature change |
| `temp_max_rate_change` | °C/5min | Maximum single-step temperature change |

### Light Features

| Column | Units | Description |
|--------|-------|-------------|
| `light_mean` | lux | Mean light level |
| `light_max` | lux | Maximum light level |
| `light_event_count` | count | Number of light-on events during the night |
| `light_total_exposure_minutes` | min | Total minutes with lux > 10 |
| `light_disruption_score` | lux·min | Weighted sum of event amplitudes × durations |

### Sound Features

| Column | Units | Description |
|--------|-------|-------------|
| `sound_mean_db` | dB SPL | Mean sound level |
| `sound_max_db` | dB SPL | Maximum sound level |
| `sound_event_count` | count | Disturbance events above 55 dB |
| `sound_above_55db_minutes` | min | Minutes above 55 dB |
| `sound_leq_db` | dB SPL | Energy-equivalent level (Leq) |

### Humidity Features

| Column | Units | Description |
|--------|-------|-------------|
| `humidity_mean` | % | Mean relative humidity |
| `humidity_out_of_range_minutes` | min | Minutes outside [30,60]% |
| `humidity_comfort_fraction` | [0,1] | Fraction of night in comfort zone |

### Sleep Quality Labels (Targets)

| Column | Range | Description |
|--------|-------|-------------|
| `sleep_efficiency` | [0.50, 0.99] | Fraction of time in bed actually asleep |
| `awakenings` | [0, 12] | Number of awakenings during the night |
| `rem_pct` | [5, 40] | REM sleep as % of total sleep time |
| `deep_pct` | [5, 40] | Deep (slow-wave) sleep % |
| `light_pct` | [5, 90] | Light sleep % (rem_pct + deep_pct + light_pct = 100) |

### Time-Series Columns

| Column | Description |
|--------|-------------|
| `ts_temperature` | JSON array of 96 temperature samples (5-min intervals, °C) |
| `ts_light` | JSON array of 96 light samples (lux) |
| `ts_sound` | JSON array of 96 sound samples (dB) |
| `ts_humidity` | JSON array of 96 humidity samples (%) |

---

## 8. Validation Results

Run `05_validation.ipynb` to populate this section. Expected targets:

| Tier | Test | Target |
|------|------|--------|
| 1 | KS-test: temperature distribution | p > 0.05 |
| 1 | KS-test: light distribution | p > 0.05 |
| 2 | Synthetic model RMSE ≤ 1.2× real baseline | PASS |
| 3 | High temp optimality → efficiency ≥ 0.78 | PASS |
| 3 | Many light events → efficiency ≤ 0.72 | PASS |
| 3 | Deep sleep ↔ awakenings correlation < −0.2 | PASS |

---

## 9. Project Architecture

```
src/
├── data_loader.py        # Kaggle downloads via kagglehub; reference stat extraction
├── signal_generator.py   # Spectral synthesis: temperature, light, sound, humidity
├── feature_extractor.py  # 30+ scalar features from raw time-series
├── sleep_quality_model.py # Random Forest regressors; env→quality label mapping
├── dataset_generator.py  # Pipeline orchestrator; stratified 5,000-session loop
└── validator.py          # Three-tier validation: KS, ML transfer, sanity checks
```

### Signal Generation Architecture

```
Temperature(t) = T_base + T_circadian(t) + T_hvac(t) + T_noise(t)
              → Butterworth LPF (order=4, fc=1/30 cpm)

Light(t)       = L_background(t) + Σ events(t)    [Poisson process]
              → Gaussian edge smoothing (σ=2 min)

Sound(t)       = S_background (pink noise) + Σ events(t)

Humidity(t)    = H_base + A_h·sin(2πt/480) + Gaussian noise
```

---

## 10. References

1. **Sleep Efficiency Dataset** — Kaggle, equilibriumm.
   https://www.kaggle.com/datasets/equilibriumm/sleep-efficiency

2. **Room Occupancy Detection IoT Dataset** — Kaggle, kukuroo3.
   https://www.kaggle.com/datasets/kukuroo3/room-occupancy-detection-data-iot-sensor

3. **Smart Home Dataset with Weather Information** — Kaggle, taranvee.
   https://www.kaggle.com/datasets/taranvee/smart-home-dataset-with-weather-information

4. Okamoto-Mizuno, K. & Mizuno, K. (2012). Effects of thermal environment on sleep and circadian rhythm. *Journal of Physiological Anthropology*, 31(1), 14.

5. Zeitzer, J. M., et al. (2000). Sensitivity of the human circadian pacemaker to nocturnal light. *Journal of Physiology*, 526(3), 695–702.

6. Walker, M. (2017). *Why We Sleep*. Scribner.

7. WHO (2009). *Night Noise Guidelines for Europe*. World Health Organization.

8. Dijk, D. J. (2009). Regulation and functional correlates of slow wave sleep. *Journal of Clinical Sleep Medicine*, 5(2 Suppl), S6–S15.
