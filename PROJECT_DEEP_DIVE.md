# Project Deep Dive: Synthetic Sleep Environment Dataset
**TECHIN 513 — Signal Processing & Machine Learning | Team 7**
Rushav Dash & Lisa Li — University of Washington

---

## The Problem We Are Solving

Sleep quality is one of the most important predictors of long-term health outcomes, and the physical environment of a bedroom — its temperature, light, sound, and humidity — is well-documented in sleep science literature as a direct driver of how well a person sleeps. Yet if you go looking for a dataset that actually captures both sides of this relationship — sensor measurements of the bedroom environment *and* polysomnographic (PSG) sleep quality metrics for the same session — you will not find one.

This gap exists for a practical reason: collecting it would require running a controlled experiment where you instrument someone's bedroom with IoT sensors and simultaneously measure their brainwaves, eye movements, and muscle activity all night with medical-grade equipment. That kind of study is expensive, requires ethics board approval, demands careful participant recruitment, and takes months. As a result, researchers who want to study how environment affects sleep either have to run such an expensive study themselves, or approximate their way around the missing data.

**Our project closes that gap by generating a synthetic dataset of 5,000 realistic sleep sessions**, each containing a full 8-hour set of bedroom environmental time-series (temperature, light, sound, humidity) alongside corresponding sleep quality labels (sleep efficiency, number of awakenings, REM percentage, deep sleep percentage, light sleep percentage). Every session is statistically grounded in real data. The key challenge — and the interesting intellectual problem — is that we built this dataset by bridging three completely unrelated Kaggle datasets that share no variables, no participants, and no overlapping context.

---

## The Three Datasets and Why They Cannot Be Joined Directly

### Dataset 1 — Sleep Efficiency Dataset
This dataset contains records of 452 real people's sleep sessions. Each row has demographic and lifestyle features: age, gender, caffeine consumption, alcohol consumption, exercise frequency, smoking status, and sleep duration. It also has the sleep quality labels we want: sleep efficiency (fraction of time in bed actually asleep), number of awakenings, and percentages of time in REM, deep, and light sleep.

**What it has:** Sleep quality labels and lifestyle features.
**What it does not have:** Any environmental sensor data. No temperature, no light level, no sound, no humidity.

### Dataset 2 — Room Occupancy Detection (IoT Sensors)
This is a real-time sensor dataset from an indoor environment (an office/lab space) with 20,560 rows of readings: temperature, light, humidity, and CO2 captured at one-minute intervals.

**What it has:** Real, physics-grounded distributions of what indoor environmental sensor readings look like — the autocorrelation structure of temperature drift, the statistical shape of light level distributions, the typical range and variance of indoor humidity.
**What it does not have:** Any sleep data. No people sleeping. No sleep quality measurements. The environment is a workplace, not a bedroom.

### Dataset 3 — Smart Home with Weather Information
A large smart home energy dataset (500,000+ rows) including thermostat data, HVAC system activity, and outdoor weather readings.

**What it has:** HVAC cycle patterns — how long heating/cooling cycles last, their periodicity. We extract the dominant period of temperature oscillation in the 20–120 minute band, which is the realistic timescale of bedroom HVAC cycling.
**What it does not have:** Sleep data. Environmental conditions specific to a bedroom.

---

## The Core Question: Why Can We Connect These?

The three datasets have no shared variables, no shared participants, and were collected in entirely different contexts. The bridge is not a mathematical join on a common key — it is **domain knowledge from sleep science acting as the translation layer**.

Here is the argument in full:

### Step 1: Sleep science establishes causal links from environment to sleep disruption

Peer-reviewed sleep medicine literature has established, through controlled laboratory studies, that:

- **Temperature:** The human body needs to drop its core temperature by ~1°C to initiate sleep. Bedroom temperatures outside the 18–21°C optimal zone interfere with this thermoregulatory process, increase arousal, and reduce sleep efficiency. (Okamoto-Mizuno & Mizuno, 2012)
- **Light:** Light at intensities as low as 10–100 lux suppresses melatonin secretion and delays sleep onset. The circadian pacemaker is extremely sensitive to nocturnal light exposure. (Zeitzer et al., 2000)
- **Sound:** Noise events above 55 dB (WHO nighttime guideline) trigger cortisol spikes that cause micro-arousals and outright awakenings. Energy-equivalent noise levels (Leq) predict sleep fragmentation.
- **Humidity:** Relative humidity outside the 30–60% comfort zone increases nasal resistance, promotes mouth breathing, and disrupts sleep architecture.

These are not correlations from observational data — they are causal mechanisms. They tell us *why* bad environmental conditions produce bad sleep outcomes.

### Step 2: The Sleep Efficiency dataset captures the same disruption through a different proxy

The Sleep Efficiency dataset does not have environmental sensors, but it does have variables that capture arousal, fragmentation, and physiological disruption through a different lens. Caffeine consumption captures arousal state. Alcohol consumption captures sleep fragmentation (alcohol suppresses REM in the second half of the night). Smoking status captures a chronic arousal/disruption signal. Exercise frequency captures baseline sleep quality.

These lifestyle variables are proxies for the same underlying construct — **how disrupted is this person's sleep architecture?** — that environmental conditions also drive.

### Step 3: We build a mathematical bridge between the two proxy spaces

Because both the environmental signals and the lifestyle features are proxies for the same underlying sleep-disruption construct, we can map one to the other using the causal mechanisms from Step 1.

Specifically, we define an **arousal index** that translates our environmental features into the equivalent of a caffeine-level disruption signal:

```
arousal_index = sensitivity_multiplier × (
    0.40 × (1 − temp_optimal_fraction)  +   # how much time was temp outside 18–21°C
    0.35 × (light_disruption_score / 2000)  +  # weighted light exposure events
    0.20 × (sound_above_55db_minutes / 120)  +  # time above WHO noise limit
    0.05 × (humidity_out_of_range_minutes / 480)  # time outside comfort zone
)
```

And a **fragmentation proxy** that translates light events and sound events into the equivalent of an alcohol-disruption signal:

```
fragmentation_proxy = sensitivity_multiplier × (
    light_event_count × 0.08 +
    (light_disruption_score / 2000) × 0.40 +
    (sound_above_55db_minutes / 120) × 0.40
)
```

These computed values are fed into the Random Forest model that was trained on the Sleep Efficiency dataset. The model then predicts what sleep efficiency, awakening count, and sleep stage percentages a person with that level of arousal and fragmentation would experience.

The weights (0.40, 0.35, 0.20, 0.05) come from the relative strength of each environmental factor's causal effect on sleep, as reported in the sleep medicine literature.

### Step 4: IoT data grounds the signals in physical reality

The Room Occupancy dataset does not tell us about bedroom sleep — but it does tell us what real indoor sensor data looks like at a statistical level. Specifically:

- The **autocorrelation structure** of temperature: real indoor temperature has a lag-1 autocorrelation of ~0.97, meaning it changes slowly and smoothly. Our signal generator uses this to set the AR(1) coefficient of the thermal inertia model, producing realistic gradual drifts instead of random jumps.
- The **distribution shape** of light levels: real indoor nighttime environments have a mean of ~3 lux with occasional bursts. We use the Poisson event rate from the IoT data to calibrate how many light events occur per night.
- The **dominant spectral frequency** of temperature variation: we extract this from FFT analysis of the real sensor data and use it to set the circadian oscillation component of our synthetic temperature signal.
- The **HVAC cycle period** from the Smart Home dataset: we fit the dominant period of thermostat cycling in the 20–120 minute band and use it to add realistic HVAC-driven temperature oscillation on top of the circadian trend.

This is the crucial point: we are not generating random numbers that happen to be in the right range. We are generating numbers whose statistical properties — autocorrelation, spectral content, event rate distributions — match what real sensors produce. The fact that the original sensors were in an office rather than a bedroom is handled by our seasonal baseline tables (which set appropriate bedroom temperature ranges by season, derived from ASHRAE comfort guidelines) and the age/sensitivity stratification.

---

## How the Pipeline Actually Works

### Stage 1: Signal Generation (Physics-Based)

For each of the 5,000 sessions we generate four 8-hour time series, sampled at 5-minute intervals (96 data points each).

**Temperature** is the most complex signal. It has three additive components:
1. A **circadian component**: a slow sinusoidal oscillation matching the body's natural temperature preference across an 8-hour night (slightly cooler in the first half, rising toward morning). Period and amplitude are calibrated from the IoT FFT.
2. An **HVAC component**: a shorter-period oscillation (typically ~45 minutes) representing heating/cooling cycles. Period extracted from Smart Home dataset spectral analysis.
3. A **noise component**: low-amplitude random fluctuations.

The sum of these three components is passed through a **4th-order Butterworth low-pass filter** (cutoff at 1/30 cycles-per-minute) to enforce thermal inertia — temperature cannot jump suddenly. The AR(1) autocorrelation of 0.97 from the IoT data is enforced by this filtering step.

The baseline temperature level is set by season (winter: 17–20°C, summer: 21–25°C) and shifted slightly upward for seniors (who prefer warmer environments) and high-sensitivity individuals.

**Light** uses a **Poisson process event model**. Most of the night is at background darkness (~3 lux with small Gaussian noise). Light-on events arrive at a Poisson rate (calibrated from IoT data) and have durations drawn from an exponential distribution. Each event is smoothed with a Gaussian kernel (σ=2 min) to avoid unphysical instantaneous transitions. High-sensitivity individuals have a higher event rate.

**Sound** uses **pink noise** (1/f noise) as the background, which matches the spectral character of real environmental sound (air conditioning hum, distant traffic). Discrete disturbance events — footsteps, doors, vehicles — are added as Poisson arrivals above the 55 dB threshold.

**Humidity** follows a mild sinusoidal pattern (breath moisture accumulates across the night and dissipates in the second half) with Gaussian noise added. Seasonal baseline humidity ranges come from the IoT dataset's observed distributions.

### Stage 2: Feature Extraction (Signal Processing)

From each raw time series we extract 29+ scalar features that summarize the environmental conditions of that night. Examples:

| Feature | What it captures |
|---------|-----------------|
| `temp_optimal_fraction` | Fraction of the night where temperature was in the optimal 18–21°C zone |
| `temp_mean_rate_change` | Mean absolute rate of temperature change (thermal stability) |
| `light_disruption_score` | Weighted sum of light event amplitudes × durations |
| `light_event_count` | Number of distinct light-on events |
| `sound_leq_db` | Energy-equivalent sound level (standard acoustics metric, Leq) |
| `sound_above_55db_minutes` | Total minutes above the WHO nighttime noise limit |
| `humidity_comfort_fraction` | Fraction of night in the 30–60% comfort zone |

These scalar features are what connect the time-series domain (signal processing) to the machine learning domain (label assignment).

### Stage 3: Label Assignment (Machine Learning)

A Random Forest model is trained on the 452 real records from the Sleep Efficiency dataset. The training target is sleep efficiency. The training features are the lifestyle columns (caffeine, alcohol, exercise, smoking, age, gender, sleep duration) after encoding and engineering proxy features (arousal index, fragmentation proxy, fitness proxy).

At inference time, each synthetic session's environmental features are **mapped through the proxy function** described above to produce the equivalent lifestyle-space inputs. The Random Forest then predicts what sleep quality that environmental profile would produce in a real person.

Predicted values are clipped to physiologically valid ranges and residual noise (calibrated from the model's out-of-bag prediction error) is added to prevent the dataset from having unrealistically perfect predictions.

Sleep stage percentages (REM, deep, light) are normalized to always sum to exactly 100%.

### Stage 4: Stratification

The 5,000 sessions are not generated uniformly at random. They are stratified across:
- **4 seasons** (1,250 sessions each): driving the baseline temperature and humidity
- **3 age groups** (young 18–34, middle 35–59, senior 60+): adjusting temperature preference and baseline sleep architecture
- **3 sensitivity levels** (low, normal, high): scaling how strongly environmental events affect the arousal and fragmentation proxies

Each session gets a deterministic seed derived from its index and the global seed (42), making the entire 5,000-session dataset exactly reproducible.

---

## Why the Synthesis Is Valid (and Where It Has Limitations)

### Why it works

1. **The causal chain is scientifically grounded.** We are not making up the direction or magnitude of environmental effects on sleep — we are encoding established experimental findings from sleep medicine into our mapping functions.

2. **The signal statistics are empirically calibrated.** Temperature autocorrelation, light event rates, HVAC periodicity — all come from real sensor measurements, not arbitrary assumptions.

3. **The ML model generalizes to the proxy space.** A Random Forest trained on lifestyle → sleep quality can be queried with the equivalent environmental proxies because both encode the same underlying physiological disruption state.

4. **The validation confirms plausibility.** Tier 2 validation shows that a model trained on our synthetic data achieves RMSE within 20% of the baseline trained on real sleep data — meaning the relationships in our dataset transfer to real-world predictive tasks.

### Where it has limitations

1. **The KS-tests fail (Tier 1).** With 5,000 sessions vs. 452 real records, the Kolmogorov-Smirnov test is extremely sensitive. Our synthetic temperature distribution differs from the office IoT distribution because we explicitly set bedroom-appropriate ranges — a deliberate design choice, not a flaw.

2. **Deep sleep ↔ awakenings correlation is near zero.** The Random Forest predicts both labels independently from the same feature vector, so it does not enforce the known negative correlation between deep sleep and awakening count. Fixing this would require post-hoc correlation enforcement or a joint multivariate model.

3. **Light events do not suppress efficiency enough at high counts.** At more than 4 light events per night, real sleep efficiency should drop significantly. Our mapping coefficient (0.08 per event in the fragmentation proxy) is too conservative. This is a tunable parameter.

4. **The proxy mapping is uni-directional.** We can go from environment → sleep quality, but not the reverse. The dataset cannot tell you what environmental profile produces a specific sleep efficiency — only what sleep efficiency a specific environment profile tends to produce.

---

## Key Numbers to Know

| Quantity | Value |
|---------|-------|
| Total sessions | 5,000 |
| Time series length | 96 points (5-min intervals over 8 hours) |
| Environmental features per session | 29+ scalar features |
| Sleep quality targets per session | 5 (efficiency, awakenings, REM%, deep%, light%) |
| Real records used for ML training | 452 (Sleep Efficiency dataset) |
| Real IoT readings used for calibration | 20,560 (Room Occupancy dataset) |
| Overall validation score | 71.4% (5/7 checks pass) |
| Output CSV size | 17 MB, 5,000 rows × 44 columns |
| Random Forest CV R² (sleep efficiency) | 0.73 |
| Synthetic model RMSE vs real baseline | Within 20% (Tier 2 PASS) |

---

## How to Explain This in One Minute

> "We had three datasets that had nothing to do with each other: one told us what IoT sensors look like in a real indoor environment, one told us how lifestyle factors affect sleep quality, and one told us how HVAC systems cycle. None of them had both environmental sensor data and sleep data at the same time.
>
> The insight is that sleep science already tells us *why* environment affects sleep — through arousal, melatonin suppression, and sleep fragmentation. So instead of needing a dataset that has both, we used the causal relationships from the literature as a translation layer. We used the IoT data to make our environmental signals physically realistic, then converted those environmental conditions into the equivalent of a caffeine-and-alcohol arousal signal that the sleep quality model already understood. The model was never told about temperature or light — it only sees an arousal index — but that index encodes the same physiological disruption that bad environmental conditions cause.
>
> The result is a dataset where the relationships between environment and sleep are grounded in real causal mechanisms, even though no single real dataset contains both sides of that relationship."

---

*References: Okamoto-Mizuno & Mizuno (2012), J. Physiol. Anthropol.; Zeitzer et al. (2000), J. Physiology; WHO Night Noise Guidelines (2009); Walker, Why We Sleep (2017); Dijk (2009), J. Clin. Sleep Med.*
