<div align="center">

# 📊 Research Diary: Multi-Area Time-Series Forecasting
### Chicago Taxi & Ride-Hailing Demand Prediction Pipeline

</div>

---

### Metadata & Project Scope

| Attribute | Specification |
| :--- | :--- |
| **Author** | Aicha Atki (`aatki`) |
| **Domain** | Spatial-Temporal Machine Learning & Predictive Modeling |
| **Dataset** | Chicago Taxi Pickups (`2019-04-01` to `2019-06-23`) |
| **Granularity** | 15-Minute Intervals across 77 Spatial Community Areas |
| **Forecasting Horizon** | 7 Days Ahead (672 consecutive steps) |
| **Target Metric** | Mean Absolute Error (MAE ≤ 10.0 on the test split) |

---

## 1. Dataset Architecture & Exploratory Data Analysis

### Overview
- **Source File:** `taxi_pickups_area.csv`
- **Temporal Span:** April 1, 2019 – June 23, 2019 (~8,064 rows per area time series)
- **Spatial Dimensions:** 77 distinct regional columns (`Pickup Community Area_0` through `_77`)
- **Submission Deliverable:** `taxi_submission_file.csv` (672 target rows, starting from `2019-06-24`)

### Data Split Strategy
- **Training Set:** Historical observations from index 0 up to N − 672
- **Validation Set:** The final 7 days (672 rows) of the observed timeline
- **Test Set:** The blind 672-row forecasting submission template

### Key Exploratory Insights
- **Spatial Sparsity:** Most community zones have low overall volume (means close to zero); a few high-density downtown hubs dominate aggregate demand.
- **Intra-Day Seasonality:** Pronounced daily cycles reflecting peak commuter travel windows (morning and evening rush hours).
- **Inter-Day Seasonality:** Distinct structural shifts between weekday traffic profiles and weekend patterns.
- **Zero-Variance Zones:** Some geographical sectors display near-zero variance across the entire timeline, presenting edge cases for predictive modeling.

---

## Background A — Evaluation Metric: Mean Absolute Error (MAE)

### What is MAE?
The **Mean Absolute Error (MAE)** measures the average magnitude of prediction errors across the forecasting horizon, without considering direction. It is the average absolute difference between actual observed demand and predicted demand.

### Mathematical Formulation

For a sequence of n evaluation steps:

$$
\text{MAE} = \frac{1}{n} \sum_{t=1}^{n} \lvert y_t - \hat{y}_t \rvert
$$

**Notation:**
- `y_t` — actual recorded trips at time step t
- `ŷ_t` — predicted trips at time step t
- `n` — total evaluation intervals (672 steps over the 7-day test horizon)

### Why MAE for this project?
- **Direct Interpretability:** MAE is in the same units as the target (number of trips). An MAE of 8.5 means predictions deviate by an average of 8.5 trips per 15-minute interval.
- **Robustness to Outliers:** Unlike MSE, MAE treats all errors on a linear scale, preventing rare traffic surges from disproportionately skewing evaluation.
- **Project Goal:** Performance threshold of **MAE ≤ 10.0** on the test split across all areas.

---

## Background B — Model Selection: Akaike Information Criterion (AIC)

### What is AIC?
When comparing models, looking only at training error is insufficient — more complex models always fit training data better but risk overfitting. The **Akaike Information Criterion (AIC)** establishes a principled trade-off between:
1. **Goodness-of-Fit:** How well the model explains the training data.
2. **Parsimony:** A penalty for the number of estimated parameters.

### Mathematical Formulation

$$
\text{AIC} = 2k - 2\ln(L)
$$

Where:
- `k` — total number of estimated parameters in the model
- `L` — the maximized likelihood value (how probable the data is given the fitted model)
- `ln(L)` — maximized log-likelihood

**Goal:** minimize AIC across all candidates:

$$
\text{Best Model} = \arg\min_{\text{model}} \; \text{AIC}(\text{model})
$$

### Understanding Likelihood (L)

The **likelihood** L answers the question: *given the fitted model parameters, how probable is the observed training data?*

For models with normally distributed errors (which all models here assume), the Gaussian log-likelihood over n training observations is:

$$
\ln(L) = -\frac{n}{2}\ln(2\pi) - \frac{n}{2}\ln(\sigma^{2}) - \frac{1}{2\sigma^{2}} \sum_{t=1}^{n} \varepsilon_{t}^{2}
$$

Where:
- `σ²` — variance of the residual errors
- `εt = yt − ŷt` — one-step-ahead prediction error (residual) at time t

**Key intuition:**
- A model that produces small residuals (εt ≈ 0) → high likelihood L → lower AIC → preferred.
- A model with many parameters (high k) → higher AIC penalty → only wins if the fit improvement justifies the complexity.

Maximising ln(L) is mathematically equivalent to **minimising the Sum of Squared Errors (SSE)**:

$$
\max \ln(L) \;\Longleftrightarrow\; \min \sum_{t=1}^{n} \varepsilon_{t}^{2}
$$

### How AIC was applied
Across the 77 areas, the pipeline:
1. Fits every candidate model configuration per area.
2. Computes AIC for each fitted model.
3. Automatically selects the configuration with the **lowest AIC** — e.g., SES for sparse areas, Holt-Winters for high-volume seasonal corridors.

---

## Method 1 — Naive Averages

### What is the Naive Average method?
The simplest possible baseline: future demand is predicted as the **global historical mean** of the series. No temporal structure is modelled.

### Mathematical Formulation

$$
\hat{y}_{n+h} = \bar{Y} = \frac{1}{n} \sum_{t=1}^{n} y_t
$$

The same constant value is broadcast across the entire 672-step horizon.

### Practical Example

| Time Interval | Recorded Trips |
| :--- | :---: |
| 08:00 | 100 |
| 08:15 | 120 |
| 08:30 | 140 |
| 08:45 | 80 |
| 09:00 | 160 |

Historical mean: (100 + 120 + 140 + 80 + 160) / 5 = **120**

| Future Interval | Predicted Demand |
| :--- | :---: |
| 09:15 | 120 |
| 09:30 | 120 |
| 09:45 | 120 |

> **Key Takeaway:** The naive model produces a completely flat, constant projection across the entire forecasting window.

### Results

| Metric | Value |
|--------|-------|
| Validation MAE | *(fill after running)* |

### Advantages & Limitations

| ✅ Advantages | ❌ Limitations |
|---|---|
| Zero training overhead | Cannot capture trends |
| Parameter-free | Ignores daily/weekly seasonality |
| Strong lower-bound baseline | Static: same output at any time of day |

---

## Method 2 — Moving Averages

### What is the Moving Average method?
Instead of averaging all historical data, the Moving Average (MA) method computes the mean over a **fixed-size sliding window** of recent observations.

### Mathematical Formulation

Given a window size k, the Simple Moving Average at the final training step n is:

$$
\bar{y}_{n} = \frac{1}{k} \sum_{i=0}^{k-1} y_{n-i}
$$

The multi-step forecast broadcasts this recent mean flat:

$$
\hat{y}_{n+h} = \bar{y}_{n}
$$

### Window Sizes Tested

| Window (intervals) | Duration | Validation MAE |
|---|---|---|
| 48 | 12 hours | *(fill after running)* |
| 96 | 1 day | *(fill after running)* |
| 192 | 2 days | *(fill after running)* |
| 336 | 3.5 days | *(fill after running)* |
| 672 | 7 days | *(fill after running)* |

**Best window:** *(fill after running)*

### Results

| Metric | Value |
|--------|-------|
| Best window | *(fill after running)* |
| Validation MAE | *(fill after running)* |

### Advantages & Limitations

| ✅ Advantages | ❌ Limitations |
|---|---|
| Adapts to recent demand levels | Prone to lag during rapid changes |
| Simple and computationally efficient | Accuracy depends on window size choice |
| Filters short-term random noise | Still produces a flat multi-step forecast |

---

## Method 3 — Exponential Smoothing (3 Algorithms)

The subject requires testing **3 different exponential smoothing algorithms** per area, with the best selected by AIC.

---

### How Weights Are Assigned in Exponential Smoothing

Unlike moving averages which give equal weight to all observations inside the window (and zero weight outside it), exponential smoothing assigns **exponentially decreasing weights** to all past observations — the further back in time, the smaller the influence.

For a smoothing parameter α, the weight given to an observation `t` periods in the past is:

$$
w_{t} = \alpha (1-\alpha)^{t}
$$

| Lag (periods ago) | Weight (α = 0.4) |
|---|---|
| 0 (most recent) | 0.400 |
| 1 | 0.240 |
| 2 | 0.144 |
| 3 | 0.086 |
| 4 | 0.052 |
| 5 | 0.031 |

**Key intuition:**
- **α close to 1** → almost all weight on the most recent observation (reacts fast, noisy).
- **α close to 0** → weight spread broadly over history (smooth, slow to adapt).
- The optimal α is found automatically by **Maximum Likelihood Estimation (MLE)** — it is the value that minimises the sum of squared one-step-ahead errors across the training set.

### How α is Estimated (MLE / SSE minimisation)

The smoothing parameter α is chosen to maximise the Gaussian log-likelihood, which is equivalent to minimising the SSE of the one-step-ahead forecast errors:

$$
\hat{\alpha} = \arg\min_{\alpha \,\in\, [0,1]} \sum_{t=1}^{n} \bigl(y_t - \hat{y}_{t \mid t-1}\bigr)^{2}
$$

This optimisation is solved numerically (e.g., L-BFGS-B) by `statsmodels` automatically when `optimized=True`.

---

### Algorithm 1: Simple Exponential Smoothing (SES)

**When to use:** Series with no trend and no seasonality.

**Idea:** The smoothed level ℓ_t is updated at every step as a weighted blend of the new observation and the previous level.

#### Smoothing Equation

$$
\ell_{t} = \alpha \, y_{t} + (1 - \alpha) \, \ell_{t-1}
$$

#### Expanded form (shows exponential weights explicitly)

Substituting repeatedly:

$$
\ell_{t} = \alpha \, y_{t} + \alpha(1-\alpha) y_{t-1} + \alpha(1-\alpha)^{2} y_{t-2} + \cdots
$$

#### Multi-Step Forecast (flat line)

$$
\hat{y}_{n+h} = \ell_{n}
$$

**Notation:**
- `ℓ_t` — smoothed level at time t
- `α ∈ [0, 1]` — level smoothing parameter (MLE-optimised)
- `h` — forecast horizon step

#### Practical Example (α = 0.4)

| Time | Recorded | Smoothed Level |
|:---|:---:|:---:|
| 08:00 | 100 | 100.0 |
| 08:15 | 120 | 0.4(120) + 0.6(100) = 108.0 |
| 08:30 | 140 | 0.4(140) + 0.6(108) = 120.8 |
| 08:45 | 80 | 0.4(80) + 0.6(120.8) = 104.5 |
| 09:00 | 160 | 0.4(160) + 0.6(104.5) = 126.7 |

Forecast for all future steps: **126.7**

---

### Algorithm 2: Holt's Linear Trend Model (Double ES)

**When to use:** Series with a clear upward or downward trend, but no seasonality.

**Idea:** Two parameters are optimised — α for the level and β* for the trend (slope). Both are found by MLE/SSE minimisation simultaneously.

#### Level & Trend Equations

$$
\ell_{t} = \alpha \, y_{t} + (1 - \alpha)(\ell_{t-1} + b_{t-1})
$$

$$
b_{t} = \beta^{*}(\ell_{t} - \ell_{t-1}) + (1 - \beta^{*}) \, b_{t-1}
$$

#### Multi-Step Forecast (linear slope)

$$
\hat{y}_{n+h} = \ell_{n} + h \, b_{n}
$$

#### How β* controls trend smoothing

| β* value | Behaviour |
|---|---|
| Close to 1 | Trend reacts fast to recent slope changes |
| Close to 0 | Trend changes slowly, more stable long-run slope |

**MLE optimisation for both parameters:**

$$
(\hat{\alpha}, \hat{\beta}^{*}) = \operatorname*{arg\,min}_{\alpha,\, \beta^{*} \in [0,1]} \sum_{t=1}^{n} \bigl(y_t - (\ell_{t-1} + b_{t-1})\bigr)^{2}
$$

**Notation:**
- `ℓ_t` — smoothed level at time t
- `b_t` — smoothed trend (slope) at time t
- `α` — level smoothing parameter
- `β*` — trend smoothing parameter

#### Practical Example (α = 0.4, β* = 0.2)

| Time | Recorded | Level | Trend |
|:---|:---:|:---:|:---:|
| 08:00 | 100 | 100.0 | 2.0 |
| 08:15 | 120 | 109.2 | 3.44 |
| 08:30 | 140 | 136.2 | 8.15 |
| 08:45 | 80 | 118.6 | 2.04 |
| 09:00 | 160 | 104.9 | −1.1 |

| Future | Calculation | Prediction |
|:---|:---:|:---:|
| 09:15 (h=1) | 104.9 + 1(−1.1) | 103.8 |
| 09:30 (h=2) | 104.9 + 2(−1.1) | 102.7 |
| 09:45 (h=3) | 104.9 + 3(−1.1) | 101.6 |

---

### Algorithm 3: Holt-Winters Seasonal Method (Triple ES)

**When to use:** Series with both trend and recurring seasonal cycles.

**Idea:** Three smoothing equations, three MLE-optimised parameters (α, β*, γ). Seasonal period `m = 96` (96 × 15 min = 1 day).

#### Level, Trend & Seasonal Equations

$$
\ell_{t} = \alpha (y_{t} - s_{t-m}) + (1 - \alpha)(\ell_{t-1} + b_{t-1})
$$

$$
b_{t} = \beta^{*}(\ell_{t} - \ell_{t-1}) + (1 - \beta^{*}) \, b_{t-1}
$$

$$
s_{t} = \gamma (y_{t} - \ell_{t-1} - b_{t-1}) + (1 - \gamma) \, s_{t-m}
$$

#### Multi-Step Forecast

$$
\hat{y}_{n+h} = \ell_{n} + h \, b_{n} + s_{n+h-m \cdot (k+1)}
$$

*(where k is the integer part of (h−1)/m, so the right seasonal index from the last observed cycle is retrieved)*

#### How γ controls seasonal smoothing

| γ value | Behaviour |
|---|---|
| Close to 1 | Seasonal pattern updates quickly (adapts to shifting peaks) |
| Close to 0 | Seasonal pattern is very stable across all cycles |

**MLE optimisation for all three parameters:**

$$
(\hat{\alpha}, \hat{\beta}^{*}, \hat{\gamma}) = \arg\min_{\alpha,\, \beta^{*},\, \gamma \,\in\, [0,1]} \sum_{t=1}^{n} \bigl(y_t - \hat{y}_{t \mid t-1}\bigr)^{2}
$$

**Notation:**
- `s_t` — seasonal component at time t
- `γ ∈ [0, 1]` — seasonal smoothing parameter
- `m = 96` — seasonal period (one day at 15-min granularity)

---

### AIC-Based Selection Summary

For each of the 77 areas, all three models are fitted and the one with the **lowest AIC** is selected:

| Area type | Expected winner |
|---|---|
| Near-zero / sparse | SES |
| Trending, low volume | Holt |
| High-traffic with daily peaks | Holt-Winters |

### Results

| Metric | Value |
|--------|-------|
| Validation MAE | *(fill after running)* |
| Most-selected algorithm | *(fill after running)* |

---

## Method 4 — SARIMA (AIC-based model selection)

### What is SARIMA?
**SARIMA (Seasonal ARIMA)** extends the standard ARIMA framework with four additional seasonal parameters, making it capable of capturing both short-term autocorrelation and recurring periodic cycles.

Notation: **ARIMA(p, d, q)(P, D, Q)_m**

| Parameter | Meaning |
|---|---|
| `p` | Non-seasonal autoregressive order |
| `d` | Non-seasonal differencing order |
| `q` | Non-seasonal moving average order |
| `P` | Seasonal autoregressive order |
| `D` | Seasonal differencing order |
| `Q` | Seasonal moving average order |
| `m` | Seasonal period (m = 96 for 15-min intervals, 1-day cycle) |

### Mathematical Formulation

Using backshift operator B (where B^k · y_t = y_{t-k}):

$$
\Phi(\text{B}^{m}) \, \phi(\text{B}) \cdot (1 - \text{B}^{m})^{D} (1 - \text{B})^{d} \cdot y_t = c + \Theta(\text{B}^{m}) \, \theta(\text{B}) \, \varepsilon_t
$$

Where:
- `φ(B)` and `θ(B)` — non-seasonal AR and MA polynomials
- `Φ(B^m)` and `Θ(B^m)` — seasonal AR and MA polynomials
- `(1−B)^d` and `(1−B^m)^D` — non-seasonal and seasonal differencing operators

### How SARIMA Parameters Are Estimated (MLE via Kalman Filter)

SARIMA parameters cannot be solved in closed form. Instead, `statsmodels` uses a **state-space representation** and the **Kalman Filter** to recursively compute the one-step-ahead prediction errors (innovations εt), then maximises the Gaussian log-likelihood:

$$
\ln(L) = -\frac{n'}{2}\ln(2\pi) - \frac{n'}{2}\ln(\sigma^{2}) - \frac{1}{2\sigma^{2}} \sum_{t=1}^{n'} \varepsilon_{t}^{2}
$$

Where `n'` is the number of observations after differencing. The optimizer (BFGS) adjusts all AR and MA coefficients simultaneously to maximise this likelihood.

**AIC is then computed** for each candidate order and the best is selected:

$$
\text{AIC} = 2k - 2\ln(L), \quad k = p + q + P + Q + 1
$$

### Candidate Grid Searched (by AIC)

| Order (p,d,q) | Seasonal (P,D,Q,m) | Description |
|---|---|---|
| (1,1,1) | (0,0,0,0) | Plain ARIMA — general baseline |
| (1,1,1) | (1,0,1,96) | SARIMA — daily seasonality |
| (2,1,2) | (0,0,0,0) | Higher-order ARIMA |
| (1,0,1) | (0,0,0,0) | Stationary ARMA |

The model with the lowest AIC is selected **per area independently**.

### Results

| Metric | Value |
|--------|-------|
| Validation MAE | *(fill after running)* |
| Most-selected order | *(fill after running)* |

> ⚠️ **Runtime note:** SARIMA fitting across 77 areas (~4 candidate models × 2 fits per area) takes approximately 10–30 minutes on CPU.

### Advantages & Limitations

| ✅ Advantages | ❌ Limitations |
|---|---|
| Captures autocorrelation and seasonality simultaneously | Slow to fit on CPU |
| Rigorous statistical grounding (MLE + Kalman Filter) | Fixed seasonal period assumption |
| AIC selection prevents overfitting | May struggle on near-zero sparse series |

---

## Final Results Summary

| # | Method | Validation MAE | Beats target (≤ 10.0) |
|---|---|---|---|
| 1 | Naive Averages | *(fill)* | *(fill)* |
| 2 | Moving Averages | *(fill)* | *(fill)* |
| 3 | Exponential Smoothing | *(fill)* | *(fill)* |
| 4 | SARIMA | *(fill)* | *(fill)* |

**Best approach:** *(fill after running)*  
**Submission saved with:** *(fill after running)*

---

## Conclusion

*(Fill after running all experiments)*

**Key takeaways:**
-
-
-

**Potential improvements (bonus):**
- **Classical ML:** Gradient-boosted trees with calendar features (hour, weekday, is_weekend, etc.)
- **Deep Learning:** LSTM / GRU encoder on the raw time series
- **Ensemble:** Average predictions from the best 2–3 methods above
