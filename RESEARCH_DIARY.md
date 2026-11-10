# Research Diary — Uber Taxi Demand Forecasting

**Author:** *(your name)*  
**Dataset:** Chicago taxi pickups — 2019-04-01 → 2019-06-23  
**Granularity:** 15-minute intervals | **Areas:** 77  
**Task:** Forecast the next 7 days (672 intervals) of taxi demand per area  
**Target metric:** Average MAE ≤ 10.0 on the test set

---

## Dataset Overview

| Property | Value |
|----------|-------|
| File | `taxi_pickups_area.csv` |
| Date range | 2019-04-01 → 2019-06-23 |
| Total rows | ~8 064 |
| Areas (columns) | 77 (`Pickup Community Area_0` … `_77`) |
| Interval | 15 minutes |
| Submission file | `taxi_submission_file.csv` (672 rows, starts 2019-06-24) |

**Train / Validation split:**  
- Train : first N − 672 rows  
- Validation : last 672 rows (7 days)  
- Test : 672 rows from the submission template

**Key observations from EDA:**
- Most areas have sparse demand (mean close to 0); a few downtown areas dominate
- Clear daily seasonality (peaks during morning/evening rush hours)
- Weekly seasonality visible (weekends differ from weekdays)
- Some areas have near-zero variance across the entire dataset

---

## Approach 1 — Naive Averages

### Description
For each area, the global mean of the entire training series is used as the prediction for every future interval. No temporal structure is modelled at all.

### Formula
$$\hat{y}_{t+h} = \bar{y} = \frac{1}{T}\sum_{t=1}^{T} y_t$$

### Implementation notes
- Simple `train_data[area].mean()` per column
- Negative predictions clipped to 0 (counts can't be negative)

### Results

| Metric | Value |
|--------|-------|
| Validation MAE | *(fill after running)* |

### Observations
- Establishes a hard upper-bound baseline — any real model should beat this
- Completely ignores trends and seasonality
- Adequate for areas with near-constant low demand

---

## Approach 2 — Moving Averages

### Description
For each area, the mean of the last `window` training intervals is used as a flat forecast. Several window sizes are tested to find the optimal one.

### Windows tested

| Window (intervals) | Duration | Validation MAE |
|--------------------|----------|----------------|
| 48 | 12 hours | *(fill after running)* |
| 96 | 1 day | *(fill after running)* |
| 192 | 2 days | *(fill after running)* |
| 336 | 3.5 days | *(fill after running)* |
| 672 | 7 days | *(fill after running)* |

**Best window:** *(fill after running)*

### Implementation notes
- `train_data[area].rolling(window=w).mean().iloc[-1]`
- Falls back to global mean if the rolling window contains NaNs

### Results

| Metric | Value |
|--------|-------|
| Best window | *(fill after running)* |
| Validation MAE | *(fill after running)* |

### Observations
- Shorter windows react faster to recent demand shifts
- Longer windows are more stable but lag behind sudden changes
- Outperforms naive averages when demand has drifted recently

---

## Approach 3 — Exponential Smoothing

### Description
Three exponential-smoothing variants are fitted per area. The one with the lowest **AIC** (Akaike Information Criterion) is selected as the final model. All weights are optimised automatically.

### The three algorithms

| Algorithm | Trend | Seasonality | Seasonal period | Notes |
|-----------|-------|-------------|-----------------|-------|
| **SES** (Simple) | None | None | — | Single smoothing parameter α |
| **Holt** (Double ES) | Additive | None | — | Adds level + trend (α, β) |
| **Holt-Winters** (Triple ES) | Additive | Additive | 96 (= 1 day) | Adds seasonal component (α, β, γ) |

### Model selection criterion
AIC = −2 log(L) + 2k  
*(lower is better; penalises complexity)*

### Implementation notes
- `statsmodels.tsa.holtwinters.ExponentialSmoothing`
- `initialization_method='estimated'` — initial values fitted from data
- `optimized=True` — smoothing parameters found by numerical optimisation
- Forecasts for test set are produced by re-fitting on train + validation data

### Results

| Metric | Value |
|--------|-------|
| Validation MAE | *(fill after running)* |
| Most-selected algorithm | *(fill after running)* |

### Observations
- Holt-Winters is expected to win for high-demand areas with clear daily cycles
- SES tends to win for low-variance / near-zero areas
- AIC selection avoids over-fitting on sparse series

---

## Approach 4 — SARIMA (AIC-based model selection)

### Description
SARIMA (Seasonal ARIMA) extends ARIMA with four additional seasonal parameters. For each area, a small candidate grid is searched and the model with the lowest **AIC** is selected.

### SARIMA parameters

| Parameter | Meaning |
|-----------|---------|
| p | Autoregressive order |
| d | Differencing order (makes series stationary) |
| q | Moving-average order |
| P | Seasonal AR order |
| D | Seasonal differencing order |
| Q | Seasonal MA order |
| m | Number of time steps per seasonal period |

**Seasonal period chosen: m = 96** (96 × 15 min = 1 day)

### Candidate grid searched

| Order (p,d,q) | Seasonal (P,D,Q,m) | Description |
|--------------|-------------------|-------------|
| (1,1,1) | (0,0,0,0) | Plain ARIMA — good general baseline |
| (1,1,1) | (1,0,1,96) | SARIMA — daily seasonality captured |
| (2,1,2) | (0,0,0,0) | Higher-order ARIMA — more complex dynamics |
| (1,0,1) | (0,0,0,0) | Stationary ARMA — no differencing |

### Implementation notes
- `statsmodels.tsa.statespace.sarimax.SARIMAX`
- `enforce_stationarity=False`, `enforce_invertibility=False` — prevents fitting errors on flat series
- Fallback to series mean if all candidates fail (e.g., constant or all-zero series)
- Computation time: ~10–30 min on CPU (77 areas × 4 candidates × 2 fits)

### Results

| Metric | Value |
|--------|-------|
| Validation MAE | *(fill after running)* |
| Most-selected order | *(fill after running)* |

### Observations
- Best suited for areas with complex autocorrelation structure
- (1,1,1)(1,0,1,96) expected to perform best on high-traffic areas with strong daily patterns
- Slower to train than exponential smoothing but can capture more dynamics

---

## Final Comparison

| # | Method | Validation MAE | Beats target (≤ 10.0) |
|---|--------|---------------|----------------------|
| 1 | Naive Averages | *(fill)* | *(fill)* |
| 2 | Moving Averages | *(fill)* | *(fill)* |
| 3 | Exponential Smoothing | *(fill)* | *(fill)* |
| 4 | SARIMA | *(fill)* | *(fill)* |

**Best approach:** *(fill after running)*  
**Submission saved with:** *(fill after running)*

---

## Conclusion

*(Fill after running all experiments)*

Key takeaways:
- 
- 
- 

Possible improvements (bonus):
- **Classical ML:** gradient-boosted trees with calendar features (hour, weekday, etc.)
- **Deep Learning:** LSTM / GRU encoder on the raw time series
- **Ensemble:** average predictions from the best 2–3 models above

