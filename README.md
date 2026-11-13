<div align="center">

# 📊 Research Diary: Multi-Area Time-Series Forecasting
### Chicago Taxi & Ride-Hailing Demand Prediction Pipeline

</div>

---

### **Metadata & Project Scope**
| Attribute | Specification |
| :--- | :--- |
| **Author** | Aicha Atki (`aatki`) |
| **Domain** | Spatial-Temporal Machine Learning & Predictive Modeling |
| **Dataset** | Chicago Taxi Pickups (`2019-04-01` to `2019-06-23`) |
| **Granularity** | 15-Minute Intervals across 77 Spatial Community Areas |
| **Forecasting Horizon** | 7 Days Ahead (672 consecutive steps) |
| **Target Metric** | Mean Absolute Error ($\text{MAE} \le 10.0$ on the test split) |

---

## **1. Dataset Architecture & Exploratory Data Analysis**

### **Overview**
* **Source File:** `taxi_pickups_area.csv`
* **Temporal Span:** April 1, 2019 – June 23, 2019 (~8,064 rows per area time series)
* **Spatial Dimensions:** 77 distinct regional columns (`Pickup Community Area_0` through `_77`)
* **Submission Deliverable:** `taxi_submission_file.csv` (672 target rows, initiating from `2019-06-24`)

### **Data Split Strategy**
* **Training Set:** Historical observations from index $0$ up to $N - 672$
* **Validation Set:** The final 7 days ($672$ rows) of the observed timeline
* **Test Set:** The blind 672-row forecasting submission template

### **Key Exploratory Insights**
* **Spatial Sparsity:** The majority of community zones exhibit low overall volume with means close to zero, while a select few high-density downtown hubs dominate aggregate demand.
* **Intra-Day Seasonality:** Pronounced daily cycles reflecting peak commuter travel windows (morning and evening rush hours).
* **Inter-Day Seasonality:** Distinct structural shifts between weekday traffic profiles and weekend patterns.
* **Zero-Variance Zones:** Isolated geographical sectors display near-zero variance across the entire timeline, presenting edge cases for predictive modeling.

---

## **2. Baseline Model: Naive Averages**

### **2.1. Conceptual Definition**
The Naive Average baseline acts as a foundational benchmark, assuming that future realizations will align with the historical mean of all previously recorded observations for a given series.

### **2.2. Mathematical Formulation**

Given a univariate time series comprising $n$ historical observations:

$$
Y = [y_1, y_2, y_3, \ldots, y_n]
$$

The historical average is computed as:

$$
\bar{Y} = \frac{1}{n}\sum_{t=1}^{n}y_t
$$

Consequently, the multi-step ahead forecast for any future time step $h$ evaluates to:

$$
\boxed{\hat{y}_{n+h} = \bar{Y}}
$$

**Notation Reference:**
- $y_t$: Observed demand value at time step $t$.
- $n$: Total cardinality of historical observations.
- $\bar{Y}$: Computed historical mean.
- $\hat{y}_{n+h}$: Point prediction for future step $n+h$.

---

### **2.3. Practical Demonstration**

Consider a localized sample of Uber demand observations for an individual sector:

| Time Interval | Recorded Trips ($y_t$) |
| :--- | :---: |
| **08:00** | 100 |
| **08:15** | 120 |
| **08:30** | 140 |
| **08:45** | 80 |
| **09:00** | 160 |

**Step 1:** Calculate the historical arithmetic mean ($\bar{Y}$):

$$
\bar{Y} = \frac{100 + 120 + 140 + 80 + 160}{5} = 120
$$

**Step 2:** Project the constant forecast across the horizon:

| Future Interval | Predicted Demand ($\hat{y}_{n+h}$) |
| :--- | :---: |
| **09:15** | 120 |
| **09:30** | 120 |
| **09:45** | 120 |
| **10:00** | 120 |
| **10:15** | 120 |

> **Key Takeaway:** The naive model produces a completely static, flat-line projection across the entire forecasting window.

![Naive Average Forecast](images/naive_average_forecast.png)

---

## **3. Pipeline Implementation**

Because the dataset encompasses 77 independent spatial community areas, the forecasting pipeline instantiates localized time-series models per zone:

* **Area A:** $\bar{Y} = 120 \text{ trips}$
* **Area B:** $\bar{Y} = 85 \text{ trips}$
* **Area C:** $\bar{Y} = 210 \text{ trips}$

Each area's calculated average is independently broadcast across its respective 672-step forecast horizon.

---

## **4. Advantages & Limitations**

### **Advantages**
* **Minimal Complexity:** Trivial to implement and scale across high-dimensional spatial datasets.
* **High Efficiency:** Computationally inexpensive with zero runtime training overhead.
* **Rigorous Baseline:** Establishes a performance floor required to validate advanced architectures (e.g., SARIMAX, RNN, and boosting frameworks).
* **Parameter-Free:** Requires no hyperparameter tuning or optimization routines.

### **Limitations**
* **Trend Blindness:** Incapable of capturing upward or downward trajectories.
* **Agnostic to Seasonality:** Completely misses recurring daily and weekly cycles.
* **Static Response:** Fails to react dynamically to sudden market shifts or external stimuli.
* **Intra-Day Flatness:** Generates identical outputs regardless of the time of day, limiting efficacy for urban transport networks where demand fluctuates heavily between early morning lows and rush hour peaks.
