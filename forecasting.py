import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX


def run_naive_averages(train_data, val_data, submission_df, num_intervals=672):
    """Predict the global mean of each area for every future interval."""
    area_columns = [col for col in train_data.columns if col != 'Trip Start Timestamp']
    global_means = train_data[area_columns].mean()

    val_predictions = pd.DataFrame(np.tile(global_means.values, (len(val_data), 1)), columns=area_columns)
    test_predictions = pd.DataFrame(np.tile(global_means.values, (num_intervals, 1)), columns=area_columns)
    test_predictions[test_predictions < 0] = 0

    mae = np.mean([mean_absolute_error(val_data[col], val_predictions[col]) for col in area_columns])
    print(f"Naive Averages Validation MAE: {mae:.4f}")

    sub_df = submission_df.copy()
    for col in area_columns:
        sub_df[col] = test_predictions[col].values
    return sub_df, mae



def run_moving_averages(train_data, val_data, submission_df, num_intervals=672, window=96):
    """Predict each area using the last rolling-window mean of the training series."""
    area_columns = [col for col in train_data.columns if col != 'Trip Start Timestamp']

    val_preds = {}
    test_preds = {}

    for area in area_columns:
        rolling_mean = train_data[area].rolling(window=window).mean().iloc[-1]
        if np.isnan(rolling_mean):
            rolling_mean = train_data[area].mean()
        val_preds[area] = [rolling_mean] * len(val_data)
        test_preds[area] = [rolling_mean] * num_intervals

    val_preds_df = pd.DataFrame(val_preds)
    test_preds_df = pd.DataFrame(test_preds)
    test_preds_df[test_preds_df < 0] = 0

    mae = np.mean([mean_absolute_error(val_data[col], val_preds_df[col]) for col in area_columns])
    print(f"Moving Averages Validation MAE: {mae:.4f}")

    sub_df = submission_df.copy()
    for col in area_columns:
        sub_df[col] = test_preds_df[col].values
    return sub_df, mae



def _best_es_forecast(series, steps):
    """
    Try three exponential-smoothing variants and pick the one with the lowest
    in-sample AIC:
        - Simple (SES)  : no trend, no seasonality
        - Double (Holt) : additive trend, no seasonality
        - Triple (H-W)  : additive trend + additive seasonal (period=96 = 1 day)
    Returns the forecast array for `steps` periods.
    """
    best_aic = np.inf
    best_forecast = None

    configs = [
        dict(trend=None,  seasonal=None,     seasonal_periods=None),
        dict(trend='add', seasonal=None,     seasonal_periods=None),
        dict(trend='add', seasonal='add',    seasonal_periods=96),
    ]

    for cfg in configs:
        try:
            m = ExponentialSmoothing(
                series,
                trend=cfg['trend'],
                seasonal=cfg['seasonal'],
                seasonal_periods=cfg['seasonal_periods'],
                initialization_method="estimated",
            ).fit(optimized=True)
            if m.aic < best_aic:
                best_aic = m.aic
                best_forecast = m.forecast(steps)
        except Exception:
            pass

    if best_forecast is None:
        best_forecast = np.full(steps, series.mean())
    return best_forecast


def run_exponential_smoothing(train_data, val_data, submission_df, num_intervals=672):
    """
    For each area, pick the best of three ES variants (SES / Holt / Holt-Winters)
    by AIC and forecast num_intervals periods.
    """
    area_columns = [col for col in train_data.columns if col != 'Trip Start Timestamp']

    val_preds = {}
    test_preds = {}

    for area in area_columns:
        val_preds[area] = _best_es_forecast(train_data[area], len(val_data))
        full_series = pd.concat([train_data[area], val_data[area]], ignore_index=True)
        test_preds[area] = _best_es_forecast(full_series, num_intervals)

    val_preds_df = pd.DataFrame(val_preds)
    test_preds_df = pd.DataFrame(test_preds)
    test_preds_df[test_preds_df < 0] = 0

    mae = np.mean([mean_absolute_error(val_data[col].values, val_preds_df[col].values)
                   for col in area_columns])
    print(f"Exponential Smoothing (best of SES/Holt/H-W) Validation MAE: {mae:.4f}")

    sub_df = submission_df.copy()
    for col in area_columns:
        sub_df[col] = test_preds_df[col].values
    return sub_df, mae



def run_sarima(train_data, val_data, submission_df, num_intervals=672):
    """
    Fit a single SARIMA model per area and forecast num_intervals periods.
    """
    area_columns = [col for col in train_data.columns if col != 'Trip Start Timestamp']

    val_preds = {}
    test_preds = {}

    for i, area in enumerate(area_columns):
        print(f"  SARIMA fitting area {i+1}/{len(area_columns)}: {area}", end='\r')
        series = train_data[area]
        full_series = pd.concat([train_data[area], val_data[area]], ignore_index=True)
        
        try:
            model = SARIMAX(
                series,
                order=(1, 1, 1),
                seasonal_order=(1, 0, 1, 96),
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False)
            
            val_preds[area] = model.forecast(len(val_data))
            
            full_model = SARIMAX(
                full_series,
                order=(1, 1, 1),
                seasonal_order=(1, 0, 1, 96),
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False)
            
            test_preds[area] = full_model.forecast(num_intervals)
            
        except Exception:
            mean_val = series.mean()
            val_preds[area] = np.full(len(val_data), mean_val)
            test_preds[area] = np.full(num_intervals, mean_val)

    print()
    val_preds_df = pd.DataFrame(val_preds)
    test_preds_df = pd.DataFrame(test_preds)
    test_preds_df[test_preds_df < 0] = 0

    mae = np.mean([mean_absolute_error(val_data[col].values, val_preds_df[col].values)
                   for col in area_columns])
    print(f"SARIMA Validation MAE: {mae:.4f}")

    sub_df = submission_df.copy()
    for col in area_columns:
        sub_df[col] = test_preds_df[col].values
    return sub_df, mae



def run_arima(train_data, val_data, submission_df, num_intervals=672):
    return run_sarima(train_data, val_data, submission_df, num_intervals)

