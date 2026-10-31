import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

def run_naive_averages(train_data, val_data, submission_df, num_intervals=672):
    area_columns = [col for col in train_data.columns if col != 'Trip Start Timestamp']
    global_means = train_data[area_columns].mean()
    
    val_predictions = pd.DataFrame(np.tile(global_means.values, (len(val_data), 1)), columns=area_columns)
    test_predictions = pd.DataFrame(np.tile(global_means.values, (num_intervals, 1)), columns=area_columns)
    test_predictions[test_predictions < 0] = 0
    
    mae = np.mean([mean_absolute_error(val_data[col], val_predictions[col]) for col in area_columns])
    print(f"Naive Averages Validation MAE: {mae:.4f}")
    
    sub_df = submission_df.copy()
    for col in area_columns:
        sub_df[col] = test_predictions[col]
    return sub_df


def run_moving_averages(train_data, val_data, submission_df, num_intervals=672, window=96):
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
        sub_df[col] = test_preds_df[col]
    return sub_df



def run_exponential_smoothing(train_data, val_data, submission_df, num_intervals=672):
    area_columns = [col for col in train_data.columns if col != 'Trip Start Timestamp']
    
    val_preds = {}
    test_preds = {}
    
    for area in area_columns:
        try:
            model = ExponentialSmoothing(train_data[area], trend='add', seasonal=None, initialization_method="estimated").fit()
            val_preds[area] = model.forecast(len(val_data))
            full_model = ExponentialSmoothing(pd.concat([train_data[area], val_data[area]]), trend='add', seasonal=None, initialization_method="estimated").fit()
            test_preds[area] = full_model.forecast(num_intervals)
        except Exception:
            mean_val = train_data[area].mean()
            val_preds[area] = [mean_val] * len(val_data)
            test_preds[area] = [mean_val] * num_intervals
            
    val_preds_df = pd.DataFrame(val_preds)
    test_preds_df = pd.DataFrame(test_preds)
    test_preds_df[test_preds_df < 0] = 0
    
    mae = np.mean([mean_absolute_error(val_data[col], val_preds_df[col]) for col in area_columns])
    print(f"Exponential Smoothing Validation MAE: {mae:.4f}")
    
    sub_df = submission_df.copy()
    for col in area_columns:
        sub_df[col] = test_preds_df[col]
    return sub_df


def run_arima(train_data, val_data, submission_df, num_intervals=672):
    area_columns = [col for col in train_data.columns if col != 'Trip Start Timestamp']
    
    val_preds = {}
    test_preds = {}
    
    for area in area_columns:
        try:
            model = SARIMAX(train_data[area], order=(1, 1, 1), seasonal_order=(0, 0, 0, 0), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
            val_preds[area] = model.forecast(len(val_data))
            full_model = SARIMAX(pd.concat([train_data[area], val_data[area]]), order=(1, 1, 1), seasonal_order=(0, 0, 0, 0), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
            test_preds[area] = full_model.forecast(num_intervals)
        except Exception:
            mean_val = train_data[area].mean()
            val_preds[area] = [mean_val] * len(val_data)
            test_preds[area] = [mean_val] * num_intervals
            
    val_preds_df = pd.DataFrame(val_preds)
    test_preds_df = pd.DataFrame(test_preds)
    test_preds_df[test_preds_df < 0] = 0
    
    mae = np.mean([mean_absolute_error(val_data[col], val_preds_df[col]) for col in area_columns])
    print(f"ARIMA Validation MAE: {mae:.4f}")
    
    sub_df = submission_df.copy()
    for col in area_columns:
        sub_df[col] = test_preds_df[col]
    return sub_df