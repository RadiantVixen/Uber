import pandas as pd
import numpy as np

def Naive_Averages(data, num_intervals=672):
    area_columns = [col for col in data.columns if col != 'Trip Start Timestamp']
    global_means = data[area_columns].mean()
    predictions = pd.DataFrame(np.tile(global_means.values, (num_intervals, 1)), columns=area_columns)
    predictions[predictions < 0] = 0
    return predictions


import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error

def evaluate_and_predict(train_data, val_data, submission_df, predictions):
    area_columns = [col for col in submission_df.columns if col != 'Trip Start Timestamp']
    
    val_predictions = pd.DataFrame(
        np.tile(train_data[area_columns].mean().values, (len(val_data), 1)), 
        columns=area_columns
    )
    
    mae_per_area = {}
    for area in area_columns:
        mae_per_area[area] = mean_absolute_error(val_data[area], val_predictions[area])
        
    average_mae = np.mean(list(mae_per_area.values()))
    print(f"Validation Average MAE: {average_mae:.4f}")
    
    for area in area_columns:
        submission_df[area] = predictions[area]
        
    submission_df.to_csv("data/taxi_submission_file.csv", index=False)