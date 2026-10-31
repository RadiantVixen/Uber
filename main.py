import argparse
import pandas as pd
import numpy as np
from parse import parse_data
from Averages_calculation import Naive_Averages

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--approach", type=str, required=True, choices=["naive", "moving", "exponential", "arima"])
    args = parser.parse_args()

    train_data, val_data, submission_df = parse_data("data/taxi_pickups_area.csv", "data/taxi_submission_file.csv")
    area_columns = [col for col in submission_df.columns if col != 'Trip Start Timestamp']

    if args.approach == "naive":
        predictions = Naive_Averages(train_data, num_intervals=672)
    elif args.approach == "moving":
        raise NotImplementedError()
    elif args.approach == "exponential":
        raise NotImplementedError()
    elif args.approach == "arima":
        raise NotImplementedError()

    for area in area_columns:
        submission_df[area] = predictions[area]
        
    submission_df.to_csv("data/taxi_submission_file.csv", index=False)
    print(f"Predictions saved for {args.approach}")

if __name__ == "__main__":
    main()