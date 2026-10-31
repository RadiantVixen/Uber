import argparse
import pandas as pd
import numpy as np
from parse import parse_data
from forecasting import (
    run_naive_averages,
    run_moving_averages,
    run_exponential_smoothing,
    run_arima
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--approach", type=str, required=True, choices=["naive", "moving", "exponential", "arima"])
    args = parser.parse_args()

    train_data, val_data, submission_df = parse_data("data/taxi_pickups_area.csv", "data/taxi_submission_file.csv")

    if args.approach == "naive":
        submission_df = run_naive_averages(train_data, val_data, submission_df, num_intervals=672)
    elif args.approach == "moving":
        submission_df = run_moving_averages(train_data, val_data, submission_df, num_intervals=672)
    elif args.approach == "exponential":
        submission_df = run_exponential_smoothing(train_data, val_data, submission_df, num_intervals=672)
    elif args.approach == "arima":
        submission_df = run_arima(train_data, val_data, submission_df, num_intervals=672)
        
    submission_df.to_csv("data/taxi_submission_file.csv", index=False)
    print(f"Predictions saved for {args.approach}")

if __name__ == "__main__":
    main()