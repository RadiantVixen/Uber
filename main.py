import argparse
import pandas as pd
import numpy as np
from parse import parse_data
from forecasting import (
    run_naive_averages,
    run_moving_averages,
    run_exponential_smoothing,
    run_sarima,
    run_arima,
)

NUM_INTERVALS = 672

def main():
    parser = argparse.ArgumentParser(description="Taxi demand forecasting")
    parser.add_argument(
        "--approach",
        type=str,
        required=True,
        choices=["naive", "moving", "exponential", "sarima", "arima", "ml", "all"],
        help="Forecasting approach to run. Use 'all' to benchmark every method.",
    )
    args = parser.parse_args()

    train_data, val_data, submission_df = parse_data(
        "data/taxi_pickups_area.csv",
        "data/taxi_submission_file.csv",
    )

    results = {}

    if args.approach in ("naive", "all"):
        sub, mae = run_naive_averages(train_data, val_data, submission_df, NUM_INTERVALS)
        results["naive"] = (sub, mae)

    if args.approach in ("moving", "all"):
        sub, mae = run_moving_averages(train_data, val_data, submission_df, NUM_INTERVALS)
        results["moving"] = (sub, mae)

    if args.approach in ("exponential", "all"):
        sub, mae = run_exponential_smoothing(train_data, val_data, submission_df, NUM_INTERVALS)
        results["exponential"] = (sub, mae)

    if args.approach in ("sarima", "arima", "all"):
        sub, mae = run_sarima(train_data, val_data, submission_df, NUM_INTERVALS)
        results["sarima"] = (sub, mae)

    if args.approach == "all":
        print("\n--- Summary ---")
        for name, (_, mae) in sorted(results.items(), key=lambda x: x[1][1]):
            print(f"  {name:15s} MAE = {mae:.4f}")
        best_name = min(results, key=lambda k: results[k][1])
        best_sub, best_mae = results[best_name]
        print(f"\nBest approach: {best_name} (MAE={best_mae:.4f})")
        best_sub.to_csv("data/taxi_submission_file.csv", index=False)
        print(f"Submission saved with predictions from '{best_name}'.")
    else:
        chosen = args.approach if args.approach != "arima" else "sarima"
        sub, mae = results[chosen]
        sub.to_csv("data/taxi_submission_file.csv", index=False)
        print(f"Predictions saved for '{chosen}' (MAE={mae:.4f}).")


if __name__ == "__main__":
    main()
