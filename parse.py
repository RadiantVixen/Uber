import pandas as pd
import numpy as np


def parse_data(path="data/taxi_pickups_area.csv", submission_path="data/taxi_submission_file.csv"):
    """
    Load the taxi pickups dataset and split into train / validation sets.

    Validation size matches the number of rows in the submission file (672),
    which corresponds to 7 days of 15-minute intervals.
    """
    df = pd.read_csv(path)

    if 'Trip Start Timestamp' in df.columns:
        df['Trip Start Timestamp'] = pd.to_datetime(df['Trip Start Timestamp'])
        df = df.sort_values('Trip Start Timestamp').reset_index(drop=True)

    val_size = 672
    train_data = df.iloc[:-val_size].copy()
    val_data = df.iloc[-val_size:].copy()

    try:
        submission_df = pd.read_csv(submission_path)
        submission_df['Trip Start Timestamp'] = pd.to_datetime(submission_df['Trip Start Timestamp'])
    except FileNotFoundError:
        submission_df = None

    print(f"Full Dataset Shape:       {df.shape}")
    print(f"Train Data Shape:         {train_data.shape}")
    print(f"Validation Data Shape:    {val_data.shape}")
    if submission_df is not None:
        print(f"Submission Template Shape: {submission_df.shape}")

    return train_data, val_data, submission_df