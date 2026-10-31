import pandas as pd
import numpy as np



def parse_data(path="data/taxi_pickups_area.csv"):
    df = pd.read_csv(path)
    print(df)
    