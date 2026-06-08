import os
import numpy as np
import pandas as pd

DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "train.csv"
)


def load_mercedes(path=DATA_PATH):
    return pd.read_csv(path)


def get_measurement(df):
    return df["y"].copy()


def get_configurations(df):
    return df["X0"].copy()


def get_ooc_by_config(df, USL=130, LSL=75):
    df = df.copy()
    df["ooc"] = ((df["y"] > USL) | (df["y"] < LSL)).astype(int)
    grouped = (
        df.groupby("X0")
        .agg(
            total=("y", "count"),
            ooc_count=("ooc", "sum"),
            mean_y=("y", "mean"),
            std_y=("y", "std"),
        )
        .reset_index()
    )
    grouped["ooc_rate"] = grouped["ooc_count"] / grouped["total"]
    grouped = grouped.sort_values("ooc_count", ascending=False).reset_index(drop=True)
    return grouped


def generate_synthetic(n=500, mean=100, std=12, shift_at=None, shift_size=1.5):
    rng = np.random.default_rng(42)
    data = rng.normal(mean, std, n)
    if shift_at is not None and 0 <= shift_at < n:
        data[shift_at:] = data[shift_at:] + shift_size * std
    return pd.Series(data, name="y")
