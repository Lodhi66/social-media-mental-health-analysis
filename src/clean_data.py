from pathlib import Path

import numpy as np
import pandas as pd

RAW_DATA_PATH = Path("data/raw/mental_health_dirty.csv")
CLEANED_DATA_PATH = Path("data/processed/mental_health_cleaned.csv")
KMEANS_DATA_PATH = Path("data/processed/kmeans_ready.csv")

NUMERIC_COLUMNS = [
    "Age",
    "Daily_Screen_Time_hrs",
    "Sleep_Quality_1_10",
    "Stress_Level_1_10",
    "Days_Without_Social_Media",
    "Exercise_Frequency_week",
]

PLATFORMS = [
    "Instagram",
    "TikTok",
    "Facebook",
    "YouTube",
    "LinkedIn",
    "X (Twitter)",
]


def clean_numeric_range(
    df, column, min_value, max_value, fill_value="median", as_int=False
):
    """Convert a column to numeric values and replace values outside a valid range."""
    df[column] = pd.to_numeric(df[column], errors="coerce")
    df.loc[(df[column] < min_value) | (df[column] > max_value), column] = np.nan

    if fill_value == "median":
        df[column] = df[column].fillna(df[column].median())

    if as_int:
        df[column] = df[column].astype(int)

    return df


def normalize_columns(df, columns):
    """Normalize selected numeric columns to a 0-1 range."""
    df_normalized = df.copy()

    for column in columns:
        min_value = df_normalized[column].min()
        max_value = df_normalized[column].max()

        if max_value != min_value:
            df_normalized[column] = (df_normalized[column] - min_value) / (
                max_value - min_value
            )

    return df_normalized


def clean_dataset(df):
    """Clean invalid values and prepare the mental health dataset for analysis."""
    df = df.drop(columns=["User_ID"])

    df = df[(df["Age"] >= 10) & (df["Age"] <= 100)]

    df["Gender"] = df["Gender"].replace({"M": "Male", "F": "Female"})
    df = df[df["Gender"].isin(["Male", "Female", "Other"])]
    df["Gender"] = df["Gender"].astype("category")

    df = clean_numeric_range(df, "Daily_Screen_Time_hrs", 0, 24)
    df = clean_numeric_range(df, "Sleep_Quality_1_10", 0, 10, as_int=True)
    df = clean_numeric_range(df, "Stress_Level_1_10", 0, 10, as_int=True)
    df = clean_numeric_range(df, "Days_Without_Social_Media", 0, 7, as_int=True)
    df = clean_numeric_range(df, "Exercise_Frequency_week", 0, 7, as_int=True)
    df = clean_numeric_range(df, "Happiness_Index_3highest", 1, 3, as_int=True)

    df.loc[~df["Social_Media_Platform"].isin(PLATFORMS), "Social_Media_Platform"] = (
        np.nan
    )
    df["Social_Media_Platform"] = df["Social_Media_Platform"].fillna(
        df["Social_Media_Platform"].mode()[0]
    )
    df["Social_Media_Platform"] = df["Social_Media_Platform"].astype("category")

    df["Happiness_Index_3highest"] = df["Happiness_Index_3highest"].astype("category")

    return df


def main():
    """Load, clean, normalize, and save datasets for later analysis."""
    print("Loading raw dataset...")
    df = pd.read_csv(RAW_DATA_PATH)

    print("Cleaning dataset...")
    cleaned_df = clean_dataset(df)

    print("Normalizing numeric columns...")
    normalized_df = normalize_columns(cleaned_df, NUMERIC_COLUMNS)

    kmeans_df = normalized_df[NUMERIC_COLUMNS]

    CLEANED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    cleaned_df.to_csv(CLEANED_DATA_PATH, index=False)
    kmeans_df.to_csv(KMEANS_DATA_PATH, index=False)

    print("Cleaning complete.")
    print(f"Cleaned dataset saved to: {CLEANED_DATA_PATH}")
    print(f"K-means dataset saved to: {KMEANS_DATA_PATH}")


if __name__ == "__main__":
    main()
