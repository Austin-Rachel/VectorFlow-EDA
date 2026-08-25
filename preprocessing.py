import numpy as np
import pandas as pd


def impute_missing_values(df):
  """Imputes numerical columns with median and categorical columns with mode."""
  print("[Process Phase] Imputing missing values...")
  df_clean = df.copy()

  for col in df_clean.select_dtypes(include=[np.number]).columns:
    if df_clean[col].isnull().any():
      median_val = df_clean[col].median()
      df_clean[col].fillna(median_val, inplace=True)

  for col in df_clean.select_dtypes(include=["object", "category"]).columns:
    if df_clean[col].isnull().any():
      mode_val = df_clean[col].mode()[0]
      df_clean[col].fillna(mode_val, inplace=True)

  return df_clean


def apply_iqr_winsorization(df, threshold=1.5):
  """Caps outliers using the Interquartile Range (IQR) method."""
  print(f"[Process Phase] Applying IQR winsorization (threshold={threshold})...")
  df_winsorized = df.copy()

  num_cols = df_winsorized.select_dtypes(include=[np.number]).columns

  for col in num_cols:
    Q1 = df_winsorized[col].quantile(0.25)
    Q3 = df_winsorized[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - (threshold * IQR)
    upper_bound = Q3 + (threshold * IQR)

    df_winsorized[col] = np.clip(
        df_winsorized[col], lower_bound, upper_bound
    )

  return df_winsorized