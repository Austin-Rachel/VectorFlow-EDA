import numpy as np
import pandas as pd


def engineer_features(df):
  """Creates interactive or ratio-based features depending on dataset attributes."""
  print("[Process Phase] Engineering new features...")
  df_fe = df.copy()

  # Generic robust feature creation template (customizable based on your specific columns)
  num_cols = df_fe.select_dtypes(include=[np.number]).columns
  if len(num_cols) >= 2:
    # Example: Creating an interaction ratio or product feature of the first two numerical columns
    col1, col2 = num_cols[0], num_cols[1]
    df_fe[f"{col1}_x_{col2}_interaction"] = df_fe[col1] * df_fe[col2]

  return df_fe


def remove_collinear_features(df, threshold=0.85):
  """Removes highly correlated features to eradicate multicollinearity."""
  print(
      f"[Process Phase] Checking and removing features with correlation >"
      f" {threshold}..."
  )
  df_numeric = df.select_dtypes(include=[np.number])

  corr_matrix = df_numeric.corr().abs()
  upper_tri = corr_matrix.where(
      np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
  )

  to_drop = [
      column for column in upper_tri.columns if any(upper_tri[column] > threshold)
  ]

  if to_drop:
    print(f"Dropping collinear features: {to_drop}")
    df_reduced = df.drop(columns=to_drop)
  else:
    print("No high multicollinearity detected.")
    df_reduced = df

  return df_reduced