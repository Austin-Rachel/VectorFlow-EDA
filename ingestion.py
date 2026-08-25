import os
import pandas as pd


def load_raw_data(filepath="data/raw/titanic.csv"):
  """Loads the raw dataset from disk with robust path checking."""
  if not os.path.exists(filepath):
    raise FileNotFoundError(
        f"Raw dataset not found at '{filepath}'. Please place your CSV file"
        " there."
    )

  print(f"[Input Phase] Loading raw dataset from: {filepath}")
  df = pd.read_csv(filepath)
  return df


def assess_missingness(df):
  """Evaluates and prints missing value statistics across features."""
  print("[Input Phase] Assessing data missingness...")
  missing_counts = df.isnull().sum()
  missing_pct = (df.isnull().mean() * 100).round(2)

  report = pd.DataFrame({"Missing Count": missing_counts, "Missing %": missing_pct})
  report = report[report["Missing Count"] > 0].sort_values(
      by="Missing %", ascending=False
  )

  if not report.empty:
    print("\nMissing Values Summary:")
    print(report)
  else:
    print("No missing values detected in the dataset.")

  return df