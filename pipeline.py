import os
import pandas as pd
from feature_engineering import engineer_features, remove_collinear_features
from ingestion import assess_missingness, load_raw_data
from preprocessing import apply_iqr_winsorization, impute_missing_values


def run_pipeline():
  print("---Starting VectorFlow-EDA Pipeline Execution---")

  # 1. INPUT PHASE
  raw_df = load_raw_data("data/raw/titanic.csv")
  validated_df = assess_missingness(raw_df)

  # 2. PROCESS PHASE
  imputed_df = impute_missing_values(validated_df)
  winsorized_df = apply_iqr_winsorization(imputed_df)
  featured_df = engineer_features(winsorized_df)
  final_processed_df = remove_collinear_features(featured_df)

  # 3. OUTPUT PHASE
  os.makedirs("data/processed", exist_ok=True)
  output_filepath = "data/processed/processed_titanic.csv"

  print(f"[Output Phase] Saving processed feature matrix to: {output_filepath}")
  final_processed_df.to_csv(output_filepath, index=False)

  print("✅ Pipeline Executed Successfully!")


if __name__ == "__main__":
  run_pipeline()