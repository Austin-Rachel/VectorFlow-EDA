# VectorFlow-EDA

## Overview
VectorFlow-EDA is a production-ready, modular data engineering pipeline designed to execute advanced exploratory data analysis (EDA) and feature engineering using a rigorous Input-Process-Output (IPO) architecture.

## Project Structure & Architecture
The project follows a clean, decoupled layout to maintain separation of concerns:
* **`data/`**: Storage repository containing raw datasets (`data/raw/`) and generated feature matrices (`data/processed/`).
* **`notebooks/`**: Houses exploratory notebooks like `exploratory_eda.ipynb` for iterative analysis.
* **`src/`**: Core modular package containing individual execution scripts:
  * `ingestion.py`: Phase 1 data loading and missingness assessment.
  * `preprocessing.py`: IQR winsorization and transformations.
  * `feature_engineering.py`: Collinearity checks and feature creation.
  * `pipeline.py`: Orchestrator for end-to-end execution.
* **`tests/`**: Automated testing suite (`test_pipeline.py`) ensuring schema and data contract validity.
* **`outputs/`**: Target directory for generated artifacts, logs, and reporting outputs.

## Setup & Execution
1. Install project dependencies:
   ```bash
   pip install -r requirements.txt