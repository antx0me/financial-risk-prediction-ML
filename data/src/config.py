"""
Configuration file for Credit Risk Prediction Project
-----------------------------------------------------

This file stores all important global configurations:
- File paths
- Model parameters
- Random seeds
"""

import os

# Base project directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data paths
DATA_DIR = os.path.join(BASE_DIR, "data")

RAW_DATA_PATH = os.path.join(DATA_DIR, "raw_data.csv")
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed_data.csv")

# Model storage paths
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOGISTIC_MODEL_PATH = os.path.join(MODELS_DIR, "logistic_model.pkl")
RANDOM_FOREST_MODEL_PATH = os.path.join(MODELS_DIR, "random_forest.pkl")

# ML settings
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Feature settings
TARGET_COLUMN = "creditworthy"  # 1 = good, 0 = bad

