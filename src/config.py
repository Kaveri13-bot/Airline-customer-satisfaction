import os
from dotenv import load_dotenv

load_dotenv()

# Data Paths
TRAIN_DATA_PATH = os.getenv("TRAIN_DATA_PATH", "data/train.csv")
TEST_DATA_PATH = os.getenv("TEST_DATA_PATH", "data/test.csv")
RAW_DATA_PATH = os.getenv("RAW_DATA_PATH", "data/Airline_info.csv")
PREDICTIONS_PATH = os.getenv("PREDICTIONS_PATH", "artifacts/predictions.csv")
MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/model.pkl")

# ML Parameters
TEST_SIZE = float(os.getenv("TEST_SIZE", 0.2))
RANDOM_STATE = int(os.getenv("RANDOM_STATE", 42))
TARGET_COLUMN = os.getenv("TARGET_COLUMN", "Satisfaction")