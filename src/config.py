"""
Configuration for World Cup 2026 Prediction Model
"""
import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

RAW_RESULTS_PATH = os.path.join(DATA_DIR, "results.csv")
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed_matches.csv")
ELO_RATINGS_PATH = os.path.join(DATA_DIR, "elo_ratings.csv")
MODEL_PATH = os.path.join(MODELS_DIR, "match_predictor.pt")
SCALER_PATH = os.path.join(MODELS_DIR, "feature_scaler.pkl")
TEAM_ENCODER_PATH = os.path.join(MODELS_DIR, "team_encoder.pkl")

# Feature engineering
ELO_K_FACTOR = 32
ELO_HOME_ADVANTAGE = 100
ELO_INITIAL = 1500
RECENT_FORM_WINDOW = 10  # Last N matches for form calculation

# Training
TRAIN_TEST_SPLIT_DATE = "2022-01-01"
VALIDATION_START_DATE = "2024-01-01"
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 200
EARLY_STOPPING_PATIENCE = 20
HIDDEN_LAYERS = [256, 128, 64]
DROPOUT_RATE = 0.3

# World Cup 2026
WC2026_START = "2026-06-11"
WC2026_END = "2026-07-19"

# Important tournaments (higher weight/importance)
IMPORTANT_TOURNAMENTS = [
    "FIFA World Cup",
    "FIFA World Cup qualification",
    "UEFA Euro",
    "Copa América",
    "African Cup of Nations",
    "AFC Asian Cup",
    "CONCACAF Gold Cup",
    "UEFA Nations League",
    "Confederations Cup",
]
