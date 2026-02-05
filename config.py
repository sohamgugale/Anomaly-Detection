"""Configuration file for fraud detection system"""

# Data paths
RAW_DATA_PATH = 'data/raw/creditcard.csv'
PROCESSED_DATA_PATH = 'data/processed/'

# Model paths
MODEL_SAVE_PATH = 'models/'

# Data split
TEST_SIZE = 0.2
VAL_SIZE = 0.1
RANDOM_STATE = 42

# Anomaly detection
CONTAMINATION = 0.002  # Expected fraud rate

# LSTM Autoencoder parameters
LSTM_UNITS = 64
ENCODING_DIM = 14
EPOCHS = 50
BATCH_SIZE = 256
LEARNING_RATE = 0.001

# Isolation Forest parameters
N_ESTIMATORS = 100
MAX_SAMPLES = 256
