import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

os.environ['KAGGLE_CONFIG_DIR'] = BASE_DIR 
DATASET_NAME = 'ellipticco/elliptic-data-set'

RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')

# main file paths
RAW_FILE_CHECK = os.path.join(RAW_DIR, 'elliptic_txs_features.csv')
PROCESSED_FILE = os.path.join(PROCESSED_DIR, 'elliptic_graph.pt')

# create directories if they don't exist
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)