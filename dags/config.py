import os
import datetime
from datetime import datetime as dt

PROJECT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FOLDER = os.path.join(PROJECT_FOLDER, 'data')
MODELRUNS_FOLDER = os.path.join(PROJECT_FOLDER, 'mlruns')
MODELS_DIR = os.path.join(PROJECT_FOLDER, 'models')

TRAIN_DATA = os.path.join(DATA_FOLDER, 'sentiment-analysis-data/train.csv')
TEST_DATA = os.path.join(DATA_FOLDER, 'sentiment-analysis-data/train.csv')

TRACKING_URI = 'http://mlflow:5000'
DEFAULT_ARGS = {'owner': 'airflow'}
CONCURRENCY = 2
SCHEDULE_INTERVAL = datetime.timedelta(hours=1)
START_DATE = dt(2025, 2, 20)

