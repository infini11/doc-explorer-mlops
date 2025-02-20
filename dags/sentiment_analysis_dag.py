from airflow.decorators import dag, task
from typing import Tuple, Dict
from scipy.sparse import csr_matrix
import pandas as pd

from src.use_cases.sentiment_analysis.train import train_model_with_io
from src.use_cases.sentiment_analysis.data_propocessing import load_and_prepare_data_with_io
from config import DEFAULT_ARGS, START_DATE, MODELRUNS_FOLDER, MODELS_DIR, CONCURRENCY, TRACKING_URI,SCHEDULE_INTERVAL, TRAIN_DATA, TEST_DATA


@dag(
    schedule=None,
    default_args=DEFAULT_ARGS,
    start_date=START_DATE,
    concurrency=CONCURRENCY,
    schedule_interval=SCHEDULE_INTERVAL
)

def sentiment_analysis_dag():
    
    @task
    def load_and_prepare_data_task(
                    train_filepath: str, 
                    test_filepath: str, type: str
                ) -> Dict[str, csr_matrix]:
        
        train_data_vectors, test_data_vectors = load_and_prepare_data_with_io(
                                                     train_filepath=train_filepath,
                                                     test_filepath=test_filepath,
                                                     type=type                  
                                            )
        
        return {"train": train_data_vectors, "test": test_data_vectors} 
    
    
    @task
    def train_model_task(
                    prefix: str,
                    mlruns_dir: str,
                    train_data_vectors: csr_matrix,
                    test_data_vectors: csr_matrix,
                    outputs: str,
                    y_train: pd.DataFrame,
                    y_test: pd.DataFrame,
                    tracking_uri: str                      
                ) -> None:
        
        train_model_with_io(
                prefix=prefix,
                mlruns_dir=mlruns_dir,
                train_data_vectors=train_data_vectors,
                test_data_vectors=test_data_vectors,
                outputs=outputs,
                y_train=y_train,
                y_test=y_test,
                tracking_uri=tracking_uri            )
        
    train_df = pd.read_csv(TRAIN_DATA)
    test_df = pd.read_csv(TEST_DATA)
    
    results = load_and_prepare_data_task(
                    train_filepath=TRAIN_DATA,
                    test_filepath=TEST_DATA,
                    type="csv"
                ) 
    train_data_vectors, test_data_vectors = results["train"], results["test"]
    
    train_model_task(
            prefix="sentiment_analysis",
            mlruns_dir=MODELRUNS_FOLDER,
            train_data_vectors=train_data_vectors,
            test_data_vectors=test_data_vectors,
            outputs=MODELS_DIR,
            y_train=train_df["Content"],
            y_test=test_df["Content"],
            tracking_uri=TRACKING_URI
        )
    
sentiment_analysis_dag()