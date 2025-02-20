import pandas as pd
from sklearn import svm
from scipy.sparse import csr_matrix
import mlflow
import pickle
from sklearn.metrics import (
        accuracy_score, f1_score, recall_score, precision_score
    )

def train_model(train_data_vectors: csr_matrix, label: pd.DataFrame) -> svm.SVC:
    """train model function

    Args:
        train_data_vectors (csr_matrix): vectorized data
        label (pd.DataFrame): label

    Returns:
        svm.SVC: SVC model
    """
    classifier = svm.SVC(kernel='linear')
    classifier.fit(train_data_vectors, label)
    
    return classifier


def compute_metrics(
        prefix: str, y_pred: csr_matrix, 
        y_test: pd.DataFrame,
        tracking_uri: str,
        mlruns_dir: str) -> None:
    """compute metrics

    Args:
        prefix (str): prefix
        y_pred (csr_matrix): predictions
        y_test (pd.DataFrame): test
    """
    mlflow.set_tracking_uri(tracking_uri)
    
    # report = classification_report(y_test, y_pred, output_dict=True)
    
    mlflow.log_metrics({f"{prefix}_Accuracy": accuracy_score(y_true=y_test, y_pred=y_pred)})
    mlflow.log_metrics({f"{prefix}_f1_score": f1_score(y_true=y_test, y_pred=y_pred, average='macro')})
    mlflow.log_metrics({f"{prefix}_recall_score": recall_score(y_true=y_test, y_pred=y_pred, average='macro')})
    mlflow.log_metrics({f"{prefix}_precision_score": precision_score(y_true=y_test, y_pred=y_pred, average='macro')})
    
    # mlflow.log_metrics({f"{prefix}_ROC_AUC_score'": roc_auc_score(y_true=y_test, y_pred=y_pred, average='macro')})
    
    

def train_model_with_io(
        prefix: str, mlruns_dir: str, train_data_vectors: csr_matrix, 
        test_data_vectors: csr_matrix, outputs: str,
        y_train: pd.DataFrame, y_test: pd.DataFrame, tracking_uri: str,
        ) -> None:
    """train model with artifacts

    Args:
        prefix (str) : use case
        mlruns_dir (str): output path
        train_data_vectors (csr_matrix): vectorized data
        test_data_vectors (csr_matrix): vectorized data
        outputs (str): models outputs
        y_train (pd.DataFrame): train target
        y_test (pd.DataFrame): test target
        tracking_uri (str): mlflow server endpoint
        artifact_path (str): artifact
    """
    mlflow.set_tracking_uri(tracking_uri)
    
    mlflow.set_experiment("sentiment_analysis_experiment")
    
    with mlflow.start_run():
        
       classfifier = train_model(
                        train_data_vectors=train_data_vectors,
                        label=y_train
                    )
              
       y_pred = classfifier.predict(test_data_vectors)
       
       compute_metrics(
            prefix=prefix,
            y_pred=y_pred,
            y_test=y_test,
            tracking_uri=tracking_uri,
            mlruns_dir=mlruns_dir
        )
       
    #    mlflow.sklearn.log_model(classfifier, artifact_path=mlruns_dir)
       
       pickle.dump(classfifier, open(f'{outputs}/sentiment_analyzer.pkl', 'wb'))
       
       
       
       