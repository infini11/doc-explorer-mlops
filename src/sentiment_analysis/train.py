import pandas as pd
from sklearn import svm
from scipy.sparse import csr_matrix

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
    
