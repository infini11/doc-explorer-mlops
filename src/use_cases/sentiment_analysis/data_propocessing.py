import pandas as pd
from typing import Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix


def load_data(filepath: str, type: str) -> pd.DataFrame:
    """read data in dataframe obj

    Args:
        filepath (str): file path

    Returns:
        pd.DataFrame: dataframe that contain data
    """
    try:
        return pd.read_csv(filepath) if type=="csv" else pd.read_json(filepath)
    except Exception as e:
        print(f"Error when reading file : {filepath} with exception : {e}")
        return


def apply_tfid_vectorizer(
            train_data: pd.DataFrame, test_data: pd.DataFrame
        ) -> Tuple[csr_matrix, csr_matrix]:
    """apply vectorizer to text

    Args:
        train_data (pd.DataFrame): train data
        test_data (pd.DataFrame): test data

    Returns:
        tuple[csr_matrix, csr_matrix]: tuple of csr_matrix
    """
    
    vectorizer = TfidfVectorizer(min_df = 5, max_df = 0.8, sublinear_tf = True, use_idf = True)
    train_vectors = vectorizer.fit_transform(train_data['Content'])
    test_vectors = vectorizer.transform(test_data['Content'])

    return train_vectors, test_vectors


def load_and_prepare_data_with_io(
            train_filepath: str, test_filepath: str, type: str
        ) -> Tuple[csr_matrix, csr_matrix]:
    """load and prepare data and generate artifact

    Args:
        train_file_path (str): train file
        test_filepath (str): test file
        type (str): type of file
    """
    
    train_data = load_data(filepath=train_filepath, type="csv")
    test_data = load_data(filepath=test_filepath, type="csv")
    
    train_data_vectors, test_data_vectors = apply_tfid_vectorizer(
                                            train_data=train_data,
                                            test_data=test_data
                                        )
    
    return (
            train_data_vectors,
            test_data_vectors
        )
    