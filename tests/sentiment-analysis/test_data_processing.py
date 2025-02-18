import pytest
import os, sys
import pandas as pd
from scipy.sparse import csr_matrix


sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.sentiment_analysis.data_propocessing import load_data, apply_tfid_vectorizer

@pytest.fixture
def sample_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_filepath = os.path.join(base_dir, "../../data/sentiment-analysis-data/train.csv")
    test_filepath = os.path.join(base_dir, "../../data/sentiment-analysis-data/test.csv")

    train_data = load_data(filepath=train_filepath, type="csv")
    test_data = load_data(filepath=test_filepath, type="csv")
    
    if train_data is None:
        raise ValueError(f"Impossible to load data {train_filepath}")
    
    if test_data is None:
        raise ValueError(f"Impossible to load data {test_filepath}")
    
    return train_data.sample(frac=0.20), test_data.sample(frac=0.20)


def test_type_of_loaded_data(sample_data):
    excepted_type = pd.DataFrame
    
    actual_train_df, actual_test_df = sample_data
    
    assert isinstance(actual_train_df, excepted_type)
    assert isinstance(actual_test_df, excepted_type)
    
    
def test_columns_in_loaded_data(sample_data):
    excepted_column = ['Content', 'Label']
    
    actual_train_df, actual_test_df = sample_data
    actual_train_columns = actual_train_df.columns.to_list()
    actual_test_columns = actual_test_df.columns.to_list()
    
    assert actual_train_columns[0] == excepted_column[0]
    assert actual_train_columns[1] == excepted_column[1]
    assert actual_test_columns[0] == excepted_column[0]
    assert actual_test_columns[1] == excepted_column[1]


def test_type_of_vectorized_vectors(sample_data):
    excepted_train_vectors_type = csr_matrix
    excepted_test_vectors_type = csr_matrix
    
    actual_train_df, actual_test_df = sample_data
    actual_train_vectors, excepted_test_vectors = apply_tfid_vectorizer(actual_train_df, actual_test_df)
    
    assert isinstance( actual_train_vectors, excepted_train_vectors_type), "train_vectors must be csr_matrix"
    assert isinstance(excepted_test_vectors, excepted_test_vectors_type), "test_vectors must be csr_matrix"
    

def test_vectorized_vectors_not_null(sample_data):
    expected_value = 0
    
    actual_train_df, actual_test_df = sample_data
    actual_train_vectors, excepted_test_vectors = apply_tfid_vectorizer(actual_train_df, actual_test_df)
    
    assert actual_train_vectors.shape[0] > expected_value
    assert excepted_test_vectors.shape[0] > expected_value
    

def test_vectorized_vectors_same_number_columns(sample_data):
    train_df, test_df = sample_data
    train_vectors, test_vectors = apply_tfid_vectorizer(train_df, test_df)
    
    assert train_vectors.shape[1] == test_vectors.shape[1]

def test_vectorized_vectors_same_number_rows(sample_data):
    train_df, test_df = sample_data
    train_vectors, test_vectors = apply_tfid_vectorizer(train_df, test_df)
    
    assert train_vectors.shape[0] > test_vectors.shape[0]    