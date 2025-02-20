import pytest
import os, sys
from sklearn import svm

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.use_cases.sentiment_analysis.data_propocessing import load_data, apply_tfid_vectorizer
from src.use_cases.sentiment_analysis.train import train_model


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


def test_type_returned_train_model(sample_data):
    actual_train_df, actual_test_df = sample_data
    actual_train_vectors, _ = apply_tfid_vectorizer(actual_train_df, actual_test_df)
    
    classifier = train_model(actual_train_vectors, actual_train_df["Label"])
    
    assert isinstance(classifier, svm.SVC)
    

def test_predict_method_in_trained_model(sample_data):
    actual_train_df, actual_test_df = sample_data
    actual_train_vectors, _ = apply_tfid_vectorizer(actual_train_df, actual_test_df)
    
    classifier = train_model(actual_train_vectors, actual_train_df["Label"])
    
    assert hasattr(classifier, "predict")
    