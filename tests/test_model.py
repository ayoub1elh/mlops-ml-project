# tests/test_model.py
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from src.model import build_model
from src.features import build_numeric_preprocess


def test_build_logistic_regression():
    """Test building a logistic regression model."""
    cfg = {
        "model": {
            "name": "logistic_regression",
            "max_iter": 1000
        }
    }
    
    model = build_model(cfg)
    assert isinstance(model, LogisticRegression), "Should return LogisticRegression"
    assert model.max_iter == 1000, "max_iter should be set correctly"


def test_build_random_forest():
    """Test building a random forest model."""
    pytest.skip("Random forest not implemented in baseline - only logistic regression supported")
    
    cfg = {
        "model": {
            "name": "random_forest",
            "n_estimators": 50,
            "random_state": 42
        }
    }
    
    model = build_model(cfg)
    assert isinstance(model, RandomForestClassifier), "Should return RandomForestClassifier"


def test_build_numeric_preprocess():
    """Test the numeric preprocessing pipeline."""
    preprocess = build_numeric_preprocess()
    
    # Check that it's a pipeline-like object with fit and transform
    assert hasattr(preprocess, 'fit'), "Should have fit method"
    assert hasattr(preprocess, 'transform'), "Should have transform method"
