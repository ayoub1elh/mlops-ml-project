# tests/test_data.py
import pytest
import pandas as pd
from src.data import load_dataset


def test_load_iris_dataset():
    """Test loading the Iris dataset."""
    cfg = {
        "data": {
            "kind": "iris"
        }
    }
    
    X, y = load_dataset(cfg)
    
    # Check return types
    assert isinstance(X, pd.DataFrame), "X should be a DataFrame"
    assert isinstance(y, pd.Series), "y should be a Series"
    
    # Check shapes
    assert X.shape[0] == 150, "Iris dataset should have 150 samples"
    assert X.shape[1] == 4, "Iris dataset should have 4 features"
    assert len(y) == 150, "Target should have 150 values"
    
    # Check no missing values
    assert not X.isnull().any().any(), "X should not have missing values"
    assert not y.isnull().any(), "y should not have missing values"


def test_load_csv_dataset():
    """Test loading a CSV dataset (if available)."""
    cfg = {
        "data": {
            "kind": "csv",
            "path": "data/dataset.csv",
            "target": "Exited"
        }
    }
    
    try:
        X, y = load_dataset(cfg)
        
        # Check return types
        assert isinstance(X, pd.DataFrame), "X should be a DataFrame"
        assert isinstance(y, pd.Series), "y should be a Series"
        
        # Check that target is not in features
        assert "Exited" not in X.columns, "Target column should not be in X"
        
    except FileNotFoundError:
        pytest.skip("CSV file not available for testing")
