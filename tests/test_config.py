# tests/test_config.py
import pytest
import yaml
from pathlib import Path


def test_config_exists():
    """Test that the train.yaml config file exists."""
    config_path = Path("config/train.yaml")
    assert config_path.exists(), "config/train.yaml should exist"


def test_config_structure():
    """Test that config has all required fields."""
    config_path = Path("config/train.yaml")
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    
    # Check main sections
    assert "data" in cfg, "Config should have 'data' section"
    assert "split" in cfg, "Config should have 'split' section"
    assert "model" in cfg, "Config should have 'model' section"
    assert "artifacts_dir" in cfg, "Config should have 'artifacts_dir' field"
    
    # Check data section
    assert "kind" in cfg["data"], "data section should have 'kind'"
    assert cfg["data"]["kind"] in ["iris", "csv"], "kind should be 'iris' or 'csv'"
    
    # Check split section
    assert "test_size" in cfg["split"], "split section should have 'test_size'"
    assert "random_state" in cfg["split"], "split section should have 'random_state'"
    assert 0 < cfg["split"]["test_size"] < 1, "test_size should be between 0 and 1"
    
    # Check model section
    assert "name" in cfg["model"], "model section should have 'name'"


def test_config_values():
    """Test that config values are reasonable."""
    config_path = Path("config/train.yaml")
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    
    # Test split values
    assert isinstance(cfg["split"]["test_size"], float), "test_size should be float"
    assert isinstance(cfg["split"]["random_state"], int), "random_state should be int"
    
    # Test artifacts directory
    assert isinstance(cfg["artifacts_dir"], str), "artifacts_dir should be string"
    assert len(cfg["artifacts_dir"]) > 0, "artifacts_dir should not be empty"
