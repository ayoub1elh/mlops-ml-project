# scripts/evaluate.py
import json
from pathlib import Path

import joblib
import yaml
from sklearn.metrics import classification_report

from src.data import load_dataset
from src.utils import setup_logger

logger = setup_logger('evaluate')

def load_cfg(path="config/train.yaml"):
    return yaml.safe_load(open(path, "r", encoding="utf-8"))

def main():
    logger.info("Starting evaluation pipeline")
    cfg = load_cfg()
    art_dir = Path(cfg.get("artifacts_dir", "artifacts"))
    
    logger.info(f"Loading model from {art_dir / 'model.joblib'}")
    model = joblib.load(art_dir / "model.joblib")
    
    logger.info(f"Loading dataset: {cfg['data']['kind']}")
    X, y = load_dataset(cfg)
    logger.info(f"Dataset loaded: {X.shape[0]} samples")
    
    logger.info("Generating predictions...")
    pred = model.predict(X)
    report = classification_report(y, pred, output_dict=True)
    
    logger.info("Saving evaluation report...")
    json.dump(report, open(art_dir / "report.json", "w"), indent=2)
    logger.info(f"Report saved to {art_dir / 'report.json'}")
    
    print("Evaluate OK: artifacts/report.json")
    logger.info("Evaluation pipeline completed successfully")
    
if __name__ == "__main__":
    main()