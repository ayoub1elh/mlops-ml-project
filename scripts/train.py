# scripts/train.py
import json
from pathlib import Path

import yaml
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.pipeline import Pipeline

from src.data import load_dataset
from src.features import build_numeric_preprocess
from src.model import build_model
from src.utils import setup_logger

logger = setup_logger('train')

def load_cfg(path="config/train.yaml"):
    return yaml.safe_load(open(path, "r", encoding="utf-8"))

def save_confusion_matrix(y_true, y_pred, out_path: Path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Pred")
    plt.ylabel("True")
    
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    
def main():
    logger.info("Starting training pipeline")
    cfg = load_cfg()
    logger.info(f"Config loaded: {cfg}")
    
    art_dir = Path(cfg.get("artifacts_dir", "artifacts"))
    art_dir.mkdir(exist_ok=True)
    
    logger.info(f"Loading dataset: {cfg['data']['kind']}")
    X, y = load_dataset(cfg)
    logger.info(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    
    Xtr, Xte, ytr, yte = train_test_split(
        X, y,
        test_size=float(cfg["split"]["test_size"]),
        random_state=int(cfg["split"]["random_state"]),
        stratify=y
    )
    logger.info(f"Train/test split: {len(Xtr)} train, {len(Xte)} test samples")
    
    preprocess = build_numeric_preprocess()
    model = build_model(cfg)
    logger.info(f"Model: {cfg['model']['name']}")

    pipe = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", model),
    ])

    logger.info("Training model...")
    pipe.fit(Xtr, ytr)
    logger.info("Model training completed")
    
    pred = pipe.predict(Xte)
    logger.info("Predictions generated")
    
    acc = float(accuracy_score(yte, pred))
    f1 = float(f1_score(yte, pred, average="macro"))
    logger.info(f"Metrics - Accuracy: {acc:.4f}, F1-macro: {f1:.4f}")
    
    # Artefacts
    logger.info("Saving artifacts...")
    joblib.dump(pipe, art_dir / "model.joblib")
    json.dump({"accuracy": acc, "f1_macro": f1}, open(art_dir / "metrics.json", "w")
        , indent=2)
    save_confusion_matrix(yte, pred, art_dir / "confusion_matrix.png")
    logger.info(f"Artifacts saved to {art_dir}")
    
    print("Train OK:", {"accuracy": acc, "f1_macro": f1})
    logger.info("Training pipeline completed successfully")
if __name__ == "__main__":
    main()