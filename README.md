# Projet MLOps - Machine Learning avec Git

Ce projet démontre un **pipeline ML reproductible** avec :
1) Construction d'une **baseline ML** (scripts, config, artefacts, tests)
2) **Versioning Git** (commits structurés, branches, tags)
3) **Logging** complet du pipeline d'entraînement et d'évaluation
4) **Tests** automatisés (configuration, données, modèle)

> **Important** : Les dossiers `data/`, `artifacts/` et les fichiers `logs/*.log` ne sont **PAS versionnés** (voir `.gitignore`).

---

## 1) Structure du projet

```
mlops-ml-project/
  README.md                             ✓ versionné
  requirements.txt                      ✓ versionné
  .gitignore                            ✓ versionné
  config/
    train.yaml                          ✓ versionné (configuration)
  src/
    __init__.py                         ✓ versionné
    data.py                             ✓ versionné (chargement données)
    features.py                         ✓ versionné (preprocessing)
    model.py                            ✓ versionné (construction modèle)
    utils.py                            ✓ versionné (logging)
  scripts/
    train.py                            ✓ versionné (entraînement)
    evaluate.py                         ✓ versionné (évaluation)
  tests/
    test_config.py                      ✓ versionné
    test_data.py                        ✓ versionné
    test_model.py                       ✓ versionné
  notebooks/                            ✓ versionné (vide)
  data/                                 ✗ NON versionné (données brutes)
    dataset.csv
  artifacts/                            ✗ NON versionné (sorties)
    model.joblib
    metrics.json
    confusion_matrix.png
    report.json
  logs/                                 ✗ fichiers *.log NON versionnés
    .gitkeep                            ✓ versionné (préserve le dossier)
    train_YYYYMMDD_HHMMSS.log
    evaluate_YYYYMMDD_HHMMSS.log
```

### Statut Git

**✓ Versionné dans Git :**
- Code source (`src/`, `scripts/`, `tests/`)
- Configuration (`config/train.yaml`)
- Documentation (`README.md`, `requirements.txt`)
- Structure (`logs/.gitkeep`)

**✗ NON versionné (ignoré) :**
- Données (`data/`)
- Artefacts ML (`artifacts/`)
- Logs d'exécution (`logs/*.log`)
- Cache Python (`__pycache__/`, `*.pyc`)
- Environnements virtuels (`.venv/`, `.env`)

---

## 2) Installation (local)

### Option A — Environnement virtuel (recommandé)

**Windows PowerShell**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Linux/macOS / Git Bash**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Option B — Installation directe
```bash
pip install -r requirements.txt
```

---

## 3) Exécuter le pipeline ML

Par défaut, la configuration utilise le **dataset Iris** (intégré dans scikit-learn), donc aucun fichier externe n'est nécessaire.

### 3.1 Entraînement

**Windows PowerShell**
```powershell
$env:PYTHONPATH = "$PWD"
python scripts/train.py
```

**Linux/macOS / Git Bash**
```bash
export PYTHONPATH=$PWD
python scripts/train.py
```

**Alternative (module)**
```bash
python -m scripts.train
```

**Sorties produites dans `artifacts/` :**
- `model.joblib` — pipeline entraîné (preprocessing + modèle)
- `metrics.json` — accuracy et F1-score
- `confusion_matrix.png` — matrice de confusion

**Log généré dans `logs/` :**
- `train_YYYYMMDD_HHMMSS.log` — log détaillé de l'entraînement

### 3.2 Évaluation

**Windows PowerShell**
```powershell
$env:PYTHONPATH = "$PWD"
python scripts/evaluate.py
```

**Linux/macOS / Git Bash**
```bash
export PYTHONPATH=$PWD
python scripts/evaluate.py
```

**Sortie produite :**
- `artifacts/report.json` — rapport de classification détaillé

**Log généré :**
- `logs/evaluate_YYYYMMDD_HHMMSS.log`

---

## 4) Tests automatisés

**Exécution des tests**

**Windows PowerShell**
```powershell
$env:PYTHONPATH = "$PWD"
python -m pytest tests/ -v
```

**Linux/macOS / Git Bash**
```bash
export PYTHONPATH=$PWD
python -m pytest tests/ -v
```

**Couverture des tests :**
- `test_config.py` — validation de la configuration YAML
- `test_data.py` — chargement datasets (Iris, CSV)
- `test_model.py` — construction modèle et preprocessing

**Résultat attendu :**
```
7 passed, 1 skipped
```

---

## 5) Configuration

Modifier `config/train.yaml` pour personnaliser le pipeline :

### Dataset Iris (par défaut)
```yaml
data:
  kind: "iris"

split:
  test_size: 0.2
  random_state: 42

model:
  name: "logistic_regression"
  max_iter: 2000

artifacts_dir: "artifacts"
```

### Dataset CSV personnalisé
1. Déposer le fichier CSV dans `data/dataset.csv`
2. Modifier la config :

```yaml
data:
  kind: "csv"
  path: "data/dataset.csv"
  target: "Exited"  # nom de la colonne cible

split:
  test_size: 0.2
  random_state: 42

model:
  name: "logistic_regression"
  max_iter: 2000

artifacts_dir: "artifacts"
```

> **Note** : Le dossier `data/` est ignoré par Git (pas versionné).

---

## 6) Logging

Tous les scripts génèrent des **logs détaillés** :

**Console** : Affichage des étapes principales
**Fichiers logs** : Historique complet dans `logs/`

**Format des fichiers :**
- `train_YYYYMMDD_HHMMSS.log`
- `evaluate_YYYYMMDD_HHMMSS.log`

**Contenu des logs :**
- Timestamps
- Configuration chargée
- Informations dataset (taille, features)
- Progression de l'entraînement
- Métriques calculées
- Chemins des artefacts sauvegardés

---

## 7) Workflow Git (exemple)

### 7.1 Initialisation + premier commit
```bash
git init

# Ajouter .gitignore en premier
git add .gitignore
git commit -m "chore: add .gitignore for ML project"

# Ajouter le code baseline
git add .
git commit -m "feat: initial ML baseline with train/eval/tests/logging"
```

### 7.2 Lier à GitHub
```bash
git branch -M main
git remote add origin <URL_DU_DEPOT>
git push -u origin main
```

### 7.3 Branches dev/feature
```bash
# Créer branche dev
git checkout -b dev
git push -u origin dev

# Créer feature branch
git checkout -b feature/amelioration
# ... modifications ...
git add .
git commit -m "feat: amélioration du preprocessing"
git push -u origin feature/amelioration
```

### 7.4 Tag de release
```bash
git checkout main
git tag -a v0.1.0 -m "Baseline ML avec logging et tests"
git push origin main
git push origin --tags
```

---

## 8) Dépendances

```
numpy              # Calcul numérique
pandas             # Manipulation données
scikit-learn       # ML (modèle, métriques, datasets)
matplotlib         # Visualisations
joblib             # Sérialisation modèle
pyyaml             # Parsing configuration
pytest             # Tests automatisés
```

---

## 9) Fonctionnalités principales

✅ **Pipeline modulaire** : séparation data/features/model  
✅ **Configuration YAML** : expérimentation facilitée  
✅ **Logging complet** : console + fichiers horodatés  
✅ **Tests automatisés** : validation configuration/données/modèle  
✅ **Multi-datasets** : Iris (baseline) + CSV custom  
✅ **Artefacts organisés** : modèle, métriques, visualisations  
✅ **Git-friendly** : `.gitignore` adapté pour projets ML  

---

## 10) Preuves d'exécution attendues

- ✅ Fichiers `artifacts/metrics.json` et `artifacts/report.json`
- ✅ Logs dans `logs/` avec timestamps
- ✅ Capture d'écran d'exécution de `train.py`
- ✅ Tests passant (7 passed, 1 skipped)
- ✅ Commits Git structurés
- ✅ Tag `v0.1.0` (si applicable)