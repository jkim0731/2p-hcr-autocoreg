# Project Instructions: ROI Classification with GBM + MLflow on Code Ocean

## Overview

Build a machine learning workflow on Code Ocean that:

- Generates features for ROIs (regions of interest).
- Classifies ROIs using a Gradient Boosted Machine (GBM) model.
- Supports iterative retraining as new manual labels are added over time.
- Tracks all model versions and experiments with MLflow (natively integrated into Code Ocean).

## Platform Concepts (Code Ocean specifics)

- **Capsule**: a Git-backed unit of code + environment + data. Think of it as a versioned, reproducible compute job.
- **Data Asset**: immutable, versioned file storage (S3-backed). Types: `dataset`, `result`, `combined`, `model`.
- **Reproducible Run**: every capsule run records the exact code commit, data asset IDs attached, and computation ID → full lineage is automatic.
- **MLflow**: natively integrated. Enable per-capsule via **Capsule Settings → MLflow tab → toggle "Track this Capsule"**. No manual `MLFLOW_TRACKING_URI` setup needed — it is injected automatically.

### Filesystem layout inside a capsule

- `/code` — your scripts (read/write)
- `/data` — attached data assets (read-only)
- `/results` — outputs that get captured as result data assets
- `/scratch` — temporary storage

## Workflow Architecture

### Capsule 1: Feature Extraction + Inference

- Input: raw ROI data (attached as a data asset at `/data/roi_data`)
- Input: current registered GBM model (attached as a data asset at `/data/model`)
- Output: features per ROI + predictions → written to `/results`
- No MLflow tracking needed here.

### Capsule 2: Labeling Capsule (manual/semi-manual)

- Each labeling session produces a new batch of labeled ROIs (features + labels CSV or parquet).
- Save output to `/results` → captured as a new result data asset after the run.
- Name each asset clearly, e.g., `roi_labels_batch_v4_2026-06-18`.
- Do not merge old and new labels here — keep each batch as its own asset.

### Capsule 3: Training Capsule (MLflow-tracked)

- Input: one or more label batch data assets, each mounted at separate paths (e.g., `/data/labels_v1`, `/data/labels_v2`, `/data/labels_v3`).
- Code reads from all mounted label paths, concatenates them, and resolves conflicts (newest label wins → use file modification time or a timestamp column in the label files).
- Trains a GBM (e.g., `sklearn.ensemble.GradientBoostingClassifier` or XGBoost / LightGBM).
- MLflow autolog handles: hyperparameters, metrics, model artifact, feature importances, input/output signature.
- Manually write the model file to `/results` as well (for use as a Code Ocean data asset).
- After a satisfactory run: register the model in the MLflow UI → becomes a versioned Registered Model in the Code Ocean Models dashboard.

## Training Capsule Code Pattern

```python
import mlflow
import mlflow.sklearn
import joblib
import pandas as pd
import glob
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

# --- Load and merge all label batches ---
label_paths = sorted(glob.glob("/data/labels_*/labels.parquet"))  # adjust glob pattern
dfs = []
for path in label_paths:
    df = pd.read_parquet(path)
    dfs.append(df)

labels_all = pd.concat(dfs)

# Resolve conflicts: newest label wins (requires a 'timestamp' column)
labels_all = (
    labels_all
    .sort_values("timestamp")
    .drop_duplicates(subset=["roi_id"], keep="last")
)

X = labels_all.drop(columns=["roi_id", "label", "timestamp"])
y = labels_all["label"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# --- MLflow tracking ---
mlflow.sklearn.autolog()  # auto-captures params, metrics, model artifact

with mlflow.start_run(run_name=f"gbm_v{os.environ.get('LABEL_VERSION', 'unknown')}"):
    model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=4)
    model.fit(X_train, y_train)

    val_score = model.score(X_val, y_val)
    mlflow.log_metric("val_accuracy", val_score)

    # Optional: log label provenance as MLflow tags for readability in MLflow UI
    # (Code Ocean already records data asset IDs in the run provenance automatically)
    mlflow.set_tag("n_total_labels", len(labels_all))
    mlflow.set_tag("n_label_batches", len(label_paths))

# --- Save model to /results for use as a Code Ocean data asset ---
os.makedirs("/results", exist_ok=True)
joblib.dump(model, "/results/gbm_model.pkl")
print(f"Validation accuracy: {val_score:.4f}")
```

## Label Data Management Strategy

**Chosen approach:** keep each label batch as a separate internal data asset, attach all batches to the training capsule as separate mounts.

- Avoids data duplication.
- Conflict resolution is handled in training code (newest label wins).
- Code Ocean's reproducible run automatically records which exact data asset IDs (and versions) were used → full data provenance is free.
- No need for External Data Assets or Combined Data Assets (which require External S3 assets and IAM configuration).

### Each iteration

1. Run labeling capsule → new label batch result data asset created.
2. Attach the new asset to the training capsule (alongside all previous batches).
3. Run the training capsule → MLflow logs the run.
4. If satisfied with metrics → register the model from the MLflow UI → new model version in Models dashboard.
5. Update inference capsule to use the new model version.

## MLflow: What Is and Isn't Automatic

### Handled automatically by `mlflow.sklearn.autolog()`

- All model hyperparameters.
- Train/val metrics (accuracy, log loss, etc.).
- Serialized model artifact (loadable via `mlflow.sklearn.load_model()`).
- Feature importances plot.
- Input/output signature.
- Code Ocean auto-tags: `codeocean.computationID`, `codeocean.userID`, `codeocean.capsuleID`, `codeocean.resultsFolder`.

### You must handle manually

- Write model to `/results/` using `joblib.dump(model, "/results/gbm_model.pkl")` — MLflow stores artifacts in its own server, not in `/results`. Writing to `/results` is required if you want a Code Ocean result data asset.
- Domain-specific metrics (e.g., per-class accuracy, conflict rate) via `mlflow.log_metric(...)`.
- Label provenance tags (optional, for MLflow UI readability) via `mlflow.set_tag(...)` — Code Ocean's run provenance already covers this at the platform level.

### Data provenance note

Code Ocean automatically records which data asset IDs were attached to every run. You do not need to manually track this for audit purposes — it is captured in the run's lineage graph and in any result data asset's provenance. MLflow tags for label provenance are a UX nicety, not a requirement.

## Environment Requirements

The training capsule needs these pip packages (add via **Capsule Settings → Environment**):

- `mlflow` (must be compatible with Code Ocean's MLflow v3.6.0)
- `scikit-learn` (or `xgboost` / `lightgbm`, depending on your GBM)
- `joblib`
- `pandas`
- `pyarrow` (if using parquet for labels)

## Post-Run: Registering the Model

After a training run:

1. In the capsule, click **"View in MLflow"** (top-right header button).
2. Go to the **Experiments** tab → select the run → click the model artifact.
3. Click **"Register Model"** → create new or add a version to an existing model.
4. The model now appears in the Code Ocean Models dashboard with full provenance (linked to the computation, code commit, and data assets used).

## Key Constraints & Gotchas

- MLflow tab is in **Capsule Settings** (gear icon, top-right of capsule IDE).
- `MLFLOW_TRACKING_URI` is auto-injected — **do not set it manually**.
- Viewers of a non-Release capsule cannot log to MLflow (runs will fail silently for them). Use a **Release capsule** if collaborators need to run tracked experiments.
- `/metadata` and `/environment` are **not** accessible during reproducible runs — do not read from them.
- `/data` is **read-only** inside runs — all outputs go to `/results` or `/scratch`.
- The run script at `/code/run` is the required headless entrypoint for all reproducible runs.



## Troubleshooting: "I don't have a model artifact"

### What is a model artifact?

A model artifact is the serialized, saved version of your trained ML model — the actual
model files written during the capsule run (weights, configuration, metadata). MLflow
treats these as a registrable **model** only when your training code explicitly logs the
model with an MLflow `log_model` call. Without that call, MLflow tracks metrics/parameters
but has **no model files to register**.

### Why you don't have one

The capsule logs metrics and parameters but is missing the MLflow **model-logging step**.
Add a `log_model` call to the training script:

```python
import mlflow

with mlflow.start_run(run_name="my_run"):
    # ... your training code ...

    # Log the trained model as an artifact (choose your flavor):
    mlflow.sklearn.log_model(model, artifact_path="my_model")      # scikit-learn
    # mlflow.pytorch.log_model(model, artifact_path="my_model")    # PyTorch
    # mlflow.tensorflow.log_model(model, artifact_path="my_model") # TensorFlow
    # mlflow.pyfunc.log_model(artifact_path="my_model", ...)       # generic / custom
```

Alternatively, use autologging, which captures model artifacts automatically for supported
libraries:

```python
mlflow.sklearn.autolog()  # or mlflow.pytorch.autolog(), mlflow.lightgbm.autolog(), etc.

with mlflow.start_run():
    # ... train your model ...
```

### Steps to fix and register the model

1. Edit the capsule's training script to add a `log_model` (or `autolog`) call.
2. Re-run the capsule with MLflow tracking enabled (toggle **ON** in **Capsule Settings → MLflow tab**).
3. Once the run completes, the model artifact appears in the Timeline under the `mlflow/` folder.
4. Right-click the model folder → **"Create New Model"** to register it in Code Ocean.