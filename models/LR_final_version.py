"""
Train final Logistic Regression model on the best feature set:
Shape_plus_PWMgauss (DNA shape + Gaussian-expanded PWM).

This script:
  - Loads the CTCF dataset with sequences and PWM scores
  - Adds four DNA shape features (MGW, ProT, Roll, HelT)
  - Cleans rows with non-finite values in shape + PWM
  - Builds the Shape_plus_PWMgauss feature matrix (405 features)
  - Performs a single 85/15 stratified train/test split
  - Trains LogisticRegression with C=0.1 and evaluates on the test set
  - Retrains the same model on the full cleaned dataset
  - Saves a text summary with dataset sizes, hyperparameters, and performance
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
)

# ------------------------------------------------------------
# Project imports
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from data_extraction.shape_features import init_shape_tracks, add_shape_arrays, shapes_to_matrix


# ============================================================
#  Helper: PWM -> Gaussian basis
# ============================================================

def pwm_to_gaussians(pwm, n_centers=5, sigma=None):
    """
    Expand 1D PWM scores into Gaussian basis features.

    Parameters
    ----------
    pwm : np.ndarray of shape (N, 1)
    n_centers : int
        Number of Gaussian basis functions.
    sigma : float or None
        Width of the Gaussians. If None, inferred from center spacing.

    Returns
    -------
    gauss : np.ndarray of shape (N, n_centers)
    """
    vals = pwm.ravel()
    min_v, max_v = vals.min(), vals.max()

    # centers evenly spaced between min and max PWM
    centers = np.linspace(min_v, max_v, n_centers)

    if sigma is None:
        # width ~ distance between adjacent centers (avoid zero if n_centers=1)
        if n_centers > 1:
            sigma = centers[1] - centers[0]
        else:
            sigma = (max_v - min_v) if max_v > min_v else 1.0

    gauss = np.exp(-0.5 * ((vals[:, None] - centers[None, :]) / sigma) ** 2)
    return gauss


# ============================================================
# 1. Load labeled windows
# ============================================================

DATA_FILE = "tf_dataset_CTCF_100_bp.csv"   # or your chosen file

print(f"Loading dataset from {DATA_FILE}...")
df = pd.read_csv(DATA_FILE)
# Expected columns: chrom, start, end, seq, label, pwm_score

# ============================================================
# 2. Add DNA shape features
# ============================================================

print("Opening DNA shape bigWig tracks...")
bw_tracks = init_shape_tracks()

print("Adding MGW / ProT / Roll / HelT arrays for each window...")
df_shapes = add_shape_arrays(df, bw_tracks)

print("Converting shape arrays to feature matrix...")
X_shape = shapes_to_matrix(df_shapes)          # shape: (N_samples, 4 * window_len)
y = df_shapes["label"].values.astype(int)

# PWM as 1D feature
pwm = df_shapes[["pwm_score"]].values          # shape: (N_samples, 1)

print("Initial shapes:")
print("  X_shape:", X_shape.shape)
print("  pwm    :", pwm.shape)
print("  y      :", y.shape)

# ============================================================
# 3. Clean non-finite rows based on Shape + PWM
# ============================================================

print("\nChecking non-finite values in (Shape + PWM)...")
X_full_tmp = np.concatenate([X_shape, pwm], axis=1)

any_nan = np.isnan(X_full_tmp).any()
any_posinf = np.isposinf(X_full_tmp).any()
any_neginf = np.isneginf(X_full_tmp).any()

print("  Any NaN   in X_full_tmp?", any_nan)
print("  Any +inf  in X_full_tmp?", any_posinf)
print("  Any -inf  in X_full_tmp?", any_neginf)

mask = np.isfinite(X_full_tmp).all(axis=1)
n_dropped = (~mask).sum()
print(f"  Dropping {n_dropped} rows with NaN/inf in shape or PWM")

X_shape = X_shape[mask]
pwm = pwm[mask]
y = y[mask]

print("After dropping non-finite rows:")
print("  X_shape:", X_shape.shape)
print("  pwm    :", pwm.shape)
print("  y      :", y.shape)

if X_shape.shape[0] == 0:
    raise RuntimeError("All rows were dropped during cleaning – no data left for training.")

# ============================================================
# 4. Build best feature set: Shape_plus_PWMgauss
# ============================================================

print("\nBuilding Gaussian basis expansion for PWM...")
pwm_gauss = pwm_to_gaussians(pwm, n_centers=5)
print("  pwm_gauss:", pwm_gauss.shape)

print("Building Shape_plus_PWMgauss feature matrix...")
X_full = np.concatenate([X_shape, pwm_gauss], axis=1)
print("  X_full:", X_full.shape)

n_samples = X_full.shape[0]
n_features = X_full.shape[1]

# ============================================================
# 5. Train/test split for final evaluation (85/15)
# ============================================================

indices = np.arange(n_samples)

idx_train, idx_test = train_test_split(
    indices, test_size=0.15, random_state=42, stratify=y
)

X_train = X_full[idx_train]
X_test = X_full[idx_test]
y_train = y[idx_train]
y_test = y[idx_test]

n_train = X_train.shape[0]
n_test = X_test.shape[0]

print("\nStratified train/test split:")
print("  Total samples:", n_samples)
print("  Train size   :", n_train)
print("  Test size    :", n_test)

# ============================================================
# 6. Define and train final Logistic Regression model
# ============================================================

BEST_C = 0.1  # from previous model selection

logreg_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        penalty="l2",
        C=BEST_C,
        max_iter=1000,
        n_jobs=-1,
        solver="lbfgs",
    )),
])

print("\nTraining final Logistic Regression (Shape_plus_PWMgauss, C=0.1) on train set...")
logreg_pipeline.fit(X_train, y_train)

print("Evaluating on held-out test set...")
y_pred_test = logreg_pipeline.predict(X_test)

clf = logreg_pipeline.named_steps["clf"]
if hasattr(clf, "predict_proba"):
    test_scores = logreg_pipeline.predict_proba(X_test)[:, 1]
else:
    test_scores = logreg_pipeline.decision_function(X_test)

test_acc = accuracy_score(y_test, y_pred_test)
test_roc = roc_auc_score(y_test, test_scores)
test_pr  = average_precision_score(y_test, test_scores)

print(f"\n=== Test set performance (Shape_plus_PWMgauss, C=0.1) ===")
print(f"TEST Accuracy: {test_acc:.6f}")
print(f"TEST ROC-AUC : {test_roc:.6f}")
print(f"TEST PR-AUC  : {test_pr:.6f}")
print("\nClassification report:\n",
      classification_report(y_test, y_pred_test, digits=3))

# ============================================================
# 7. Retrain on full cleaned dataset
# ============================================================

print("\nRetraining final model on the full cleaned dataset...")
logreg_pipeline.fit(X_full, y)

print("Final model trained on full dataset.")
full_clf_params = logreg_pipeline.named_steps["clf"].get_params()

# ============================================================
# 8. Save summary to file
# ============================================================

results_dir = PROJECT_ROOT / "results"
results_dir.mkdir(parents=True, exist_ok=True)

data_suffix = Path(DATA_FILE).stem.replace("tf_dataset_", "")
summary_path = results_dir / f"logreg_final_shape_pwmgauss_{data_suffix}.txt"

with open(summary_path, "w") as f:
    f.write("FINAL LOGISTIC REGRESSION MODEL (Shape_plus_PWMgauss)\n")
    f.write(f"Dataset file: {DATA_FILE}\n")
    f.write(f"Total samples used after cleaning: {n_samples}\n")
    f.write(f"Number of features: {n_features}\n\n")

    f.write("Train/test split (for evaluation):\n")
    f.write(f"  Train size: {n_train}\n")
    f.write(f"  Test size : {n_test}\n\n")

    f.write("Test set performance (85/15 split):\n")
    f.write(f"  TEST Accuracy: {test_acc:.6f}\n")
    f.write(f"  TEST ROC-AUC : {test_roc:.6f}\n")
    f.write(f"  TEST PR-AUC  : {test_pr:.6f}\n\n")

    f.write("Model hyperparameters (LogisticRegression):\n")
    f.write(json.dumps(full_clf_params, indent=2))
    f.write("\n")

print(f"\nSaved final model summary to: {summary_path}")
