import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import sys
from pathlib import Path
import json  # for saving hyperparameters

from tqdm import tqdm

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
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
#  Helper functions
# ============================================================

def pwm_to_gaussians(pwm, n_centers=5, sigma=None):
    """
    Expand 1D PWM scores into Gaussian basis features.

    pwm: array of shape (N, 1)
    returns: array of shape (N, n_centers)
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


class TQDMKFold(StratifiedKFold):
    """
    StratifiedKFold that wraps CV splits with a tqdm progress bar.
    Used by GridSearchCV via the 'cv' argument.
    """
    def split(self, X, y=None, groups=None):
        base_splitter = super().split(X, y, groups)
        for train_idx, val_idx in tqdm(base_splitter, desc="CV splits", leave=False):
            yield train_idx, val_idx


# ============================================================
# 1. Load labeled windows
# ============================================================

DATA_FILE = "tf_dataset_CTCF_100_testbp.csv"   # change if you picked another TF/window

print(f"Loading dataset from {DATA_FILE}...")
df = pd.read_csv(DATA_FILE)
# df columns expected: chrom, start, end, seq, label, pwm_score

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

# --- Sanity checks before cleaning (shape+PWM only) ---
X_full_tmp = np.concatenate([X_shape, pwm], axis=1)
print("Any NaN in X_full?", np.isnan(X_full_tmp).any())
print("Any +inf in X_full?", np.isposinf(X_full_tmp).any())
print("Any -inf in X_full?", np.isneginf(X_full_tmp).any())

# ---- Drop rows with any non-finite values (based on shape+PWM) ----
mask = np.isfinite(X_full_tmp).all(axis=1)
print(f"Dropping {(~mask).sum()} rows with NaN/inf in shape or PWM")

X_shape = X_shape[mask]
pwm = pwm[mask]
y = y[mask]
seqs = df_shapes["seq"].values[mask]   # keep sequences aligned with the same mask

print("After dropping non-finite rows:")
print("  X_shape:", X_shape.shape)
print("  pwm    :", pwm.shape)
print("  y      :", y.shape)
print("  Any remaining non-finite in shapes?",
      ~np.isfinite(np.concatenate([X_shape, pwm], axis=1)).all())

# ============================================================
# 3. Build extended features: PWM Gaussians 
# ============================================================

print("\nBuilding Gaussian basis expansion for PWM...")
pwm_gauss = pwm_to_gaussians(pwm, n_centers=5)
print("  pwm_gauss:", pwm_gauss.shape)

# -------- Plot vanilla PWM distribution (optional) --------
plt.figure(figsize=(7, 4))
plt.hist(pwm, bins=50)
plt.title("Distribution of raw PWM scores")
plt.xlabel("PWM score")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# -------- Plot Gaussian basis functions over PWM range (optional) --------
vals = np.linspace(pwm.min(), pwm.max(), 200)
centers = np.linspace(pwm.min(), pwm.max(), 5)
sigma = centers[1] - centers[0] if len(centers) > 1 else 1.0

plt.figure(figsize=(7, 4))
for c in centers:
    plt.plot(vals, np.exp(-0.5 * ((vals - c) / sigma) ** 2))
plt.title("Gaussian basis functions for PWM")
plt.xlabel("PWM score")
plt.ylabel("Basis value")
plt.tight_layout()
plt.show()

# -------- Define FEATURE SETS to compare --------
feature_sets = {
    "PWM_only": pwm,                                        # (N, 1)
    "PWM_gauss": pwm_gauss,                                 # (N, 5)
    "Shape_only": X_shape,                                  # (N, 4 * L)
    "Shape_plus_PWM": np.concatenate([X_shape, pwm], axis=1),
    "Shape_plus_PWMgauss": np.concatenate([X_shape, pwm_gauss], axis=1),
}

print("\nDefined feature sets:")
for name, X_fs in feature_sets.items():
    print(f"  {name:28s} -> X shape: {X_fs.shape}")

# ============================================================
# 4. Create a single train/test split (by indices)
# ============================================================

n_samples = y.shape[0]
indices = np.arange(n_samples)

idx_trainval, idx_test = train_test_split(
    indices, test_size=0.15, random_state=42, stratify=y
)

print("\nIndex-based split:")
print("  Total samples:", n_samples)
print("  Train+Val size:", len(idx_trainval))
print("  Test size     :", len(idx_test))

y_trainval = y[idx_trainval]
y_test = y[idx_test]

n_trainval = len(idx_trainval)
n_test = len(idx_test)

# ============================================================
# 5. Define Logistic Regression pipeline + hyperparameter grid
# ============================================================

logreg_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        penalty="l2",
        max_iter=1000,
        n_jobs=-1,
        solver="lbfgs",
    )),
])

param_grid = {
    "clf__C": [0.01, 0.1, 1.0, 10.0],
    # you could also add:
    # "clf__class_weight": [None, "balanced"],
}

cv = TQDMKFold(n_splits=5, shuffle=True, random_state=42)

# ============================================================
# 6. Run GridSearchCV for each FEATURE SET
# ============================================================

best_estimators = {}   # feat_name -> best_estimator
cv_scores = {}         # feat_name -> best_cv_score (mean ROC-AUC)
best_params_map = {}   # feat_name -> best hyperparameters
cv_results_list = []   # list of DataFrames (one per feature set) for detailed CV results

print("\n===== Hyperparameter search with 5-fold CV on train+val (Logistic Regression) =====")

for feat_name, X_fs in feature_sets.items():
    print(f"\n### Feature set: {feat_name} ###")

    X_trainval_fs = X_fs[idx_trainval]

    print("  Running GridSearchCV for LogisticRegression ...")
    grid = GridSearchCV(
        estimator=logreg_pipeline,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        verbose=0,
        return_train_score=True,
    )

    grid.fit(X_trainval_fs, y_trainval)

    best_estimators[feat_name] = grid.best_estimator_
    cv_scores[feat_name] = grid.best_score_
    best_params_map[feat_name] = grid.best_params_

    print(f"  Best ROC-AUC (CV mean): {grid.best_score_:.3f}")
    print("  Best params:", grid.best_params_)

    # Save detailed CV results for this feature set
    cv_res = pd.DataFrame(grid.cv_results_)
    cv_res["feature_set"] = feat_name
    cv_res["n_samples_trainval"] = X_trainval_fs.shape[0]
    cv_res["n_features"] = X_trainval_fs.shape[1]
    cv_results_list.append(cv_res)

# ============================================================
# 7. Evaluate best Logistic Regression model per FEATURE SET on TEST set
# ============================================================

print("\n===== Test set evaluation (Logistic Regression) =====")

test_results = {}  # feat_name -> metrics dict

for feat_name, X_fs in feature_sets.items():
    print(f"\n### Feature set: {feat_name} ###")

    X_trainval_fs = X_fs[idx_trainval]
    X_test_fs = X_fs[idx_test]

    best_model = best_estimators[feat_name]
    print("  Retraining best LogisticRegression on full train+val...")
    best_model.fit(X_trainval_fs, y_trainval)

    print("  Evaluating on TEST set...")
    y_pred_test = best_model.predict(X_test_fs)

    clf = best_model.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        test_scores = best_model.predict_proba(X_test_fs)[:, 1]
    elif hasattr(clf, "decision_function"):
        test_scores = best_model.decision_function(X_test_fs)
    else:
        raise ValueError("Don't know how to get scores for LogisticRegression")

    test_acc = accuracy_score(y_test, y_pred_test)
    test_roc = roc_auc_score(y_test, test_scores)
    test_pr  = average_precision_score(y_test, test_scores)

    print(f"  TEST Accuracy: {test_acc:.3f}")
    print(f"  TEST ROC-AUC : {test_roc:.3f}")
    print(f"  TEST PR-AUC  : {test_pr:.3f}")
    print("  TEST Classification report:\n",
          classification_report(y_test, y_pred_test, digits=3))

    # full model params (not just best C) if you ever want them
    full_params = clf.get_params()

    test_results[feat_name] = {
        "model": "LogisticRegression",
        "acc": test_acc,
        "roc": test_roc,
        "pr": test_pr,
        "cv_roc": cv_scores[feat_name],
        "best_params": best_params_map[feat_name],
        "full_clf_params": full_params,
        "n_samples": n_samples,
        "n_trainval": n_trainval,
        "n_test": n_test,
        "n_features": X_fs.shape[1],
    }

# ============================================================
# 8. Summary + PWM-only vs PWM-gauss + pick BEST feature set
# ============================================================

print("\n===== Summary across feature sets (Logistic Regression) =====")
for feat_name, res in test_results.items():
    print(f"{feat_name:28s}  "
          f"Acc={res['acc']:.3f}  ROC-AUC={res['roc']:.3f}  "
          f"PR-AUC={res['pr']:.3f}  CV-ROC={res['cv_roc']:.3f}  "
          f"Params={res['best_params']}  "
          f"(n={res['n_samples']}, d={res['n_features']})")

# Explicit comparison: vanilla PWM vs Gaussian PWM
if "PWM_only" in test_results and "PWM_gauss" in test_results:
    print("\n===== PWM-only vs PWM-Gaussian comparison =====")
    print(f"PWM_only   : TEST ROC-AUC = {test_results['PWM_only']['roc']:.3f}, "
          f"CV ROC-AUC = {test_results['PWM_only']['cv_roc']:.3f}")
    print(f"PWM_gauss  : TEST ROC-AUC = {test_results['PWM_gauss']['roc']:.3f}, "
          f"CV ROC-AUC = {test_results['PWM_gauss']['cv_roc']:.3f}")

# Choose best: prioritize TEST ROC, then CV ROC, then Accuracy
best_feat = max(
    test_results.items(),
    key=lambda kv: (kv[1]["roc"], kv[1]["cv_roc"], kv[1]["acc"])
)
best_name, best_res = best_feat

print("\n===== BEST FEATURE SET (by TEST ROC-AUC) =====")
print(f"Best feature set: {best_name}")
print(f"  TEST Accuracy: {best_res['acc']:.3f}")
print(f"  TEST ROC-AUC : {best_res['roc']:.3f}")
print(f"  TEST PR-AUC  : {best_res['pr']:.3f}")
print(f"  CV ROC-AUC   : {best_res['cv_roc']:.3f}")
print(f"  Best hyperparameters: {best_res['best_params']}")
print(f"  Dataset: n={best_res['n_samples']}, train+val={best_res['n_trainval']}, "
      f"test={best_res['n_test']}, d={best_res['n_features']}")

# ============================================================
# 9. Save results to files (CSV + TXT)
# ============================================================

# Per-feature-set summary (one row per feature set)
rows = []
for feat_name, res in test_results.items():
    rows.append({
        "feature_set": feat_name,
        "model": res["model"],
        "test_accuracy": res["acc"],
        "test_roc_auc": res["roc"],
        "test_pr_auc": res["pr"],
        "cv_roc_auc": res["cv_roc"],
        "best_params": json.dumps(res["best_params"]),
        "n_samples": res["n_samples"],
        "n_trainval": res["n_trainval"],
        "n_test": res["n_test"],
        "n_features": res["n_features"],
    })

results_df = pd.DataFrame(rows)

# Create results directory
results_dir = PROJECT_ROOT / "results"
results_dir.mkdir(parents=True, exist_ok=True)

# Generate name suffix based on dataset filename
data_suffix = Path(DATA_FILE).stem.replace("tf_dataset_", "")

# Output paths
csv_path = results_dir / f"logreg_results_{data_suffix}.csv"
txt_path = results_dir / f"logreg_best_feature_set_{data_suffix}.txt"
cv_csv_path = results_dir / f"logreg_cv_results_{data_suffix}.csv"

# Save per-feature summary CSV
results_df.to_csv(csv_path, index=False)

# Save best result summary (including dataset sizes)
# Save result summary (including dataset sizes) for EACH feature set
with open(txt_path, "w") as f:
    f.write("LOGISTIC REGRESSION SUMMARY PER FEATURE SET\n")
    f.write(f"Dataset file: {DATA_FILE}\n")
    f.write(f"Total samples used (after cleaning): {n_samples}\n")
    f.write(f"Train+Val size: {n_trainval}\n")
    f.write(f"Test size: {n_test}\n\n")

    for feat_name, res in test_results.items():
        f.write("========================================\n")
        f.write(f"Feature set: {feat_name}\n")
        f.write(f"TEST Accuracy: {res['acc']:.6f}\n")
        f.write(f"TEST ROC-AUC : {res['roc']:.6f}\n")
        f.write(f"TEST PR-AUC  : {res['pr']:.6f}\n")
        f.write(f"CV ROC-AUC   : {res['cv_roc']:.6f}\n")
        f.write(f"Best hyperparameters: {json.dumps(res['best_params'])}\n")
        f.write(f"Number of features: {res['n_features']}\n\n")

    # Highlight the best feature set at the end
    f.write("===== BEST FEATURE SET (by TEST ROC-AUC) =====\n")
    f.write(f"Feature set: {best_name}\n")
    f.write(f"TEST Accuracy: {best_res['acc']:.6f}\n")
    f.write(f"TEST ROC-AUC : {best_res['roc']:.6f}\n")
    f.write(f"TEST PR-AUC  : {best_res['pr']:.6f}\n")
    f.write(f"CV ROC-AUC   : {best_res['cv_roc']:.6f}\n")
    f.write(f"Best hyperparameters: {json.dumps(best_res['best_params'])}\n")
    f.write(f"Number of features: {best_res['n_features']}\n")


# Save detailed CV results (one row per hyperparameter config per feature set)
if cv_results_list:
    all_cv_df = pd.concat(cv_results_list, ignore_index=True)
    all_cv_df.to_csv(cv_csv_path, index=False)

print(f"\nSaved per-feature results to: {csv_path}")
print(f"Saved best feature summary to: {txt_path}")
print(f"Saved detailed CV results to: {cv_csv_path}")
