import numpy as np
import pandas as pd
import sys
from pathlib import Path
from itertools import product

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
)

# ---------- make project root importable ----------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from data_extraction.shape_features import init_shape_tracks, add_shape_arrays, shapes_to_matrix


# ============================================================
# 1. Load labeled windows
# ============================================================

DATA_FILE = "tf_dataset_CTCF_100_testbp.csv"   # change name if needed

print(f"Loading dataset from {DATA_FILE}...")
df = pd.read_csv(DATA_FILE)
# expected columns: chrom, start, end, seq, label, pwm_score


# ============================================================
# 2. Add DNA shape features
# ============================================================

print("Opening DNA shape bigWig tracks...")
bw_tracks = init_shape_tracks()

print("Adding MGW / ProT / Roll / HelT arrays for each window...")
df_shapes = add_shape_arrays(df, bw_tracks)

print("Converting shape arrays to feature matrix...")
X_shape = shapes_to_matrix(df_shapes)          # (N, 4 * window_len)
y = df_shapes["label"].values.astype(int)

# PWM as 1D feature
pwm = df_shapes[["pwm_score"]].values          # (N, 1)

print("Initial shapes:")
print("  X_shape:", X_shape.shape)
print("  pwm    :", pwm.shape)
print("  y      :", y.shape)

# --- Sanity checks before cleaning (shape+PWM) ---
X_full_tmp = np.concatenate([X_shape, pwm], axis=1)
print("Any NaN in X_full?", np.isnan(X_full_tmp).any())
print("Any +inf in X_full?", np.isposinf(X_full_tmp).any())
print("Any -inf in X_full?", np.isneginf(X_full_tmp).any())

# ---- Drop rows with any non-finite values ----
mask = np.isfinite(X_full_tmp).all(axis=1)
print(f"Dropping {(~mask).sum()} rows with NaN/inf in shape or PWM")

X_shape = X_shape[mask]
pwm = pwm[mask]
y = y[mask]
seqs = df_shapes["seq"].values[mask]   # keep sequences aligned

print("After dropping non-finite rows:")
print("  X_shape:", X_shape.shape)
print("  pwm    :", pwm.shape)
print("  y      :", y.shape)
print("  Any remaining non-finite in shapes?",
      ~np.isfinite(np.concatenate([X_shape, pwm], axis=1)).all())


# ============================================================
# 3. Build k-mer features and define FEATURE SETS
# ============================================================

print("Building 3-mer sequence features...")

feature_sets = {
    "PWM_only": pwm,                            # (N, 1)
    "Shape_only": X_shape,                      # (N, 4 * window_len)
    "Shape_plus_PWM": np.concatenate([X_shape, pwm], axis=1),
}

print("\nDefined feature sets:")
for name, X_fs in feature_sets.items():
    print(f"  {name:24s} -> X shape: {X_fs.shape}")


# ============================================================
# 4. Train/test split
# ============================================================

n_samples = y.shape[0]
indices = np.arange(n_samples)

idx_trainval, idx_test = train_test_split(
    indices, test_size=0.15, random_state=42, stratify=y
)

print("\nIndex-based split:")
print("  Train+Val size:", len(idx_trainval))
print("  Test size     :", len(idx_test))

y_trainval = y[idx_trainval]
y_test = y[idx_test]


# ============================================================
# 5. MLP pipeline + hyperparameter grid
# ============================================================

mlp_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", MLPClassifier(
        solver="adam",
        activation="relu",
        max_iter=200,
        early_stopping=True,
        random_state=42,
    )),
])

mlp_param_grid = {
    "clf__hidden_layer_sizes": [
        (64,),
        (128,),
        (128, 64),
    ],
    "clf__alpha": [1e-4, 1e-3, 1e-2],          # L2 regularization
    "clf__learning_rate_init": [1e-3, 1e-4],  # learning rate
    "clf__batch_size": [64, 128],             # minibatch size
    # optionally could also tune activation:
    # "clf__activation": ["relu", "tanh"],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# ============================================================
# 6. GridSearchCV per feature set
# ============================================================

best_estimators = {}
cv_scores = {}

print("\n===== Hyperparameter search with 5-fold CV on train+val (MLP + Adam) =====")

for feat_name, X_fs in feature_sets.items():
    print(f"\n### Feature set: {feat_name} ###")
    X_trainval_fs = X_fs[idx_trainval]

    print("  Running GridSearchCV for MLPClassifier ...")
    grid = GridSearchCV(
        estimator=mlp_pipeline,
        param_grid=mlp_param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        verbose=0,
    )

    grid.fit(X_trainval_fs, y_trainval)

    best_estimators[feat_name] = grid.best_estimator_
    cv_scores[feat_name] = grid.best_score_

    print(f"  Best ROC-AUC (CV mean): {grid.best_score_:.3f}")
    print("  Best params:", grid.best_params_)


# ============================================================
# 7. Test evaluation
# ============================================================

print("\n===== Test set evaluation (MLP + Adam) =====")

test_results = {}

for feat_name, X_fs in feature_sets.items():
    print(f"\n### Feature set: {feat_name} ###")

    X_trainval_fs = X_fs[idx_trainval]
    X_test_fs = X_fs[idx_test]

    best_model = best_estimators[feat_name]
    print("  Retraining best MLP on full train+val...")
    best_model.fit(X_trainval_fs, y_trainval)

    print("  Evaluating on TEST set...")
    y_pred_test = best_model.predict(X_test_fs)

    clf = best_model.named_steps["clf"]
    # MLPClassifier supports predict_proba
    test_scores = clf.predict_proba(X_test_fs)[:, 1]

    test_acc = accuracy_score(y_test, y_pred_test)
    test_roc = roc_auc_score(y_test, test_scores)
    test_pr  = average_precision_score(y_test, test_scores)

    print(f"  TEST Accuracy: {test_acc:.3f}")
    print(f"  TEST ROC-AUC : {test_roc:.3f}")
    print(f"  TEST PR-AUC  : {test_pr:.3f}")
    print("  TEST Classification report:\n",
          classification_report(y_test, y_pred_test, digits=3))

    test_results[feat_name] = {
        "model": "MLP_Adam",
        "acc": test_acc,
        "roc": test_roc,
        "pr": test_pr,
        "cv_roc": cv_scores[feat_name],
    }


# ============================================================
# 8. Summary + best feature set (with tie-breaking)
# ============================================================

print("\n===== Summary across feature sets (MLP + Adam) =====")
for feat_name, res in test_results.items():
    print(f"{feat_name:24s}  "
          f"Acc={res['acc']:.3f}  ROC-AUC={res['roc']:.3f}  "
          f"PR-AUC={res['pr']:.3f}  CV-ROC={res['cv_roc']:.3f}")

# Choose best: prioritize TEST ROC, then CV ROC, then Accuracy
best_feat = max(
    test_results.items(),
    key=lambda kv: (kv[1]["roc"], kv[1]["cv_roc"], kv[1]["acc"])
)
best_name, best_res = best_feat

print("\n===== BEST FEATURE SET (by TEST ROC-AUC, MLP + Adam) =====")
print(f"Best feature set: {best_name}")
print(f"  TEST Accuracy: {best_res['acc']:.3f}")
print(f"  TEST ROC-AUC : {best_res['roc']:.3f}")
print(f"  TEST PR-AUC  : {best_res['pr']:.3f}")
print(f"  CV ROC-AUC   : {best_res['cv_roc']:.3f}")


# ============================================================
# 9. Save results to files
# ============================================================
# === NEW: save results to files ===

# Build a DataFrame from test_results
rows = []
for feat_name, res in test_results.items():
    rows.append({
        "feature_set": feat_name,
        "model": res["model"],
        "test_accuracy": res["acc"],
        "test_roc_auc": res["roc"],
        "test_pr_auc": res["pr"],
        "cv_roc_auc": res["cv_roc"],
    })
results_df = pd.DataFrame(rows)

# Derive a suffix from DATA_FILE, e.g. "CTCF_100bp" from "tf_dataset_CTCF_100bp.csv"
data_suffix = Path(DATA_FILE).stem.replace("tf_dataset_", "")

results_dir = PROJECT_ROOT / "results"
results_dir.mkdir(parents=True, exist_ok=True)

csv_path = results_dir / f"mlp_results_{data_suffix}.csv"
txt_path = results_dir / f"mlp_best_feature_set_{data_suffix}.txt"

results_df.to_csv(csv_path, index=False)

with open(txt_path, "w") as f:
    f.write("BEST FEATURE SET (by TEST ROC-AUC, MLP + Adam)\n")
    f.write(f"Feature set: {best_name}\n")
    f.write(f"TEST Accuracy: {best_res['acc']:.6f}\n")
    f.write(f"TEST ROC-AUC : {best_res['roc']:.6f}\n")
    f.write(f"TEST PR-AUC  : {best_res['pr']:.6f}\n")
    f.write(f"CV ROC-AUC   : {best_res['cv_roc']:.6f}\n")

print(f"\nSaved per-feature results to: {csv_path}")
print(f"Saved best feature summary to: {txt_path}")
