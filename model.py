import numpy as np
import pandas as pd

from data_extraction.shape_features import init_shape_tracks, add_shape_arrays, shapes_to_matrix

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
)


# -------- 1. Load labeled windows --------
DATA_FILE = "tf_dataset_CTCF_101bp.csv"   # change name if you picked another TF/window

print(f"Loading dataset from {DATA_FILE}...")
df = pd.read_csv(DATA_FILE)
# df columns expected: chrom, start, end, seq, label, pwm_score


# -------- 2. Add DNA shape features --------
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

# --- Sanity checks before cleaning ---
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

print("After dropping non-finite rows:")
print("  X_shape:", X_shape.shape)
print("  pwm    :", pwm.shape)
print("  y      :", y.shape)
print("  Any remaining non-finite in shapes?",
      ~np.isfinite(np.concatenate([X_shape, pwm], axis=1)).all())


# -------- 3. Define FEATURE SETS to compare --------
#  - This is where you "prove" shape helps: compare PWM-only vs Shape-only vs Shape+PWM

feature_sets = {
    "PWM_only": pwm,                            # shape: (N, 1)
    "Shape_only": X_shape,                      # shape: (N, 4 * window_len)
    "Shape_plus_PWM": np.concatenate([X_shape, pwm], axis=1),
    # If  later build sequence features, we can add:
    # "Sequence_only": X_seq,
    # "Seq_plus_Shape_plus_PWM": np.concatenate([X_seq, X_shape, pwm], axis=1)
}

print("\nDefined feature sets:")
for name, X_fs in feature_sets.items():
    print(f"  {name:16s} -> X shape: {X_fs.shape}")


# -------- 4. Create a single train/test split (by indices) --------
# We want the same split across all feature sets for fair comparison.

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


# -------- 5. Define models + hyperparameter grids --------
# Pipelines (scaler + classifier)
base_models = {
    "LogisticRegression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="l2",
            max_iter=1000,
            n_jobs=-1,
            solver="lbfgs",
        )),
    ]),
    "LinearSVM": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LinearSVC(
            max_iter=5000,
        )),
    ]),
    "MLP": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            activation="relu",
            batch_size=128,
            max_iter=100,
            early_stopping=True,
            random_state=42,
        )),
    ]),
}

# Hyperparameters for each model
# (All 'clf__...' because the classifier step is named 'clf' in the pipeline.)
param_grids = {
    "LogisticRegression": {
        "clf__C": [0.01, 0.1, 1.0, 10.0],
        # we can also vary class_weight:
        # "clf__class_weight": [None, "balanced"],
    },
    "LinearSVM": {
        "clf__C": [0.01, 0.1, 1.0, 10.0],
    },
    "MLP": {
        "clf__hidden_layer_sizes": [(32,), (64,), (128,)],
        "clf__alpha": [1e-4, 1e-3, 1e-2],          # L2 regularization strength
        "clf__learning_rate_init": [1e-3, 1e-4],   # learning rate
        "clf__momentum": [0.9, 0.95],              # momentum for SGD
        # we could add:
        # "clf__beta_1": [0.9, 0.95]  # if using Adam
    },
}

# We will use ROC-AUC as the primary CV metric
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# -------- 6. Run GridSearchCV for each (feature_set, model) --------
best_estimators = {}  # (feature_set, model_name) -> best_estimator
cv_scores = {}        # (feature_set, model_name) -> best_cv_score (mean ROC-AUC)

print("\n===== Hyperparameter search with 5-fold CV on train+val =====")

for feat_name, X_fs in feature_sets.items():
    print(f"\n### Feature set: {feat_name} ###")

    X_trainval_fs = X_fs[idx_trainval]
    X_test_fs = X_fs[idx_test]   # we will use this later

    for model_name, base_model in base_models.items():
        print(f"\nRunning GridSearchCV for model: {model_name} ...")

        param_grid = param_grids[model_name]

        grid = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            verbose=0,
        )

        grid.fit(X_trainval_fs, y_trainval)

        best_estimators[(feat_name, model_name)] = grid.best_estimator_
        cv_scores[(feat_name, model_name)] = grid.best_score_

        print(f"  Best ROC-AUC (CV mean): {grid.best_score_:.3f}")
        print("  Best params:", grid.best_params_)


# -------- 7. Evaluate best model per FEATURE SET on the TEST set --------
# For each feature set, pick the model with best CV ROC-AUC, then test.

print("\n===== Test set evaluation =====")

test_results = {}  # feat_name -> (best_model_name, metrics dict)

for feat_name, X_fs in feature_sets.items():
    print(f"\n### Feature set: {feat_name} ###")

    X_trainval_fs = X_fs[idx_trainval]
    X_test_fs = X_fs[idx_test]

    # choose model with best CV ROC-AUC for this feature set
    model_candidates = {
        model_name: cv_scores[(feat_name, model_name)]
        for model_name in base_models.keys()
    }
    best_model_name = max(model_candidates.keys(), key=lambda m: model_candidates[m])
    print(f"  Best model by CV ROC-AUC: {best_model_name} "
          f"(CV ROC-AUC={model_candidates[best_model_name]:.3f})")

    # retrain best model on ALL train+val data for this feature set
    best_model = best_estimators[(feat_name, best_model_name)]
    print("  Retraining on full train+val...")
    best_model.fit(X_trainval_fs, y_trainval)

    print("  Evaluating on TEST set...")
    y_pred_test = best_model.predict(X_test_fs)

    # get scores for ROC/PR
    clf = best_model.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        test_scores = best_model.predict_proba(X_test_fs)[:, 1]
    elif hasattr(clf, "decision_function"):
        test_scores = best_model.decision_function(X_test_fs)
    else:
        raise ValueError(f"Don't know how to get scores for model {best_model_name}")

    test_acc = accuracy_score(y_test, y_pred_test)
    test_roc = roc_auc_score(y_test, test_scores)
    test_pr  = average_precision_score(y_test, test_scores)

    print(f"  TEST Accuracy: {test_acc:.3f}")
    print(f"  TEST ROC-AUC : {test_roc:.3f}")
    print(f"  TEST PR-AUC  : {test_pr:.3f}")
    print("  TEST Classification report:\n",
          classification_report(y_test, y_pred_test, digits=3))

    test_results[feat_name] = {
        "model": best_model_name,
        "acc": test_acc,
        "roc": test_roc,
        "pr": test_pr,
    }


# -------- 8. Summary: does DNA shape help? --------
print("\n===== Summary across feature sets (best model per set) =====")
for feat_name, res in test_results.items():
    print(f"{feat_name:16s}  "
          f"Model={res['model']:18s}  "
          f"Acc={res['acc']:.3f}  ROC-AUC={res['roc']:.3f}  PR-AUC={res['pr']:.3f}")
