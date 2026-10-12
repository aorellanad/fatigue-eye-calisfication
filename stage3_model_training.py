import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score
)
from sklearn.model_selection import train_test_split, LeaveOneGroupOut
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC

print(sklearn.__version__)

# ==========================
# load dataset
# ==========================
DATASET_PATH = "dataset/training_dataset.csv"
df = pd.read_csv(DATASET_PATH, sep=";")

FATIGUE_LABELS = {
    0: "No fatigue",
    1: "Slight fatigue",
    2: "Moderated fatigue",
    3: "High fatigue"
}

# ========================
# set target and features
# ========================
TARGET = "self_report_final_score"

FEATURES = [
    "age",
    "glasses",
    "hours_sleep",
    "caffeine_last_6h",

    "session_minutes",
    "blink_baseline_ratio",
    "total_blinks",
    "avg_blinks",
    "std_blinks",
]

X = df[FEATURES]
y = df[TARGET]

# ===================
# feature types
# ===================
numeric_scaled_features = [
    "age",
    "hours_sleep",
    "session_minutes",
    "blink_baseline_ratio",
    "total_blinks",
    "avg_blinks",
    "std_blinks",
]

categorical_features = [
    "glasses",
    "caffeine_last_6h",
]

# =================
# preprocessing
# =================
numeric_scaled_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num_scaled", numeric_scaled_transformer, numeric_scaled_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# ===================
# train/test split
# ===================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==============
# models
# ==============
models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ),
    "SVM": SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale"
    ),
    "MLP": MLPClassifier(
        hidden_layer_sizes=(20,),
        solver="lbfgs",
        max_iter=1000,
        random_state=42
    )
}

# ========================
# training + evaluation
# ========================
results = []

for name, model in models.items():
    print(f"\n🔹 Training {name}")

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    results.append({
        "Model": name,
        "Accuracy": acc,
        "F1-score": f1
    })

    print("Accuracy:", acc)
    print("F1-score:", f1)

    print("\nClassification report:")
    target_names = [FATIGUE_LABELS[c] for c in pipeline.classes_]
    print(classification_report(y_test,
                                y_pred,
                                labels=pipeline.classes_,
                                target_names=target_names,
                                zero_division=0
                                ))

    # confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=pipeline.classes_)

    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=target_names,
                yticklabels=target_names
                )
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# =======================
# results comparison
# =======================
results_df = pd.DataFrame(results)

best_model_name = results_df.sort_values(
    by="F1-score", ascending=False
).iloc[0]["Model"]

if best_model_name == "MLP":
    best_model = models["MLP"]

    joblib.dump(best_model, "models/mlp_ocular_fatigue_model.pkl")
    print("Modelo MLP exportado como 'mlp_ocular_fatigue_model.pkl'")

results_df.set_index("Model")[["Accuracy", "F1-score"]].plot(
    kind="bar",
    figsize=(8, 5),
    ylim=(0, 1),
    rot=0
)

plt.title("Model Comparison – Ocular Fatigue Classification")
plt.ylabel("Score")
plt.grid(axis="y")
plt.tight_layout()
plt.show()


GROUP_COL = "participant_id"
groups = df[GROUP_COL]

logo = LeaveOneGroupOut()
n_subjects = groups.nunique()
print(f"\n===== LOSO: {n_subjects} subjects -> {n_subjects} folds =====")

loso_results = []
loso_pooled = {name: {"y_true": [], "y_pred": []} for name in models}

N_BOOT = 10_000
rng = np.random.default_rng(42)
boot_idx = rng.integers(0, n_subjects, size=(N_BOOT, n_subjects))

for name, model in models.items():
    print(f"\n🔹 LOSO - {name}")

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    fold_accuracies = []
    fold_f1_scores = []

    for train_idx, test_idx in logo.split(X, y, groups):
        X_train_fold, X_test_fold = X.iloc[train_idx], X.iloc[test_idx]
        y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]

        pipeline.fit(X_train_fold, y_train_fold)
        y_pred_fold = pipeline.predict(X_test_fold)

        fold_accuracies.append(accuracy_score(y_test_fold, y_pred_fold))
        fold_f1_scores.append(
            f1_score(y_test_fold, y_pred_fold, average="macro", zero_division=0)
        )

        loso_pooled[name]["y_true"].extend(y_test_fold.tolist())
        loso_pooled[name]["y_pred"].extend(y_pred_fold.tolist())

    mean_acc, std_acc = np.mean(fold_accuracies), np.std(fold_accuracies)
    mean_f1, std_f1 = np.mean(fold_f1_scores), np.std(fold_f1_scores)
    acc_ci = np.percentile(np.asarray(fold_accuracies)[boot_idx].mean(axis=1), [2.5, 97.5])
    f1_ci = np.percentile(np.asarray(fold_f1_scores)[boot_idx].mean(axis=1), [2.5, 97.5])

    loso_results.append({
        "Model": name,
        "LOSO Accuracy (mean)": mean_acc,
        "LOSO Accuracy (std)": std_acc,
        "LOSO F1-macro (mean)": mean_f1,
        "LOSO F1-macro (std)": std_f1,
        "LOSO Accuracy CI low": acc_ci[0],
        "LOSO Accuracy CI high": acc_ci[1],
        "LOSO F1-macro CI low": f1_ci[0],
        "LOSO F1-macro CI high": f1_ci[1],
        "N folds": len(fold_accuracies)
    })

    print(f"Per-subject accuracy: mean={mean_acc:.3f}  std={std_acc:.3f}")
    print(f"Per-subject F1-macro: mean={mean_f1:.3f}  std={std_f1:.3f}")
    print(f"Accuracy 95% CI: [{acc_ci[0]:.2f}, {acc_ci[1]:.2f}]")
    print(f"F1-macro 95% CI: [{f1_ci[0]:.2f}, {f1_ci[1]:.2f}]")


    y_true_all = loso_pooled[name]["y_true"]
    y_pred_all = loso_pooled[name]["y_pred"]

    present_labels = sorted(set(y_true_all) | set(y_pred_all))
    target_names = [FATIGUE_LABELS[c] for c in present_labels]

    print("\nPooled out-of-fold classification report (LOSO):")
    print(classification_report(
        y_true_all, y_pred_all,
        labels=present_labels,
        target_names=target_names,
        zero_division=0
    ))

    cm_loso = confusion_matrix(y_true_all, y_pred_all, labels=present_labels)
    sns.heatmap(cm_loso, annot=True, fmt="d",
                xticklabels=target_names,
                yticklabels=target_names)
    plt.title(f"Confusion Matrix - {name} (LOSO, pooled)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


loso_results_df = pd.DataFrame(loso_results)
print("\n===== LOSO summary (subject-independent) =====")
print(loso_results_df)

summary_means = loso_results_df.set_index("Model")[
    ["LOSO Accuracy (mean)", "LOSO F1-macro (mean)"]
]
summary_stds = loso_results_df.set_index("Model")[
    ["LOSO Accuracy (std)", "LOSO F1-macro (std)"]
].values.T

summary_means.plot(
    kind="bar",
    figsize=(8, 5),
    ylim=(0, 1),
    rot=0,
    yerr=summary_stds,
    capsize=4
)
plt.title("Model Comparison – LOSO Validation (subject-independent)")
plt.ylabel("Score")
plt.grid(axis="y")
plt.tight_layout()
plt.show()