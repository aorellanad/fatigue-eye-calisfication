import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC

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
]

numeric_unscaled_features = [
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

numeric_unscaled_transformer = Pipeline(steps=[
    ("passthrough", "passthrough")
])

categorical_transformer = Pipeline(steps=[
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num_scaled", numeric_scaled_transformer, numeric_scaled_features),
        ("num_unscaled", numeric_unscaled_transformer, numeric_unscaled_features),
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
    f1 = f1_score(y_test, y_pred, average="weighted")

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
