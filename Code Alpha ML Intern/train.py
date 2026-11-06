import pandas as pd
from pathlib import Path
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)

# --------------------------------------------------
# 1. PATHS AND DATA
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Credit Score" / "data" / "engineered_credit_data.csv"

# All generated files will be saved here:
OUTPUT_DIR = BASE_DIR / "Credit Score" / "model_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

target = "default payment next month"
if target not in df.columns:
    raise ValueError(f"Target column '{target}' was not found.")

X = df.drop(columns=[target])
y = df[target]

# Ensure all features are numeric and contain no missing values.
X = X.apply(pd.to_numeric, errors="coerce")
if X.isnull().any().any():
    raise ValueError("Features contain missing or non-numeric values. Clean them first.")

# --------------------------------------------------
# 2. TRAIN-TEST SPLIT
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# --------------------------------------------------
# 3. DEFINE MODELS (same settings as your old script)
# --------------------------------------------------
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000))
    ]),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=8,
        random_state=42,
        class_weight="balanced"
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )
}

results = []
test_probabilities = {}
trained_models = {}

# --------------------------------------------------
# 4. TRAIN, EVALUATE AND SAVE CONFUSION MATRICES
# --------------------------------------------------
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    trained_models[name] = model

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    test_probabilities[name] = probabilities

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1 Score": f1_score(y_test, predictions, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, probabilities)
    })

    print(f"\n{name} classification report:")
    print(classification_report(y_test, predictions, zero_division=0))

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        display_labels=["No Default", "Default"],
        cmap="Blues",
        values_format="d"
    )
    plt.title(f"{name} - Confusion Matrix")
    plt.tight_layout()

    safe_name = name.lower().replace(" ", "_")
    plt.savefig(OUTPUT_DIR / f"{safe_name}_confusion_matrix.png", dpi=300)
    plt.close()

# --------------------------------------------------
# 5. SAVE MODEL COMPARISON
# --------------------------------------------------
results_df = pd.DataFrame(results)
print("\nMODEL COMPARISON")
print(results_df.to_string(index=False))
results_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

# --------------------------------------------------
# 6. SAVE COMBINED ROC CURVE
# --------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))
for name, probabilities in test_probabilities.items():
    RocCurveDisplay.from_predictions(
        y_test, probabilities, name=name, ax=ax
    )

ax.plot([0, 1], [0, 1], "k--", label="Random Classifier")
ax.set_title("ROC Curve Comparison")
ax.legend()
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "roc_curve_comparison.png", dpi=300)
plt.close(fig)

# --------------------------------------------------
# 7. SAVE RANDOM FOREST FEATURE IMPORTANCE
# --------------------------------------------------
random_forest = trained_models["Random Forest"]

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": random_forest.feature_importances_
}).sort_values("Importance", ascending=False)

importance_df.to_csv(
    OUTPUT_DIR / "random_forest_feature_importance.csv",
    index=False
)

top_features = importance_df.head(15).sort_values("Importance")
fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(top_features["Feature"], top_features["Importance"])
ax.set_title("Random Forest - Top 15 Feature Importances")
ax.set_xlabel("Importance")
fig.tight_layout()
fig.savefig(
    OUTPUT_DIR / "random_forest_feature_importance.png",
    dpi=300
)
plt.close(fig)

# --------------------------------------------------
# 8. SAVE TRAINED RANDOM FOREST MODEL
# --------------------------------------------------
joblib.dump(
    random_forest,
    OUTPUT_DIR / "random_forest_credit_model.joblib"
)

print("\nTraining and evaluation completed successfully.")
print(f"Output directory: {OUTPUT_DIR.resolve()}")
print("\nFiles saved:")
for saved_file in sorted(OUTPUT_DIR.iterdir()):
    print(saved_file.name)
