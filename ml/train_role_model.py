import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score, f1_score, classification_report

from xgboost import XGBClassifier


DATA_PATH = "data/gold_v3/ml_job_role_dataset.csv"
MODEL_DIR = "models"
BEST_MODEL_PATH = "models/best_role_model.pkl"
LABEL_ENCODER_PATH = "models/role_label_encoder.pkl"

os.makedirs(MODEL_DIR, exist_ok=True)

print("=" * 70)
print("JOB ROLE CLASSIFICATION - MULTI MODEL TRAINING")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

print("Original shape:", df.shape)
print("Columns:", df.columns.tolist())

df = df.dropna(subset=["combined_text", "role_category"])

X = df["combined_text"].astype(str)
y = df["role_category"].astype(str)

print("\nTarget Distribution:")
print(y.value_counts())

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ),

    "Linear SVM": LinearSVC(
        class_weight="balanced"
    ),

    "Naive Bayes": MultinomialNB(),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    ),

    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        objective="multi:softmax",
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1
    ),

    "MLP Neural Network": MLPClassifier(
        hidden_layer_sizes=(128, 64),
        max_iter=30,
        random_state=42,
        early_stopping=True
    )
}

results = []
best_model = None
best_name = None
best_f1 = 0

for model_name, algorithm in models.items():
    print("\n" + "-" * 70)
    print("Training:", model_name)
    print("-" * 70)

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            stop_words="english"
        )),
        ("model", algorithm)
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    print("Accuracy:", round(accuracy, 4))
    print("Weighted F1:", round(weighted_f1, 4))
    print("Macro F1:", round(macro_f1, 4))

    results.append({
        "model": model_name,
        "accuracy": accuracy,
        "weighted_f1": weighted_f1,
        "macro_f1": macro_f1
    })

    if weighted_f1 > best_f1:
        best_f1 = weighted_f1
        best_model = pipeline
        best_name = model_name
        best_pred = y_pred

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="weighted_f1", ascending=False)
print(results_df)

results_df.to_csv("models/role_model_comparison.csv", index=False)

print("\n" + "=" * 70)
print("BEST MODEL")
print("=" * 70)
print("Best Model:", best_name)
print("Best Weighted F1:", round(best_f1, 4))

print("\nClassification Report for Best Model:")
print(
    classification_report(
        y_test,
        best_pred,
        target_names=label_encoder.classes_
    )
)

joblib.dump(best_model, BEST_MODEL_PATH)
joblib.dump(label_encoder, LABEL_ENCODER_PATH)

print("\nSaved Best Model:", BEST_MODEL_PATH)
print("Saved Label Encoder:", LABEL_ENCODER_PATH)
print("Saved Comparison CSV: models/role_model_comparison.csv")

print("=" * 70)
print("TRAINING COMPLETED")
print("=" * 70)
