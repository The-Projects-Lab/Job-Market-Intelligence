import os
import pickle
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

INPUT_PATH = os.path.join(BASE_DIR, "data", "gold_v3", "jobs_gold.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models", "job_recommender")

os.makedirs(MODEL_DIR, exist_ok=True)

print("=" * 60)
print("TRAINING ML3 JOB RECOMMENDATION ENGINE")
print("=" * 60)

df = pd.read_csv(INPUT_PATH, low_memory=False)

print("Input shape:", df.shape)
print("Columns:", df.columns.tolist())

required_cols = [
    "job_title",
    "company",
    "location",
    "experience",
    "skills",
    "description",
    "role_category"
]

for col in required_cols:
    if col not in df.columns:
        df[col] = ""

df = df.fillna("")

df["recommendation_text"] = (
    df["job_title"].astype(str) + " " +
    df["role_category"].astype(str) + " " +
    df["skills"].astype(str) + " " +
    df["description"].astype(str) + " " +
    df["location"].astype(str) + " " +
    df["experience"].astype(str)
)

df["recommendation_text"] = (
    df["recommendation_text"]
    .str.lower()
    .str.replace(r"[^a-zA-Z0-9+#.\s]", " ", regex=True)
    .str.replace(r"\s+", " ", regex=True)
)

vectorizer = TfidfVectorizer(
    max_features=7000,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2
)

job_vectors = vectorizer.fit_transform(df["recommendation_text"])

print("TF-IDF matrix shape:", job_vectors.shape)

df_export = df[
    [
        "job_title",
        "company",
        "location",
        "experience",
        "skills",
        "role_category",
        "source",
        "job_url",
        "recommendation_text"
    ]
].copy()

df_export.to_csv(os.path.join(MODEL_DIR, "recommender_jobs.csv"), index=False)

with open(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"), "wb") as f:
    pickle.dump(vectorizer, f)

with open(os.path.join(MODEL_DIR, "job_vectors.pkl"), "wb") as f:
    pickle.dump(job_vectors, f)

print("Saved:")
print(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
print(os.path.join(MODEL_DIR, "job_vectors.pkl"))
print(os.path.join(MODEL_DIR, "recommender_jobs.csv"))

print("=" * 60)
print("ML3 JOB RECOMMENDER TRAINING COMPLETED")
print("=" * 60)
