import os
import pickle
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE_DIR, "models", "job_recommender")

JOBS_PATH = os.path.join(MODEL_DIR, "recommender_jobs.csv")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
VECTORS_PATH = os.path.join(MODEL_DIR, "job_vectors.pkl")


def load_recommender():
    jobs_df = pd.read_csv(JOBS_PATH, low_memory=False)

    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)

    with open(VECTORS_PATH, "rb") as f:
        job_vectors = pickle.load(f)

    return jobs_df, vectorizer, job_vectors


def recommend_jobs(candidate_profile, top_n=10):
    jobs_df, vectorizer, job_vectors = load_recommender()

    candidate_profile = candidate_profile.lower()

    user_vector = vectorizer.transform([candidate_profile])

    similarity_scores = cosine_similarity(user_vector, job_vectors).flatten()

    top_indices = similarity_scores.argsort()[::-1][:top_n]

    results = jobs_df.iloc[top_indices].copy()
    results["match_score"] = similarity_scores[top_indices]
    results["match_percent"] = (results["match_score"] * 100).round(2)

    return results[
        [
            "job_title",
            "company",
            "location",
            "experience",
            "skills",
            "role_category",
            "source",
            "match_percent",
            "job_url"
        ]
    ]


if __name__ == "__main__":
    print("=" * 60)
    print("ML3 JOB RECOMMENDATION ENGINE")
    print("=" * 60)

    profile = input("\nEnter your skills/profile:\n")

    recommendations = recommend_jobs(profile, top_n=10)

    print("\nTOP MATCHING JOBS:\n")
    print(recommendations.to_string(index=False))
