import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


MODEL_PATH = "models/best_role_model.pkl"
ENCODER_PATH = "models/role_label_encoder.pkl"
GOLD_JOBS_PATH = "data/gold_v3/jobs_gold.csv"

REC_VECTORIZER_PATH = "models/job_recommender/tfidf_vectorizer.pkl"
JOB_VECTORS_PATH = "models/job_recommender/job_vectors.pkl"
REC_JOBS_PATH = "models/job_recommender/recommender_jobs.csv"


print("Loading ML1 Role Classifier...")
role_model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)

print("Loading Gold Data...")
jobs_df = pd.read_csv(GOLD_JOBS_PATH, low_memory=False)

print("Loading ML3 Job Recommender...")
rec_vectorizer = joblib.load(REC_VECTORIZER_PATH)
job_vectors = joblib.load(JOB_VECTORS_PATH)
rec_jobs_df = pd.read_csv(REC_JOBS_PATH, low_memory=False)


print("=" * 60)
print("CAREER ADVISOR ENGINE: ML1 + ML2 + ML3")
print("=" * 60)
print("Gold Records:", jobs_df.shape)
print("Recommender Jobs:", rec_jobs_df.shape)


# Later this will come from resume / Streamlit UI
user_skills = """
python sql pandas numpy machine learning
"""

user_skills_clean = user_skills.lower()


# ============================================================
# ML1: ROLE PREDICTION
# ============================================================

prediction_encoded = role_model.predict([user_skills_clean])[0]

predicted_role = label_encoder.inverse_transform(
    [prediction_encoded]
)[0]

print("\nPredicted Job Role:")
print(predicted_role)


# ============================================================
# ML2: ROLE-WISE SKILL GAP
# ============================================================

role_jobs = jobs_df[
    jobs_df["role_category"] == predicted_role
].copy()

print("\nMarket Jobs Analyzed:", role_jobs.shape[0])


all_skills = []

for skills in role_jobs["extracted_skills"].dropna():
    for skill in str(skills).split(","):
        skill = (
            skill
            .replace("[", "")
            .replace("]", "")
            .replace("'", "")
            .replace('"', "")
            .strip()
            .lower()
        )

        if skill:
            all_skills.append(skill)


skill_df = (
    pd.Series(all_skills)
    .value_counts()
    .reset_index()
)

skill_df.columns = ["skill", "job_count"]

skill_df["demand_percent"] = round(
    skill_df["job_count"] / role_jobs.shape[0] * 100,
    2
)

top_market_skills = skill_df.head(15)


available = []
missing = []

for _, row in top_market_skills.iterrows():
    skill = row["skill"]

    record = {
        "skill": skill,
        "job_count": row["job_count"],
        "demand_percent": row["demand_percent"]
    }

    if skill in user_skills_clean:
        record["status"] = "Available"
        available.append(record)
    else:
        record["status"] = "Missing"
        missing.append(record)


available_df = pd.DataFrame(available)
missing_df = pd.DataFrame(missing)

profile_match = round(
    len(available) / len(top_market_skills) * 100,
    2
)


# ============================================================
# ML3: JOB RECOMMENDATION
# ============================================================

recommendation_query = f"""
{user_skills_clean}
{predicted_role}
"""

query_vector = rec_vectorizer.transform([recommendation_query])

similarity_scores = cosine_similarity(
    query_vector,
    job_vectors
).flatten()

rec_jobs_df["similarity_score"] = similarity_scores


if "role_category" in rec_jobs_df.columns:
    filtered_jobs = rec_jobs_df[
        rec_jobs_df["role_category"] == predicted_role
    ].copy()
else:
    filtered_jobs = rec_jobs_df.copy()


recommended_jobs = (
    filtered_jobs
    .sort_values("similarity_score", ascending=False)
    .head(10)
)


# ============================================================
# FINAL REPORT
# ============================================================

print("\nPROFILE MATCH:")
print(profile_match, "%")

print("\nAVAILABLE SKILLS:")
if available_df.empty:
    print("None")
else:
    print(available_df)

print("\nMISSING SKILLS:")
if missing_df.empty:
    print("No missing skills")
else:
    print(missing_df)

print("\nLEARNING PRIORITY:")
if missing_df.empty:
    print("No priority needed.")
else:
    for i, skill in enumerate(missing_df["skill"].head(5), start=1):
        print(f"{i}. {skill}")


print("\nTOP RECOMMENDED JOBS:")

display_cols = [
    "job_title",
    "company",
    "location",
    "experience",
    "salary",
    "similarity_score"
]

available_cols = [
    col for col in display_cols
    if col in recommended_jobs.columns
]

print(
    recommended_jobs[available_cols]
    .reset_index(drop=True)
)


print("=" * 60)
print("CAREER ADVISOR COMPLETED")
print("=" * 60)
