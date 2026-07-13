import os
import re
import pandas as pd
from datetime import datetime

SILVER_PATH = "data/processed/silver_jobs.csv"
GOLD_DIR = "data/gold_v2"

os.makedirs(GOLD_DIR, exist_ok=True)

SKILL_KEYWORDS = [
    "python", "sql", "mysql", "postgresql", "mongodb",
    "pandas", "numpy", "excel", "power bi", "tableau",
    "machine learning", "deep learning", "nlp", "gen ai", "llm",
    "spark", "pyspark", "hadoop", "hive", "kafka",
    "aws", "azure", "gcp", "databricks",
    "etl", "airflow", "docker", "git",
    "statistics", "data analysis", "data visualization"
]

ROLE_MAP = {
    "data analyst": ["data analyst", "business analyst", "mis analyst"],
    "data engineer": ["data engineer", "etl developer", "big data", "spark", "hadoop"],
    "data scientist": ["data scientist", "machine learning", "ml engineer"],
    "ai engineer": ["ai engineer", "genai", "llm", "nlp"],
    "software/backend": ["backend", "software engineer", "python developer"]
}


def clean_text(x):
    if pd.isna(x):
        return ""
    return str(x).lower().strip()


def extract_skills(text):
    text = clean_text(text)
    found = []
    for skill in SKILL_KEYWORDS:
        if skill in text:
            found.append(skill)
    return ", ".join(sorted(set(found)))


def classify_role(title):
    title = clean_text(title)
    for role, keywords in ROLE_MAP.items():
        for kw in keywords:
            if kw in title:
                return role
    return "other"


def experience_group(exp):
    exp = clean_text(exp)

    nums = re.findall(r"\d+", exp)
    if not nums:
        return "not specified"

    min_exp = int(nums[0])

    if min_exp == 0:
        return "fresher"
    elif min_exp <= 2:
        return "0-2 years"
    elif min_exp <= 5:
        return "3-5 years"
    elif min_exp <= 10:
        return "6-10 years"
    else:
        return "10+ years"


def main():
    print("=" * 60)
    print("STARTING GOLD V2 PIPELINE")
    print("=" * 60)

    df = pd.read_csv(SILVER_PATH, low_memory=False)

    print("Silver Input:", df.shape)

    df["gold_processed_at"] = datetime.now()

    # Main enrichments
    df["combined_text"] = (
        df["job_title"].fillna("") + " " +
        df["skills"].fillna("") + " " +
        df["description"].fillna("")
    )

    df["extracted_skills"] = df["combined_text"].apply(extract_skills)
    df["role_category"] = df["job_title"].apply(classify_role)
    df["experience_group"] = df["experience"].apply(experience_group)

    # 1. Gold fact table
    fact_jobs = df[
        [
            "job_id", "job_title", "company", "location",
            "experience", "salary", "source", "posted_date",
            "job_url", "role_category", "experience_group",
            "extracted_skills", "gold_processed_at"
        ]
    ]

    # 2. Skill demand table
    skill_rows = []

    for _, row in df.iterrows():
        for skill in str(row["extracted_skills"]).split(","):
            skill = skill.strip()
            if skill:
                skill_rows.append({
                    "job_id": row["job_id"],
                    "skill": skill,
                    "role_category": row["role_category"],
                    "source": row["source"]
                })

    skill_demand = pd.DataFrame(skill_rows)

    # 3. Aggregation tables
    top_skills = (
        skill_demand.groupby("skill")
        .size()
        .reset_index(name="job_count")
        .sort_values("job_count", ascending=False)
    )

    role_demand = (
        df.groupby("role_category")
        .size()
        .reset_index(name="job_count")
        .sort_values("job_count", ascending=False)
    )

    company_demand = (
        df.groupby("company")
        .size()
        .reset_index(name="job_count")
        .sort_values("job_count", ascending=False)
    )

    location_demand = (
        df.groupby("location")
        .size()
        .reset_index(name="job_count")
        .sort_values("job_count", ascending=False)
    )

    experience_demand = (
        df.groupby("experience_group")
        .size()
        .reset_index(name="job_count")
        .sort_values("job_count", ascending=False)
    )

    source_summary = (
        df.groupby("source")
        .size()
        .reset_index(name="job_count")
        .sort_values("job_count", ascending=False)
    )

    # 4. ML dataset
    ml_dataset = df[
        [
            "job_title", "company", "location",
            "experience_group", "role_category",
            "extracted_skills", "source"
        ]
    ].copy()

    # 5. RAG documents
    rag_documents = pd.DataFrame({
        "job_id": df["job_id"],
        "document_text": (
            "Job Title: " + df["job_title"].fillna("") +
            "\nCompany: " + df["company"].fillna("") +
            "\nLocation: " + df["location"].fillna("") +
            "\nExperience: " + df["experience"].fillna("") +
            "\nSalary: " + df["salary"].fillna("") +
            "\nSkills: " + df["extracted_skills"].fillna("") +
            "\nDescription: " + df["description"].fillna("")
        )
    })

    # 6. Gold quality summary
    quality_summary = pd.DataFrame([
        ["silver_input_rows", len(df)],
        ["gold_fact_rows", len(fact_jobs)],
        ["unique_companies", df["company"].nunique()],
        ["unique_locations", df["location"].nunique()],
        ["unique_roles", df["role_category"].nunique()],
        ["skill_records", len(skill_demand)],
        ["rag_documents", len(rag_documents)],
        ["ml_dataset_rows", len(ml_dataset)]
    ], columns=["metric", "value"])

    # Save outputs
    fact_jobs.to_csv(f"{GOLD_DIR}/fact_jobs.csv", index=False)
    skill_demand.to_csv(f"{GOLD_DIR}/skill_demand.csv", index=False)
    top_skills.to_csv(f"{GOLD_DIR}/top_skills.csv", index=False)
    role_demand.to_csv(f"{GOLD_DIR}/role_demand.csv", index=False)
    company_demand.to_csv(f"{GOLD_DIR}/company_demand.csv", index=False)
    location_demand.to_csv(f"{GOLD_DIR}/location_demand.csv", index=False)
    experience_demand.to_csv(f"{GOLD_DIR}/experience_demand.csv", index=False)
    source_summary.to_csv(f"{GOLD_DIR}/source_summary.csv", index=False)
    ml_dataset.to_csv(f"{GOLD_DIR}/ml_dataset.csv", index=False)
    rag_documents.to_csv(f"{GOLD_DIR}/rag_documents.csv", index=False)
    quality_summary.to_csv(f"{GOLD_DIR}/gold_quality_summary.csv", index=False)

    print("Gold Fact Jobs:", fact_jobs.shape)
    print("Skill Demand:", skill_demand.shape)
    print("Top Skills:", top_skills.shape)
    print("Role Demand:", role_demand.shape)
    print("Company Demand:", company_demand.shape)
    print("Location Demand:", location_demand.shape)
    print("Experience Demand:", experience_demand.shape)
    print("ML Dataset:", ml_dataset.shape)
    print("RAG Documents:", rag_documents.shape)

    print("=" * 60)
    print("GOLD V2 PIPELINE COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
