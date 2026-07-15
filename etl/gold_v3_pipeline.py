import os
import re
import pandas as pd
from datetime import datetime

SILVER_PATH = "data/processed/silver_jobs.csv"
GOLD_DIR = "data/gold_v3"

os.makedirs(GOLD_DIR, exist_ok=True)

SKILL_KEYWORDS = [
    "python", "sql", "excel", "power bi", "tableau", "machine learning",
    "deep learning", "nlp", "genai", "llm", "rag", "pandas", "numpy",
    "spark", "pyspark", "hadoop", "hive", "kafka", "airflow",
    "aws", "azure", "gcp", "docker", "kubernetes", "git",
    "etl", "data analysis", "data visualization", "statistics",
    "mysql", "mongodb", "postgresql", "snowflake", "databricks",
    "tensorflow", "pytorch", "scikit-learn"
]


def clean_text(x):
    if pd.isna(x):
        return ""
    return str(x).lower().strip()


def extract_skills(text):
    text = clean_text(text)
    found = []

    for skill in SKILL_KEYWORDS:
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, text):
            found.append(skill)

    return list(set(found))


def map_role(title):
    title = clean_text(title)

    if any(x in title for x in ["genai", "llm", "rag"]):
        return "GenAI Engineer"
    elif any(x in title for x in ["data engineer", "big data", "etl", "spark", "hadoop"]):
        return "Data Engineer"
    elif any(x in title for x in ["data scientist", "machine learning", "ml engineer", "ai engineer"]):
        return "Data Scientist / ML"
    elif any(x in title for x in ["business analyst", "business intelligence"]):
        return "Business Analyst"
    elif any(x in title for x in ["data analyst", "mis analyst", "power bi", "tableau", "bi analyst"]):
        return "Data Analyst / BI"
    elif any(x in title for x in ["python developer", "backend", "software engineer", "developer"]):
        return "Software / Backend"
    else:
        return "Other"


def extract_exp_min(exp):
    exp = clean_text(exp)
    nums = re.findall(r"\d+", exp)

    if nums:
        return int(nums[0])

    return None


def exp_bucket(exp):
    years = extract_exp_min(exp)

    if years is None:
        return "Not Mentioned"
    elif years == 0:
        return "Fresher"
    elif years <= 2:
        return "0-2 Years"
    elif years <= 5:
        return "3-5 Years"
    elif years <= 8:
        return "6-8 Years"
    else:
        return "9+ Years"


def parse_salary(salary):
    salary = clean_text(salary)

    if salary == "" or salary in ["not disclosed", "not mentioned", "na", "none", "nan"]:
        return pd.Series([None, None, None])

    if any(x in salary for x in ["not disclosed", "undisclosed", "not mentioned"]):
        return pd.Series([None, None, None])

    nums = re.findall(r"\d+(?:\.\d+)?", salary)

    if len(nums) == 0:
        return pd.Series([None, None, None])

    nums = [float(x) for x in nums]

    if len(nums) == 1:
        min_sal = nums[0]
        max_sal = nums[0]
    else:
        min_sal = nums[0]
        max_sal = nums[1]

    if "lakh" in salary or "lac" in salary or "lpa" in salary:
        min_sal *= 100000
        max_sal *= 100000

    avg_sal = (min_sal + max_sal) / 2

    return pd.Series([min_sal, max_sal, avg_sal])


def save_csv(df, name):
    path = os.path.join(GOLD_DIR, name)
    df.to_csv(path, index=False)
    print(f"Saved {name}: {df.shape}")


print("=" * 60)
print("STARTING GOLD V3 PIPELINE")
print("=" * 60)

df = pd.read_csv(SILVER_PATH, low_memory=False)

print("Silver input:", df.shape)

required_cols = [
    "job_title", "company", "location", "experience", "salary",
    "skills", "description", "source", "job_url", "posted_date", "job_id"
]

for col in required_cols:
    if col not in df.columns:
        df[col] = ""

df["job_title_clean"] = df["job_title"].apply(clean_text)
df["company_clean"] = df["company"].apply(clean_text)
df["location_clean"] = df["location"].apply(clean_text)
df["skills_clean"] = df["skills"].apply(clean_text)
df["description_clean"] = df["description"].apply(clean_text)

df["combined_text"] = (
    df["job_title_clean"] + " " +
    df["skills_clean"] + " " +
    df["description_clean"]
)

df["role_category"] = df["job_title"].apply(map_role)
df["experience_bucket"] = df["experience"].apply(exp_bucket)
df["extracted_skills"] = df["combined_text"].apply(extract_skills)
df["skill_count"] = df["extracted_skills"].apply(len)

df[["salary_min", "salary_max", "salary_avg"]] = df["salary"].apply(parse_salary)

df["gold_processed_at"] = datetime.now()


valid_salary_df = df[
    (df["salary_avg"].notna()) &
    (df["salary_avg"] >= 50000) &
    (df["salary_avg"] <= 10000000)
].copy()


jobs_gold = df[
    [
        "job_title", "company", "location", "experience", "salary",
        "skills", "description", "job_url", "posted_date", "source",
        "job_id", "role_category", "experience_bucket",
        "extracted_skills", "skill_count",
        "salary_min", "salary_max", "salary_avg",
        "gold_processed_at"
    ]
].copy()

jobs_gold["extracted_skills"] = jobs_gold["extracted_skills"].apply(lambda x: ", ".join(x))

save_csv(jobs_gold, "jobs_gold.csv")


skill_rows = []

for _, row in df.iterrows():
    for skill in row["extracted_skills"]:
        skill_rows.append({
            "skill": skill,
            "role_category": row["role_category"],
            "source": row["source"]
        })

skill_df = pd.DataFrame(skill_rows)

if not skill_df.empty:
    skill_demand = (
        skill_df.groupby("skill")
        .size()
        .reset_index(name="job_count")
        .sort_values("job_count", ascending=False)
    )
else:
    skill_demand = pd.DataFrame(columns=["skill", "job_count"])

save_csv(skill_demand, "skill_demand.csv")


role_demand = (
    df.groupby("role_category")
    .size()
    .reset_index(name="job_count")
    .sort_values("job_count", ascending=False)
)

save_csv(role_demand, "role_demand.csv")


company_demand = (
    df.groupby("company")
    .size()
    .reset_index(name="job_count")
    .sort_values("job_count", ascending=False)
)

save_csv(company_demand, "company_demand.csv")


location_demand = (
    df.groupby("location")
    .size()
    .reset_index(name="job_count")
    .sort_values("job_count", ascending=False)
)

save_csv(location_demand, "location_demand.csv")


experience_demand = (
    df.groupby("experience_bucket")
    .size()
    .reset_index(name="job_count")
    .sort_values("job_count", ascending=False)
)

save_csv(experience_demand, "experience_demand.csv")


salary_insights = (
    valid_salary_df
    .groupby("role_category")
    .agg(
        job_count=("job_title", "count"),
        avg_salary=("salary_avg", "mean"),
        min_salary=("salary_min", "min"),
        max_salary=("salary_max", "max")
    )
    .reset_index()
    .sort_values("avg_salary", ascending=False)
)

save_csv(salary_insights, "salary_insights.csv")


source_quality = (
    df.groupby("source")
    .agg(
        total_jobs=("job_title", "count"),
        unique_companies=("company", "nunique"),
        unique_locations=("location", "nunique"),
        salary_available_raw=("salary_avg", lambda x: x.notna().sum()),
        avg_skill_count=("skill_count", "mean")
    )
    .reset_index()
)

valid_salary_by_source = (
    valid_salary_df.groupby("source")
    .size()
    .reset_index(name="valid_salary_records")
)

source_quality = source_quality.merge(
    valid_salary_by_source,
    on="source",
    how="left"
)

source_quality["valid_salary_records"] = source_quality["valid_salary_records"].fillna(0).astype(int)

source_quality["valid_salary_availability_pct"] = (
    source_quality["valid_salary_records"] / source_quality["total_jobs"] * 100
).round(2)

source_quality["avg_skill_count"] = source_quality["avg_skill_count"].round(2)

save_csv(source_quality, "source_quality.csv")


ml_job_role_dataset = df[
    ["combined_text", "role_category", "experience_bucket", "skill_count", "source"]
].copy()

ml_job_role_dataset = ml_job_role_dataset[
    ml_job_role_dataset["role_category"] != "Other"
]

save_csv(ml_job_role_dataset, "ml_job_role_dataset.csv")


ml_salary_dataset = valid_salary_df[
    [
        "job_title", "company", "location", "role_category",
        "experience_bucket", "skill_count", "source", "salary_avg"
    ]
].copy()

save_csv(ml_salary_dataset, "ml_salary_dataset.csv")


rag_docs = df.copy()

rag_docs["document_text"] = (
    "Job Title: " + rag_docs["job_title"].astype(str) + "\n" +
    "Company: " + rag_docs["company"].astype(str) + "\n" +
    "Location: " + rag_docs["location"].astype(str) + "\n" +
    "Experience: " + rag_docs["experience"].astype(str) + "\n" +
    "Role Category: " + rag_docs["role_category"].astype(str) + "\n" +
    "Skills: " + rag_docs["skills"].astype(str) + "\n" +
    "Extracted Skills: " + rag_docs["extracted_skills"].apply(lambda x: ", ".join(x)) + "\n" +
    "Description: " + rag_docs["description"].astype(str)
)

rag_documents = rag_docs[
    ["job_id", "role_category", "source", "document_text"]
].copy()

save_csv(rag_documents, "rag_documents.csv")


quality_data = {
    "metric": [
        "total_gold_jobs",
        "unique_companies",
        "unique_locations",
        "unique_sources",
        "jobs_with_raw_salary",
        "jobs_with_valid_salary",
        "jobs_without_valid_salary",
        "jobs_with_extracted_skills",
        "avg_skills_per_job",
        "ml_role_dataset_rows",
        "ml_salary_dataset_rows",
        "rag_documents"
    ],
    "value": [
        len(jobs_gold),
        df["company"].nunique(),
        df["location"].nunique(),
        df["source"].nunique(),
        df["salary_avg"].notna().sum(),
        len(valid_salary_df),
        len(df) - len(valid_salary_df),
        (df["skill_count"] > 0).sum(),
        round(df["skill_count"].mean(), 2),
        len(ml_job_role_dataset),
        len(ml_salary_dataset),
        len(rag_documents)
    ]
}

gold_quality_report = pd.DataFrame(quality_data)

save_csv(gold_quality_report, "gold_quality_report.csv")

print("=" * 60)
print("GOLD V3 PIPELINE COMPLETED")
print("=" * 60)
