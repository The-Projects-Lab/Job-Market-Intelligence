import pandas as pd
import os
import re
from datetime import datetime


SILVER_PATH = "data/processed/silver_jobs.csv"

GOLD_DIR = "data/gold"

os.makedirs(GOLD_DIR, exist_ok=True)


print("="*50)
print("STARTING GOLD LAYER")
print("="*50)


# ==========================
# READ SILVER
# ==========================

df = pd.read_csv(SILVER_PATH)

print("Silver Input:", df.shape)


# ==========================
# GOLD JOB FACT TABLE
# ==========================

gold_jobs = df.copy()

gold_jobs["gold_processed_at"] = datetime.now()


gold_jobs.to_csv(
    f"{GOLD_DIR}/gold_jobs.csv",
    index=False
)


print("Gold Jobs:", gold_jobs.shape)



# ==========================
# SKILL DEMAND MART
# ==========================

skills = []


for skill_text in df["skills"].dropna():

    skill_list = re.split(
        ",|/|\\||;",
        str(skill_text).lower()
    )

    for skill in skill_list:

        skill = skill.strip()

        if len(skill) > 1:
            skills.append(skill)



skill_df = (
    pd.DataFrame(
        skills,
        columns=["skill"]
    )
    .value_counts()
    .reset_index(name="demand_count")
)


skill_df.to_csv(
    f"{GOLD_DIR}/skill_demand.csv",
    index=False
)

print(
    "Skills extracted:",
    skill_df.shape
)



# ==========================
# COMPANY DEMAND MART
# ==========================

company_df = (
    df.groupby("company")
    .size()
    .reset_index(name="job_count")
    .sort_values(
        "job_count",
        ascending=False
    )
)


company_df.to_csv(
    f"{GOLD_DIR}/company_demand.csv",
    index=False
)


print(
    "Companies:",
    company_df.shape
)



# ==========================
# LOCATION DEMAND MART
# ==========================


location_df = (
    df.groupby("location")
    .size()
    .reset_index(name="job_count")
    .sort_values(
        "job_count",
        ascending=False
    )
)


location_df.to_csv(
    f"{GOLD_DIR}/location_demand.csv",
    index=False
)


print(
    "Locations:",
    location_df.shape
)




# ==========================
# ROLE DEMAND MART
# ==========================


role_df = (
    df.groupby("job_title")
    .size()
    .reset_index(name="job_count")
    .sort_values(
        "job_count",
        ascending=False
    )
)


role_df.to_csv(
    f"{GOLD_DIR}/role_demand.csv",
    index=False
)


print(
    "Roles:",
    role_df.shape
)




# ==========================
# EXPERIENCE ANALYSIS
# ==========================


exp_df = (
    df.groupby("experience")
    .size()
    .reset_index(name="count")
)


exp_df.to_csv(
    f"{GOLD_DIR}/experience_analysis.csv",
    index=False
)


print(
    "Experience groups:",
    exp_df.shape
)



# ==========================
# ML DATASET
# ==========================

ml_df = df[
    [
        "job_title",
        "skills",
        "description",
        "experience"
    ]
].copy()


ml_df.to_csv(
    f"{GOLD_DIR}/ml_dataset.csv",
    index=False
)


print(
    "ML dataset:",
    ml_df.shape
)




# ==========================
# GENAI RAG DATA
# ==========================


rag_df = pd.DataFrame()


rag_df["document"] = (

    "Job Title: "
    + df["job_title"].astype(str)

    + "\nCompany: "
    + df["company"].astype(str)

    + "\nSkills Required: "
    + df["skills"].astype(str)

    + "\nExperience: "
    + df["experience"].astype(str)

    + "\nDescription: "
    + df["description"].astype(str)
)


rag_df["created_at"] = datetime.now()


rag_df.to_csv(
    f"{GOLD_DIR}/rag_documents.csv",
    index=False
)


print(
    "RAG Documents:",
    rag_df.shape
)


print("="*50)
print("GOLD LAYER COMPLETED")
print("="*50)
