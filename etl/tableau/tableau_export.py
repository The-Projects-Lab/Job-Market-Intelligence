import os
import ast
import pandas as pd

BASE_DIR = "/home/sunbeam/job-market-intelligence"
GOLD_DIR = os.path.join(BASE_DIR, "data", "gold_v3")
OUT_DIR = os.path.join(BASE_DIR, "data", "tableau")

os.makedirs(OUT_DIR, exist_ok=True)


def read_csv(name):
    path = os.path.join(GOLD_DIR, name)
    return pd.read_csv(path, low_memory=False)


def save(df, name):
    path = os.path.join(OUT_DIR, name)
    df.to_csv(path, index=False)
    print(f"Saved {name}: {df.shape}")


def parse_skills(value):
    """
    Handles extracted_skills stored as:
    - list string: ['python', 'sql']
    - comma string: python, sql
    - normal text
    """
    if pd.isna(value):
        return []

    value = str(value).strip()

    if value == "":
        return []

    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return [str(x).strip().lower() for x in parsed if str(x).strip()]
    except Exception:
        pass

    if "," in value:
        return [x.strip().lower() for x in value.split(",") if x.strip()]

    return [value.lower()]


print("=" * 60)
print("CREATING TABLEAU EXPORT FILES")
print("=" * 60)

jobs = read_csv("jobs_gold.csv")
skill_demand = read_csv("skill_demand.csv")
role_demand = read_csv("role_demand.csv")
company_demand = read_csv("company_demand.csv")
location_demand = read_csv("location_demand.csv")
experience_demand = read_csv("experience_demand.csv")
salary_insights = read_csv("salary_insights.csv")
source_quality = read_csv("source_quality.csv")

print("Gold Jobs:", jobs.shape)

# -------------------------------------------------------
# 1. KPI Summary
# -------------------------------------------------------

kpi_summary = pd.DataFrame({
    "metric": [
        "Total Jobs",
        "Unique Companies",
        "Unique Locations",
        "Sources",
        "Role Categories",
        "Skills Extracted",
        "Records for RAG",
        "ML Role Dataset Records"
    ],
    "value": [
        len(jobs),
        jobs["company"].nunique() if "company" in jobs.columns else 0,
        jobs["location"].nunique() if "location" in jobs.columns else 0,
        jobs["source"].nunique() if "source" in jobs.columns else 0,
        jobs["role_category"].nunique() if "role_category" in jobs.columns else 0,
        len(skill_demand),
        len(jobs),
        56063
    ]
})

save(kpi_summary, "tableau_kpi_summary.csv")


# -------------------------------------------------------
# 2. Main Job Overview File
# -------------------------------------------------------

overview_cols = [
    "job_id",
    "job_title",
    "company",
    "location",
    "experience",
    "salary",
    "source",
    "role_category",
    "experience_bucket",
    "skill_count",
    "salary_min",
    "salary_max",
    "salary_avg",
    "posted_date"
]

available_cols = [col for col in overview_cols if col in jobs.columns]

job_overview = jobs[available_cols].copy()

if "salary_avg" in job_overview.columns:
    job_overview["salary_avg"] = pd.to_numeric(job_overview["salary_avg"], errors="coerce")

if "skill_count" in job_overview.columns:
    job_overview["skill_count"] = pd.to_numeric(job_overview["skill_count"], errors="coerce").fillna(0)

save(job_overview, "tableau_job_overview.csv")


# -------------------------------------------------------
# 3. Skill Demand with Percentage
# -------------------------------------------------------

skill_demand_clean = skill_demand.copy()

if "job_count" in skill_demand_clean.columns:
    skill_demand_clean["demand_percent"] = round(
        (skill_demand_clean["job_count"] / len(jobs)) * 100, 2
    )

skill_demand_clean = skill_demand_clean.sort_values(
    by="job_count", ascending=False
)

save(skill_demand_clean, "tableau_skill_demand.csv")


# -------------------------------------------------------
# 4. Role Demand with Percentage
# -------------------------------------------------------

role_demand_clean = role_demand.copy()

if "job_count" in role_demand_clean.columns:
    role_demand_clean["demand_percent"] = round(
        (role_demand_clean["job_count"] / len(jobs)) * 100, 2
    )

role_demand_clean = role_demand_clean.sort_values(
    by="job_count", ascending=False
)

save(role_demand_clean, "tableau_role_demand.csv")


# -------------------------------------------------------
# 5. Top Companies
# -------------------------------------------------------

company_demand_clean = company_demand.copy()

if "job_count" in company_demand_clean.columns:
    company_demand_clean = company_demand_clean.sort_values(
        by="job_count", ascending=False
    ).head(100)

save(company_demand_clean, "tableau_top_companies.csv")


# -------------------------------------------------------
# 6. Top Locations
# -------------------------------------------------------

location_demand_clean = location_demand.copy()

if "job_count" in location_demand_clean.columns:
    location_demand_clean = location_demand_clean.sort_values(
        by="job_count", ascending=False
    ).head(100)

save(location_demand_clean, "tableau_top_locations.csv")


# -------------------------------------------------------
# 7. Source-wise Role Matrix
# -------------------------------------------------------

if {"source", "role_category"}.issubset(jobs.columns):
    source_role_matrix = (
        jobs.groupby(["source", "role_category"])
        .size()
        .reset_index(name="job_count")
        .sort_values("job_count", ascending=False)
    )

    source_role_matrix["demand_percent"] = round(
        (source_role_matrix["job_count"] / len(jobs)) * 100, 2
    )

    save(source_role_matrix, "tableau_source_role_matrix.csv")


# -------------------------------------------------------
# 8. Role-Skill Matrix
# -------------------------------------------------------

if {"role_category", "extracted_skills"}.issubset(jobs.columns):
    rows = []

    for _, row in jobs[["role_category", "extracted_skills"]].iterrows():
        role = row["role_category"]
        skills = parse_skills(row["extracted_skills"])

        for skill in skills:
            rows.append({
                "role_category": role,
                "skill": skill
            })

    role_skill_df = pd.DataFrame(rows)

    if not role_skill_df.empty:
        role_skill_matrix = (
            role_skill_df.groupby(["role_category", "skill"])
            .size()
            .reset_index(name="job_count")
            .sort_values(["role_category", "job_count"], ascending=[True, False])
        )

        save(role_skill_matrix, "tableau_role_skill_matrix.csv")


# -------------------------------------------------------
# 9. Salary Insights
# -------------------------------------------------------

salary_clean = salary_insights.copy()
save(salary_clean, "tableau_salary_insights.csv")


# -------------------------------------------------------
# 10. Source Quality
# -------------------------------------------------------

source_quality_clean = source_quality.copy()
save(source_quality_clean, "tableau_source_quality.csv")


# -------------------------------------------------------
# 11. Experience Demand
# -------------------------------------------------------

experience_clean = experience_demand.copy()

if "job_count" in experience_clean.columns:
    experience_clean["demand_percent"] = round(
        (experience_clean["job_count"] / len(jobs)) * 100, 2
    )

save(experience_clean, "tableau_experience_demand.csv")


print("=" * 60)
print("TABLEAU EXPORT COMPLETED")
print("Output folder:", OUT_DIR)
print("=" * 60)
