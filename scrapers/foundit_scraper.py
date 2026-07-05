import sys
import time
import re
from pathlib import Path
from datetime import datetime

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.roles import ROLE_CATEGORIES

SOURCE = "foundit"
BASE_URL = "https://www.foundit.in/middleware/jobsearch"

OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / SOURCE
BATCH_DIR = OUTPUT_DIR / "batches"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
BATCH_DIR.mkdir(parents=True, exist_ok=True)

LIMIT = 15
MAX_START = 3000

HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "X-Source-Country": "IN",
    "X-Language-Code": "EN",
    "X-Source-Site-Context": "rexmonster",
    "Referer": "https://www.foundit.in/",
    "User-Agent": "Mozilla/5.0",
}


def get_all_roles():
    roles = []
    for category_roles in ROLE_CATEGORIES.values():
        roles.extend(category_roles)
    return list(dict.fromkeys(roles))


def safe_text(value):
    if value is None:
        return None
    if isinstance(value, list):
        return ", ".join(map(str, value))
    if isinstance(value, dict):
        return str(value)
    return str(value).strip()


def extract_job(job, searched_role):
    min_exp = job.get("minimumExperience") or {}
    max_exp = job.get("maximumExperience") or {}
    min_salary = job.get("minimumSalary") or {}
    max_salary = job.get("maximumSalary") or {}

    return {
        "job_id": job.get("jobId"),
        "title": safe_text(job.get("title")),
        "company": safe_text(job.get("companyName")),
        "location": safe_text(job.get("locations")),
        "experience": f"{min_exp.get('years')} - {max_exp.get('years')}",
        "min_exp": min_exp.get("years"),
        "max_exp": max_exp.get("years"),
        "salary": safe_text(job.get("salary")),
        "min_salary": min_salary.get("absoluteValue"),
        "max_salary": max_salary.get("absoluteValue"),
        "skills": safe_text(job.get("skills")),
        "description": safe_text(job.get("description")),
        "posted_date": safe_text(job.get("postedDate")),
        "updated_at": safe_text(job.get("updatedAt")),
        "employment_type": safe_text(job.get("employmentTypes")),
        "industry": safe_text(job.get("industries")),
        "apply_url": safe_text(job.get("redirectUrl")),
        "searched_role": searched_role,
        "source": SOURCE,
        "scraped_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def scrape_role(session, role):
    rows = []
    start = 0

    print(f"\nScraping Foundit: {role}")

    while start <= MAX_START:
        params = {
            "start": start,
            "sort": 1,
            "limit": LIMIT,
            "query": role,
        }

        try:
            response = session.get(BASE_URL, params=params, timeout=30)
            print(f"{role} | start={start} | status={response.status_code}")

            if response.status_code != 200:
                break

            data = response.json()
            jobs = data.get("jobSearchResponse", {}).get("data", [])

            valid_jobs = [
                job for job in jobs
                if isinstance(job, dict) and job.get("title")
            ]

            if not valid_jobs:
                break

            for job in valid_jobs:
                rows.append(extract_job(job, role))

            start += LIMIT
            time.sleep(1)

        except Exception as e:
            print("Error:", e)
            break

    return rows


def build_master_from_batches():
    files = list(BATCH_DIR.glob("foundit_*.csv"))

    if not files:
        return

    dfs = [pd.read_csv(file) for file in files]
    master = pd.concat(dfs, ignore_index=True)

    master.to_csv(OUTPUT_DIR / "foundit_master_raw.csv", index=False)

    dedup = master.drop_duplicates(subset=["job_id"], keep="last")
    dedup.to_csv(OUTPUT_DIR / "foundit_master_dedup.csv", index=False)

    print("\nFOUNDIt MASTER UPDATED")
    print("Raw:", master.shape)
    print("Dedup:", dedup.shape)


def main():
    session = requests.Session()
    session.headers.update(HEADERS)

    all_rows = []

    for role in get_all_roles():
        all_rows.extend(scrape_role(session, role))
        time.sleep(2)

    batch_df = pd.DataFrame(all_rows)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_file = BATCH_DIR / f"foundit_{timestamp}.csv"

    batch_df.to_csv(batch_file, index=False)

    print("\nBatch saved:", batch_file)
    print("Batch shape:", batch_df.shape)

    build_master_from_batches()


if __name__ == "__main__":
    main()
