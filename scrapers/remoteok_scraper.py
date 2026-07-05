import sys
from pathlib import Path
from datetime import datetime
import time

import requests
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.roles import ROLE_CATEGORIES


BASE_URL = "https://remoteok.com/api"
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "remoteok"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json"
}


def fetch_remoteok_jobs():
    response = requests.get(
        BASE_URL,
        headers=HEADERS,
        timeout=30
    )

    response.raise_for_status()

    data = response.json()

    return data[1:]   # first record is metadata


def normalize_job(job, searched_role=None, category=None):
    return {
        "job_id": job.get("id"),
        "title": job.get("position"),
        "company": job.get("company"),
        "location": job.get("location"),
        "salary_min": job.get("salary_min"),
        "salary_max": job.get("salary_max"),
        "skills": ",".join(job.get("tags", [])),
        "description": job.get("description"),
        "posted_date": job.get("date"),
        "searched_role": searched_role,
        "category": category,
        "source": "remoteok",
        "scrape_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }


def job_matches_role(job, role):
    text = " ".join([
        str(job.get("position", "")),
        str(job.get("company", "")),
        str(job.get("location", "")),
        " ".join(job.get("tags", [])),
        str(job.get("description", ""))
    ]).lower()

    return role.lower() in text


def main():
    print("Fetching RemoteOK API...")
    jobs = fetch_remoteok_jobs()

    print(f"Total RemoteOK API jobs: {len(jobs)}")

    all_rows = []

    # Full dump
    full_rows = [
        normalize_job(job)
        for job in jobs
    ]

    full_df = pd.DataFrame(full_rows)
    full_df.to_csv(
        OUTPUT_DIR / "remoteok_full_dump.csv",
        index=False
    )

    print("Saved full dump:", full_df.shape)

    # Role-wise filtered files
    for category, roles in ROLE_CATEGORIES.items():
        for role in roles:
            role_rows = []

            for job in jobs:
                if job_matches_role(job, role):
                    role_rows.append(
                        normalize_job(
                            job,
                            searched_role=role,
                            category=category
                        )
                    )

            role_df = pd.DataFrame(role_rows)

            file_name = role.replace(" ", "_") + ".csv"

            role_df.to_csv(
                OUTPUT_DIR / file_name,
                index=False
            )

            all_rows.extend(role_rows)

            print(f"{role}: {len(role_rows)}")

            time.sleep(0.5)

    master_raw = pd.DataFrame(all_rows)

    master_raw.to_csv(
        OUTPUT_DIR / "remoteok_master_raw.csv",
        index=False
    )

    if not master_raw.empty:
        master_dedup = master_raw.drop_duplicates(
            subset=["job_id"]
        )
    else:
        master_dedup = master_raw

    master_dedup.to_csv(
        OUTPUT_DIR / "remoteok_master_dedup.csv",
        index=False
    )

    print("\nDONE")
    print("Master raw:", master_raw.shape)
    print("Master dedup:", master_dedup.shape)
    print("Saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
