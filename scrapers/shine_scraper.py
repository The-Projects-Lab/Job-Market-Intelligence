import sys
import time
import re
from pathlib import Path
from datetime import datetime

import pandas as pd
import requests
from bs4 import BeautifulSoup

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.roles import ROLE_CATEGORIES

SOURCE = "shine"
BASE_URL = "https://www.shine.com/api/v2/search/simple/"

OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / SOURCE
BATCH_DIR = OUTPUT_DIR / "batches"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
BATCH_DIR.mkdir(parents=True, exist_ok=True)

MAX_PAGES_PER_ROLE = 100

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json",
}


def get_all_roles():
    roles = []
    for category_roles in ROLE_CATEGORIES.values():
        roles.extend(category_roles)
    return list(dict.fromkeys(roles))


def clean_html(text):
    if not text:
        return None
    return BeautifulSoup(str(text), "lxml").get_text(" ", strip=True)


def safe_text(value):
    if value is None:
        return None
    if isinstance(value, list):
        return ", ".join(map(str, value))
    return str(value).strip()


def make_query(role):
    return role.replace(" ", "-") + "-jobs"


def extract_job(job, searched_role):
    return {
        "job_id": job.get("id"),
        "title": safe_text(job.get("jJT")),
        "company": safe_text(job.get("jCName")),
        "location": safe_text(job.get("jLoc")),
        "experience": safe_text(job.get("jExp")),
        "salary": safe_text(job.get("jSal")),
        "skills": safe_text(job.get("jKwd")),
        "description": clean_html(job.get("jJD")),
        "posted_date": safe_text(job.get("jPDate")),
        "industry": safe_text(job.get("jInd")),
        "searched_role": searched_role,
        "source": SOURCE,
        "scraped_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def scrape_role(session, role):
    rows = []

    print(f"\nScraping Shine: {role}")

    for page in range(1, MAX_PAGES_PER_ROLE + 1):
        params = {
            "q": make_query(role),
            "page": page,
        }

        try:
            response = session.get(BASE_URL, params=params, timeout=30)
            print(f"{role} | page={page} | status={response.status_code}")

            if response.status_code != 200:
                break

            data = response.json()
            jobs = data.get("results", [])

            if not jobs:
                break

            for job in jobs:
                rows.append(extract_job(job, role))

            if not data.get("next"):
                break

            time.sleep(1)

        except Exception as e:
            print("Error:", e)
            break

    return rows


def build_master_from_batches():
    files = list(BATCH_DIR.glob("shine_*.csv"))

    if not files:
        return

    dfs = [pd.read_csv(file) for file in files]
    master = pd.concat(dfs, ignore_index=True)

    master.to_csv(OUTPUT_DIR / "shine_master_raw.csv", index=False)

    dedup = master.drop_duplicates(subset=["job_id"], keep="last")
    dedup.to_csv(OUTPUT_DIR / "shine_master_dedup.csv", index=False)

    print("\nSHINE MASTER UPDATED")
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
    batch_file = BATCH_DIR / f"shine_{timestamp}.csv"

    batch_df.to_csv(batch_file, index=False)

    print("\nBatch saved:", batch_file)
    print("Batch shape:", batch_df.shape)

    build_master_from_batches()


if __name__ == "__main__":
    main()
