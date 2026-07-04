import sys
import os
import time
import re
from pathlib import Path
from datetime import datetime

import pandas as pd

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.roles import ROLE_CATEGORIES

SOURCE = "naukri"

OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / SOURCE
BATCH_DIR = OUTPUT_DIR / "batches"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
BATCH_DIR.mkdir(parents=True, exist_ok=True)

MAX_PAGES_PER_ROLE = 50


def get_all_roles():
    roles = []
    for category_roles in ROLE_CATEGORIES.values():
        roles.extend(category_roles)
    return list(dict.fromkeys(roles))


def safe_text(value):
    if value is None:
        return None
    return str(value).strip()


def build_driver():
    import shutil
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service

    chrome_path = shutil.which("chromium-browser") or shutil.which("chromium")

    if chrome_path is None:
        chrome_path = "/snap/bin/chromium"

    print("Using Chromium:", chrome_path)

    options = Options()
    options.binary_location = chrome_path

    options.add_argument("--start-maximized")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--remote-debugging-port=9222")
    options.add_argument("--user-data-dir=/tmp/selenium_chrome_profile")

    service = Service("/usr/bin/chromedriver")

    return webdriver.Chrome(
        service=service,
        options=options
    )
    
def extract_from_card(card, searched_role):
    lines = [line.strip() for line in card.text.split("\n") if line.strip()]

    job = {
        "job_id": None,
        "title": None,
        "company": None,
        "location": None,
        "experience": None,
        "salary": None,
        "skills": None,
        "description": None,
        "posted_date": None,
        "searched_role": searched_role,
        "source": SOURCE,
        "scraped_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    if len(lines) > 0:
        job["title"] = lines[0]

    if len(lines) > 1:
        job["company"] = lines[1]

    city_keywords = [
        "pune", "mumbai", "bangalore", "bengaluru",
        "hyderabad", "chennai", "delhi", "gurgaon",
        "gurugram", "noida", "kolkata", "remote",
        "ahmedabad", "india"
    ]

    for line in lines:
        lower = line.lower()

        if ("yrs" in lower or "year" in lower) and job["experience"] is None:
            job["experience"] = line

        elif ("₹" in line or "lpa" in lower or "pa" in lower) and job["salary"] is None:
            job["salary"] = line

        elif any(city in lower for city in city_keywords) and job["location"] is None:
            job["location"] = line

        elif ("day ago" in lower or "days ago" in lower or "week ago" in lower or "month ago" in lower):
            job["posted_date"] = line

    if len(lines) >= 4:
        job["description"] = lines[-3]

    if len(lines) >= 3:
        possible_skills = lines[-2]
        if "ago" not in possible_skills.lower():
            job["skills"] = possible_skills

    return job


def scrape_role(driver, role):
    rows = []
    keyword = role.replace(" ", "-")

    print(f"\nScraping Naukri: {role}")

    for page in range(1, MAX_PAGES_PER_ROLE + 1):
        url = f"https://www.naukri.com/{keyword}-jobs-{page}"
        print(url)

        try:
            driver.get(url)
            time.sleep(5)

            cards = driver.find_elements(By.CSS_SELECTOR, "div.srp-jobtuple-wrapper")

            if not cards:
                print("No jobs found")
                break

            for card in cards:
                try:
                    rows.append(extract_from_card(card, role))
                except Exception:
                    continue

        except Exception as e:
            print("Error:", e)
            break

    return rows


def build_master_from_batches():
    files = list(BATCH_DIR.glob("naukri_*.csv"))

    if not files:
        return

    dfs = [pd.read_csv(file) for file in files]
    master = pd.concat(dfs, ignore_index=True)

    master.to_csv(OUTPUT_DIR / "naukri_master_raw.csv", index=False)

    dedup = master.drop_duplicates(
        subset=["title", "company", "location"],
        keep="last"
    )

    dedup.to_csv(OUTPUT_DIR / "naukri_master_dedup.csv", index=False)

    print("\nNAUKRI MASTER UPDATED")
    print("Raw:", master.shape)
    print("Dedup:", dedup.shape)


def main():
    driver = build_driver()
    all_rows = []

    try:
        for role in get_all_roles():
            all_rows.extend(scrape_role(driver, role))
            time.sleep(3)

    finally:
        driver.quit()

    batch_df = pd.DataFrame(all_rows)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_file = BATCH_DIR / f"naukri_{timestamp}.csv"

    batch_df.to_csv(batch_file, index=False)

    print("\nBatch saved:", batch_file)
    print("Batch shape:", batch_df.shape)

    build_master_from_batches()


if __name__ == "__main__":
    main()
