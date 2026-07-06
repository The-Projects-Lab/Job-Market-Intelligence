import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime
import time

BASE_URL = "https://internshala.com/jobs/page-{}/"

headers = {
    "User-Agent": "Mozilla/5.0"
}

all_jobs = []

MAX_PAGES = 50

for page in range(1, MAX_PAGES + 1):

    if page == 1:
        url = "https://internshala.com/jobs/"
    else:
        url = BASE_URL.format(page)

    print(f"Scraping Page {page}")

    try:
        response = requests.get(
            url,
            headers=headers,
            timeout=30
        )

        soup = BeautifulSoup(
            response.text,
            "lxml"
        )

        jobs = soup.select(
            "div.individual_internship"
        )

        print(f"Found {len(jobs)} jobs")

        if len(jobs) == 0:
            break

        for job in jobs:

            try:

                title = job.select_one(
                    "a.job-title-href"
                )

                company = job.select_one(
                    "p.company-name"
                )

                location = job.select_one(
                    "p.locations"
                )

                description = job.select_one(
                    ".about_job .text"
                )

                skills = job.select(
                    ".job_skill"
                )

                posted = job.select_one(
                    ".status-inactive span"
                )

                job_url = ""

                if title:
                    href = title.get("href")
                    job_url = (
                        "https://internshala.com"
                        + href
                    )

                all_jobs.append({
                    "job_id": job.get(
                        "internshipid"
                    ),

                    "job_title":
                        title.text.strip()
                        if title else None,

                    "company":
                        company.text.strip()
                        if company else None,

                    "location":
                        location.text.strip()
                        if location else None,

                    "salary": None,

                    "experience": None,

                    "employment_type":
                        "Job",

                    "industry": None,

                    "skills":
                        ",".join(
                            [
                                s.text.strip()
                                for s in skills
                            ]
                        ),

                    "description":
                        description.text.strip()
                        if description else None,

                    "posted_date":
                        posted.text.strip()
                        if posted else None,

                    "job_url":
                        job_url,

                    "source":
                        "Internshala",

                    "scrape_time":
                        datetime.now()
                })

            except Exception as e:
                print("Job Error:", e)

        time.sleep(2)

    except Exception as e:
        print("Page Error:", e)

df = pd.DataFrame(all_jobs)

print("\nTotal Jobs:", len(df))

filename = (
    "data/raw/internshala/"
    f"internshala_jobs_"
    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
)

df.to_csv(
    filename,
    index=False
)

print(f"\nSaved -> {filename}")
