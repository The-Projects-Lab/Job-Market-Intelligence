import os
import pandas as pd
import hashlib
from datetime import datetime

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

os.makedirs(PROCESSED_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(PROCESSED_DIR, "silver_jobs.csv")
AUDIT_FILE = os.path.join(PROCESSED_DIR, "silver_audit_report.csv")


def create_job_id(row):
    text = (
        str(row.get("job_title", "")) +
        str(row.get("company", "")) +
        str(row.get("location", "")) +
        str(row.get("source", ""))
    )
    return hashlib.md5(text.lower().encode()).hexdigest()


def standardize_columns(df, source):
    df.columns = df.columns.str.lower().str.strip()

    column_map = {
        "title": "job_title",
        "jobtitle": "job_title",
        "job_title": "job_title",
        "companyname": "company",
        "company": "company",
        "locations": "location",
        "location": "location",
        "skills": "skills",
        "description": "description",
        "jobdescription": "description",
        "jd": "description",
        "exp": "experience",
        "experience": "experience",
        "salary": "salary",
        "createdat": "posted_date",
        "posted_date": "posted_date",
        "jdurl": "job_url",
        "redirecturl": "job_url",
        "url": "job_url",
    }

    df = df.rename(columns={c: column_map[c] for c in df.columns if c in column_map})

    required_cols = [
        "job_title", "company", "location", "experience",
        "salary", "skills", "description", "job_url", "posted_date"
    ]

    for col in required_cols:
        if col not in df.columns:
            df[col] = None

    df["source"] = source
    return df[required_cols + ["source"]]


def read_all_csv_files():
    all_dfs = []
    audit_rows = []

    for source in os.listdir(RAW_DIR):
        source_path = os.path.join(RAW_DIR, source)

        if not os.path.isdir(source_path):
            continue

        for file in os.listdir(source_path):
            if file.endswith(".csv"):
                file_path = os.path.join(source_path, file)

                try:
                    df = pd.read_csv(file_path)
                    raw_count = len(df)

                    std_df = standardize_columns(df, source)
                    all_dfs.append(std_df)

                    audit_rows.append({
                        "source": source,
                        "file_name": file,
                        "raw_count": raw_count,
                        "status": "success"
                    })

                    print(f"Loaded {source}/{file}: {raw_count}")

                except Exception as e:
                    audit_rows.append({
                        "source": source,
                        "file_name": file,
                        "raw_count": 0,
                        "status": f"failed: {e}"
                    })
                    print(f"Failed {source}/{file}: {e}")

    return all_dfs, audit_rows


def clean_silver(df):
    before_cleaning = len(df)

    df = df.dropna(subset=["job_title", "company"], how="any").copy()

    df["job_title"] = df["job_title"].astype(str).str.strip().str.lower()
    df["company"] = df["company"].astype(str).str.strip().str.lower()
    df["location"] = df["location"].astype(str).str.strip().str.lower()
    df["skills"] = df["skills"].astype(str).str.lower()
    df["description"] = df["description"].astype(str).str.lower()

    before_dedup = len(df)

    df["job_id"] = df.apply(create_job_id, axis=1)
    df = df.drop_duplicates(subset=["job_id"])

    after_dedup = len(df)

    df["processed_at"] = datetime.now()

    metrics = {
        "total_raw_rows": before_cleaning,
        "rows_after_null_removal": before_dedup,
        "null_rows_removed": before_cleaning - before_dedup,
        "duplicates_removed": before_dedup - after_dedup,
        "final_silver_rows": after_dedup
    }

    return df, metrics


def main():
    all_dfs, audit_rows = read_all_csv_files()

    if not all_dfs:
        print("No CSV files found.")
        return

    master_df = pd.concat(all_dfs, ignore_index=True)

    silver_df, metrics = clean_silver(master_df)

    silver_df.to_csv(OUTPUT_FILE, index=False)

    audit_df = pd.DataFrame(audit_rows)
    for k, v in metrics.items():
        audit_df[k] = v

    audit_df.to_csv(AUDIT_FILE, index=False)

    print("\nSILVER PIPELINE COMPLETED")
    print("=========================")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    print(f"\nSaved: {OUTPUT_FILE}")
    print(f"Audit: {AUDIT_FILE}")


if __name__ == "__main__":
    main()
