from pathlib import Path
import re
import pandas as pd

RAW = Path("data/raw")
PROCESSED = Path("data/processed")
PROCESSED.mkdir(parents=True, exist_ok=True)


def clean_text(x):
    if pd.isna(x):
        return ""
    x = str(x).lower().strip()
    x = re.sub(r"\s+", " ", x)
    x = re.sub(r"[^a-z0-9,+.# ]", "", x)
    return x


def normalize_company(x):
    x = clean_text(x)
    x = re.sub(r"\bprivate limited\b|\bpvt ltd\b|\bltd\b|\blimited\b|\binc\b|\bllp\b", "", x)
    return re.sub(r"\s+", " ", x).strip()


def read_csv_safe(path):
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception as e:
        print("Skipping:", path, "|", e)
        return pd.DataFrame()


def standardize(df, source):
    if df.empty:
        return df

    col_map = {
        "companyName": "company",
        "company_name": "company",
        "job_title": "title",
        "jobRole": "title",
        "job_location": "location",
        "locations": "location",
        "job_description": "description",
        "jd": "description",
        "scrape_time": "scraped_at",
    }

    df = df.rename(columns=col_map)

    required = [
        "job_id", "title", "company", "location", "experience",
        "salary", "skills", "description", "posted_date",
        "searched_role", "source", "scraped_at"
    ]

    for col in required:
        if col not in df.columns:
            df[col] = None

    df["source"] = source

    return df[required]


def load_source(source):
    source_dir = RAW / source

    files = []

    if (source_dir / "batches").exists():
        files.extend(list((source_dir / "batches").glob("*.csv")))

    for f in source_dir.glob("*.csv"):
        if "sample" not in f.name.lower():
            files.append(f)

    files = list(dict.fromkeys(files))

    dfs = []

    for f in files:
        print(f"Reading {source}:", f)
        df = read_csv_safe(f)
        df = standardize(df, source)
        if not df.empty:
            dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


def main():
    sources = [
        "foundit",
        "shine",
        "naukri",
        "internshala",
        "remoteok"
    ]

    all_dfs = []

    source_report = []

    for source in sources:
        df = load_source(source)

        if df.empty:
            print("No data for:", source)
            continue

        source_report.append({
            "source": source,
            "rows_loaded": len(df)
        })

        all_dfs.append(df)

    master_raw = pd.concat(all_dfs, ignore_index=True)

    master_raw.to_csv(PROCESSED / "jobs_master_raw.csv", index=False)

    print("\nMASTER RAW:", master_raw.shape)

    df = master_raw.copy()

    df["title_clean"] = df["title"].apply(clean_text)
    df["company_clean"] = df["company"].apply(normalize_company)
    df["location_clean"] = df["location"].apply(clean_text)

    df = df[
        (df["title_clean"] != "") &
        (df["company_clean"] != "")
    ]

    before_dedup = len(df)

    df["dedup_key"] = (
        df["title_clean"] + "|" +
        df["company_clean"] + "|" +
        df["location_clean"]
    )

    source_count = (
        df.groupby("dedup_key")["source"]
        .nunique()
        .reset_index()
        .rename(columns={"source": "source_count"})
    )

    sources_joined = (
        df.groupby("dedup_key")["source"]
        .apply(lambda x: ",".join(sorted(set(x))))
        .reset_index()
        .rename(columns={"source": "sources_available"})
    )

    df = df.merge(source_count, on="dedup_key", how="left")
    df = df.merge(sources_joined, on="dedup_key", how="left")

    df = df.sort_values(
        by=["source_count", "scraped_at"],
        ascending=[False, False]
    )

    clean = df.drop_duplicates(subset=["dedup_key"], keep="first")

    clean = clean.drop(columns=[
        "title_clean", "company_clean", "location_clean"
    ])

    clean.to_csv(PROCESSED / "jobs_clean.csv", index=False)

    report = pd.DataFrame(source_report)

    summary = pd.DataFrame([
        {"metric": "total_raw_rows_loaded", "value": len(master_raw)},
        {"metric": "rows_after_title_company_filter", "value": before_dedup},
        {"metric": "final_clean_unique_rows", "value": len(clean)},
        {"metric": "duplicates_removed_cross_source", "value": before_dedup - len(clean)}
    ])

    report.to_csv(PROCESSED / "source_report.csv", index=False)
    summary.to_csv(PROCESSED / "dedup_report.csv", index=False)

    print("\nFINAL REPORT")
    print(summary)

    print("\nSaved:")
    print(PROCESSED / "jobs_master_raw.csv")
    print(PROCESSED / "jobs_clean.csv")
    print(PROCESSED / "source_report.csv")
    print(PROCESSED / "dedup_report.csv")


if __name__ == "__main__":
    main()
