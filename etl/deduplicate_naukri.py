import pandas as pd
import os

INPUT = "data/raw/naukri/naukri_master_raw.csv"
OUTPUT = "data/raw/naukri/naukri_master_dedup.csv"

df = pd.read_csv(INPUT)

print("Before duplicate removal:")
print(df.shape)


# normalize fields
for col in ["title", "company", "location"]:
    df[col] = (
        df[col]
        .astype(str)
        .str.lower()
        .str.strip()
    )


# remove duplicate jobs
df = df.drop_duplicates(
    subset=[
        "title",
        "company",
        "location"
    ],
    keep="first"
)


print("\nAfter duplicate removal:")
print(df.shape)


df.to_csv(
    OUTPUT,
    index=False
)

print("\nSaved:")
print(OUTPUT)
