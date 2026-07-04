import pandas as pd
from pathlib import Path


files = {

    "internshala":
    "data/raw/internshala/internshala_jobs_20260531_151341.csv",

    "remoteok":
    "data/raw/remoteok_jobs_20260531_144303.csv"

}


for name, path in files.items():

    print("\n" + "="*70)
    print(name.upper())
    print("="*70)


    file = Path(path)


    if not file.exists():

        print("File missing")
        continue


    df = pd.read_csv(file)


    print("\nShape:")
    print(df.shape)


    print("\nColumns:")
    print(df.columns.tolist())


    print("\nMissing values:")
    print(
        df.isnull()
        .sum()
        .sort_values(
            ascending=False
        )
    )


    print("\nDuplicate rows:")

    print(
        df.duplicated()
        .sum()
    )


    print("\nSample:")

    print(
        df.head(3)
    )
