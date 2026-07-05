from pathlib import Path
import pandas as pd

sources = [
    "foundit",
    "shine",
    "naukri",
    "internshala",
    "remoteok"
]

BASE = Path("data/raw")

for source in sources:

    print("\n====================")
    print(source.upper())
    print("====================")

    folder = BASE / source

    files = list(folder.glob("*.csv"))

    for f in files:
        try:
            df = pd.read_csv(f, low_memory=False)

            print(
                f.name,
                " --> ",
                df.shape
            )

        except Exception as e:
            print(
                f.name,
                "ERROR",
                e
            )
