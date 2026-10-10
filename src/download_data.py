from pathlib import Path
from ucimlrepo import fetch_ucirepo
import pandas as pd

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

def main() -> None:
    heart = fetch_ucirepo(id=45)  # UCI Heart Disease (Cleveland)
    X = heart.data.features
    y = heart.data.targets.iloc[:, 0].rename("target")  # original column is "num"
    df = pd.concat([X, y], axis=1)
    out = RAW_DIR / "heart.csv"
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows to {out}")

if __name__ == "__main__":
    main()