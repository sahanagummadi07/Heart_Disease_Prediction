"""
Download heart disease CSV (Google Drive per brief, with public fallbacks).
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "heart.csv"

GOOGLE_DRIVE_ID = "1k3Yhgzrgzl9CbdGXuZvK7WgbZ8kVx56I"
FALLBACK_URLS = [
    f"https://drive.google.com/uc?export=download&id={GOOGLE_DRIVE_ID}",
    "https://raw.githubusercontent.com/selva86/datasets/master/heart.csv",
]


def fetch() -> Path:
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DATA_PATH.exists() and DATA_PATH.stat().st_size > 100:
        return DATA_PATH
    try:
        import gdown

        gdown.download(
            f"https://drive.google.com/uc?id={GOOGLE_DRIVE_ID}",
            str(DATA_PATH),
            quiet=False,
        )
    except Exception:
        pass
    if DATA_PATH.exists() and DATA_PATH.stat().st_size > 100:
        return DATA_PATH
    for url in FALLBACK_URLS:
        try:
            urllib.request.urlretrieve(url, DATA_PATH)
            if DATA_PATH.exists() and DATA_PATH.stat().st_size > 100:
                return DATA_PATH
        except Exception:
            continue
    raise RuntimeError(
        "Could not download heart.csv. Place the file at data/heart.csv manually "
        f"(see dataset link in project brief). Tried: Google Drive id {GOOGLE_DRIVE_ID} and fallbacks."
    )


if __name__ == "__main__":
    p = fetch()
    print("Dataset ready at:", p)
