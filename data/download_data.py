"""
Downloader and preprocessor for 2WikiMultiHopQA.
Fetches dev.parquet from Hugging Face and extracts a fast local sample JSON for offline dev and benchmarks.
"""

import os
import urllib.request
import pandas as pd
import json
from tqdm import tqdm


HF_DEV_URL = "https://huggingface.co/datasets/xanhho/2WikiMultihopQA/resolve/main/dev.parquet"
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
DEV_PARQUET_PATH = os.path.join(DATA_DIR, "2wikimultihopqa_dev.parquet")
SAMPLE_JSON_PATH = os.path.join(DATA_DIR, "2wikimultihopqa_sample.json")


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_dataset(force: bool = False, sample_size: int = 150):
    os.makedirs(DATA_DIR, exist_ok=True)

    if not os.path.exists(DEV_PARQUET_PATH) or force:
        print(f"Downloading 2WikiMultiHopQA dev dataset from {HF_DEV_URL}...")
        headers = {"User-Agent": "Mozilla/5.0"}
        req = urllib.request.Request(HF_DEV_URL, headers=headers)
        with urllib.request.urlopen(req) as resp, open(DEV_PARQUET_PATH, "wb") as out_file:
            total_size = int(resp.headers.get("content-length", 0))
            with tqdm(total=total_size, unit="B", unit_scale=True, desc="2wikimultihopqa_dev.parquet") as pbar:
                while True:
                    chunk = resp.read(1024 * 64)
                    if not chunk:
                        break
                    out_file.write(chunk)
                    pbar.update(len(chunk))
        print(f"Saved dev dataset to: {DEV_PARQUET_PATH}")
    else:
        print(f"Dataset already exists at: {DEV_PARQUET_PATH}")

    # Create sample JSON if not present
    if not os.path.exists(SAMPLE_JSON_PATH) or force:
        print(f"Creating balanced sample subset of {sample_size} questions...")
        df = pd.read_parquet(DEV_PARQUET_PATH)

        # Try to balance across question types
        types = df["type"].unique().tolist()
        per_type = sample_size // len(types)
        samples = []
        for t in types:
            sub = df[df["type"] == t]
            samples.append(sub.head(per_type))
        sample_df = pd.concat(samples).reset_index(drop=True)
        if len(sample_df) < sample_size:
            remaining = sample_size - len(sample_df)
            extra = df[~df["_id"].isin(sample_df["_id"])].head(remaining)
            sample_df = pd.concat([sample_df, extra]).reset_index(drop=True)

        records = sample_df.to_dict(orient="records")
        with open(SAMPLE_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(records)} sample questions to: {SAMPLE_JSON_PATH}")


if __name__ == "__main__":
    download_dataset()
