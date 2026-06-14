r"""STEP 0 (recommended) - download the dataset via the Kaggle API (kagglehub).

Run:  python download_data.py

Downloads the NIH Chest X-ray *sample* (~5 GB) into kagglehub's local cache and
records the path so the rest of the pipeline finds it automatically.

The download is RESUMABLE: kagglehub continues from the partial file, and this
script auto-retries on connection drops, so a flaky connection will still finish
(it just makes progress in chunks across retries).

ONE-TIME SETUP:
  1. Create a free account at https://www.kaggle.com
  2. Kaggle -> avatar -> Settings -> API -> "Create New Token" -> downloads token.
  3. Save the KGAT_... token to:  C:\Users\<you>\.kaggle\access_token
     (or set env var KAGGLE_API_TOKEN)
"""
import sys
import time
from pathlib import Path

import config

DATASET = "nih-chest-xrays/sample"
MAX_RETRIES = 40
RETRY_WAIT_SECONDS = 5


def main():
    try:
        import kagglehub
    except ImportError:
        sys.exit("kagglehub is not installed. Run:  pip install kagglehub")

    print(f"Downloading '{DATASET}' via kagglehub (~5 GB, resumable)...")
    last_error = None
    path = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            path = Path(kagglehub.dataset_download(DATASET))
            break
        except KeyboardInterrupt:
            raise
        except Exception as e:
            last_error = e
            print(f"\n[attempt {attempt}/{MAX_RETRIES}] interrupted: {e}")
            print(f"Retrying in {RETRY_WAIT_SECONDS}s (resumes from the partial file)...")
            time.sleep(RETRY_WAIT_SECONDS)

    if path is None:
        sys.exit(f"Gave up after {MAX_RETRIES} attempts. Last error: {last_error}\n"
                 "Check your connection and re-run — it will resume where it left off.")

    print(f"\nDownloaded to: {path}")
    pointer = config.ARTIFACTS_DIR / "data_path.txt"
    pointer.write_text(str(path))
    print(f"Recorded data path -> {pointer}")
    print("Next: python prepare_data.py")


if __name__ == "__main__":
    main()
