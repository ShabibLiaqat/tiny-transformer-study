"""Fetch the MovieLens sequence dataset inside a container, where TLS works.

The local machine cannot verify the GroupLens certificate (no usable CA roots for that
host), so the download happens in a Modal container and the extracted ratings file is
pulled back through the shared volume.

    python -m modal run fetch_sequences.py
"""

from pathlib import Path

import modal

APP = modal.App("sequence-data-fetch")
VOLUME = modal.Volume.from_name("posthog-attention-runs", create_if_missing=True)
URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"

@APP.function(image=modal.Image.debian_slim().apt_install("curl", "unzip"),
              volumes={"/runs": VOLUME}, timeout=900)
def fetch() -> dict:
    import subprocess
    subprocess.run(["curl", "-sSfL", URL, "-o", "/tmp/ml.zip"], check=True)
    subprocess.run(["unzip", "-o", "-q", "/tmp/ml.zip", "-d", "/tmp/out"], check=True)
    source = Path("/tmp/out/ml-latest-small/ratings.csv")
    data = source.read_bytes()
    Path("/runs/ratings.csv").write_bytes(data)
    Path("/runs/ml-latest-small.zip").write_bytes(Path("/tmp/ml.zip").read_bytes())
    VOLUME.commit()
    return {"bytes": len(data), "lines": data.count(b"\n")}

@APP.local_entrypoint()
def main():
    print(fetch.remote())
