"""Deduplicate the corpus and split it by page, so the held-out number means something.

The first version of this experiment concatenated every page and held out the last
10% of the text. Two problems with that, both fair criticisms:

* PostHog's pages share boilerplate (navigation, calls to action, the same cookie
  sentence), so the same sentences appeared on both sides of the split and the
  held-out loss was optimistic.
* A slice of concatenated text is not a natural boundary; the model has seen the
  same page's earlier paragraphs.

This script builds `data/train.txt` and `data/test.txt` from whole pages: near
duplicate pages are dropped by shingle overlap, and the remaining pages are assigned
to train or test by a hash of their URL, so no page can appear on both sides.

    python -B build_pages.py
"""

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
SHINGLE = 5
TEST_SHARE = 5  # one page in five


def pages():
    raw = (DATA / "corpus.txt").read_text(encoding="utf-8")
    parts = re.split(r"### SOURCE (\S+)", raw)
    for index in range(1, len(parts) - 1, 2):
        url, body = parts[index], parts[index + 1]
        text = re.sub(r"[ \t]+", " ", body).strip()
        if len(text) > 800:
            yield url, text


def shingles(text):
    words = re.findall(r"[a-z0-9']+", text.casefold())
    return {" ".join(words[i:i + SHINGLE]) for i in range(0, max(len(words) - SHINGLE, 1), SHINGLE)}


def main():
    kept, dropped, seen = [], [], {}
    for url, text in pages():
        signature = hashlib.sha256(text[:2000].encode()).hexdigest()
        if signature in seen:
            dropped.append((url, f"identical prefix to {seen[signature]}"))
            continue
        current = shingles(text)
        duplicate_of = None
        for other_url, other in kept:
            if len(other["shingles"]) < 20:
                continue
            overlap = len(current & other["shingles"]) / max(len(current | other["shingles"]), 1)
            if overlap > 0.6:
                duplicate_of = other_url
                break
        if duplicate_of:
            dropped.append((url, f"near-duplicate of {duplicate_of}"))
            continue
        seen[signature] = url
        kept.append((url, {"text": text, "shingles": current}))

    train, test = [], []
    for url, record in kept:
        bucket = int(hashlib.sha256(url.encode()).hexdigest(), 16) % TEST_SHARE
        (test if bucket == 0 else train).append(record["text"])

    (DATA / "train.txt").write_text("\n\n".join(train), encoding="utf-8")
    (DATA / "test.txt").write_text("\n\n".join(test), encoding="utf-8")
    report = {
        "pages_read": len(kept) + len(dropped),
        "pages_kept": len(kept),
        "pages_dropped": len(dropped),
        "train_pages": len(train),
        "test_pages": len(test),
        "train_chars": sum(len(t) for t in train),
        "test_chars": sum(len(t) for t in test),
        "dropped": dropped[:20],
    }
    (DATA / "split.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({k: report[k] for k in
                      ("pages_read", "pages_kept", "pages_dropped", "train_pages",
                       "test_pages", "train_chars", "test_chars")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
