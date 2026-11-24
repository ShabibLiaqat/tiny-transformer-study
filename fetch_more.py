"""Extend the corpus from PostHog's sitemap so the large model has enough data.

348 KB was enough to show the pipeline works, and not enough to show a model that
generalises: the base run overfits (2.39 bits/byte in-sample against 3.34 held out).
This walks the published sitemap, keeps documentation and handbook pages, and
appends them to the same corpus with the source list updated.

    python -B fetch_more.py --limit 250
"""

import argparse
import json
import re
import time
import urllib.request
from pathlib import Path
from xml.etree import ElementTree

from fetch_corpus import DATA, GAP, UA, text_of

ROOT = Path(__file__).resolve().parent
PREFIXES = ("https://posthog.com/docs", "https://posthog.com/handbook",
            "https://posthog.com/blog", "https://posthog.com/questions")


def get(url):
    request = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(request, timeout=30) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        return response.read().decode(charset, "replace")


def sitemap_urls():
    """Read every <loc> from the site's sitemap, following one level of index."""
    urls = []
    queue = ["https://posthog.com/sitemap.xml"]
    while queue:
        current = queue.pop(0)
        try:
            raw = get(current)
        except Exception as error:
            print(f"sitemap skip {current}: {type(error).__name__}")
            continue
        try:
            root = ElementTree.fromstring(raw)
        except ElementTree.ParseError:
            continue
        for element in root.iter():
            if element.tag.endswith("loc") and element.text:
                loc = element.text.strip()
                if loc.endswith(".xml"):
                    if len(queue) < 3:
                        queue.append(loc)
                elif loc.startswith(PREFIXES):
                    urls.append(loc)
    return sorted(set(urls))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=250)
    args = parser.parse_args()

    existing = (DATA / "sources.json")
    sources = json.loads(existing.read_text(encoding="utf-8")) if existing.is_file() else []
    corpus_path = DATA / "corpus.txt"
    corpus = corpus_path.read_text(encoding="utf-8") if corpus_path.is_file() else ""

    candidates = [url for url in sitemap_urls() if url not in sources]
    print(f"{len(candidates)} sitemap pages not yet in the corpus; taking {args.limit}")
    added = 0
    for url in candidates[:args.limit]:
        try:
            html = get(url)
        except Exception as error:
            print(f"skip {url}: {type(error).__name__}")
            time.sleep(GAP)
            continue
        body = text_of(html)
        if len(body) < 800:
            time.sleep(GAP)
            continue
        corpus += f"\n\n### SOURCE {url}\n\n{body}"
        sources.append(url)
        added += 1
        if added % 25 == 0:
            print(f"{added} pages, corpus now {len(corpus):,} chars")
        time.sleep(GAP)
    corpus_path.write_text(corpus, encoding="utf-8")
    existing.write_text(json.dumps(sources, indent=1), encoding="utf-8")
    print(f"added {added} pages; corpus {len(corpus):,} chars from {len(sources)} sources")


if __name__ == "__main__":
    main()
