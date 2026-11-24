"""Collect a public text corpus from PostHog's own site for the tiny-model experiment.

The point is to train on *their* material rather than toy data: the docs, the
handbook and the changelog are all published for anyone to read. It stays polite
(one request at a time, a fixed gap) and writes plain text, so the training run is
reproducible from the URLs recorded alongside it.

    python -B fetch_corpus.py
"""

import json
import re
import time
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
UA = {"User-Agent": "Mozilla/5.0 (compatible; research-experiment/1.0)"}
GAP = 0.35

SEEDS = [
    "https://posthog.com/docs",
    "https://posthog.com/handbook",
    "https://posthog.com/changelog",
    "https://posthog.com/docs/product-analytics",
    "https://posthog.com/docs/session-replay",
    "https://posthog.com/docs/ai-engineering",
    "https://posthog.com/docs/data-warehouse",
    "https://posthog.com/docs/cdp",
    "https://posthog.com/docs/experiments",
    "https://posthog.com/handbook/company",
    "https://posthog.com/handbook/engineering",
    "https://posthog.com/handbook/people",
    "https://posthog.com/blog",
    "https://posthog.com/docs/model-context-protocol",
]


class Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "nav", "footer", "svg"}:
            self.skip += 1
        if tag in {"p", "li", "h1", "h2", "h3", "h4", "pre", "code", "tr"}:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in {"script", "style", "nav", "footer", "svg"} and self.skip:
            self.skip -= 1

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def text_of(html):
    parser = Text()
    parser.feed(html)
    raw = "".join(parser.parts)
    raw = re.sub(r"[ \t\r\f\v]+", " ", raw)
    raw = re.sub(r"\n\s*\n+", "\n\n", raw)
    return raw.strip()


def get(url):
    request = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(request, timeout=30) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        return response.read().decode(charset, "replace")


def main():
    DATA.mkdir(exist_ok=True)
    pieces, sources = [], []
    for url in SEEDS:
        try:
            html = get(url)
        except Exception as error:
            print(f"skip {url}: {type(error).__name__}")
            continue
        body = text_of(html)
        if len(body) < 400:
            print(f"skip {url}: only {len(body)} chars of text")
            continue
        pieces.append(f"\n\n### SOURCE {url}\n\n{body}")
        sources.append(url)
        print(f"{len(body):>7} chars  {url}")
        time.sleep(GAP)
    corpus = "\n".join(pieces)
    (DATA / "corpus.txt").write_text(corpus, encoding="utf-8")
    (DATA / "sources.json").write_text(json.dumps(sources, indent=1), encoding="utf-8")
    print(f"\ncorpus: {len(corpus):,} chars from {len(sources)} public pages")


if __name__ == "__main__":
    main()
