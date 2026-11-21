"""Fetch openly licensed papers from arXiv for local experiments (never committed).

Usage:
    python scripts/download_open_access.py 2310.xxxxx 2402.yyyyy --out data/open_access

For each arXiv id the script reads the paper's license from the arXiv Atom API
(``arxiv:license`` element) and downloads the PDF only if the license is one that permits
redistribution of the work (CC BY, CC BY-SA, CC0 / public domain). Everything else (including
arXiv's default non-exclusive distribution license) is skipped with a message. The
``data/open_access/`` directory is git-ignored: downloaded PDFs are for your local use and
must not be committed to this repository.

Please respect arXiv's API terms: at most one request every three seconds.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

API = "https://export.arxiv.org/api/query?id_list={id}"
PDF = "https://arxiv.org/pdf/{id}"
ALLOWED = (
    "creativecommons.org/licenses/by/",
    "creativecommons.org/licenses/by-sa/",
    "creativecommons.org/publicdomain/zero/",
    "creativecommons.org/share-your-work/public-domain/",
)
ID_RE = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")
NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
USER_AGENT = "ResearchRAG-open-access-fetcher/0.1"


def license_of(arxiv_id: str) -> str | None:
    """Return the license URL arXiv reports for a paper (``None`` if absent)."""
    req = urllib.request.Request(API.format(id=arxiv_id), headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        root = ET.fromstring(resp.read())
    entry = root.find("atom:entry", NS)
    node = entry.find("arxiv:license", NS) if entry is not None else None
    return node.text.strip() if node is not None and node.text else None


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("ids", nargs="+", help="arXiv ids, e.g. 2401.01234")
    parser.add_argument("--out", type=Path, default=Path("data/open_access"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    status = 0
    for arxiv_id in args.ids:
        if not ID_RE.match(arxiv_id):
            print(f"skip {arxiv_id}: not an arXiv id")
            status = 1
            continue
        lic = license_of(arxiv_id)
        if not lic or not any(a in lic for a in ALLOWED):
            print(f"skip {arxiv_id}: license {lic or 'unknown'} does not clearly permit redistribution")
            time.sleep(3)
            continue
        target = args.out / f"{arxiv_id}.pdf"
        req = urllib.request.Request(PDF.format(id=arxiv_id), headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=60) as resp:
            target.write_bytes(resp.read())
        (args.out / f"{arxiv_id}.LICENSE.txt").write_text(
            f"arXiv:{arxiv_id}\nlicense: {lic}\n", encoding="utf-8"
        )
        print(f"saved {target} ({lic})")
        time.sleep(3)
    return status


if __name__ == "__main__":
    sys.exit(main())
