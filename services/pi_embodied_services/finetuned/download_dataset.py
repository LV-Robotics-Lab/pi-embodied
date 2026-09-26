# Copyright 2026 The Show-Harness Authors. Licensed under the Apache License, Version 2.0.
# Modified by pi-embodied: train/scripts/download_dataset.sh of github.com/showlab/Show-Harness @137d571
# as a standard-library downloader pinned to one dataset revision and checked file by file.

"""Fetch the released Show-Harness training data (showlab/Show-Harness-Data), pinned and verified.

    python -m pi_embodied_services.finetuned.download_dataset --splits real sim \\
        --dest /root/autodl-tmp/data/Show-Harness-Data [--register --lf-root $LF_ROOT]

writes ``<dest>/<split>/`` (``rollouts.json`` with image paths relative to it, the frames and
``episodes.jsonl``) plus ``README.md``, all from ``$HF_ENDPOINT`` (default https://hf-mirror.com) at
``REVISION``. 43k files cannot be listed here, so the pin is one digest per split: the sha256 of
the split's sorted ``path<TAB>size<TAB>id`` lines, ``id`` being ``sha256:<content>`` for an LFS file
and ``git-sha1:<blob id>`` otherwise, taken from huggingface.co's
``/api/datasets/<repo>/revision/<rev>?blobs=true`` on 2026-09-26. The file list is fetched from the
endpoint, accepted only if it hashes to that digest (a mirror serving other content would have to
serve other metadata), then every file is downloaded and checked against it (a mismatch aborts).
Re-running only fetches what is missing or wrong. ``--register`` then adds ``<prefix>_<split>``
to LLaMA-Factory's dataset_info.json through the vendored register_dataset.py, as upstream's
script does. Standard library only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pi_embodied_services.finetuned.download import HEADERS, digest, endpoint, verified

REPO = "showlab/Show-Harness-Data"
REVISION = "0c568ffa757ed02cc01780e553a54e8289af2b49"
#: split -> (files, bytes, sha256 of its canonical listing)
SPLITS: dict[str, tuple[int, int, str]] = {
    "real": (
        15714,
        873126786,
        "b1027a7df4fdddb7a0376aa3409f7e14a9acc89f64744678e780e40c6a0bbe51",
    ),
    "sim": (
        27278,
        1324629192,
        "59d129a7f1fd5a508ba507fd3990b137156997a6b3e0ee32bb0bad91654e10c6",
    ),
}
ROOT_FILES = {"README.md": (4806, "git-sha1:d5f12134607c78a8b9af1a4f9d95d8f220605cc5")}
REGISTER = (
    Path(__file__).resolve().parent
    / "showharness/train/data_preparation/register_dataset.py"
)


def listing_entry(sibling: dict) -> tuple[str, int, str]:
    lfs = sibling.get("lfs")
    ident = f"sha256:{lfs['sha256']}" if lfs else f"git-sha1:{sibling['blobId']}"
    return sibling["rfilename"], int(sibling["size"]), ident


def listing_digest(entries: list[tuple[str, int, str]]) -> str:
    lines = sorted(f"{p}\t{s}\t{i}" for p, s, i in entries)
    return hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest()


def pinned_listing(siblings: list[dict], split: str) -> list[tuple[str, int, str]]:
    """The split's entries of an endpoint's listing, refused unless they hash to the pin."""
    entries = [
        listing_entry(s) for s in siblings if s["rfilename"].startswith(f"{split}/")
    ]
    count, size, want = SPLITS[split]
    got = listing_digest(entries)
    if got != want:
        raise SystemExit(
            f"{endpoint()} lists {len(entries)} files / {sum(e[1] for e in entries)} B for "
            f"{REPO}@{REVISION}/{split} (digest {got}); pinned {count} / {size} B ({want})"
        )
    return entries


def fetch_listing() -> list[dict]:
    url = f"{endpoint()}/api/datasets/{REPO}/revision/{REVISION}?blobs=true"
    with urllib.request.urlopen(
        urllib.request.Request(url, headers=HEADERS), timeout=300
    ) as r:
        return json.load(r)["siblings"]


def fetch_one(dest: Path, entry: tuple[str, int, str]) -> bool:
    """Download one file unless it is already there and right; True if it was fetched."""
    name, size, want = entry
    out = dest / name
    if verified(out, size, want):
        return False
    out.parent.mkdir(parents=True, exist_ok=True)
    url = f"{endpoint()}/datasets/{REPO}/resolve/{REVISION}/{name}"
    tmp = out.with_name(out.name + ".part")
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as w:
                while block := r.read(1 << 20):
                    w.write(block)
            break
        except OSError:
            if attempt == 4:
                raise
    have = digest(tmp, want.split(":")[0])
    if tmp.stat().st_size != size or have != want:
        bad = out.with_name(out.name + ".mismatch")
        tmp.replace(bad)
        raise SystemExit(
            f"{bad}: {have} ({bad.stat().st_size} B), pinned {want} ({size} B)"
        )
    tmp.replace(out)
    return True


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--splits", nargs="+", default=["real", "sim"], choices=sorted(SPLITS)
    )
    p.add_argument("--dest", default="/root/autodl-tmp/data/Show-Harness-Data")
    p.add_argument("--jobs", type=int, default=16)
    p.add_argument(
        "--register",
        action="store_true",
        help="register <prefix>_<split> in LLaMA-Factory",
    )
    p.add_argument("--prefix", default="showharness")
    p.add_argument(
        "--lf-root", default=None, help="LLaMA-Factory checkout (default $LF_ROOT)"
    )
    args = p.parse_args(argv)
    dest = Path(args.dest)

    siblings = fetch_listing()
    entries = [(n, *ROOT_FILES[n]) for n in ROOT_FILES]
    for split in args.splits:
        entries += pinned_listing(siblings, split)
        print(
            f"[listing] {split}: {SPLITS[split][0]} files match the pinned digest",
            flush=True,
        )
    fetched = done = 0
    with ThreadPoolExecutor(args.jobs) as pool:
        for got in pool.map(lambda e: fetch_one(dest, e), entries):
            fetched += got
            done += 1
            if done % 2000 == 0:
                print(
                    f"[download] {done}/{len(entries)} checked ({fetched} fetched)",
                    flush=True,
                )
    (dest / ".revision").write_text(f"{REPO}@{REVISION} {' '.join(args.splits)}\n")
    print(f"[download] {len(entries)} files verified ({fetched} fetched) -> {dest}")
    if args.register:
        for split in args.splits:
            cmd = [
                sys.executable,
                str(REGISTER),
                f"{args.prefix}_{split}",
                "--samples",
                str(dest / split / "rollouts.json"),
            ]
            if args.lf_root:
                cmd += ["--lf-root", args.lf_root]
            subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
