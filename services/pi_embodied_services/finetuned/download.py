"""Fetch a released Show-Harness adapter and the base model it names, pinned and verified.

    python -m pi_embodied_services.finetuned.download --adapter qwen3_5_2b_sim \\
        --dest /root/autodl-tmp/checkpoints

writes ``<dest>/Show-Harness-VLMs/<adapter>/`` and ``<dest>/<base name>/`` (e.g. ``Qwen3.5-2B``).
Files come from ``$HF_ENDPOINT`` (default https://hf-mirror.com) at a pinned revision, big ones
through ``aria2c -x16``; every file is checked against the repo's size and, for LFS files, its
sha256. Standard library only, so any Python runs it. The released adapters
(showlab/Show-Harness-VLMs @ ADAPTER_REVISION): qwen3_5_0_8b, qwen3_5_2b, qwen3_5_4b, qwen3_5_9b,
gemma4_e4b (the real-robot ``ft`` split) and qwen3_5_2b_sim (RoboLab + ManiSkill).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import urllib.request
from pathlib import Path

ADAPTER_REPO = "showlab/Show-Harness-VLMs"
ADAPTER_REVISION = "c1c94dfa0f5f1e8827acbc9698ba5f765f1605e5"
# Base revisions the adapters were checked against (others resolve `main` at download time).
BASE_REVISIONS = {"Qwen/Qwen3.5-2B": "15852e8c16360a2fea060d615a32b45270f8a8fc"}
BIG = 64 << 20
# hf-mirror answers 403 to urllib's default User-Agent.
HEADERS = {"User-Agent": "pi-embodied-finetuned/1"}


def endpoint() -> str:
    return os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")


def get_json(url: str):
    with urllib.request.urlopen(
        urllib.request.Request(url, headers=HEADERS), timeout=60
    ) as r:
        return json.load(r)


def siblings(repo: str, revision: str) -> tuple[str, list[dict]]:
    """(resolved sha, [{rfilename, size, lfs?}]) of a model repo at `revision`."""
    info = get_json(f"{endpoint()}/api/models/{repo}/revision/{revision}?blobs=true")
    return info["sha"], info["siblings"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(
    repo: str, revision: str, files: list[dict], dest: Path, strip: str = ""
) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for f in files:
        name = f["rfilename"]
        out = dest / name[len(strip) :]
        size = f.get("size")
        lfs = (f.get("lfs") or {}).get("sha256")
        if (
            out.exists()
            and out.stat().st_size == size
            and (not lfs or sha256(out) == lfs)
        ):
            print(f"ok   {out} ({size} B)")
            continue
        url = f"{endpoint()}/{repo}/resolve/{revision}/{name}"
        out.parent.mkdir(parents=True, exist_ok=True)
        if size and size > BIG and shutil.which("aria2c"):
            subprocess.run(
                [
                    "aria2c",
                    "-x16",
                    "-s16",
                    "-k8M",
                    "--file-allocation=none",
                    "--continue=true",
                    "--max-tries=0",
                    "--retry-wait=5",
                    "--timeout=60",
                    "--console-log-level=warn",
                    "--summary-interval=60",
                    "-d",
                    str(out.parent),
                    "-o",
                    out.name,
                    url,
                ],
                check=True,
            )
        else:
            tmp = out.with_suffix(out.suffix + ".part")
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as w:
                shutil.copyfileobj(r, w)
            tmp.replace(out)
        got = out.stat().st_size
        if size is not None and got != size:
            raise SystemExit(f"{out}: {got} B, expected {size} B")
        if lfs and sha256(out) != lfs:
            raise SystemExit(f"{out}: sha256 mismatch (expected {lfs})")
        print(f"got  {out} ({got} B{', sha256 ok' if lfs else ''})")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--adapter",
        default="qwen3_5_2b_sim",
        help="Folder in showlab/Show-Harness-VLMs",
    )
    p.add_argument("--dest", default="/root/autodl-tmp/checkpoints")
    p.add_argument("--no-base", action="store_true", help="Only the adapter")
    args = p.parse_args()
    dest = Path(args.dest)

    _, files = siblings(ADAPTER_REPO, ADAPTER_REVISION)
    prefix = f"{args.adapter}/"
    mine = [f for f in files if f["rfilename"].startswith(prefix)]
    if not mine:
        folders = sorted(
            {f["rfilename"].split("/")[0] for f in files if "/" in f["rfilename"]}
        )
        raise SystemExit(f"no adapter {args.adapter!r}; released: {', '.join(folders)}")
    adapter_dir = dest / "Show-Harness-VLMs" / args.adapter
    fetch(ADAPTER_REPO, ADAPTER_REVISION, mine, adapter_dir, strip=prefix)
    base = json.loads((adapter_dir / "adapter_config.json").read_text())[
        "base_model_name_or_path"
    ]
    print(f"adapter {args.adapter} -> {adapter_dir} (base {base})")
    if args.no_base:
        return
    sha, base_files = siblings(base, BASE_REVISIONS.get(base, "main"))
    base_dir = dest / base.split("/")[-1]
    fetch(base, sha, base_files, base_dir)
    (base_dir / ".revision").write_text(f"{base}@{sha}\n")
    print(f"base {base}@{sha} -> {base_dir}")


if __name__ == "__main__":
    main()
