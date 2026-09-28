"""Download only the official HSSD-Hab files HumanCLAW's 41 val scenes use.

``humanclaw-bench prepare-hssd --hssd-root DIR`` wants an official HSSD-Hab install
(``hssd-hab.scene_dataset_config.json``, ``objects/``, ``stages/``, ``semantics/``) and checks
every mesh it links against ``resources/hssd/humanclaw-hssd-val41/asset_requirements.json`` (size
and sha256). The full ``hssd/hssd-hab`` dataset is far larger than those 12,112 object meshes, 82
stage files and 43 semantic files (~3.8 GB), so this fetches exactly the listed files, in HSSD's
own layout, and verifies each one the same way. ``hssd/hssd-hab`` is gated on Hugging Face (accept
its terms first); huggingface_hub reads ``HF_TOKEN`` and ``HF_ENDPOINT`` (e.g. hf-mirror.com).

    python -m pi_embodied_services.robots.humanclaw.fetch_hssd \\
        --humanclaw ~/HumanCLAW --out ~/data/hssd-hab
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REQUIREMENTS = "resources/hssd/humanclaw-hssd-val41/asset_requirements.json"
SUPPLEMENT = "resources/hssd/humanclaw-hssd-val41/supplement.json"
SCENE_CONFIG = "hssd-hab.scene_dataset_config.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def plan(
    requirements: dict, files: list[str], supplemental: frozenset[str] = frozenset()
) -> list[tuple[str, dict | None]]:
    """The repo paths to fetch with their expected size/sha256 (None: the scene config, unchecked).

    Objects are found by file name anywhere under ``objects/`` (HSSD shards them by the first
    character), except the baked meshes of HumanCLAW's own supplement (``supplemental``, which
    prepare-hssd downloads from HumanCLAW/HumanCLAW-HSSD); stages and semantics are ``<group>/<name>`` exactly as prepare-hssd reads them.
    Raises when a required file is not in the repo.
    """
    assets = requirements["assets"]
    by_name: dict[str, list[str]] = {}
    for path in files:
        if path.startswith("objects/"):
            by_name.setdefault(path.rsplit("/", 1)[-1], []).append(path)
    out: list[tuple[str, dict | None]] = [(SCENE_CONFIG, None)]
    missing: list[str] = []
    for name, spec in sorted(assets["objects"].items()):
        if name in supplemental:
            continue
        paths = by_name.get(name, [])
        if not paths:
            missing.append(f"objects/**/{name}")
            continue
        out.extend((p, spec) for p in sorted(paths))
    present = set(files)
    for group in ("stages", "semantics"):
        for name, spec in sorted(assets[group].items()):
            path = f"{group}/{name}"
            if path not in present:
                missing.append(path)
            out.append((path, spec))
    if missing:
        raise SystemExit(
            f"{len(missing)} required HSSD files are not in the repo, e.g. {missing[:3]!r}"
        )
    return out


def ok(path: Path, spec: dict | None, verify: bool) -> bool:
    if not path.is_file():
        return False
    if spec is None:
        return True
    if path.stat().st_size != int(spec["size_bytes"]):
        return False
    return not verify or sha256(path) == spec["sha256"]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--humanclaw", required=True, help="HumanCLAW checkout")
    p.add_argument("--out", required=True, help="HSSD root to fill")
    p.add_argument("--repo", default="hssd/hssd-hab")
    p.add_argument("--revision", default="main")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--retries", type=int, default=6)
    args = p.parse_args(argv)

    from huggingface_hub import HfApi, hf_hub_download

    requirements = json.loads(
        (Path(args.humanclaw).expanduser() / REQUIREMENTS).read_text(encoding="utf-8")
    )
    # One request (hf-mirror's paginated tree API points its cursor at huggingface.co).
    info = HfApi().repo_info(args.repo, repo_type="dataset", revision=args.revision)
    supplement = json.loads(
        (Path(args.humanclaw).expanduser() / SUPPLEMENT).read_text(encoding="utf-8")
    )
    todo = plan(
        requirements,
        [s.rfilename for s in info.siblings or []],
        frozenset(supplement["files"]),
    )
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    left = [
        (path, spec) for path, spec in todo if not ok(out / path, spec, verify=False)
    ]
    print(
        f"[fetch_hssd] {len(todo)} files required, {len(todo) - len(left)} present, "
        f"{len(left)} to fetch from {args.repo}@{info.sha}",
        flush=True,
    )

    def fetch(path: str, spec: dict | None) -> str:
        for attempt in range(args.retries):
            try:
                hf_hub_download(
                    args.repo,
                    path,
                    repo_type="dataset",
                    revision=info.sha,
                    local_dir=out,
                )
                if ok(out / path, spec, verify=True):
                    return path
                (out / path).unlink(missing_ok=True)
                raise ValueError("size/sha256 mismatch")
            except Exception as error:  # noqa: BLE001
                if attempt == args.retries - 1:
                    raise RuntimeError(f"{path}: {error}") from error
                time.sleep(min(60, 2 ** (attempt + 1)))
        return path

    failed = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(fetch, path, spec) for path, spec in left]
        for i, future in enumerate(as_completed(futures), 1):
            try:
                future.result()
            except Exception as error:  # noqa: BLE001
                failed += 1
                print(f"[fetch_hssd] FAILED {error}", file=sys.stderr, flush=True)
            if i % 250 == 0 or i == len(futures):
                print(
                    f"[fetch_hssd] {i}/{len(futures)} fetched, {failed} failed",
                    flush=True,
                )
    if failed:
        return 1
    bad = [path for path, spec in todo if not ok(out / path, spec, verify=True)]
    if bad:
        print(
            f"[fetch_hssd] {len(bad)} files fail size/sha256, e.g. {bad[:3]!r}",
            file=sys.stderr,
        )
        return 1
    print(f"[fetch_hssd] complete: {len(todo)} files verified under {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
