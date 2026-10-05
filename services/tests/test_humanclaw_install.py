"""robots/humanclaw/install.sh's weights step and its echo, with every tool stubbed: a tarball that
fails its sha256 check is replaced by one fresh download (never kept to fail every run), a fresh
download that fails is removed with a message, and the command echo hides a URL's userinfo."""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

INSTALL = (
    Path(__file__).resolve().parents[1]
    / "pi_embodied_services"
    / "robots"
    / "humanclaw"
    / "install.sh"
)
WEIGHTS = "HumanCLAW_pretrained_weights_paper_fullval_v1_20260816.tar.gz"
GOOD = b"the paper_fullval_v1 motion weights"

pytestmark = pytest.mark.skipif(
    shutil.which("sha256sum") is None or shutil.which("bash") is None,
    reason="needs bash and sha256sum",
)


def _tool(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!/usr/bin/env bash\n{body}\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


def _sandbox(tmp_path: Path, download: bytes) -> dict:
    """install.sh with the tarball's pin set to GOOD's sha256, its tools stubbed and logged; the
    stubbed `hf` writes `download` as the tarball."""
    log = tmp_path / "calls.log"
    scripts = tmp_path / "a" / "b" / "scripts"
    scripts.mkdir(parents=True)
    text = INSTALL.read_text(encoding="utf-8")
    pin = "WEIGHTS_SHA256=" + hashlib.sha256(GOOD).hexdigest()
    lines = [pin if ln.startswith("WEIGHTS_SHA256=") else ln for ln in text.split("\n")]
    assert pin in lines, "install.sh pins WEIGHTS_SHA256"
    (scripts / "install.sh").write_text("\n".join(lines), encoding="utf-8")
    _tool(scripts / "build_habitat.sh", f'echo "build_habitat $*" >>"{log}"')
    bin_ = tmp_path / "bin"
    _tool(
        bin_ / "git",
        f'echo "git $*" >>"{log}"\ncase "$*" in *rev-parse*) echo c4f9351 ;; esac',
    )
    _tool(bin_ / "uv", f'echo "uv $*" >>"{log}"')
    _tool(
        bin_ / "tar",
        f'echo "tar $*" >>"{log}"\nmkdir -p weights/paper_fullval_v1/base && : >weights/paper_fullval_v1/base/motion_dit.pt',
    )
    venv = tmp_path / "venv"
    _tool(venv / "bin" / "python", "exit 0")
    _tool(venv / "bin" / "humanclaw-bench", f'echo "humanclaw-bench $*" >>"{log}"')
    blob = tmp_path / "download.bin"
    blob.write_bytes(download)
    _tool(
        venv / "bin" / "hf",
        f'echo "hf $*" >>"{log}"\nfor a in "$@"; do case $a in *.tar.gz) f=$a ;; esac; done\ncp "{blob}" "$f"',
    )
    hc = tmp_path / "root" / "HumanCLAW"
    (hc / ".git").mkdir(parents=True)
    (hc / "constraints").mkdir()
    (hc / "constraints" / "eval-cu124.txt").write_text(
        "torch==2.6.0\nnumpy==1.26\n", encoding="utf-8"
    )
    hssd = tmp_path / "hssd"
    hssd.mkdir()
    (hssd / "hssd-hab.scene_dataset_config.json").write_text("{}", encoding="utf-8")
    env = {
        **os.environ,
        "PATH": f"{bin_}:{os.environ['PATH']}",
        "UV_INDEX_URL": "https://user:s3cr3t@mirror.example/simple",
        "HUMANCLAW_TORCH_INDEX": "https://u:p4ss@idx.example/whl/cu128",
    }
    run = lambda: subprocess.run(  # noqa: E731
        ["bash", str(scripts / "install.sh"), str(venv), str(hc), str(hssd)],
        env=env,
        capture_output=True,
        text=True,
    )
    return {"run": run, "log": log, "tarball": hc.parent / WEIGHTS, "hc": hc}


def test_stale_tarball_is_replaced_by_one_fresh_download_and_urls_are_redacted(
    tmp_path,
):
    sb = _sandbox(tmp_path, GOOD)
    sb["tarball"].write_bytes(b"truncated")
    r = sb["run"]()
    assert r.returncode == 0, r.stdout + r.stderr
    assert "fails its sha256 check" in r.stderr
    calls = sb["log"].read_text(encoding="utf-8").splitlines()
    assert sum(c.startswith("hf download") for c in calls) == 1
    assert any(c.startswith("tar -xzf") for c in calls)
    assert sb["tarball"].read_bytes() == GOOD
    assert (
        sb["hc"] / "weights" / "paper_fullval_v1" / "base" / "motion_dit.pt"
    ).is_file()
    # HC-8: the echo of `uv pip install --index-url ... --extra-index-url ...` hides both tokens.
    assert "s3cr3t" not in r.stdout + r.stderr and "p4ss" not in r.stdout + r.stderr
    assert (
        "://***@mirror.example/simple" in r.stdout
        and "://***@idx.example/whl/cu128" in r.stdout
    )


def test_fresh_download_that_fails_its_checksum_is_removed_with_a_message(tmp_path):
    sb = _sandbox(tmp_path, b"a stale mirror copy")
    r = sb["run"]()
    assert r.returncode == 1
    assert "does not match sha256" in r.stderr
    assert not sb["tarball"].exists(), "the bad tarball is not kept"
    calls = sb["log"].read_text(encoding="utf-8").splitlines()
    assert sum(c.startswith("hf download") for c in calls) == 1
    assert not any(c.startswith("tar") for c in calls), "nothing is extracted"
    # The next run downloads again rather than failing on the kept file.
    sb2 = _sandbox(tmp_path / "again", GOOD)
    assert sb2["run"]().returncode == 0


def test_good_tarball_is_kept_and_not_downloaded_again(tmp_path):
    sb = _sandbox(tmp_path, b"never fetched")
    sb["tarball"].write_bytes(GOOD)
    r = sb["run"]()
    assert r.returncode == 0, r.stdout + r.stderr
    calls = sb["log"].read_text(encoding="utf-8").splitlines()
    assert not any(c.startswith("hf download") for c in calls)
    assert sb["tarball"].read_bytes() == GOOD
