"""Fetch a released Show-Harness adapter and the base model it names, pinned and verified.

    python -m pi_embodied_services.finetuned.download --adapter qwen3_5_2b_sim \\
        --dest /root/autodl-tmp/checkpoints

writes ``<dest>/Show-Harness-VLMs/<adapter>/`` and ``<dest>/<base name>/`` (e.g. ``Qwen3.5-2B``).
Files come from ``$HF_ENDPOINT`` (default https://hf-mirror.com) at a pinned revision, big ones
through ``aria2c -x16``. The file list, sizes and hashes are pinned in this script (``PINS``), not
fetched from the endpoint: a mirror serving tampered weights would also serve matching metadata.
Every file is checked against its pinned size and hash (LFS files: the sha256 of the content;
small files: the git blob sha1 ``sha1("blob <size>\\0" + content)``) and a mismatch aborts. Standard
library only, so any Python runs it. The released adapters
(showlab/Show-Harness-VLMs @ ADAPTER_REVISION): qwen3_5_0_8b, qwen3_5_2b, qwen3_5_4b, qwen3_5_9b,
gemma4_e4b (the real-robot ``ft`` split) and qwen3_5_2b_sim (RoboLab + ManiSkill).

``PINS`` was taken on 2026-09-25 from huggingface.co's own API
(``/api/models/<repo>/revision/<rev>?blobs=true``), each LFS sha256 and size checked against the
LFS pointer at ``huggingface.co/<repo>/raw/<rev>/<file>``, and compared with hf-mirror's answer for
the same revisions (identical). Base revisions: Qwen3.5-2B is the one the adapters were checked
against; the other bases are their ``main`` on that date. To move a pin, regenerate that repo's
entry the same way, from huggingface.co, never from the mirror.
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
# repo -> {revision, files: {path: (size, "sha256:<hex>" for LFS | "git-sha1:<git blob id>")}}.
PINS: dict[str, dict] = {
    "showlab/Show-Harness-VLMs": {
        "revision": "c1c94dfa0f5f1e8827acbc9698ba5f765f1605e5",
        "files": {
            ".gitattributes": (
                2209,
                "git-sha1:186d317fc7b74a7d5d89f8c4b6bbf1c592c80614",
            ),
            "README.md": (4685, "git-sha1:41abf5978b61a7a737a75f9b314a1ebef008c40e"),
            "gemma4_e4b/README.md": (
                177,
                "git-sha1:e6abfc13305bb686c64bdc28fb2ae02ae5cbb009",
            ),
            "gemma4_e4b/adapter_config.json": (
                9936,
                "git-sha1:31487c3de6e92f025e11f12d3fad2f654c991755",
            ),
            "gemma4_e4b/adapter_model.safetensors": (
                311137912,
                "sha256:9089179579bace066f663add550a9c2ba6f9c5ffb4b87bc5df90a5c5e2c6922e",
            ),
            "gemma4_e4b/chat_template.jinja": (
                2429,
                "git-sha1:9dcd869580fb561be66701d5464a95eb15039247",
            ),
            "gemma4_e4b/processor_config.json": (
                1689,
                "git-sha1:5465974d23e1eca2c46c2809b26c997946ce0d90",
            ),
            "gemma4_e4b/tokenizer.json": (
                32169626,
                "sha256:cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f",
            ),
            "gemma4_e4b/tokenizer_config.json": (
                2777,
                "git-sha1:7d97c436291f6ed764835968e9c87d2f0c09ea0a",
            ),
            "qwen3_5_0_8b/README.md": (
                178,
                "git-sha1:d7448c8e7e7c09ed2ad74c2971210e004fa7462c",
            ),
            "qwen3_5_0_8b/adapter_config.json": (
                1135,
                "git-sha1:97b4c4c51f2c61a5a45adeca9d7845bb27c968b3",
            ),
            "qwen3_5_0_8b/adapter_model.safetensors": (
                86637528,
                "sha256:4be1683776d077701ddc309f51f41f57f2af8df80698fe0d5fc9569eaec59110",
            ),
            "qwen3_5_0_8b/chat_template.jinja": (
                9665,
                "git-sha1:567088a3bda735237f5cb709a418ff4f6ee611ed",
            ),
            "qwen3_5_0_8b/processor_config.json": (
                1191,
                "git-sha1:33818c7f9e991ad735fd240209f4fa73e6c28c50",
            ),
            "qwen3_5_0_8b/tokenizer.json": (
                19989325,
                "sha256:06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
            ),
            "qwen3_5_0_8b/tokenizer_config.json": (
                1192,
                "git-sha1:a24f83f0b01c5220e20b7b299ee543c74433383b",
            ),
            "qwen3_5_2b/README.md": (
                176,
                "git-sha1:65b2a82f3734fcdb17f617d883424dde0986549a",
            ),
            "qwen3_5_2b/adapter_config.json": (
                1133,
                "git-sha1:97dab891d2bd1adae573debede48a4d5589a3319",
            ),
            "qwen3_5_2b/adapter_model.safetensors": (
                134610104,
                "sha256:77e6a10bf5ebd9027ea4cd74ad2f1142236e57200fe1ff23318d0df33b3187e1",
            ),
            "qwen3_5_2b/chat_template.jinja": (
                9665,
                "git-sha1:567088a3bda735237f5cb709a418ff4f6ee611ed",
            ),
            "qwen3_5_2b/processor_config.json": (
                1191,
                "git-sha1:33818c7f9e991ad735fd240209f4fa73e6c28c50",
            ),
            "qwen3_5_2b/tokenizer.json": (
                19989325,
                "sha256:06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
            ),
            "qwen3_5_2b/tokenizer_config.json": (
                1192,
                "git-sha1:a24f83f0b01c5220e20b7b299ee543c74433383b",
            ),
            "qwen3_5_2b_sim/README.md": (
                176,
                "git-sha1:65b2a82f3734fcdb17f617d883424dde0986549a",
            ),
            "qwen3_5_2b_sim/adapter_config.json": (
                1133,
                "git-sha1:c570a2eac3c6be14eb0ed93cbe25989abe58c76b",
            ),
            "qwen3_5_2b_sim/adapter_model.safetensors": (
                134610104,
                "sha256:82f8cb6d177722b05a3705dbd7ce9fb558435321a114e70572a0c89378a5aa49",
            ),
            "qwen3_5_2b_sim/chat_template.jinja": (
                9665,
                "git-sha1:567088a3bda735237f5cb709a418ff4f6ee611ed",
            ),
            "qwen3_5_2b_sim/processor_config.json": (
                1191,
                "git-sha1:33818c7f9e991ad735fd240209f4fa73e6c28c50",
            ),
            "qwen3_5_2b_sim/tokenizer.json": (
                19989325,
                "sha256:06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
            ),
            "qwen3_5_2b_sim/tokenizer_config.json": (
                1192,
                "git-sha1:a24f83f0b01c5220e20b7b299ee543c74433383b",
            ),
            "qwen3_5_4b/README.md": (
                176,
                "git-sha1:4051e766e08929289acc1b4c425a7a41b8b8896f",
            ),
            "qwen3_5_4b/adapter_config.json": (
                1133,
                "git-sha1:569843c798016006097fbd8004c1fae97ca4f258",
            ),
            "qwen3_5_4b/adapter_model.safetensors": (
                259794944,
                "sha256:52821d443398dd3aed51957c86ab52999721b933ca171eb8eaed6eb6f02a5d14",
            ),
            "qwen3_5_4b/chat_template.jinja": (
                9665,
                "git-sha1:567088a3bda735237f5cb709a418ff4f6ee611ed",
            ),
            "qwen3_5_4b/processor_config.json": (
                1191,
                "git-sha1:33818c7f9e991ad735fd240209f4fa73e6c28c50",
            ),
            "qwen3_5_4b/tokenizer.json": (
                19989325,
                "sha256:06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
            ),
            "qwen3_5_4b/tokenizer_config.json": (
                1192,
                "git-sha1:a24f83f0b01c5220e20b7b299ee543c74433383b",
            ),
            "qwen3_5_9b/README.md": (
                176,
                "git-sha1:d71b4e08c19201bd66fabda6141a244d06197cb0",
            ),
            "qwen3_5_9b/adapter_config.json": (
                1133,
                "git-sha1:96f5be231bb02bc971bdfdac9599cde219aa0451",
            ),
            "qwen3_5_9b/adapter_model.safetensors": (
                346302672,
                "sha256:bdbea8b69c0e8f319c9d6dca76bba570886d06ba0f24f557b82742430547966a",
            ),
            "qwen3_5_9b/chat_template.jinja": (
                9665,
                "git-sha1:567088a3bda735237f5cb709a418ff4f6ee611ed",
            ),
            "qwen3_5_9b/processor_config.json": (
                1191,
                "git-sha1:33818c7f9e991ad735fd240209f4fa73e6c28c50",
            ),
            "qwen3_5_9b/tokenizer.json": (
                19989325,
                "sha256:06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
            ),
            "qwen3_5_9b/tokenizer_config.json": (
                1192,
                "git-sha1:a24f83f0b01c5220e20b7b299ee543c74433383b",
            ),
        },
    },
    "Qwen/Qwen3.5-2B": {
        "revision": "15852e8c16360a2fea060d615a32b45270f8a8fc",
        "files": {
            ".gitattributes": (
                1570,
                "git-sha1:52373fe24473b1aa44333d318f578ae6bf04b49b",
            ),
            "LICENSE": (11544, "git-sha1:f938136e3adacfd92be087f6e113b5d6d97f678f"),
            "README.md": (62814, "git-sha1:2efa49d346c547ea81cd3d1310f414df25934f45"),
            "chat_template.jinja": (
                7755,
                "git-sha1:0ef09f214eaa6d9bca297988afc1454b5827b2c7",
            ),
            "config.json": (2908, "git-sha1:d30a15be1bbfd610f58ad80f1fbeb6778f58a80f"),
            "merges.txt": (
                3353259,
                "git-sha1:a494e019ca1502219fd0128658b979e5f05ae8e8",
            ),
            "model.safetensors-00001-of-00001.safetensors": (
                4548221488,
                "sha256:aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1",
            ),
            "model.safetensors.index.json": (
                64460,
                "git-sha1:69a41758e1138259916d08b99b56fada441fd502",
            ),
            "preprocessor_config.json": (
                390,
                "git-sha1:2ea84a437d448ff71b08df68fdd949d5cc4ebb64",
            ),
            "tokenizer.json": (
                12807982,
                "sha256:5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
            ),
            "tokenizer_config.json": (
                16709,
                "git-sha1:fae3ce993e07c092ad024dde45e592379fde91bb",
            ),
            "video_preprocessor_config.json": (
                385,
                "git-sha1:3ba673a5ad7d4d13f54155ecd38b2a94a6dac8fe",
            ),
            "vocab.json": (
                6722759,
                "git-sha1:0aa0ce0658d60ac4a5d609f4eadb0e8e43514176",
            ),
        },
    },
    "Qwen/Qwen3.5-0.8B": {
        "revision": "2fc06364715b967f1860aea9cf38778875588b17",
        "files": {
            ".gitattributes": (
                1570,
                "git-sha1:52373fe24473b1aa44333d318f578ae6bf04b49b",
            ),
            "LICENSE": (11544, "git-sha1:f938136e3adacfd92be087f6e113b5d6d97f678f"),
            "README.md": (61705, "git-sha1:5824f1761b2b3a55a2141a9a1172a7f92c7c2ad9"),
            "chat_template.jinja": (
                7755,
                "git-sha1:0ef09f214eaa6d9bca297988afc1454b5827b2c7",
            ),
            "config.json": (2907, "git-sha1:715f0448b9d38103211f0ad88bbb4d6e4f4be8c9"),
            "merges.txt": (
                3353259,
                "git-sha1:a494e019ca1502219fd0128658b979e5f05ae8e8",
            ),
            "model.safetensors-00001-of-00001.safetensors": (
                1746942600,
                "sha256:04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696",
            ),
            "model.safetensors.index.json": (
                50900,
                "git-sha1:f691cefdb79d73270895ebd6d9594ddcecfc1838",
            ),
            "preprocessor_config.json": (
                390,
                "git-sha1:2ea84a437d448ff71b08df68fdd949d5cc4ebb64",
            ),
            "tokenizer.json": (
                12807982,
                "sha256:5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
            ),
            "tokenizer_config.json": (
                16709,
                "git-sha1:fae3ce993e07c092ad024dde45e592379fde91bb",
            ),
            "video_preprocessor_config.json": (
                385,
                "git-sha1:3ba673a5ad7d4d13f54155ecd38b2a94a6dac8fe",
            ),
            "vocab.json": (
                6722759,
                "git-sha1:0aa0ce0658d60ac4a5d609f4eadb0e8e43514176",
            ),
        },
    },
    "Qwen/Qwen3.5-4B": {
        "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "files": {
            ".gitattributes": (
                1570,
                "git-sha1:52373fe24473b1aa44333d318f578ae6bf04b49b",
            ),
            "LICENSE": (11544, "git-sha1:f938136e3adacfd92be087f6e113b5d6d97f678f"),
            "README.md": (77661, "git-sha1:7950a3aadf378cd13758097bc52f0ed849a59007"),
            "chat_template.jinja": (
                7756,
                "git-sha1:a585dec894e63da457d9440ec6aa7caa16d20860",
            ),
            "config.json": (3161, "git-sha1:557d961b205319c6a7da5f757f565b69b3967b7d"),
            "merges.txt": (
                3353259,
                "git-sha1:a494e019ca1502219fd0128658b979e5f05ae8e8",
            ),
            "model.safetensors-00001-of-00002.safetensors": (
                5329398688,
                "sha256:26a93f066e1916adb13453dae5a0c707c0fbc71299ed98779571a907b8e74c61",
            ),
            "model.safetensors-00002-of-00002.safetensors": (
                3990429408,
                "sha256:cb544bd9bfae93dc59b0f22b292f5933573854a7f9b97835c67060d7d910e188",
            ),
            "model.safetensors.index.json": (
                76196,
                "git-sha1:fddda6039f7c1d17260c9e923b8a72fd025d9a86",
            ),
            "preprocessor_config.json": (
                390,
                "git-sha1:2ea84a437d448ff71b08df68fdd949d5cc4ebb64",
            ),
            "tokenizer.json": (
                12807982,
                "sha256:5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
            ),
            "tokenizer_config.json": (
                16710,
                "git-sha1:eda48d3e75a8e59a8479ee4ec8b37f76e711d9c1",
            ),
            "video_preprocessor_config.json": (
                385,
                "git-sha1:3ba673a5ad7d4d13f54155ecd38b2a94a6dac8fe",
            ),
            "vocab.json": (
                6722759,
                "git-sha1:0aa0ce0658d60ac4a5d609f4eadb0e8e43514176",
            ),
        },
    },
    "Qwen/Qwen3.5-9B": {
        "revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        "files": {
            ".gitattributes": (
                1570,
                "git-sha1:52373fe24473b1aa44333d318f578ae6bf04b49b",
            ),
            "LICENSE": (11544, "git-sha1:f938136e3adacfd92be087f6e113b5d6d97f678f"),
            "README.md": (77643, "git-sha1:0f3972cb2c995a86bde9ac92440da530fe2b2c68"),
            "chat_template.jinja": (
                7756,
                "git-sha1:a585dec894e63da457d9440ec6aa7caa16d20860",
            ),
            "config.json": (3126, "git-sha1:273ce437e01baf96a07cd9eb3d5f48bac8d7c657"),
            "merges.txt": (
                3353259,
                "git-sha1:a494e019ca1502219fd0128658b979e5f05ae8e8",
            ),
            "model.safetensors-00001-of-00004.safetensors": (
                5276436216,
                "sha256:db6f444b43d318c92f360a13a25561a6a65b10c0631b8ed305a426dbaa6c380e",
            ),
            "model.safetensors-00002-of-00004.safetensors": (
                5335161512,
                "sha256:31c7d7e2dd5d207840b31cc59083c8f4c4718959149e0358c0364052bb9a0330",
            ),
            "model.safetensors-00003-of-00004.safetensors": (
                5368717440,
                "sha256:7ec36ba3a4176a44c3c0876ad80c56a2f70c84bf008d82e9501df642f17dadec",
            ),
            "model.safetensors-00004-of-00004.safetensors": (
                3325995712,
                "sha256:b62b0c4cd7e44edee103ee8f4fe225f246d5e768e07bfd5f25b63a8aa1fdd0c6",
            ),
            "model.safetensors.index.json": (
                79657,
                "git-sha1:e4c1cb7dba5096b43b9d92bc781aba5e3aa8acd8",
            ),
            "preprocessor_config.json": (
                390,
                "git-sha1:2ea84a437d448ff71b08df68fdd949d5cc4ebb64",
            ),
            "tokenizer.json": (
                12807982,
                "sha256:5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
            ),
            "tokenizer_config.json": (
                16710,
                "git-sha1:eda48d3e75a8e59a8479ee4ec8b37f76e711d9c1",
            ),
            "video_preprocessor_config.json": (
                385,
                "git-sha1:3ba673a5ad7d4d13f54155ecd38b2a94a6dac8fe",
            ),
            "vocab.json": (
                6722759,
                "git-sha1:0aa0ce0658d60ac4a5d609f4eadb0e8e43514176",
            ),
        },
    },
    "google/gemma-4-E4B-it": {
        "revision": "ee0ef6023621cff504d758262d4e04895a5af4a2",
        "files": {
            ".gitattributes": (
                1570,
                "git-sha1:52373fe24473b1aa44333d318f578ae6bf04b49b",
            ),
            "README.md": (27956, "git-sha1:0a4a3bf163f3a4248d7c46ca32b96dcabd1ab2f4"),
            "chat_template.jinja": (
                18569,
                "git-sha1:fbe3b59b625cd1b8850ea592d4203df6ec04684b",
            ),
            "config.json": (5145, "git-sha1:d68960fdcce766f2bfe41436325a8a483a74d125"),
            "generation_config.json": (
                208,
                "git-sha1:e605bb4523b1462ea9d9a3810b9e3ecf7ab7b1f6",
            ),
            "model.safetensors": (
                15992595884,
                "sha256:cfbd3d2f1cd71bd471c37fe2bf8546d5028d41e5736f64e1ca6c6b8893125503",
            ),
            "processor_config.json": (
                1689,
                "git-sha1:5465974d23e1eca2c46c2809b26c997946ce0d90",
            ),
            "tokenizer.json": (
                32169626,
                "sha256:cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f",
            ),
            "tokenizer_config.json": (
                3082,
                "git-sha1:6068e357379d36f823c377e30efb101fa2c67fe4",
            ),
        },
    },
}
ADAPTER_REVISION = PINS[ADAPTER_REPO]["revision"]
BIG = 64 << 20
# hf-mirror answers 403 to urllib's default User-Agent.
HEADERS = {"User-Agent": "pi-embodied-finetuned/1"}


def endpoint() -> str:
    return os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")


def digest(path: Path, kind: str) -> str:
    """`kind` "sha256" (the content's) or "git-sha1" (the git blob id: sha1 of a header + content)."""
    h = hashlib.sha256() if kind == "sha256" else hashlib.sha1()
    if kind == "git-sha1":
        h.update(b"blob %d\0" % path.stat().st_size)
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return f"{kind}:{h.hexdigest()}"


def verified(path: Path, size: int, want: str) -> bool:
    return (
        path.exists()
        and path.stat().st_size == size
        and digest(path, want.split(":")[0]) == want
    )


def fetch(repo: str, files: list[str], dest: Path, strip: str = "") -> None:
    """Download `files` of the pinned `repo` into `dest`, each checked against its pinned hash."""
    pin = PINS[repo]
    revision = pin["revision"]
    dest.mkdir(parents=True, exist_ok=True)
    for name in files:
        size, want = pin["files"][name]
        out = dest / name[len(strip) :]
        if verified(out, size, want):
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
        if got != size:
            raise SystemExit(f"{out}: {got} B, expected {size} B (pinned)")
        have = digest(out, want.split(":")[0])
        if have != want:
            bad = out.with_name(out.name + ".mismatch")
            out.replace(bad)
            raise SystemExit(
                f"{bad}: {have} does not match the pinned {want}; {endpoint()} served "
                f"different content for {repo}@{revision}/{name}"
            )
        print(f"got  {out} ({got} B, {want.split(':')[0]} ok)")


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

    files = PINS[ADAPTER_REPO]["files"]
    prefix = f"{args.adapter}/"
    mine = [f for f in files if f.startswith(prefix)]
    if not mine:
        folders = sorted({f.split("/")[0] for f in files if "/" in f})
        raise SystemExit(f"no adapter {args.adapter!r}; released: {', '.join(folders)}")
    adapter_dir = dest / "Show-Harness-VLMs" / args.adapter
    fetch(ADAPTER_REPO, mine, adapter_dir, strip=prefix)
    base = json.loads((adapter_dir / "adapter_config.json").read_text())[
        "base_model_name_or_path"
    ]
    print(f"adapter {args.adapter} -> {adapter_dir} (base {base})")
    if args.no_base:
        return
    if base not in PINS:
        raise SystemExit(f"base {base} has no pinned revision and hashes in PINS")
    sha = PINS[base]["revision"]
    base_dir = dest / base.split("/")[-1]
    fetch(base, list(PINS[base]["files"]), base_dir)
    (base_dir / ".revision").write_text(f"{base}@{sha}\n")
    print(f"base {base}@{sha} -> {base_dir}")


if __name__ == "__main__":
    main()
