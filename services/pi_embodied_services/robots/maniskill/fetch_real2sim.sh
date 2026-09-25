#!/usr/bin/env bash
# Fetch the RLinf real2sim rigs (BlockPAP-v1 / BlockStack-v1) that ./scenes.py registers.
#
#   bash fetch_real2sim.sh [dest]          # default /root/autodl-tmp/assets/RLinf-real2sim
#   HTTPS_PROXY=http://127.0.0.1:1056 bash fetch_real2sim.sh   # when github.com is flaky (code only)
#
# 1. Code: real_franka/ of github.com/AaronCaoZJ/RLinf (Apache-2.0, a fork of RLinf/RLinf by a
#    Show-Harness co-author), sparse, pinned to REV. Upstream RLinf does not carry these envs.
# 2. The wood tabletop textures pick_and_place.py loads from
#    <root>/rlinf/envs/maniskill/assets/carrot/more_table/textures/<id>.png: the 21 PNGs of the HF
#    dataset RLinf/maniskill_assets (11.9 MB, via hf-mirror), checked against their LFS sha256.
# 3. BlockStack's panda_v2_extended.urdf (only in RLinf's own ManiSkill build), rebuilt from its
#    spec into the ManiSkill venv's assets by scenes.write_extended_urdf (PYTHON= that venv).
# Then point the env server at it: RLINF_ROOT=<dest> (scenes.py's default is the same path).
set -euo pipefail

DEST="${1:-/root/autodl-tmp/assets/RLinf-real2sim}"
REPO=https://github.com/AaronCaoZJ/RLinf.git
REV=fd52554870dc9c0dbd4243c052f5ec32c56f1340
PYTHON="${PYTHON:-/root/autodl-tmp/venvs/maniskill/bin/python}"
SERVICES="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
HF="${HF_ENDPOINT:-https://hf-mirror.com}/datasets/RLinf/maniskill_assets/resolve/c23fc1880ed7861686d4f995360101eaee4d18a0"

if [ ! -d "$DEST/.git" ]; then
  git clone -q --filter=blob:none --no-checkout "$REPO" "$DEST"
fi
cd "$DEST"
git sparse-checkout set --no-cone /LICENSE /real_franka/ '!/real_franka/real2sim_env/render/' \
  '!/real_franka/data_inspector/**/*.png'
git fetch -q origin "$REV" 2>/dev/null || true
git checkout -q "$REV"
echo "code: $(git rev-parse HEAD) $(git log -1 --format=%ci)"

TEX=rlinf/envs/maniskill/assets/carrot/more_table/textures
mkdir -p "$TEX"
while read -r sha name; do
  f="$TEX/$name"
  if [ ! -f "$f" ] || [ "$(sha256sum "$f" | cut -d' ' -f1)" != "$sha" ]; then
    # Assets never ride the code proxy.
    env -u HTTPS_PROXY -u https_proxy -u ALL_PROXY -u all_proxy curl -sSfL --retry 3 --max-time 120 -o "$f" "$HF/carrot/more_table/textures/$name"
  fi
  [ "$(sha256sum "$f" | cut -d' ' -f1)" = "$sha" ] || { echo "sha256 mismatch: $f" >&2; exit 1; }
done <<'EOF'
1f331921425c90507349c4214be5bdefff0831ca7c11aed488af39ead2d1eddd 001.png
b165f05e4d6572f3e2b6a1cf7d4fadc803b02be3383359f1ece58b4027c34e92 002.png
81d032981718fd3a500d3b7d239c3fb7e460c73fa665bd0d2f3337c963461cd0 003.png
0eabfddc3c7c05c2a5d0639ee506cdd54f0db7cbbfc801aa319fd9650741a314 004.png
e90470b12008a4268ae79a3f2888fedf85a4bad26907d3ea9c84c2216be8101a 005.png
86f6edc07674562e70c1426d60896baeba405d29285f3f76831ec102fcc154fc 006.png
01c43bcee1cbab8c3976e661cf3aff439fe36f54b5b4b3d31af35ad6f372850b 007.png
83c309cf3e2dd86523230e95e60396a2263f4a673c6ee31e6df4fe299d67e822 008.png
54e6c9d4e175b43dd9fb06e91a5ba81ddc3bb6d93a4d68bf4c3f59e85eb43f8b 009.png
41999b50accdb26dae33ff20c8083d65521e2e56b9c6c3a7cd73ed3bffda968d 010.png
78c8926e761919f31491f0f01aadd6b71982b56ad59bb69ae355d5fa6edc4e75 011.png
fb8f10d196e2f9d7b537f9e329ba595fe1983e39d0cf40983cfcdd2b6e33b45c 012.png
ed10c39bef6988e6a8efb3e08e47f093d4433e5ad23c5fa7b04951fb5f8d55fb 013.png
be0f97da6df071f3d66ba3a8a1e70a90ac1706aa6735109701fe2495f77b9a1b 014.png
89027b05c7b414aaec0665db22c8486baf1ebf216030f28ed1abd095409896e0 015.png
96680c9f686864114dc8f37c18a862a0f94264b742f3c065475d9b0814fd1b15 016.png
493a00c9e51bd324773dafda37ebbb5d51d131ce9a94ae697bc7d60adf6777bb 017.png
e805e65c68f97d88d54b8736fbaab939aeb4725e173fdbad9d13000f1d99f892 018.png
561984a7cf85dec16f9ddf94f2d09cda691787ef6c722872318d9203a937c7cf 019.png
748d052846925f2b80cab52b85192dab014ab205526e2d5786072be8df3cf7c4 020.png
c86d3be6a65125014ae0003d539a7b8943dadd161b3550d28ede2e3e7aa6d74c 021.png
EOF
echo "textures: $(ls "$TEX" | wc -l) files, $(du -sh "$TEX" | cut -f1) in $DEST/$TEX"
echo "urdf: $(PYTHONPATH="$SERVICES" "$PYTHON" -m pi_embodied_services.robots.maniskill.scenes 2>/dev/null | tail -1)"
