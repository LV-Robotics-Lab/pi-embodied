#!/usr/bin/env bash
# Build HumanCLAW's patched Habitat-Sim into a venv, as its README (patches/habitat-sim/README.md)
# prescribes: habitat-sim at the pinned commit, humanclaw_halfphysics.patch applied (sha256 checked),
# a headless + CUDA + Bullet build, the editable install with the same switches, and the magnum
# bindings. Idempotent: an existing checkout at the pin is reused, an applied patch is kept.
#
#   build_habitat.sh <venv> <humanclaw-checkout> [habitat-src-dir]
#
# habitat-src-dir defaults to <humanclaw-checkout>/../habitat-sim. Extra CMake arguments for the
# host's pitfalls (patches/habitat-sim/README.md: a newer CMake's policy floor, a slim CUDA without
# cudart, a broken ccache) go in HABITAT_CMAKE_ARGS, e.g.
#   HABITAT_CMAKE_ARGS="-DCMAKE_POLICY_VERSION_MINIMUM=3.5" build_habitat.sh ...
# Git honours GIT_PROXY (e.g. http://127.0.0.1:1056) for the clone and its submodules.
set -euo pipefail
venv=${1:?usage: build_habitat.sh <venv> <humanclaw-checkout> [habitat-src-dir]}
hc=$(cd "${2:?humanclaw checkout}" && pwd)
src=${3:-$hc/../habitat-sim}
PIN=acbe6f4922e68145e401e55c30f9dfea460a3f24
PATCH=$hc/patches/habitat-sim/humanclaw_halfphysics.patch
PATCH_SHA256=6f57ec8130b4ccca7d208a0754ba691d52e44b469cd6dcee220fa193fcad6766
PY=$venv/bin/python
[ -x "$PY" ] || { echo "build_habitat.sh: no python in $venv" >&2; exit 1; }
git_() { if [ -n "${GIT_PROXY:-}" ]; then git -c http.proxy="$GIT_PROXY" -c https.proxy="$GIT_PROXY" "$@"; else git "$@"; fi; }
run() { printf '+'; printf ' %q' "$@"; printf '\n'; "$@"; }

got=$(sha256sum "$PATCH" | cut -d' ' -f1)
[ "$got" = "$PATCH_SHA256" ] || { echo "build_habitat.sh: $PATCH sha256 $got, expected $PATCH_SHA256" >&2; exit 1; }
[ -d "$src/.git" ] || run git_ clone https://github.com/facebookresearch/habitat-sim.git "$src"
cd "$src"
if [ "$(git rev-parse HEAD)" != "$PIN" ]; then
	git diff --quiet || { echo "build_habitat.sh: $src has local changes and is not at $PIN" >&2; exit 1; }
	run git_ fetch origin "$PIN" || true
	run git checkout -q "$PIN"
fi
run git_ submodule update --init --recursive
if git apply --reverse --check "$PATCH" 2>/dev/null; then
	echo "== $PATCH already applied"
else
	run git apply --check "$PATCH"
	run git apply "$PATCH"
fi
run "$PY" -m pip install -r requirements.txt
# The paper build: headless, CUDA and Bullet (Bullet runs on the CPU; CUDA is still part of the build).
cmake_args=${HABITAT_CMAKE_ARGS:-}
run "$PY" setup.py build_ext --inplace --headless --with-cuda --bullet --cache-args \
	${cmake_args:+--cmake --cmake-args="$cmake_args"}
# The in-place build is the editable install: put src_python on the venv's path. (`pip install -e .`,
# with the switches again, re-runs CMake, which re-fetches OpenEXR's Imath from GitHub; offline or
# behind a flaky mirror that step alone fails after a complete build.)
site=$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
echo "$src/src_python" >"$site/habitat_sim.pth"
run "$PY" -m pip install -e build/deps/magnum-bindings/src/python
TORCH_LIB=$("$PY" -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).parent / "lib")')
LD_LIBRARY_PATH="$TORCH_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" "$PY" - <<'PY'
import habitat_sim
from habitat_sim._ext import habitat_sim_bindings

assert habitat_sim.built_with_bullet, "habitat-sim was built without Bullet"
print("habitat_sim", habitat_sim.__file__, habitat_sim_bindings.__file__)
PY
