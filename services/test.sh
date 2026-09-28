#!/usr/bin/env bash
# The services' unit tests and lint, with the test dependencies only. The robot extras in
# pyproject.toml are mutually incompatible by design (one venv per backend, as setup.sh builds
# them), so plain `uv run pytest` - which resolves every extra into one environment - does not
# work; --no-project skips that resolution. Extra arguments go to pytest.
set -euo pipefail
cd "$(dirname "$0")"
uv run -q --no-project --with numpy --with pytest --with pyyaml --with scipy --with omegaconf \
	--with msgpack --with pillow --with pyarrow python -m pytest -q "${@:-tests}"
uvx ruff check .
uvx ruff format --check .
