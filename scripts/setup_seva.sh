#!/bin/bash
# Install Stable Virtual Camera (SEVA) into the project venv, pinned to the commit we studied.
# SEVA is cloned into external/ (git-ignored: its code is not redistributed in this repo) and
# installed editable, because its pyproject only packages the top-level `seva` module, so a
# normal install drops `seva.modules`. --no-deps: SEVA pins numpy==1.24.4; the libraries its
# code actually imports come from our `seva` extra in pyproject.toml.
set -euo pipefail
cd "$(dirname "$0")/.."
SEVA_COMMIT=fe19948e9b7bea261ab2db780a59656131404a83
if [ ! -d external/stable-virtual-camera ]; then
  git clone -q https://github.com/Stability-AI/stable-virtual-camera.git external/stable-virtual-camera
fi
git -C external/stable-virtual-camera checkout -q "$SEVA_COMMIT"
VIRTUAL_ENV="$PWD/.venv" uv pip install -q -e ".[seva]"
VIRTUAL_ENV="$PWD/.venv" uv pip install -q --no-deps -e external/stable-virtual-camera
.venv/bin/python -c "import seva.model, seva.sampling, seva.eval; print('SEVA ready:', seva.model.__file__)"
