#!/usr/bin/env bash
# setup.sh
# -----------------------------------------------------------------------------
# Combined RunPod setup for openpi (SAFE adaptation) + LIBERO rollout environment.
# Idempotent — safe to re-run on a pod that already has some pieces installed.
#
# Target template: runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
# Recommended GPU: RTX 4090 / A5000+ (sm_89, native bf16, >=24GB)
#
# NOTE: openpi's policy server runs on JAX (jax[cuda12]==0.5.0), NOT PyTorch.
# The torch baked into the RunPod template is irrelevant here — JAX ships its own
# bundled CUDA 12 wheels, so all it needs from the host is a recent NVIDIA driver.
# LIBERO (the rollout client) is a SEPARATE Python 3.8 venv that talks to the
# policy server over a websocket, exactly as in the repo README.
#
# What this script does, in order:
#   1. Sanity-check the host (NVIDIA driver visible, Python present)
#   2. Install uv
#   3. Install apt-level system libs for headless MuJoCo + git-lfs + ffmpeg
#   4. Init the LIBERO git submodule (rewriting its SSH URL to HTTPS)
#   5. Build the base openpi JAX env with `uv sync` (LFS smudge skipped)
#   6. Smoke-test that JAX sees the GPU and that openpi (incl. our SAFE edits) imports
#   7. Build the LIBERO Python 3.8 venv + editable openpi-client / libero installs
#   8. Configure MuJoCo headless EGL rendering + pre-write LIBERO config (no prompt)
#   9. Smoke-test mujoco/robosuite/libero import + a real env reset/render/step
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh
#
# After this completes, run a rollout in two terminals (see notes at the end):
#   T1 (base env):   uv run scripts/serve_policy.py --env LIBERO --record --save_name ...
#   T2 (libero env): source examples/libero/.venv/bin/activate && python examples/libero/main.py ...
# -----------------------------------------------------------------------------

set -euo pipefail

# ----- Config ----------------------------------------------------------------
# Resolve the repo root from this script's own location so it works regardless of cwd.
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="${WORKSPACE:-/workspace}"

LIBERO_DIR="${REPO_DIR}/third_party/libero"
LIBERO_VENV="${REPO_DIR}/examples/libero/.venv"
LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${WORKSPACE}/.libero}"
ROBOSUITE_ASSETS="${WORKSPACE}/robosuite_assets"

# Caches on the (persistent) workspace volume so re-pulls survive pod restarts.
export UV_CACHE_DIR="${WORKSPACE}/.cache/uv"
export PIP_CACHE_DIR="${WORKSPACE}/.cache/pip"
export HF_HOME="${WORKSPACE}/.cache/huggingface"
export XDG_CACHE_HOME="${WORKSPACE}/.cache"
export TMPDIR="${WORKSPACE}/tmp"

mkdir -p \
    "${UV_CACHE_DIR}" \
    "${PIP_CACHE_DIR}" \
    "${HF_HOME}" \
    "${XDG_CACHE_HOME}" \
    "${TMPDIR}" \
    "${LIBERO_CONFIG_PATH}" \
    "${ROBOSUITE_ASSETS}"

EXPECTED_PYTHON_MAJORMINOR="3.11"   # openpi base env (pyproject: requires-python >=3.11)

# LIBERO's requirements pin an old torch/torchvision; the cu113 index has the
# matching GPU wheels. This mirrors the repo README exactly.
LIBERO_TORCH_INDEX_URL="https://download.pytorch.org/whl/cu113"

# ----- Helpers ---------------------------------------------------------------
log()     { printf '\033[1;36m[setup] %s\033[0m\n' "$*"; }
warn()    { printf '\033[1;33m[setup WARN] %s\033[0m\n' "$*" >&2; }
fail()    { printf '\033[1;31m[setup FAIL] %s\033[0m\n' "$*" >&2; exit 1; }
section() { printf '\n\033[1;35m===== %s =====\033[0m\n' "$*"; }

cd "${REPO_DIR}"
log "Repo:       ${REPO_DIR}"
log "Workspace:  ${WORKSPACE}"

# ----- Step 1: Host sanity check ---------------------------------------------
section "Step 1 — Verify host (NVIDIA driver + Python)"

if command -v nvidia-smi >/dev/null 2>&1; then
    DRIVER_VER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 || echo '?')"
    GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 || echo '?')"
    log "GPU:        ${GPU_NAME}"
    log "Driver:     ${DRIVER_VER}"
else
    warn "nvidia-smi not found. JAX will fall back to CPU and rollouts will be unusably slow."
fi

PY_MM="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo '?')"
log "System py:  ${PY_MM}"
if [[ "${PY_MM}" != "${EXPECTED_PYTHON_MAJORMINOR}" ]]; then
    warn "System Python is ${PY_MM}, base env expects ${EXPECTED_PYTHON_MAJORMINOR}. uv will fetch the right one for the venv."
fi

# ----- Step 2: Install uv ----------------------------------------------------
section "Step 2 — Install uv"

if command -v uv >/dev/null 2>&1; then
    log "uv already present: $(uv --version)"
else
    pip install -q uv
    log "Installed: $(uv --version)"
fi

# ----- Step 3: System-level apt packages -------------------------------------
section "Step 3 — Install apt packages (MuJoCo headless deps + git-lfs + ffmpeg)"

# EGL: GPU-accelerated offscreen rendering (preferred)
# OSMesa: software CPU fallback if EGL fails on a particular pod
# X11 stubs: some MuJoCo paths still touch them even without a display
# git-lfs: openpi tracks some assets via LFS (we skip smudge, but need the binary)
# build-essential + cmake: build deps for any source-only LIBERO deps
# ffmpeg: imageio uses it for rollout video encoding

if [[ "$(id -u)" -eq 0 ]]; then APT="apt-get"; else APT="sudo apt-get"; fi

export DEBIAN_FRONTEND=noninteractive
${APT} update -qq
${APT} install -y --no-install-recommends \
    libegl1 libgles2 libglvnd0 \
    libosmesa6 libosmesa6-dev patchelf \
    libgl1 libxext6 libxrandr2 libxcursor1 libxinerama1 libxi6 \
    build-essential cmake \
    git git-lfs ffmpeg \
    > /tmp/apt.log 2>&1 || warn "apt install had warnings — see /tmp/apt.log"

git lfs install --skip-smudge >/dev/null 2>&1 || true
${APT} clean || true
rm -rf /var/lib/apt/lists/* /tmp/apt.log || true

log "System libs installed."

# ----- Step 4: Init the LIBERO submodule -------------------------------------
section "Step 4 — Initialize LIBERO submodule"

# .gitmodules pins git@github.com (SSH) URLs, which fail on a pod with no SSH key.
# Rewrite SSH -> HTTPS just for this operation (without touching .gitmodules).
if [[ -f "${LIBERO_DIR}/setup.py" ]]; then
    log "LIBERO submodule already populated."
else
    log "Fetching LIBERO submodule over HTTPS..."
    git -c url."https://github.com/".insteadOf="git@github.com:" \
        submodule update --init --recursive "${LIBERO_DIR}"
fi
[[ -f "${LIBERO_DIR}/requirements.txt" ]] || fail "LIBERO submodule missing requirements.txt — init failed."

# ----- Step 5: Build the base openpi JAX env ---------------------------------
section "Step 5 — Build base openpi env (uv sync)"

# GIT_LFS_SKIP_SMUDGE=1: don't pull large LFS blobs during the sync (checkpoints
# are downloaded on demand by serve_policy.py at runtime).
GIT_LFS_SKIP_SMUDGE=1 uv sync

log "Base env synced to ${REPO_DIR}/.venv"

# ----- Step 6: Base env smoke test (JAX GPU + openpi import) ------------------
section "Step 6 — Base env smoke test (JAX GPU + openpi import)"

uv run python <<'PY'
import jax, jax.numpy as jnp
print("jax:        ", jax.__version__)
print("backend:    ", jax.default_backend())
print("devices:    ", jax.devices())

# Real GPU op — forces CUDA init, surfaces driver/runtime mismatches early.
x = jnp.ones((256, 256))
y = float((x @ x).sum())
print(f"matmul OK:   {y}")

if jax.default_backend() != "gpu":
    raise SystemExit("JAX is not using the GPU — check the NVIDIA driver / pod GPU assignment.")

# Validate the openpi policy model imports (incl. the SAFE multi-layer hidden-state edits).
import openpi.models.pi0 as pi0
print("openpi.pi0:  imported OK | SAFE_HIDDEN_LAYERS =", pi0.SAFE_HIDDEN_LAYERS)
PY

log "Base env smoke test passed."

# ----- Step 7: Build the LIBERO Python 3.8 env -------------------------------
section "Step 7 — Build LIBERO env (Python 3.8) + editable installs"

if [[ -x "${LIBERO_VENV}/bin/python" ]]; then
    log "LIBERO venv already exists at ${LIBERO_VENV}"
else
    uv venv --python 3.8 "${LIBERO_VENV}"
fi

# CRITICAL: LIBERO's repo is missing this top-level __init__.py. Without it,
# find_packages() in its setup.py discovers no `libero` package and the editable
# install registers nothing — `import libero` then fails. find_packages() runs at
# INSTALL time, so this MUST be created BEFORE `uv pip install -e` below.
if [[ ! -f "${LIBERO_DIR}/libero/__init__.py" ]]; then
    echo "# Placeholder so find_packages() discovers libero.* (LIBERO repo is missing this)." \
        > "${LIBERO_DIR}/libero/__init__.py"
    log "Created ${LIBERO_DIR}/libero/__init__.py"
fi

# uv pip sync into the LIBERO venv (note: VIRTUAL_ENV targets that venv explicitly,
# so we don't need to `activate`). cu113 index supplies the old torch wheels.
VIRTUAL_ENV="${LIBERO_VENV}" uv pip sync \
    "${REPO_DIR}/examples/libero/requirements.txt" \
    "${LIBERO_DIR}/requirements.txt" \
    --extra-index-url "${LIBERO_TORCH_INDEX_URL}" \
    --index-strategy=unsafe-best-match

VIRTUAL_ENV="${LIBERO_VENV}" uv pip install -e "${REPO_DIR}/packages/openpi-client"
VIRTUAL_ENV="${LIBERO_VENV}" uv pip install -e "${LIBERO_DIR}"

log "LIBERO env ready at ${LIBERO_VENV}"

# ----- Step 8: Configure MuJoCo headless rendering + LIBERO config -----------
section "Step 8 — Configure MuJoCo EGL rendering + pre-write LIBERO config"

PROFILE_FILE="/etc/profile.d/openpi-libero.sh"
PROFILE_MARKER="# === setup.sh env vars ==="

if [[ ! -f "${PROFILE_FILE}" ]] || ! grep -q "${PROFILE_MARKER}" "${PROFILE_FILE}" 2>/dev/null; then
    if [[ "$(id -u)" -eq 0 ]]; then WRITE_CMD="tee"; else WRITE_CMD="sudo tee"; fi
    ${WRITE_CMD} "${PROFILE_FILE}" > /dev/null <<EOF
${PROFILE_MARKER}
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export ROBOSUITE_DEFAULT_ASSET_PATH=${ROBOSUITE_ASSETS}
export LIBERO_CONFIG_PATH=${LIBERO_CONFIG_PATH}
EOF
    log "Wrote ${PROFILE_FILE}"
else
    log "${PROFILE_FILE} already configured."
fi

# Apply to the current shell so the smoke test below works.
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export ROBOSUITE_DEFAULT_ASSET_PATH="${ROBOSUITE_ASSETS}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH}"

# LIBERO prompts via input() on first import if its config.yaml doesn't exist,
# which breaks non-interactive runs. Pre-write it with sensible defaults.
"${LIBERO_VENV}/bin/python" <<PY
import os, yaml

libero_pkg = "${LIBERO_DIR}/libero/libero"
config = {
    "benchmark_root": libero_pkg,
    "bddl_files":     os.path.join(libero_pkg, "bddl_files"),
    "init_states":    os.path.join(libero_pkg, "init_files"),
    "datasets":       os.path.join(libero_pkg, "../datasets"),
    "assets":         os.path.join(libero_pkg, "assets"),
}
os.makedirs("${LIBERO_CONFIG_PATH}", exist_ok=True)
os.makedirs(config["datasets"], exist_ok=True)
config_file = os.path.join("${LIBERO_CONFIG_PATH}", "config.yaml")
with open(config_file, "w") as f:
    yaml.dump(config, f)
print(f"Wrote {config_file}")
PY

log "MuJoCo EGL + LIBERO config set; first import won't prompt."

# ----- Step 9: LIBERO env + render smoke test --------------------------------
section "Step 9 — LIBERO smoke test (import + reset + render + step)"

"${LIBERO_VENV}/bin/python" <<'PY'
import os
print(f"MUJOCO_GL={os.environ.get('MUJOCO_GL', '<unset>')}")

# Import mujoco first for a clear error if EGL libs are missing.
import mujoco
print(f"mujoco:    {mujoco.__version__}")
import robosuite
print(f"robosuite: {robosuite.__version__}")
import openpi_client
print("openpi_client: imported OK")

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np

task_suite = benchmark.get_benchmark_dict()["libero_10"]()
task = task_suite.get_task(0)
bddl_path = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
print(f"Task 0: {task.language}")

env = OffScreenRenderEnv(bddl_file_name=bddl_path, camera_heights=224, camera_widths=224)
env.seed(0)
obs = env.reset()

agentview = obs.get("agentview_image")
if agentview is None:
    raise RuntimeError("agentview_image missing from obs — render failed silently")
mean_pixel = float(agentview.mean())
print(f"agentview_image: shape={agentview.shape}, dtype={agentview.dtype}, mean={mean_pixel:.1f}")
if mean_pixel < 5.0:
    raise RuntimeError(
        f"Mean pixel value {mean_pixel:.1f} — render is black, EGL not actually rendering. "
        f"Edit /etc/profile.d/openpi-libero.sh and switch MUJOCO_GL=egl -> osmesa."
    )

obs, reward, done, info = env.step(np.zeros(env.env.action_dim))
print(f"step OK: reward={reward}, done={done}")
env.close()
print("LIBERO env smoke test passed.")
PY

# ----- Done ------------------------------------------------------------------
section "All done"

log "Environment ready:"
log "  base (JAX) env:   ${REPO_DIR}/.venv          (use via 'uv run ...')"
log "  LIBERO env:       ${LIBERO_VENV}"
log "  render env vars:  ${PROFILE_FILE}            (auto-loads in new shells)"
log ""
log "Run a recorded LIBERO rollout (two terminals):"
log "  # Terminal 1 — pi0-FAST policy server (base env):"
log "  export CUDA_VISIBLE_DEVICES=0"
log "  uv run scripts/serve_policy.py --env LIBERO --record --save_name pi0fast-libero_10"
log ""
log "  # Terminal 1 — OR pi0 policy server (downloads checkpoint on first run):"
log "  uv run scripts/serve_policy.py --env LIBERO --record --save_name pi0-libero_10 \\"
log "      policy:checkpoint --policy.config pi0_libero --policy.dir s3://openpi-assets/checkpoints/pi0_libero"
log ""
log "  # Terminal 2 — LIBERO rollout client:"
log "  source ${LIBERO_VENV}/bin/activate"
log "  export PYTHONPATH=\$PYTHONPATH:${LIBERO_DIR}"
log "  python examples/libero/main.py --args.task_suite_name libero_10 --args.save_name pi0fast-libero_10"
log ""
log "Recorded policy outputs (incl. the new per-layer 'hidden_states') land in the"
log "policy_records/*.pkl files written by the --record server."
log ""
log "If a future pod renders black frames (mean pixel 0), switch EGL -> OSMesa in ${PROFILE_FILE}."
