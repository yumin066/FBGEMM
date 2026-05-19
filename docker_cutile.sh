#!/usr/bin/env bash
set -euo pipefail

IMAGE="${CUTILE_DOCKER_IMAGE:-nvcr.io/nvidia/pytorch:26.04-py3}"
NAME="${CUTILE_DOCKER_NAME:-cutile-hstu-$(id -un)-$(date +%Y%m%d-%H%M%S)}"
REPO_ROOT="${HSTU_REPO_ROOT:-/home/scratch.minyu_gpu/project/shopee/fbgemm-hstu}"
CUTILE_ROOT="${CUTILE_ROOT:-/home/scratch.minyu_gpu/project/shopee/cutile-python}"
PIP_USER_BASE="${PYTHONUSERBASE:-/home/scratch.minyu_gpu/project/.cache/pip-user}"
CUTILE_TILEIRAS_RC_VERSION="${CUTILE_TILEIRAS_RC_VERSION:-}"
if [[ -n "${CUTILE_TILEIRAS_RC_VERSION}" ]]; then
  CUTILE_DEFAULT_BUILD_ROOT="/home/scratch.minyu_gpu/project/.cache/cutile-cext-dev-tileiras-${CUTILE_TILEIRAS_RC_VERSION}"
else
  CUTILE_DEFAULT_BUILD_ROOT="/home/scratch.minyu_gpu/project/.cache/cutile-cext-dev"
fi
CUTILE_BUILD_ROOT="${CUTILE_BUILD_ROOT:-${CUTILE_DEFAULT_BUILD_ROOT}}"
CUTILE_BUILD_LIB="${CUTILE_BUILD_LIB:-${CUTILE_BUILD_ROOT}/build_lib}"
CUTILE_BUILD_DIR="${CUDA_TILE_CEXT_BUILD_DIR:-${CUTILE_BUILD_ROOT}/build}"
CUTILE_BUILD_JOBS="${CUTILE_BUILD_JOBS:-32}"
CUTILE_DOCKER_GPUS="${CUTILE_DOCKER_GPUS:-2}"
CUTILE_USE_PIP_NIGHTLY="${CUTILE_USE_PIP_NIGHTLY:-0}"
CUTILE_USE_NVTRITON_TILEIRAS="${CUTILE_USE_NVTRITON_TILEIRAS:-0}"
if [[ "${CUTILE_USE_PIP_NIGHTLY}" == "1" ]]; then
  CUTILE_PYTHONPATH="${REPO_ROOT}/fbgemm_gpu/experimental/hstu/src:${REPO_ROOT}/fbgemm_gpu/experimental/hstu"
else
  CUTILE_PYTHONPATH="${CUTILE_BUILD_LIB}:${REPO_ROOT}/fbgemm_gpu/experimental/hstu/src:${REPO_ROOT}/fbgemm_gpu/experimental/hstu:${CUTILE_ROOT}/src"
fi
CUTILE_PYTHON_VERSION="${CUTILE_PYTHON_VERSION:-3.12}"
CUTILE_PIP_CUDA_ROOT="${CUTILE_PIP_CUDA_ROOT:-${PIP_USER_BASE}/lib/python${CUTILE_PYTHON_VERSION}/site-packages/nvidia/cu13}"
CUTILE_NVTRITON_TILEIRAS_ROOT="${CUTILE_NVTRITON_TILEIRAS_ROOT:-${PIP_USER_BASE}/lib/python${CUTILE_PYTHON_VERSION}/site-packages/triton/backends/tileir/bin}"
if [[ -n "${CUTILE_TILEIRAS_RC_VERSION}" ]]; then
  CUTILE_CONTAINER_PATH="${CUTILE_PIP_CUDA_ROOT}/bin:${CUTILE_PIP_CUDA_ROOT}/nvvm/bin:${PIP_USER_BASE}/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
  CUTILE_CONTAINER_LD_LIBRARY_PATH="${CUTILE_PIP_CUDA_ROOT}/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
else
  CUTILE_CONTAINER_PATH="${PIP_USER_BASE}/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
  CUTILE_CONTAINER_LD_LIBRARY_PATH="/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
fi
if [[ "${CUTILE_USE_NVTRITON_TILEIRAS}" == "1" ]]; then
  CUTILE_CONTAINER_PATH="${CUTILE_NVTRITON_TILEIRAS_ROOT}:${CUTILE_CONTAINER_PATH}"
fi
CUTILE_NIGHTLY_INDEX_URL="${CUTILE_NIGHTLY_INDEX_URL:-https://urm.nvidia.com/artifactory/api/pypi/nv-shared-pypi-local/simple}"
CUTILE_PIP_SPEC="${CUTILE_PIP_SPEC:-cuda-tile}"
CUTILE_NVTRITON_PIP_SPEC="${CUTILE_NVTRITON_PIP_SPEC:-nv-triton==9.9.99.dev20260517+git6eb9b7f2.cudatile}"
CUTILE_TILEIRAS_RC_INDEX_URL="${CUTILE_TILEIRAS_RC_INDEX_URL:-https://urm.nvidia.com/artifactory/api/pypi/sw-gpu-cuda-installer-pypi-local/simple}"

if [[ "${CUTILE_DOCKER_PULL:-1}" == "1" ]]; then
  docker pull "${IMAGE}"
fi

if [[ "${1:-}" == "install-tileiras-rc" ]]; then
  CMD=(bash -lc '
set -euo pipefail
if [[ -z "${CUTILE_TILEIRAS_RC_VERSION}" ]]; then
  echo "CUTILE_TILEIRAS_RC_VERSION must be set, for example 13.3.7" >&2
  exit 2
fi
python -m pip install --user --pre --upgrade \
  --index-url "${CUTILE_NIGHTLY_INDEX_URL}" \
  "${CUTILE_PIP_SPEC}"
python -m pip install --user \
  --index-url "${CUTILE_TILEIRAS_RC_INDEX_URL}" \
  "nvidia-cuda-tileiras==${CUTILE_TILEIRAS_RC_VERSION}" \
  "nvidia-cuda-nvcc==${CUTILE_TILEIRAS_RC_VERSION}" \
  "nvidia-nvvm==${CUTILE_TILEIRAS_RC_VERSION}" \
  "nvidia-nvjitlink==${CUTILE_TILEIRAS_RC_VERSION}" \
  "nvidia-cuda-crt==${CUTILE_TILEIRAS_RC_VERSION}" \
  "nvidia-cuda-runtime==${CUTILE_TILEIRAS_RC_VERSION}"
python - <<'"'"'PY'"'"'
import shutil

print("tileiras path", shutil.which("tileiras"))
print("nvcc path", shutil.which("nvcc"))
print("nvvm path", shutil.which("nvvm"))
PY
tileiras --version
nvcc --version | head -n 1
')
elif [[ "${1:-}" == "install-nvtriton-tileiras" ]]; then
  CMD=(bash -lc '
set -euo pipefail
python -m pip install --user --pre --upgrade \
  --index-url "${CUTILE_NIGHTLY_INDEX_URL}" \
  "${CUTILE_PIP_SPEC}" \
  "${CUTILE_NVTRITON_PIP_SPEC}" \
  pytest pytest-benchmark cuda-python
python - <<'"'"'PY'"'"'
import glob
import os
import sys

for root in sys.path:
    if "site-packages" not in root:
        continue
    for path in glob.glob(os.path.join(root, "triton", "backends", "tileir", "bin", "tileiras")):
        print("nv-triton tileiras", path, "exec", os.access(path, os.X_OK))
PY
')
elif [[ "${1:-}" == "test-mma-scaled" ]]; then
  CMD=(bash -lc '
set -euo pipefail
echo "tileiras=$(which tileiras)"
tileiras --version
python - <<'"'"'PY'"'"'
import cuda.tile as ct
from cuda.tile._cext import dev_features_enabled
from cuda.tile._compile import _get_max_supported_bytecode_version
import tempfile

print("cuda.tile", ct.__version__, ct.__file__)
print("dev_features", dev_features_enabled())
print("tileiras_bytecode", _get_max_supported_bytecode_version(tempfile.gettempdir(), allow_dev=True))
PY
cd "${CUTILE_ROOT}"
python -m pytest -q test/test_mma_scaled.py::test_mma_scaled_fp8 --tb=short -s
')
elif [[ "${1:-}" == "bootstrap-cutile" ]]; then
  CMD=(bash -lc '
set -euo pipefail
mkdir -p "${CUTILE_BUILD_LIB}/cuda"
set +e
python "${CUTILE_ROOT}/setup.py" build_ext --enable-dev-features --parallel "${CUTILE_BUILD_JOBS}" --build-lib "${CUTILE_BUILD_LIB}"
build_status=$?
set -e
if [[ ! -f "${CUDA_TILE_CEXT_BUILD_DIR}/cext/lib_cext.so" ]]; then
  exit "${build_status}"
fi
if [[ "${build_status}" -ne 0 ]]; then
  echo "cuTile build_ext returned ${build_status}; using built cext artifact overlay"
fi
cp -a "${CUTILE_ROOT}/src/cuda/." "${CUTILE_BUILD_LIB}/cuda/"
cp "${CUDA_TILE_CEXT_BUILD_DIR}/cext/lib_cext.so" "${CUTILE_BUILD_LIB}/cuda/tile/_cext.so"
python -c "import cuda.tile as ct; import cuda.tile._cext as cext; print(\"cuda.tile\", ct.__version__, ct.__file__); print(\"_cext\", cext.__file__); print(\"dev_features\", cext.dev_features_enabled())"
')
elif [[ "$#" -eq 0 ]]; then
  CMD=(/bin/bash)
else
  CMD=("$@")
fi

docker run \
  --rm \
  --cap-add SYS_ADMIN --security-opt seccomp=unconfined \
  --ipc=host \
  --gpus "device=${CUTILE_DOCKER_GPUS}" \
  --shm-size=16g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --name "${NAME}" \
  --user "$(id -u):$(id -g)" \
  --workdir "${REPO_ROOT}" \
  -e HOME="/home/$(id -un)" \
  -e PYTHONUSERBASE="${PIP_USER_BASE}" \
  -e PATH="${CUTILE_CONTAINER_PATH}" \
  -e LD_LIBRARY_PATH="${CUTILE_CONTAINER_LD_LIBRARY_PATH}" \
  -e HSTU_REPO_ROOT="${REPO_ROOT}" \
  -e CUTILE_ROOT="${CUTILE_ROOT}" \
  -e CUTILE_PIP_CUDA_ROOT="${CUTILE_PIP_CUDA_ROOT}" \
  -e CUDA_TILE_CEXT_BUILD_DIR="${CUTILE_BUILD_DIR}" \
  -e CUTILE_BUILD_LIB="${CUTILE_BUILD_LIB}" \
  -e CUTILE_BUILD_JOBS="${CUTILE_BUILD_JOBS}" \
  -e PYTHONPATH="${CUTILE_PYTHONPATH}" \
  -e CUTILE_NIGHTLY_INDEX_URL="${CUTILE_NIGHTLY_INDEX_URL}" \
  -e CUTILE_PIP_SPEC="${CUTILE_PIP_SPEC}" \
  -e CUTILE_NVTRITON_PIP_SPEC="${CUTILE_NVTRITON_PIP_SPEC}" \
  -e CUTILE_TILEIRAS_RC_INDEX_URL="${CUTILE_TILEIRAS_RC_INDEX_URL}" \
  -e CUTILE_TILEIRAS_RC_VERSION="${CUTILE_TILEIRAS_RC_VERSION}" \
  -e HSTU_ARCH_LIST="12.0" \
  -e HSTU_DISABLE_BACKWARD="TRUE" \
  -e HSTU_DISABLE_DETERMINISTIC="FALSE" \
  -e HSTU_DISABLE_HDIM32="FALSE" \
  -e HSTU_DISABLE_HDIM64="FALSE" \
  -e HSTU_DISABLE_HDIM256="FALSE" \
  -v /home/scratch.minyu_gpu/:/home/scratch.minyu_gpu/ \
  -v /home/scratch.trt_llm_data/:/home/scratch.trt_llm_data/ \
  -v /home/scratch.svc_compute_arch/:/home/scratch.svc_compute_arch/ \
  -v /home/scratch.junyiq_gpu_1/:/home/scratch.junyiq_gpu_1/ \
  -v /home/minyu/:/home/minyu/ \
  -v /home/tools_ai/:/home/tools_ai/ \
  "${IMAGE}" \
  "${CMD[@]}"
