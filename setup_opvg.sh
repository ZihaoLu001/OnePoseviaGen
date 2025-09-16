#!/usr/bin/env bash
# Clean setup for OnePoseviaGen (PyTorch 2.4.1 + CUDA 12.1)
set -eo pipefail   # ← 去掉 -u，避免 conda 的 deactivate 钩子报 unbound variable

echo "🚀 Starting clean setup for OnePoseviaGen"

# ---------- 0) Conda env ----------
source ~/miniconda3/etc/profile.d/conda.sh
if conda env list | grep -qE '^\s*opvg\s'; then
  echo "🟢 Using existing conda env 'opvg'"
else
  conda create -y -n opvg python=3.11 pip
fi
conda activate opvg

# conda 换为 strict + forge
conda config --env --set solver classic
conda config --env --set channel_priority strict
conda config --env --remove channels defaults 2>/dev/null || true
conda config --env --add channels conda-forge

python -m pip install -U pip setuptools wheel

# ---------- 1) CUDA & Torch ----------
echo "📦 Installing CUDA 12.1 dev toolchain (headers included)"
# 避免与 nvidia 频道的 cuda-toolkit 混装
conda remove -y cuda-toolkit 2>/dev/null || true

conda install -y -c conda-forge \
  "cuda-version=12.1" \
  "cuda-nvcc=12.1" \
  "cuda-cudart=12.1.*" "cuda-cudart-dev=12.1.*" \
  "libnvjitlink=12.1.*" \
  "libcusparse=12.*" "libcusparse-dev=12.*" \
  "libcublas=12.*"  "libcublas-dev=12.*" \
  "libcurand=10.*"  "libcurand-dev=10.*" \
  "libcusolver=11.*" "libcusolver-dev=11.*" \
  "libcufft=11.*"   "libcufft-dev=11.*" \
  "libnpp=12.*"     "libnpp-dev=12.*"

export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
# 两处 include/lib 都加上（conda 的 CUDA 头文件可能在 targets 下）
export CPATH="$CONDA_PREFIX/include:$CONDA_PREFIX/targets/x86_64-linux/include:${CPATH-}"
export LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/targets/x86_64-linux/lib:${LIBRARY_PATH-}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64:$CONDA_PREFIX/targets/x86_64-linux/lib:${LD_LIBRARY_PATH-}"

echo "📦 Installing PyTorch cu121 wheels"
python -m pip install \
  torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
  --index-url https://download.pytorch.org/whl/cu121

python - <<'PY'
import torch, os
inc1=os.path.join(os.environ["CONDA_PREFIX"],"include","cusparse.h")
inc2=os.path.join(os.environ["CONDA_PREFIX"],"targets","x86_64-linux","include","cusparse.h")
print("Torch:", torch.__version__, "CUDA avail:", torch.cuda.is_available(), "CUDA ver:", torch.version.cuda)
assert torch.cuda.is_available(), "CUDA not available — check drivers/toolkit"
assert (torch.version.cuda or "").startswith("12.1"), "Expect torch cu121"
assert (os.path.exists(inc1) or os.path.exists(inc2)), "Missing cusparse.h — install *-dev packages"
PY

# ---------- 2) Build deps ----------
echo "📦 Installing build deps (cmake<3.31 / ninja / pybind11 / eigen)"
conda install -y -c conda-forge "cmake<3.31" ninja pybind11 eigen=3.4.0

P11_DIR="$CONDA_PREFIX/lib/python3.11/site-packages/pybind11/share/cmake/pybind11"
[ -d "$P11_DIR" ] || P11_DIR="$CONDA_PREFIX/share/cmake/pybind11"

# ---------- 3) Python requirements ----------
echo "📦 pip requirements..."
pip install -r requirements.txt

# ---------- 4) 外部 CUDA 扩展 ----------
mkdir -p tmp/extensions

ARCH_AUTO=$(python - <<'PY'
import torch
try:
    m,n = torch.cuda.get_device_capability(0)
    print(f"{m}.{n}")
except Exception:
    print("8.0")
PY
)
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST-${ARCH_AUTO}}"
echo "🧮 TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"

echo "🛠️ diffoctreerast..."
rm -rf tmp/extensions/diffoctreerast
git clone --recurse-submodules https://github.com/JeffreyXiang/diffoctreerast.git tmp/extensions/diffoctreerast
CUDA_HOME="$CONDA_PREFIX" FORCE_CUDA=1 \
  pip install --no-build-isolation tmp/extensions/diffoctreerast

echo "🛠️ mip-splatting / diff-gaussian-rasterization..."
rm -rf tmp/extensions/mip-splatting
git clone https://github.com/autonomousvision/mip-splatting.git tmp/extensions/mip-splatting
CUDA_HOME="$CONDA_PREFIX" FORCE_CUDA=1 \
  pip install --no-build-isolation tmp/extensions/mip-splatting/submodules/diff-gaussian-rasterization/

echo "🛠️ PyTorch3D (build from source)..."
CUDA_HOME="$CONDA_PREFIX" FORCE_CUDA=1 \
  pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git"

# ---------- 5) Build F-Pose（替代原 build_all_conda.sh） ----------
echo "🛠️ Building F-Pose (robust mode)..."

# 只装 C++ Boost（不要装带 Python 绑定的 boost），优先 1.84，找不到就 1.74
if ! conda list boost-cpp | grep -q '^boost-cpp'; then
  conda install -y -c conda-forge "boost-cpp=1.84" || conda install -y -c conda-forge "boost-cpp=1.74"
fi

# 自动定位 F-Pose 的 CMakeLists.txt（最多往下 4 层）
FPOSE_SRC=$(find oneposeviagen -maxdepth 4 -name CMakeLists.txt -printf '%h\n' | grep -E '/fpose($|/)' | head -n1 || true)
if [ -z "${FPOSE_SRC}" ]; then
  echo "❌ 未找到 F-Pose 的 CMakeLists.txt，请检查路径（例如 oneposeviagen/fpose）"; exit 1
fi
echo "📁 F-Pose source: ${FPOSE_SRC}"

# 查找 Boost 的 CMake config 目录（避免 zsh 的 NOMATCH）
BoostCFG=$(find "$CONDA_PREFIX/lib/cmake" -maxdepth 1 -type d -name 'Boost-*' -print -quit 2>/dev/null || true)
[ -n "${BoostCFG}" ] && echo "🔎 Boost_DIR=${BoostCFG}" || echo "⚠️ 未检测到 BoostConfig 目录，将仅用 BOOST_ROOT"

# CMake 配置与构建（Ninja）
cmake -S "${FPOSE_SRC}" -B "${FPOSE_SRC}/build" \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="${P11_DIR};${CONDA_PREFIX}" \
  -DBOOST_ROOT="$CONDA_PREFIX" \
  ${BoostCFG:+-DBoost_DIR="$BoostCFG"} \
  -DBoost_NO_SYSTEM_PATHS=ON

cmake --build "${FPOSE_SRC}/build" -j"$(nproc)"

# ---------- 6) 安装仓内各子包（editable，禁用隔离） ----------
echo "🛠️ Installing local packages (editable, no build isolation)"
pushd oneposeviagen >/dev/null
for pkg in fpose SAM2-in-video trellis Amodal3R SpaTrackerV2; do
  echo "📦 pip install -e $pkg"
  (cd "$pkg" && PIP_NO_BUILD_ISOLATION=1 pip install -e .)
done
popd >/dev/null

# ---------- 7) 预训练权重 ----------
echo "📦 Downloading pretrained weights..."
python oneposeviagen/scripts/download_weights.py

# ---------- 8) 快速自检 ----------
python - <<'PY'
mods = ["pytorch3d", "diff_gaussian_rasterization", "diffoctreerast"]
for m in mods:
    try:
        __import__(m)
        print(f"[OK] import {m}")
    except Exception as e:
        print(f"[FAIL] import {m} -> {e}")
PY

echo "🎉 Setup completed successfully!"
