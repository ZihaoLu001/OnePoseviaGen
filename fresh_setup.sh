#!/usr/bin/env bash
set -euo pipefail

echo "🚀 Starting clean setup for OnePoseviaGen"

# ---------- 0) 新建并进入环境 ----------
source ~/miniconda3/etc/profile.d/conda.sh
conda create -y -n opvg python=3.11 pip
conda activate opvg

# 避免 libmamba 求解器抽风
conda config --env --set solver classic
conda config --env --set channel_priority strict
conda config --env --remove channels defaults 2>/dev/null || true
conda config --env --add channels conda-forge

python -m pip install -U pip setuptools wheel

# ---------- 1) CUDA/torch ----------
echo "📦 Installing CUDA toolkit 12.1 (for building nvcc-based extensions)"
conda install -y -c "nvidia/label/cuda-12.1.0" cuda-toolkit
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH-}"

echo "📦 Installing PyTorch cu121 wheels"
python -m pip install \
  torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
  --index-url https://download.pytorch.org/whl/cu121

python - <<'PY'
import torch
print("Torch:", torch.__version__, "CUDA avail:", torch.cuda.is_available(), "CUDA ver:", torch.version.cuda)
assert torch.cuda.is_available(), "CUDA not available — check drivers/toolkit"
assert torch.version.cuda and torch.version.cuda.startswith("12.1")
PY

# ---------- 2) 基础构建依赖 ----------
echo "📦 Installing build deps (cmake/ninja/pybind11/eigen)"
conda install -y -c conda-forge cmake ninja pybind11 eigen=3.4.0

# pybind11 的 cmake 包路径（供 F-Pose 构建）
P11_DIR="$CONDA_PREFIX/lib/python3.11/site-packages/pybind11/share/cmake/pybind11"
[ -d "$P11_DIR" ] || P11_DIR="$CONDA_PREFIX/share/cmake/pybind11"

# ---------- 3) Python 依赖 ----------
echo "📦 pip requirements..."
pip install -r requirements.txt

# ---------- 4) 外部 CUDA 扩展 ----------
mkdir -p tmp/extensions

echo "🛠️ diffoctreerast..."
rm -rf tmp/extensions/diffoctreerast
git clone --recurse-submodules https://github.com/JeffreyXiang/diffoctreerast.git tmp/extensions/diffoctreerast
# 4090/Ada -> 8.9；根据你的显卡可改
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST-8.9}" CUDA_HOME="$CONDA_PREFIX" \
  pip install --no-build-isolation tmp/extensions/diffoctreerast

echo "🛠️ mip-splatting / diff-gaussian-rasterization..."
rm -rf tmp/extensions/mip-splatting
git clone https://github.com/autonomousvision/mip-splatting.git tmp/extensions/mip-splatting
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST-8.9}" CUDA_HOME="$CONDA_PREFIX" \
  pip install --no-build-isolation tmp/extensions/mip-splatting/submodules/diff-gaussian-rasterization/

echo "🛠️ PyTorch3D..."
# 从源码装以匹配当前 torch/cuda
pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git"

# ---------- 5) Build F-Pose ----------
echo "🛠️ Building F-Pose..."
pushd oneposeviagen/fpose/fpose >/dev/null
CMAKE_PREFIX_PATH="${P11_DIR};${CONDA_PREFIX}" bash build_all_conda.sh
popd >/dev/null

# ---------- 6) 安装仓内各子包（editable） ----------
echo "🛠️ Installing local packages (editable)"
pushd oneposeviagen >/dev/null
for pkg in fpose SAM2-in-video trellis Amodal3R SpaTrackerV2; do
  echo "📦 pip install -e $pkg"
  (cd "$pkg" && pip install -e .)
done
popd >/dev/null

# ---------- 7) 预训练权重 ----------
echo "📦 Downloading pretrained weights..."
python oneposeviagen/scripts/download_weights.py

echo "🎉 Setup completed successfully!"
