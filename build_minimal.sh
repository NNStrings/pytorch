#!/usr/bin/bash
set -euo pipefail

# =========================
# 基本配置
# =========================
BUILD_DIR=build
TOOLCHAIN_FILE=~/toolchains/gcc-clang.cmake

# 并行度（建议 1~2）
JOBS=4

# 每批 clang-tidy 文件数
BATCH_SIZE=20

# =========================
# 环境变量（核心优化）
# =========================

export MAX_JOBS=$JOBS
export CMAKE_BUILD_PARALLEL_LEVEL=$JOBS
export USE_NINJA=1

# 禁用重型组件（极关键）
export USE_CUDA=0
export USE_CUDNN=0
export USE_MKLDNN=0
export USE_NCCL=0
export USE_DISTRIBUTED=0
export BUILD_TEST=0
export BUILD_CAFFE2=0
export USE_XNNPACK=0
export USE_QNNPACK=0
export USE_PYTORCH_QNNPACK=0
export USE_FBGEMM=0

# 降低编译开销
export CFLAGS="-O0 -g0"
export CXXFLAGS="-O0 -g0"

# 强制 CMake 行为
export CMAKE_GENERATOR=Ninja
export CMAKE_TOOLCHAIN_FILE=$TOOLCHAIN_FILE
export CMAKE_EXPORT_COMPILE_COMMANDS=ON

# 限制 clang analyzer 内存行为
export CCC_ANALYZER_MAX_LOOP=4
export CCC_ANALYZER_STORE_MODEL=region

# clang 错误屏蔽
export CXXFLAGS="$CXXFLAGS -Wno-unused-command-line-argument"
export CFLAGS="$CFLAGS -Wno-unused-command-line-argument"

# =========================
# Step 1: 清理
# =========================
echo "[+] Cleaning build directory"
rm -rf $BUILD_DIR

# =========================
# Step 2: 生成 compile_commands.json
# =========================
echo "[+] Running minimal build_ext (no install)"

python3 setup.py build_ext

# =========================
# Step 3: 链接 compile_commands.json
# =========================
echo "[+] Linking compile_commands.json"

if [ -f "$BUILD_DIR/compile_commands.json" ]; then
    ln -sf $BUILD_DIR/compile_commands.json .
else
    echo "[!] compile_commands.json not found!"
    exit 1
fi

# =========================
# Step 4: 提取源文件列表
# =========================
# echo "[+] Extracting file list"

# jq '.[].file' compile_commands.json | sed 's/"//g' > files.txt

# 去重（避免重复 TU）
# sort -u files.txt -o files.txt

# =========================
# Step 5: 分批
# =========================
# echo "[+] Splitting into batches (size=$BATCH_SIZE)"

# rm -f batch_*
# split -l $BATCH_SIZE files.txt batch_

# =========================
# Step 6: clang-tidy 分析
# =========================
# echo "[+] Running clang-tidy (batched)"

# for f in batch_*; do
#     echo "[+] Processing $f"

#     xargs -a $f -I{} clang-tidy {} \
#         -p $BUILD_DIR \
#         -j $JOBS \
#         -checks='-*,clang-analyzer-*' \
#         --quiet \
#         || true
# done

echo "[+] Done"