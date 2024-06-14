#!/bin/bash

# Usage: ./cmake.sh performs "configure + generate" && build && install,
# --no-conda flag avoids conda CUDA and conda cuDNN
# --no-install flag skips automatic installation

# NOTE: if not using the g++ and gcc compilers from the conda env,
# comment out the "-DCMAKE_*_COMPILER=..." lines

# The directory of this script, taken from:
# https://stackoverflow.com/questions/59895/how-do-i-get-the-directory-where-a-bash-script-is-located-from-within-the-script
_src_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
_build_dir="$_src_dir/build"

if [ "${1}" = "--no-conda" ] || [ "${2}" = "--no-conda" ]; then
    _use_conda=OFF
else
    _use_conda=ON
fi

cmake \
    -S"$_src_dir" \
    -B"$_build_dir" \
    -DUSE_ACTIVE_CONDA_PYTORCH=$_use_conda \
    -DUSE_ACTIVE_CONDA_CUDA_TOOLKIT=$_use_conda \
    -DUSE_ACTIVE_CONDA_CUDNN=$_use_conda \
    -DCMAKE_C_COMPILER="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc" \
    -DCMAKE_CXX_COMPILER="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++" \
    -DCMAKE_CUDA_HOST_COMPILER="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++" \
&& cmake \
    --build "$_build_dir"

if [ "${2}" = "--no-install" ] || [ "${1}" = "--no-install" ]; then
    :
else
    cmake --install "$_build_dir"
fi
