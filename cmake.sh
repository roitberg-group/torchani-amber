#!/bin/bash

# Usage: ./cmake.sh performs configure+generate && build && install,
# --conda flag uses conda CUDA and conda cuDNN

# The directory of this script, taken from:
# https://stackoverflow.com/questions/59895/how-do-i-get-the-directory-where-a-bash-script-is-located-from-within-the-script
_src_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
_build_dir="$_src_dir/build"

if [ "${1}" = "--conda" ]; then
    _use_conda=ON
else
    _use_conda=OFF
fi

cmake \
    --fresh \
    -S"$_src_dir" \
    -B"$_build_dir" \
    -DCONDA_CUDA=$_use_conda \
    -DCONDA_CUDNN=$_use_conda \
&& cmake \
    --build "$_build_dir" \
&& cmake \
    --install "$_build_dir"
