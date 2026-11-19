#!/bin/bash
set -euo pipefail

usage="Usage: $0 [-v cuda_version] [-p torch_version] [output]"
cuda_version="12.8"
torch_version="2.11"
while getopts "v:p:h" opt; do
    case "${opt}" in
        v) cuda_version="${OPTARG}" ;;
        p) torch_version="${OPTARG}" ;;
        h) echo "${usage}"; exit 0 ;;
        *) echo "${usage}" >&2; exit 1 ;;
    esac
done
shift $((OPTIND - 1))

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bundle_name="mlip-interface"
output="$(realpath "${1:-${PWD}/${bundle_name}.tar.gz}")"
torchani_csrc="./submodules/torchani/torchani/csrc"

patched_dir="$(mktemp -d)"
trap 'rm -rf "${patched_dir}"' EXIT
sed \
    -e "s,^name: ani-amber$,name: amber-mlip," \
    -e "s,nvidia::cuda-toolkit=.*,nvidia::cuda-toolkit=${cuda_version}," \
    -e "s,download.pytorch.org/whl/cu[0-9]*,download.pytorch.org/whl/cu${cuda_version//./}," \
    -e "s,torch==.*,torch==${torch_version}.*," \
    "${repo_root}/environment.yaml" > "${patched_dir}/environment.yaml"
sed \
    -e "s,^set(TORCHANI_REQUIRED_TORCH_VERSION \"[^\"]*\",set(TORCHANI_REQUIRED_TORCH_VERSION \"${torch_version}\"," \
    -e "s,^set(TORCHANI_REQUIRED_CUDATK_VERSION \"[^\"]*\",set(TORCHANI_REQUIRED_CUDATK_VERSION \"${cuda_version}\"," \
    "${repo_root}/CMakeLists.txt" > "${patched_dir}/CMakeLists.txt"

skipped_names=(
    -name .git
    -o -name .github
    -o -name .gitignore
    -o -name .gitmodules
    -o -name .pre-commit-config.yaml
    -o -name .vscode
    -o -name .cache
    -o -name .mypy_cache
    -o -name .pytest_cache
    -o -name __pycache__
    -o -name 'dump_sander_*'
    -o -name 'dump_pmemd_*'
    -o -name '*.pt'
    -o -name '*.png'
    -o -name '*.swp'
    -o -name '*.swo'
)

cd "${repo_root}"
{
    find . \
        \( "${skipped_names[@]}" \
        -o -path ./AmberTools-patch \
        -o -path ./make_amber_bundle.sh \
        -o -path ./run-cmake \
        -o -path ./environment.yaml \
        -o -path ./CMakeLists.txt \
        -o -path ./inputs \
        -o -path ./tests/expect \
        -o -path ./external \
        -o -path ./jit/aimnet2 \
        -o -path ./jit/nutmeg \
        -o -path ./submodules/Catch2 \
        -o -path ./submodules/torchani \
        -o -path "./${bundle_name}.tar.gz" \) -prune \
        -o -print
    echo ./submodules/torchani
    echo ./submodules/torchani/torchani
    find "${torchani_csrc}" \( "${skipped_names[@]}" \) -prune -o -print
    echo "-C ${patched_dir}"
    echo ./environment.yaml
    echo ./CMakeLists.txt
} | tar -czf "${output}" \
    --no-recursion \
    --transform "s,^\.,${bundle_name}," \
    -T -

echo "WARNING: tests/expect/ and inputs/ are not packed" >&2
echo "Created ${output} (CUDA Toolkit ${cuda_version}, PyTorch ${torch_version})"
