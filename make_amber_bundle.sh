#!/bin/bash
set -euo pipefail

usage="Usage: $0 [output]"
while getopts "h" opt; do
    case "${opt}" in
        h) echo "${usage}"; exit 0 ;;
        *) echo "${usage}" >&2; exit 1 ;;
    esac
done
shift $((OPTIND - 1))

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bundle_name="mlip-interface"
output="$(realpath "${1:-${PWD}/${bundle_name}.tar.gz}")"
torchani_csrc="./submodules/torchani/torchani/csrc"

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
        -o -path ./build \
        -o -path ./inputs \
        -o -path ./jit \
        -o -path ./tests \
        -o -path ./submodules/Catch2 \
        -o -path ./submodules/torchani \
        -o -path "./${bundle_name}.tar.gz" \) -prune \
        -o -print
    echo ./submodules/torchani
    echo ./submodules/torchani/torchani
    find "${torchani_csrc}" \( "${skipped_names[@]}" \) -prune -o -print
} | tar -czf "${output}" \
    --no-recursion \
    --transform "s,^\.,${bundle_name}," \
    -T -

echo "WARNING: tests/, inputs/ and jit/ are not packed" >&2
echo "Created ${output}"
