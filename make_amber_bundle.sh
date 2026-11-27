#!/bin/bash
set -euo pipefail

usage="Usage: $0 [-h] [--apply] [output]"
do_apply=false
positional_args=()
for arg in "$@"; do
    case "${arg}" in
        -h|--help) echo "${usage}"; exit 0 ;;
        --apply) do_apply=true ;;
        -*) echo "${usage}" >&2; exit 1 ;;
        *) positional_args+=("${arg}") ;;
    esac
done
if (( ${#positional_args[@]} > 1 )); then
    echo "${usage}" >&2
    exit 1
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bundle_name="mlip-interface"
output="$(realpath "${positional_args[0]:-${PWD}/${bundle_name}.tar.gz}")"
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
        -o -path ./compile_commands.json \
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

if [[ "${do_apply}" == true ]]; then
    amber_src="${HOME}/Repos/amber/AmberTools/src"
    if [[ ! -d "${amber_src}/${bundle_name}" ]]; then
        echo "ERROR: ${amber_src}/${bundle_name} not found, ${output} was not applied" >&2
        exit 1
    fi
    rm -rf "${amber_src:?}/${bundle_name}"
    tar -xzmf "${output}" -C "${amber_src}"
    rm "${output}"
    echo "Replaced ${amber_src}/${bundle_name} and removed ${output}"
fi
