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
torchani_csrc="submodules/torchani/torchani/csrc"

skipped_names=(
    ':!*.gitignore'
    ':!*.clang-format'
    ':!*.pre-commit-config.yaml'
    ':!*dump_sander_*'
    ':!*dump_pmemd_*'
    ':!*.pt'
    ':!*.png'
    ':!*.md'
)

cd "${repo_root}"
if [[ -n "$(git status --porcelain)" ]]; then
    echo "ERROR: ${repo_root} has uncommitted or untracked changes" >&2
    exit 1
fi
{
    git ls-files -- . \
        "${skipped_names[@]}" \
        ':!.github' \
        ':!.gitmodules' \
        ':!AmberTools-patch' \
        ':!make_vendored_tarball.sh' \
        ':!run-cmake' \
        ':!environment.yaml' \
        ':!ci-environment.yaml' \
        ':!compile_commands.json' \
        ':!inputs' \
        ':!jit' \
        ':!tests' \
        ':!submodules'
    git -C "${torchani_csrc}" ls-files -- \
        . "${skipped_names[@]}" \
        | sed "s,^,${torchani_csrc}/,"
} | tar -czf "${output}" \
    --no-recursion \
    --transform "s,^,${bundle_name}/," \
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
