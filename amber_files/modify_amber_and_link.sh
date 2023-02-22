#!/bin/bash

# Modify Amber files in place (WARNING, this may screw up your amber installation)
# configure Amber with torchani and add the library to the Amber library path
# and make pmemd and sander with some default flags

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

echo "Modifying, Linking and configuring Amber"
if [ -z "${configure_amber}" ]; then
    if [ -z "${1}" ]; then
        echo "Configure amber not set, please pass 'true' or 'false' as the first command-line argument"
        exit 1
    fi
    configure_amber=${1}
fi

# shellcheck disable=SC1090
source "${DIR}/../install/python.sh"
# shellcheck disable=SC2154
if [ -z "${python_command}" ]; then
    set_python_command
fi


if [ -z "${AMBERHOME}" ]; then
    echo "\nError: AMBERHOME is not set\n"
    exit
fi
echo "\nUsing the following AMBERHOME variable: ${AMBERHOME}\n"

# Copy pmemd files
pmemd_torchani_dir="${DIR}/pmemd"
sander_torchani_dir="${DIR}/sander"
config_torchani_dir="${DIR}"

pmemd_amber_dir="${AMBERHOME}/src/pmemd/src"
sander_amber_dir="${AMBERHOME}/AmberTools/src/sander"
config_amber_dir="${AMBERHOME}/AmberTools/src"


function copy_file() {
    if [ -f "${2}/${3}" ]; then
        cp "${1}/${3}" "${2}/${3}"
    else
        echo "${2}/${3} not found"
        exit 1
    fi
}

# Copy pmemd files
if [ -d "${pmemd_amber_dir}" ]; then
    copy_file "${pmemd_torchani_dir}" "${pmemd_amber_dir}" "external.F90"
    copy_file "${pmemd_torchani_dir}" "${pmemd_amber_dir}" "external_dat.F90"
    copy_file "${pmemd_torchani_dir}" "${pmemd_amber_dir}" "mdin_ctrl_dat.F90"
else
    echo "${pmemd_amber_dir} not found"
    exit 1
fi

# Copy sander files
if [ -d "${sander_amber_dir}" ]; then
    copy_file "${sander_torchani_dir}" "${sander_amber_dir}" "external.F90"
    copy_file "${sander_torchani_dir}" "${sander_amber_dir}" "mdread2.F90"
else
    echo "${sander_amber_dir} not found"
    exit 1
fi

# Copy other files
if [ -d "${config_amber_dir}" ]; then
    copy_file "${config_torchani_dir}" "${config_amber_dir}" "configure2"
else
    echo "${config_amber_dir} not found"
    exit 1
fi

# Symlink libtorchani.so to amber library path
torchani_lib_file="${DIR}/../build/libtorchani.so"
if [ -f "${torchani_lib_file}" ]; then
    cd "${AMBERHOME}/lib64/" || echo "Could not find path" && exit 1
    ln -s "${torchani_lib_file}" .
    cd "${AMBERHOME}/lib/" || echo "Could not find path" && exit 1
    ln -s "${torchani_lib_file}" .
else
    echo "${torchani_lib_file}" "not found"
    exit 1
fi

# Configure amber with default configuration, not installing updates
if ${configure_amber}; then
    cd "${AMBERHOME}" || exit 1
    yes n | ./configure -noX11 -torchani --skip-python gnu
    cd - || exit 1
else
    echo "Skipping Amber configuration"
fi
echo "Done modifying, linking and configuring Amber"
