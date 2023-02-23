#!/bin/bash

# The directory where this file is located
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# amber
test_amber=false
make_amber=false
configure_amber=false
modify_amber_and_link=false
install_python_packages=ON

# Command line arguments are read manually, without using getopts. The
# shift command is used to shift the labelling of the command line arguments
# so as to always have the last argument be named $1
# This code is based on
# https://stackoverflow.com/questions/192249/how-do-i-parse-command-line-arguments-in-bash
usage="
This script installs the shared library libtorchani, along with LibTorch and
link it to an existing amber installation. Before running this script be sure
that you have an existing directory with your Amber installation, the path to
which is AMBERHOME. This script compiles libtorchani, JIT compiles models in
the torchani python library and also performs the necessary modifications to
the Amber source for libtorchani to be successfully liked with Amber. By
default it also performs tests to make sure that the interface works properly
typical usage is ./install.sh --amber --amberhome <AMBERHOME>, but other
commands for fine grained control are available.

WARNING: Note that this interfase is a work in progress and this script may
fail silently with some option combinations, currently only --no-libtorch,
--no-jit, --no-tests, --amber and --amberhome are tested and seem to work
correctly.
--no-python-packages:
    Skips installation of torchani, torch and numpy.
--no-libtorch:
    Don't download and extract the LibTorch tensor library (LibTorch is only
    extracted if a ./lib/libtorch directory does not exist).
--no-jit:
    Skip JIT compiling of torchani models.
--no-libtorchani:
    Don't build the library (performs other steps).
--no-tests:
    Does not perform unit tests for libtorchani.
--make-amber:
    Builds sander and pmemd from the existing Amber installation.
--config-amber:
    Configures amber with default configuration.
--test-amber:
    Performs some short dynamics with Amber to test the interface.
--modify-amber:
    Modifies the Amber installation's files and links libtorchani with it.
--amber:
    Same as setting --make-amber --config-amber --modify-amber
--only-amber:
    Only modifies, builds and tests Amber
--amberhome /path/to/amber/home/:
    Sets the variable AMBERHOME=/path/to/amber/home/ (the directory where Amber
    was installed)"

# An explicit while loop is used since long form options are clearer than short
# form and getopts can only parse short form
positional=()
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -h|--help)
        echo "${usage}"
        exit 0
        ;;
        --cuda-includes)
        set_cuda_includes=ON
        shift # past argument
        ;;
        --no-custom-cudnn)
        custom_cudnn=OFF
        shift # past argument
        ;;
        --no-python-packages)
        install_python_packages=OFF
        shift # past argument
        ;;
        --no-cxx11abi)
        cxx11abi=OFF
        shift # past argument
        ;;
        --no-jit)
        jit_compile_models=OFF
        shift # past argument
        ;;
        --no-libtorchani)
        build_libtorchani=false
        shift # past argument
        ;;
        --config-amber)
        configure_amber=false
        shift # past argument
        ;;
        --no-tests)
        run_tests=false
        shift # past argument
        ;;
        --make-amber)
        make_amber=true
        shift # past argument
        ;;
        --test-amber)
        test_amber=true
        shift # past argument
        ;;
        --modify-amber)
        modify_amber_and_link=true
        shift # past argument
        ;;
        --amber)
        make_amber=true
        configure_amber=true
        modify_amber_and_link=true
        shift # past argument
        ;;

        --cuda)
        cuda_toolkit_version=$2
        shift # past argument
        shift # past value
        ;;
        --pytorch)
        pytorch_version=$2
        shift # past argument
        shift # past value
        ;;
        --pytorch-cuda)
        pytorch_cuda_version=${2}
        shift # past argument
        shift # past value
        ;;

        --cudnn)
        cudnn_version=$2
        shift # past argument
        shift # past value
        ;;
        --cudnn-cuda)
        cudnn_cuda_version=${2}
        shift # past argument
        shift # past value
        ;;

        --libtorch)
        libtorch_version=${2}
        shift # past argument
        shift # past value
        ;;
        --libtorch-cuda)
        libtorch_cuda_version=${2}
        shift # past argument
        shift # past value
        ;;

        --amberhome)
        export AMBERHOME=$2
        shift # past argument
        shift # past value
        ;;
        --only-amber)
        # this option overrides other options
        # set all to false
        jit_compile_models=OFF
        build_libtorchani=false
        run_tests=false

        # only do amber related stuff
        make_amber=true
        configure_amber=true
        modify_amber_and_link=true
        shift # past argument
        ;;
        *)    # unknown option
        echo "ERROR: Unknown option $1"
        echo "Usage: "
        echo "${usage}"
        exit 1
        shift # past argument
        ;;
    esac
done
set -- "${positional[@]}" # restore positional parameters

# Create Makefile for libtorchani with cmake and build libtorchani into ./build
# this requires CMake version 3.0 at least this builds libtorchani AND unit
# tests. To build unit tests python, torch and torchani are required
if ${build_libtorchani}; then
    rm -rf ${DIR}/build/*
    cd "${DIR}/build" || exit 1
    _CUDA=11.6
    _TORCH=1.13
    cmake \
        -DCMAKE_CXX_COMPILER=g++\
        -DCMAKE_C_COMPILER=gcc\
        -DCUDA_TOOLKIT_VERSION=$_CUDA\
        -DCUDA_TOOLKIT_SET_VARS=ON\
        -DCUSTOM_CUDNN=ON\
        -DCUSTOM_CUDNN_CUDA_VERSION=$_CUDA\
        -DCUSTOM_CUDNN_VERSION=8\
        -DJIT_AVOID_OPTIMIZATIONS=ON\
        -DJIT_COMPILE_MODELS=ON\
        -DJIT_EXTERNAL_CELL_LIST=ON\
        -DJIT_TORCH_CELL_LIST=ON\
        -DLIBTORCH_VERSION=$_TORCH\
        -DLIBTORCH_USES_CXX11ABI=ON\
        -DLIBTORCH_CUDA_VERSION=$_CUDA\
        -DPYTORCH_VERSION=$_TORCH\
        -DPYTORCH_TORCHVISION_VERSION=0.14\
        -DPYTORCH_CUDA_VERSION=$_CUDA\
        -DTORCHANI_INSTALL=ON\
        ..
    cmake --build . --config Release
    cd - || exit 1
else
    echo "Flag no-libtorchani models specified, skipping libtorchani.so compilation"
fi

# Run unit tests for libtorchani only
if ${run_tests}; then
    # Only run cuda tests if a cuda device could be detected by torch
    if $(hash nvidia-smi); then
        "${DIR}/build/test/unit_tests" ${1} [CUDA]
    else
        echo "WARNING: Only running CPU tests, no CUDA device detected."
        echo "To rerun all tests in the future execute './test/unit_tests'."
        echo "To run only CUDA tests in the future './test/unit_tests [CUDA]'."
    fi
    "${DIR}/build/test/unit_tests" ${1} [CPU]
else
    echo "Flag no-tests specified, skipping libtorchani.so compilation"
fi

# Modifys sander and pmemd files, and modifies configure2 inside AmberTools
# then configures amber and copies libtorchani to the correct location
if ${modify_amber_and_link}; then
    # probably better to execute scripts than to source them to be
    # sure that they don't redefine stuff
    "${DIR}/amber_files/modify_amber_and_link.sh" "${configure_amber}"
fi

# make pmemd and sander only, not the whole of Amber
if ${make_amber}; then
    pmemd_amber_dir="${AMBERHOME}/src/pmemd/src"
    sander_amber_dir="${AMBERHOME}/AmberTools/src/sander"
    cd "${pmemd_amber_dir}" || exit 1
    make
    cd - || exit 1
    cd "${sander_amber_dir}" || exit 1
    make
    cd - || exit 1
fi

# run some test dynamics to check that CUDA and CPU work fine
if ${test_amber}; then
    "${DIR}/amber_examples/run.sh"
fi
echo "Done"
