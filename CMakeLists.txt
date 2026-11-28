cmake_minimum_required(
    VERSION 3.19
    FATAL_ERROR
)

project(
    Torchani
    LANGUAGES C CXX
    VERSION 0.7
    DESCRIPTION "Fortran, C and C++ interface to execute general MLIPs, with special support for the TorchANI library"
    HOMEPAGE_URL "https://github.com/roitberg-group/torchani-amber"
)

# Amber sets global compile definitions, flags, and other settings that would otherwise
# be inherited by this project. Unset them to expose the user's cached values
if(INSIDE_AMBER)
    set_directory_properties(PROPERTIES COMPILE_DEFINITIONS "")
    foreach(
        VAR_NAME
        CMAKE_C_FLAGS
        CMAKE_CXX_FLAGS
        CMAKE_C_FLAGS_DEBUG
        CMAKE_CXX_FLAGS_DEBUG
        CMAKE_C_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_RELEASE
        CMAKE_SHARED_LINKER_FLAGS
        CMAKE_C_STANDARD
        CMAKE_CXX_STANDARD
        CMAKE_NO_SYSTEM_FROM_IMPORTED
        CMAKE_MODULE_PATH
    )
        unset(${VAR_NAME})
    endforeach()
endif()

# Necessary CMake built-in modules
include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

# Make private internal modules available
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

# Internal modules
include(Torchani_Msg)
include(Torchani_PlatformHelper)
include(Torchani_OutOfSourceBuild)
include(Torchani_LibtorchHelper)
include(Torchani_PytorchHelper)
include(Torchani_CudaToolkitHelper)
include(Torchani_AmberHelper)
include(Torchani_TorchScriptUtils)
include(Torchani_MlipsHelper)

Torchani_OutOfSourceBuild_ensure()  # Fail with fatal error if out-of-source
Torchani_PlatformHelper_ensure_linux_x86_64()  # Fail with fatal error if not in x86_64 linux

# PyTorch/LibTorch and their backends: options and flags

# NOTE: The pre-built libtorch zip requires extra cuda libraries to be present.
# These libs are *not* in the plain toolkit:
# libcudnn.so, libnccl.so, libcusparseLt.so, libnvshmem_host.so
# Without those installed, libtorch fails at runtime
#
# When a CUDA libtorch download is requested, the libs and headers of these extra
# libraries are also downloaded from their nvidia-*-cu<cuda-major> PyPI wheels
# (directly by fetching from the json API, no pip needed).
#
# The pytorch option is simpler, since the pytorch wheel pulls the required nvidia
# wheels and toolkit libraries directly.

# General: options and flags
# Exact major.minor versions must match for these
set(TORCHANI_REQUIRED_TORCH_VERSION "2.13" CACHE STRING "Require this version of PyTorch/LibTorch, M.m or M.m.p")
set(TORCHANI_REQUIRED_CUDATK_VERSION "13.0" CACHE STRING "Require this version of the CUDA Toolkit")
option(TORCHANI_BUILD_CUDA "Build with CUDA support. If OFF, a CPU-only PyTorch/LibTorch is required" ON)
option(TORCHANI_LINK_TO_PYTORCH_LIBS "Try to find and link to a local PyTorch installation" ON)
option(TORCHANI_SET_LIBRARIES_INSTALL_RPATH "Hardcode the RPATH used during build-time for the installed libs ($ORIGIN is always added)" ON)
option(TORCHANI_DOWNLOAD_AND_INSTALL_LIBTORCH "Download LibTorch library, extract it into the build dir, and install its libs into TORCHANI_LIBRARY_INSTALL_DIR/libtorch" OFF)
if("$ENV{XDG_CACHE_HOME}" STREQUAL "")
    set(TORCHANI_DEFAULT_DOWNLOAD_CACHE_DIR "$ENV{HOME}/.cache/Torchani")
else()
    set(TORCHANI_DEFAULT_DOWNLOAD_CACHE_DIR "$ENV{XDG_CACHE_HOME}/Torchani")
endif()
set(TORCHANI_DOWNLOAD_CACHE_DIR "${TORCHANI_DEFAULT_DOWNLOAD_CACHE_DIR}" CACHE PATH "Dir where the downloaded LibTorch archive, nvidia wheels, and JIT-compiled MLIPs are kept, if a download is requested")
set(TORCHANI_EXTERNAL_LIBTORCH_ROOT "" CACHE PATH "Existing LibTorch to link against. Must contain cuDNN, NCCL, cuSPARSELt, and NVSHMEM")
option(TORCHANI_PREFER_CONDA_LIBS "Prefer libraries under $CONDA_PREFIX (the active conda env)" ON)
option(TORCHANI_FORCE_SYSTEM_CUDATK "Hardcode CUDA Toolkit to /usr/local/cuda-<version>" OFF)
option(TORCHANI_ALLOW_UNTESTED_COMPILERS "Allow compiling with an untested compiler" OFF)
option(TORCHANI_CHECK_PYTORCH_AND_CUDA_VERSIONS "If linking to PyTorch, require its version, and its CUDA version to match the ones in this CMake config" ON)
option(TORCHANI_SUPPORT_PYTHON_MODELS "Support Python models (a dir with a model.py), run by an embedded Python interpreter. Requires TORCHANI_LINK_TO_PYTORCH_LIBS" OFF)

# NOTE: Amber doesn't ship NCCL and the CUDA Toolkit doesn't include it, but there is a
# -DNCCL flag, which if used makes Amber attempt to find NCCL according to NCCL_HOME
# (system paths first, NCCL_HOME last)
# It is very unlekely that loading both NCCL libs would cause any issue whatsoever, but
# for safety it could be a good idea to always point pmemd to libtorch's NCCL if NCCL is
# requested

# Validate options
if(TORCHANI_LINK_TO_PYTORCH_LIBS AND TORCHANI_DOWNLOAD_AND_INSTALL_LIBTORCH)
    Torchani_Msg_error("'TORCHANI_LINK_TO_PYTORCH_LIBS' is incompatible with 'TORCHANI_DOWNLOAD_AND_INSTALL_LIBTORCH'")
endif()
# Python models need libtorch_python, and the libtorch that "import torch" loads
if(TORCHANI_SUPPORT_PYTHON_MODELS AND NOT TORCHANI_LINK_TO_PYTORCH_LIBS)
    Torchani_Msg_error("'TORCHANI_SUPPORT_PYTHON_MODELS' requires 'TORCHANI_LINK_TO_PYTORCH_LIBS'")
endif()
if(NOT TORCHANI_EXTERNAL_LIBTORCH_ROOT STREQUAL "")
    if(TORCHANI_LINK_TO_PYTORCH_LIBS)
        Torchani_Msg_error("'TORCHANI_EXTERNAL_LIBTORCH_ROOT' is incompatible with 'TORCHANI_LINK_TO_PYTORCH_LIBS'")
    endif()
    if(TORCHANI_DOWNLOAD_AND_INSTALL_LIBTORCH)
        Torchani_Msg_error("'TORCHANI_EXTERNAL_LIBTORCH_ROOT' is incompatible with 'TORCHANI_DOWNLOAD_AND_INSTALL_LIBTORCH'")
    endif()
endif()

if("$ENV{CONDA_DEFAULT_ENV}" STREQUAL "base")
    Torchani_Msg_warn("You are building from the conda 'base' env")
endif()

set(TORCHANI_USE_CONDA_LIBS OFF)
if(NOT "$ENV{CONDA_PREFIX}" STREQUAL "")
    if(TORCHANI_PREFER_CONDA_LIBS)
        Torchani_Msg_info("You are running inside a conda env with prefix $ENV{CONDA_PREFIX}")
        Torchani_Msg_info("Libs will be searched here before the system dirs")
        Torchani_Msg_info("If you don't want this, deactivate your conda env before configuring")
        Torchani_Msg_info("Or reconfigure with -DTORCHANI_PREFER_CONDA_LIBS=OFF")
        list(APPEND CMAKE_PREFIX_PATH $ENV{CONDA_PREFIX})
        if(TORCHANI_BUILD_CUDA)
            # Workaround since torch can't find cuda installed in a conda env otherwise
            # Hardcode x86_64-linux, since the MLIP interface can only build there
            set(ENV{CUDA_INC_PATH} $ENV{CONDA_PREFIX}/targets/x86_64-linux/include)
        endif()
        set(TORCHANI_USE_CONDA_LIBS ON)
    else()
        Torchani_Msg_warn("You seem to be installing inside a conda env but with TORCHANI_PREFER_CONDA_LIBS=OFF")
        Torchani_Msg_warn("This is probably not what you want, and it is *HIGHLY RECOMMENDED* to avoid this")
    endif()
endif()

if(TORCHANI_USE_CONDA_LIBS AND TORCHANI_FORCE_SYSTEM_CUDATK)
    Torchani_Msg_error("'TORCHANI_PREFER_CONDA_LIBS' is incompatible with TORCHANI_FORCE_SYSTEM_CUDATK")
endif()

if(NOT TORCHANI_BUILD_CUDA AND TORCHANI_FORCE_SYSTEM_CUDATK)
    Torchani_Msg_error("'TORCHANI_BUILD_CUDA=OFF' is incompatible with TORCHANI_FORCE_SYSTEM_CUDATK")
endif()

# Validate compilers
if(NOT TORCHANI_ALLOW_UNTESTED_COMPILERS)
    set(COMPILER_IDS ${CMAKE_CXX_COMPILER_ID} ${CMAKE_C_COMPILER_ID} ${CMAKE_Fortran_COMPILER_ID})
    foreach(COMPILER_ID ${COMPILER_IDS})
        if(NOT (COMPILER_ID STREQUAL GNU))
            Torchani_Msg_error("The TorchANI-Amber MLIP Interface requires GNU compilers (g++, gcc, gfortran)")
        endif()
    endforeach()
    set(COMPILER_VERSIONS ${CMAKE_CXX_COMPILER_VERSION} ${CMAKE_C_COMPILER_VERSION} ${CMAKE_Fortran_COMPILER_VERSION})
    foreach(COMPILER_VERSION ${COMPILER_VERSIONS})
        if(COMPILER_VERSION VERSION_LESS 11.0 OR COMPILER_VERSION VERSION_GREATER_EQUAL 16.0)
            Torchani_Msg_error("The TorchANI-Amber MLIP Interface requires GNU compilers with version >=11 and <16")
        endif()
    endforeach()
endif()

# JIT: Options and flags
option(TORCHANI_JIT_COMPILE_MODELS "JIT-compile built-in TorchANI models" OFF)
option(TORCHANI_JIT_COMPILE_AIMNET2 "Also JIT-compile AimNet2 models" ON)
option(TORCHANI_JIT_FORCE_RECOMPILATION "Force JIT-recompilation of all models, even if jit-compiled *.pt files with the expected names are found" OFF)
option(TORCHANI_DOWNLOAD_JIT_MODELS "Download the JIT-compiled MLIPs that match the PyTorch/LibTorch version from Hugging Face, and install them into TORCHANI_MLIPS_INSTALL_DIR" ON)

if(TORCHANI_JIT_COMPILE_MODELS AND NOT TORCHANI_LINK_TO_PYTORCH_LIBS)
    Torchani_Msg_error("'TORCHANI_JIT_COMPILE_MODELS' requires 'TORCHANI_LINK_TO_PYTORCH_LIBS'")
endif()

# PyTorch/LibTorch setup
if(TORCHANI_LINK_TO_PYTORCH_LIBS)
    Torchani_Msg_info("Will try to find and link to an existing PyTorch installation")
    # Set var PYTORCH_ROOT
    Torchani_PytorchHelper_detect(
        USE_CUDA ${TORCHANI_BUILD_CUDA}
        CHECK_VERSIONS ${TORCHANI_CHECK_PYTORCH_AND_CUDA_VERSIONS}
        LIB_VERSION ${TORCHANI_REQUIRED_TORCH_VERSION}
        CUDATK_VERSION ${TORCHANI_REQUIRED_CUDATK_VERSION}
    )
    list(APPEND CMAKE_PREFIX_PATH ${PYTORCH_ROOT})
elseif(TORCHANI_DOWNLOAD_AND_INSTALL_LIBTORCH)
    Torchani_Msg_info("Will download and install user-specified LibTorch")
    # NOTE: If the requested version already exists it is *not* extracted
    # Set vars LIBTORCH_ROOT, LIBTORCH_LIBRARY_DIR
    Torchani_LibtorchHelper_download_and_install(
        USE_CUDA ${TORCHANI_BUILD_CUDA}
        LIB_VERSION ${TORCHANI_REQUIRED_TORCH_VERSION}
        CUDATK_VERSION ${TORCHANI_REQUIRED_CUDATK_VERSION}
        EXTRACT_DIR "${PROJECT_BINARY_DIR}"
        CACHE_DIR "${TORCHANI_DOWNLOAD_CACHE_DIR}"
    )
    list(APPEND CMAKE_PREFIX_PATH ${LIBTORCH_ROOT})
elseif(NOT TORCHANI_EXTERNAL_LIBTORCH_ROOT STREQUAL "")
    Torchani_Msg_info("Will use the existing LibTorch at ${TORCHANI_EXTERNAL_LIBTORCH_ROOT}")
    Torchani_LibtorchHelper_check_external(
        ROOT "${TORCHANI_EXTERNAL_LIBTORCH_ROOT}"
        USE_CUDA ${TORCHANI_BUILD_CUDA}
        LIB_VERSION ${TORCHANI_REQUIRED_TORCH_VERSION}
        CUDATK_VERSION ${TORCHANI_REQUIRED_CUDATK_VERSION}
    )
    list(PREPEND CMAKE_PREFIX_PATH ${TORCHANI_EXTERNAL_LIBTORCH_ROOT})
endif()

if(TORCHANI_BUILD_CUDA)
    # CUDA Toolkit setup
    if(TORCHANI_USE_CONDA_LIBS)
        Torchani_Msg_info("Will try to use CUDA Toolkit from the active Conda env")
        # Set CUDA_TOOLKIT_ROOT_DIR and CMAKE_CUDA_COMPILER
        Torchani_CudaToolkitHelper_set_conda_paths()
        Torchani_Msg_banner(
            "The TorchANI-Amber MLIP Interface requires CUDA Toolkit dev version, which is *not* bundled with PyTorch.
            At this stage of the configuration, we don't check whether all necessary CUDA
            Toolkit features are present.
            Make sure you have the required headers, NVCC, and *all* CUDA Toolkit libs. To
            ensure this, for example, use the provided conda env, which has all
            dependencies, or directly download and install the CUDA Toolkit
            ${TORCHANI_REQUIRED_CUDATK_VERSION} from Nvidia's web page. Failures related to CUDA
            after this point may indicate you are missing a required part of the Toolkit."
        )
    elseif(TORCHANI_FORCE_SYSTEM_CUDATK)
        Torchani_Msg_info("Will manually set variables that point to the CUDA Toolkit")
        Torchani_Msg_info("assuming it is located in /usr/local/cuda-<version>")
        # Set CUDA_TOOLKIT_ROOT_DIR and CMAKE_CUDA_COMPILER
        Torchani_CudaToolkitHelper_set_custom_paths(CUDATK_VERSION ${TORCHANI_REQUIRED_CUDATK_VERSION})
    else()
        Torchani_Msg_info("Will follow LibTorch's default lookup procedure for the CUDA Toolkit")
        set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
    endif()
    set(TORCHANI_REQUESTED_CUDA_COMPILER "${CMAKE_CUDA_COMPILER}")
    enable_language(CUDA)
    # CMake keeps the CUDA compiler detected on the first configure of a build dir
    if(
        (TORCHANI_USE_CONDA_LIBS OR TORCHANI_FORCE_SYSTEM_CUDATK)
        AND NOT CMAKE_CUDA_COMPILER STREQUAL TORCHANI_REQUESTED_CUDA_COMPILER
    )
        Torchani_Msg_error(
            "Requested CUDA compiler ${TORCHANI_REQUESTED_CUDA_COMPILER}, but this build dir already uses ${CMAKE_CUDA_COMPILER}. If you need to change compiler, delete the build dir and reconfigure."
        )
    endif()

    if(INSIDE_AMBER)
        Torchani_AmberHelper_check_nvcc_matches(
            TORCHANI_NVCC "${CMAKE_CUDA_COMPILER}"
            AMBER_NVCC "${CUDA_NVCC_EXECUTABLE}"
        )
        Torchani_AmberHelper_hide_amber_cuda_sources(AMBER_SOURCE_DIR "${CMAKE_SOURCE_DIR}")
    endif()
else()
    Torchani_Msg_info("CUDA support is disabled, building CPU-only")
endif()

message(STATUS "The modified CMake prefix path is: ${CMAKE_PREFIX_PATH}")

# LibTorch is a required dep
# Set TORCH_FOUND, TORCH_LIBRARIES, TORCH_INCLUDE_DIRS
Torchani_Msg_banner(
    "CMake warnings about 'kineto not being found', or 'CMAKE_CUDA_ARCHITECTURES being
    incompatible with pytorch' are safe to ignore. Other warnings should not be ignored."
)

if(INSIDE_AMBER)
    Torchani_AmberHelper_disable_torch_mkl_if_not_found(MKL_FOUND "${MKL_FOUND}")
    Torchani_AmberHelper_set_torch_cuda_arch_list(AMBER_NVCC_FLAGS ${CUDA_NVCC_FLAGS})
endif()
find_package(Torch ${TORCHANI_REQUIRED_TORCH_VERSION} REQUIRED)

if(TORCHANI_SUPPORT_PYTHON_MODELS)
    Torchani_Msg_info("Python models are supported, libtorchani will embed a Python interpreter")
    find_package(Python REQUIRED COMPONENTS Interpreter Development.Embed)
    find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib" NO_DEFAULT_PATH REQUIRED)
endif()

# Must match torchani.paths.extensions_build_tag in the TorchANI Python package
if(TORCHANI_BUILD_CUDA)
    string(REGEX MATCH "^[0-9]+" CUDA_VERSION_MAJOR "${CMAKE_CUDA_COMPILER_VERSION}")
    set(TORCHANI_EXTENSIONS_BUILD_TAG "${Torch_VERSION_MAJOR}.${Torch_VERSION_MINOR}-cu${CUDA_VERSION_MAJOR}")
else()
    set(TORCHANI_EXTENSIONS_BUILD_TAG "${Torch_VERSION_MAJOR}.${Torch_VERSION_MINOR}-cpu")
endif()

# *_INSTALL_DIR are all relative to the CMAKE_INSTALL_PREFIX
set(TORCHANI_INCLUDE_INSTALL_DIR "${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}" CACHE STRING "Install dir for the C header")

# NOTE: Technically, from the Linux File System Hierarchy, the convention should be
# lib/<arch-id>/Torchani/<tag>, but it is lib/Torchani/<tag> for now, for simplicity.
# TorchANI follows the same convention.
set(TORCHANI_LIBRARY_INSTALL_DIR "" CACHE STRING "Install dir for libtorchani, cuaev and cell_list. Defaults to lib/${PROJECT_NAME}/<build tag>")
if(TORCHANI_LIBRARY_INSTALL_DIR STREQUAL "")
    set(TORCHANI_LIBRARY_INSTALL_DIR "lib/${PROJECT_NAME}/${TORCHANI_EXTENSIONS_BUILD_TAG}")
endif()
set(TORCHANI_MLIPS_INSTALL_DIR "${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/mlips/jit" CACHE STRING "Install dir for the downloaded JIT-compiled MLIPs")
if(IS_ABSOLUTE "${TORCHANI_LIBRARY_INSTALL_DIR}" OR IS_ABSOLUTE "${TORCHANI_MLIPS_INSTALL_DIR}")
    Torchani_Msg_error("'TORCHANI_LIBRARY_INSTALL_DIR' and 'TORCHANI_MLIPS_INSTALL_DIR' must be relative to CMAKE_INSTALL_PREFIX")
endif()
# The build tree mirrors the install tree, so libtorchani finds the MLIPs at the same
# path relative to itself in both
set(TORCHANI_LIBRARY_BUILD_DIR "${PROJECT_BINARY_DIR}/${TORCHANI_LIBRARY_INSTALL_DIR}")
set(TORCHANI_MLIPS_BUILD_DIR "${PROJECT_BINARY_DIR}/${TORCHANI_MLIPS_INSTALL_DIR}")
file(RELATIVE_PATH TORCHANI_MLIPS_RELATIVE_DIR "/${TORCHANI_LIBRARY_INSTALL_DIR}" "/${TORCHANI_MLIPS_INSTALL_DIR}")
# The TorchANI Python package only loads the extensions if this file matches its tag
file(WRITE "${TORCHANI_LIBRARY_BUILD_DIR}/_extensions_build_tag.txt" "${TORCHANI_EXTENSIONS_BUILD_TAG}")

# Build both libcuaev, libcell_list, and libtorchani
set(CELL_ROOT "${CMAKE_CURRENT_LIST_DIR}/submodules/torchani/torchani/csrc")
add_library(
    torchani_cell_list
    SHARED
    "${CELL_ROOT}/cell_list.cpp"
)
target_compile_features(torchani_cell_list PRIVATE cxx_std_20)
if(TORCHANI_USE_CONDA_LIBS)
    target_link_directories(torchani_cell_list PRIVATE $ENV{CONDA_PREFIX}/lib)
endif()
target_link_libraries(torchani_cell_list PRIVATE ${TORCH_LIBRARIES})
target_include_directories(torchani_cell_list PRIVATE ${CELL_ROOT})

set(CUAEV_ROOT "${CMAKE_CURRENT_LIST_DIR}/submodules/torchani/torchani/csrc")
add_library(
    torchani_cuaev
    SHARED
    "${CUAEV_ROOT}/cuaev_schema.cpp"
    "${CUAEV_ROOT}/aev_cpu.cpp"
)
if(TORCHANI_BUILD_CUDA)
    target_sources(
        torchani_cuaev
        PRIVATE
            "${CUAEV_ROOT}/cuaev.cpp"
            "${CUAEV_ROOT}/aev.cu"
            "${CUAEV_ROOT}/aev.h"
            "${CUAEV_ROOT}/cuaev_cub.cuh"
    )
endif()
target_compile_features(torchani_cuaev PRIVATE cxx_std_20)
# TORCHANI_OPT and "-use_fast_math" make calculations faster in libcuaev.so.
# TORCHANI_OPT commands libcuaev to use cuda intrinsics for cos, sin, etc.
# "-use_fast_math" commands nvcc to use the fast math library
# The wrapped cub namespace flags are needed to safely use cub:
# https://github.com/pytorch/pytorch/pull/55292
# https://github.com/pytorch/pytorch/pull/66219
# TODO: CUB_NS_* are deprecated in CCCL in favor of CUB_WRAPPED_NAMESPACE
set(CUAEV_CUB_NAMESPACE_DEFS "CUB_NS_QUALIFIER=::cuaev::cub" "CUB_NS_PREFIX=namespace cuaev {" "CUB_NS_POSTFIX=}")
target_compile_definitions(
    torchani_cuaev
    PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:TORCHANI_OPT>
        $<$<COMPILE_LANGUAGE:CUDA>:${CUAEV_CUB_NAMESPACE_DEFS}>
)
target_compile_options(torchani_cuaev PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math>)
if(TORCHANI_USE_CONDA_LIBS)
    target_link_directories(torchani_cuaev PRIVATE $ENV{CONDA_PREFIX}/lib)
endif()
target_link_libraries(torchani_cuaev PRIVATE ${TORCH_LIBRARIES})
target_include_directories(torchani_cuaev PRIVATE ${CUAEV_ROOT})

# The extensions are named as the TorchANI Python package expects them (cuaev.so,
# cell_list.so). After installation, the TorchANI Python package finds them in
# <prefix>/lib/Torchani/<TORCHANI_EXTENSIONS_BUILD_TAG>, so they don't need to be built
# a second time by pip or "ani build-extensions"
set_target_properties(torchani_cuaev torchani_cell_list PROPERTIES PREFIX "")
set_target_properties(torchani_cuaev PROPERTIES OUTPUT_NAME cuaev EXPORT_NAME cuaev)
set_target_properties(torchani_cell_list PROPERTIES OUTPUT_NAME cell_list EXPORT_NAME cell_list)

file(GLOB TORCHANI_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_LIST_DIR}/src/*.cpp")
if(NOT TORCHANI_SUPPORT_PYTHON_MODELS)
    list(REMOVE_ITEM TORCHANI_SOURCES "${CMAKE_CURRENT_LIST_DIR}/src/python_model.cpp")
endif()
add_library(torchani SHARED ${TORCHANI_SOURCES})
target_compile_options(
    torchani
    PRIVATE
        -pedantic
        -Wall
        -Wextra
        -Wcast-align
        -Wcast-qual
        -Wdisabled-optimization
        -Winit-self
        -Wmissing-include-dirs
        -Woverloaded-virtual
        -Wredundant-decls
        -Wshadow
)
target_compile_features(torchani PRIVATE cxx_std_20)
target_compile_definitions(
    torchani
    PRIVATE
        $<$<CONFIG:Debug>:DEBUG>
        TORCHANI_MLIPS_RELATIVE_DIR="${TORCHANI_MLIPS_RELATIVE_DIR}"
        TORCHANI_HOMEPAGE_URL="${PROJECT_HOMEPAGE_URL}"
        $<$<BOOL:${TORCHANI_BUILD_CUDA}>:TORCHANI_BUILD_CUDA>
        $<$<BOOL:${TORCHANI_SUPPORT_PYTHON_MODELS}>:TORCHANI_SUPPORT_PYTHON_MODELS>
)
# libtorchani references none of the symbols of cuaev and cell_list, but loading them is
# what registers their Torch ops. libtorchani dlopens them at init from its own dir
if(TORCHANI_USE_CONDA_LIBS)
    target_link_directories(torchani PRIVATE $ENV{CONDA_PREFIX}/lib)
endif()
target_link_libraries(torchani PRIVATE ${TORCH_LIBRARIES} ${CMAKE_DL_LIBS})
if(TORCHANI_SUPPORT_PYTHON_MODELS)
    target_link_libraries(torchani PRIVATE ${TORCH_PYTHON_LIBRARY} Python::Python)
endif()
add_dependencies(torchani torchani_cuaev torchani_cell_list)
set_target_properties(
    torchani
    torchani_cuaev
    torchani_cell_list
    PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${TORCHANI_LIBRARY_BUILD_DIR}"
)
target_include_directories(torchani PRIVATE "${CMAKE_CURRENT_LIST_DIR}/include" ${TORCH_INCLUDE_DIRS})

# For cuaev, cell_list and libtorchani, the install RPATH is set by default to be
# the same as the link path to avoid issues if the correct CUDA is not
# installed system-wide. If this is not done then CMake tries to re-link the
# binaries at install-time, and prefers the system CUDA, which may be bad. In addition,
# if the Conda PyTorch is used, TorchANI would not be able to find the necessary
# libraries even if the env is active, since PyTorch doesn't expose them in
# ${CONDA_PREFIX}/lib.
#
# NOTE: When doing this, both the build-tree and the install-tree binaries are
# still linked to the CUDA driver library from the system (libcuda.so) by
# default. This behavior seems to be a design choice of Conda, since envs don't
# have a "real" libcuda.so, just a "stub" libcuda.so. I belive this is ok,
# since the CUDA driver is unrelated to the CUDA Toolkit, but I'm documenting
# it here for completeness.
#
# INSTALL_RPATH_USE_LINK_PATH bakes the linked library paths from *outside* this project
# into the install rpath of the targets (it doesn't bake the lib paths *inside* the
# project). The link paths inside the project are baked in manually
# (the installed LibTorch dir, which is $ORIGIN/libtorch, and the Torchani installation
# dir, which is $ORIGIN).
# NOTE: INSTALL_RPATH_USE_LINK_PATH also adds /lib/intel64, /lib/intel64_win and
# /lib/win-x64, derived from TORCH_LIBRARIES. These don't exist, but this is harmless
# NOTE: INSTALL_RPATH_USE_LINK_PATH skips CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES, which
# contain the linked cudart and nvrtc, so their dirs are also baked in manually
# NOTE: CUDA_cudart_LIBRARY and CUDA_nvrtc_LIBRARY are internal cache vars of
# FindCUDAToolkit (called by Torch), and they may change in the future
set(TORCHANI_LIBRARIES_INSTALL_RPATH "$ORIGIN")
if(TORCHANI_SET_LIBRARIES_INSTALL_RPATH)
    get_filename_component(CUDART_LIBRARY_DIR "${CUDA_cudart_LIBRARY}" DIRECTORY)
    get_filename_component(NVRTC_LIBRARY_DIR "${CUDA_nvrtc_LIBRARY}" DIRECTORY)
    list(PREPEND TORCHANI_LIBRARIES_INSTALL_RPATH
        ${CUDART_LIBRARY_DIR}
        ${NVRTC_LIBRARY_DIR}
    )
    list(REMOVE_DUPLICATES TORCHANI_LIBRARIES_INSTALL_RPATH)
endif()
if(TORCHANI_DOWNLOAD_AND_INSTALL_LIBTORCH)
    list(PREPEND TORCHANI_LIBRARIES_INSTALL_RPATH "$ORIGIN/libtorch")
endif()
if(NOT TORCHANI_EXTERNAL_LIBTORCH_ROOT STREQUAL "")
    list(PREPEND TORCHANI_LIBRARIES_INSTALL_RPATH "${TORCHANI_EXTERNAL_LIBTORCH_ROOT}/lib")
endif()
set_target_properties(
    torchani_cuaev
    torchani_cell_list
    torchani
    PROPERTIES
        INSTALL_RPATH "${TORCHANI_LIBRARIES_INSTALL_RPATH}"
        INSTALL_RPATH_USE_LINK_PATH ${TORCHANI_SET_LIBRARIES_INSTALL_RPATH}
)

# The models are JIT-compiled after the extensions are built, since scripting them
# requires the cuAEV ops to be registered
if(TORCHANI_JIT_COMPILE_MODELS)
    Torchani_Msg_info("Will attempt to JIT-compile ANI models after building the extensions")
    Torchani_TorchScriptUtils_add_jit_compile_models_target(
        FORCE_RECOMPILATION ${TORCHANI_JIT_FORCE_RECOMPILATION}
        AIMNET2 ${TORCHANI_JIT_COMPILE_AIMNET2}
        EXTENSIONS_DIR $<TARGET_FILE_DIR:torchani_cuaev>
    )
    add_dependencies(torchani_jit_compile_models torchani_cuaev torchani_cell_list)
else()
    Torchani_Msg_info("Skipping JIT-compilation of models, make sure to manually JIT-compile needed ANI models")
endif()

# Build Catch2 subproject (unit test framework) and Torchani unit tests
# TODO: Reinstate internal mlip-interface tests inside Amber if possible in the future
if(NOT INSIDE_AMBER)
    include(CTest)
    if(BUILD_TESTING)
        set(CATCH2_ROOT "${CMAKE_CURRENT_LIST_DIR}/submodules/Catch2")
        add_subdirectory(${CATCH2_ROOT})
        # Make Catch2 provided cmake-modules available
        list(APPEND CMAKE_MODULE_PATH "${CATCH2_ROOT}/extras")
        include(Catch)

        add_executable(
            torchani_tests
            "${CMAKE_CURRENT_LIST_DIR}/tests/tests.cpp"
            "${CMAKE_CURRENT_LIST_DIR}/tests/test_fortran_utils.cpp"
            "${CMAKE_CURRENT_LIST_DIR}/tests/test_config.cpp"
            "${CMAKE_CURRENT_LIST_DIR}/tests/test_model_formats.cpp"
        )
        target_include_directories(
            torchani_tests
            PRIVATE
                "${CMAKE_CURRENT_LIST_DIR}/include"
                "${CMAKE_CURRENT_LIST_DIR}/src"
        )
        target_link_libraries(torchani_tests PRIVATE Catch2::Catch2WithMain torchani ${TORCH_LIBRARIES})
        target_compile_definitions(
            torchani_tests
            PRIVATE
                TORCHANI_MLIPS_BUILD_DIR="${TORCHANI_MLIPS_BUILD_DIR}"
                $<$<BOOL:${TORCHANI_BUILD_CUDA}>:TORCHANI_BUILD_CUDA>
                $<$<BOOL:${TORCHANI_SUPPORT_PYTHON_MODELS}>:TORCHANI_SUPPORT_PYTHON_MODELS>
        )
        catch_discover_tests(torchani_tests)

        # Used by tests/test_namelists.py to compare the namelist parser with f90nml
        add_executable(
            torchani_namelist_dump
            "${CMAKE_CURRENT_LIST_DIR}/tests/namelist_dump.cpp"
            "${CMAKE_CURRENT_LIST_DIR}/src/fortran_utils.cpp"
            "${CMAKE_CURRENT_LIST_DIR}/src/utils.cpp"
        )
        target_include_directories(torchani_namelist_dump PRIVATE "${CMAKE_CURRENT_LIST_DIR}/src")
        target_compile_features(torchani_namelist_dump PRIVATE cxx_std_20)
    endif()
endif()

# Same names as the exported targets, for consumers that use add_subdirectory
add_library(${PROJECT_NAME}::torchani ALIAS torchani)
add_library(${PROJECT_NAME}::cuaev ALIAS torchani_cuaev)
add_library(${PROJECT_NAME}::cell_list ALIAS torchani_cell_list)

set(PKG_EXPORT_FILESET_NAME "${PROJECT_NAME}Targets")

# Install corresponding targets
install(
    TARGETS
        torchani
        torchani_cuaev
        torchani_cell_list
    EXPORT ${PKG_EXPORT_FILESET_NAME}  # Add ref to targs in the pkg-export-fileset
    LIBRARY
        DESTINATION "${TORCHANI_LIBRARY_INSTALL_DIR}"
)
if(TORCHANI_DOWNLOAD_AND_INSTALL_LIBTORCH)
    install(
        DIRECTORY "${LIBTORCH_LIBRARY_DIR}/"
        DESTINATION "${TORCHANI_LIBRARY_INSTALL_DIR}/libtorch"
        USE_SOURCE_PERMISSIONS
    )
endif()
if(TORCHANI_DOWNLOAD_JIT_MODELS)
    Torchani_MlipsHelper_fetch_jit_models(
        REPO "roitberg-group/compiled-mlips"
        REVISION "main"
        TORCH_VERSION "${Torch_VERSION}"
        CACHE_DIR "${TORCHANI_DOWNLOAD_CACHE_DIR}"
        DESTINATION "${TORCHANI_MLIPS_BUILD_DIR}"
    )
    install(
        DIRECTORY "${TORCHANI_MLIPS_BUILD_DIR}/"
        DESTINATION "${TORCHANI_MLIPS_INSTALL_DIR}"
    )
endif()
# Install C-interface include file
install(
    FILES
        "${CMAKE_CURRENT_LIST_DIR}/include/torchani.h"
    DESTINATION "${TORCHANI_INCLUDE_INSTALL_DIR}"
)

# Amber's CMake setup doesn't support exported targets, so the files for discovery and
# config by downstream pkgs are only installed if not building as an Amber subproject
if(NOT INSIDE_AMBER)
    # Create and install files for discovery and config by downstream pkgs
    # - pkg-config-file: <Pkg>Config.cmake
    # - pkg-ver-file: <Pkg>ConfigVersion.cmake
    set(PKG_VERSION_FILE_BLDTREE_PATH "${CMAKE_CURRENT_BINARY_DIR}/cmake/${PROJECT_NAME}ConfigVersion.cmake")
    write_basic_package_version_file(
        "${PKG_VERSION_FILE_BLDTREE_PATH}"
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion
    )
    set(PKG_CONFIG_FILE_BLDTREE_PATH "${CMAKE_CURRENT_BINARY_DIR}/cmake/${PROJECT_NAME}Config.cmake")

    set(PKGDATA_INSTALL_DIR "${CMAKE_INSTALL_DATAROOTDIR}/cmake/${PROJECT_NAME}")
    set(DATA_INSTALL_DIR "${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}")
    set(FORTRAN_INTERFACE_INSTALL_SRC_FILE "${DATA_INSTALL_DIR}/Fortran/torchani.F90")
    configure_package_config_file(
        "${CMAKE_CURRENT_LIST_DIR}/cmake/${PROJECT_NAME}Config.cmake.in"
        "${PKG_CONFIG_FILE_BLDTREE_PATH}"
        INSTALL_DESTINATION "${PKGDATA_INSTALL_DIR}"
        PATH_VARS
            TORCHANI_INCLUDE_INSTALL_DIR
            DATA_INSTALL_DIR
            TORCHANI_LIBRARY_INSTALL_DIR
            FORTRAN_INTERFACE_INSTALL_SRC_FILE
    )
    # Install CMake config-file and config-ver-file for downstream usage:
    install(
        FILES
            "${PKG_CONFIG_FILE_BLDTREE_PATH}"
            "${PKG_VERSION_FILE_BLDTREE_PATH}"
        DESTINATION "${PKGDATA_INSTALL_DIR}"
    )

    # Install CMake targets for downstream usage:
    # pkg-export-fileset: {<Pkg>Targets.cmake, <Pkg>Targets-*.cmake}
    install(
        EXPORT ${PKG_EXPORT_FILESET_NAME}
        FILE "${PROJECT_NAME}Targets.cmake"  # Primary filename in the pkg-export-fileset
        NAMESPACE "${PROJECT_NAME}::"  # Exported targets namespace
        DESTINATION "${PKGDATA_INSTALL_DIR}"
    )

    # Install module file necessary for a portable fortran interface
    install(
        FILES
            "${CMAKE_CURRENT_LIST_DIR}/interface/torchani.F90"
        DESTINATION "${DATA_INSTALL_DIR}/Fortran"
    )
endif()
# TODO Add model.jit.pt files
