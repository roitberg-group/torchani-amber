cmake_minimum_required(
    VERSION 3.24
    FATAL_ERROR
)

project(
    Torchani
    LANGUAGES C CXX
    VERSION 0.1
    DESCRIPTION "C++ wrapper for Torchani"
    HOMEPAGE_URL "https://github.com/roitberg-group/torchani-amber.git"
)

# CMake builtin includes
include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

# Append private internal modules path
set(PRIVATE_MODULES_DIR "${CMAKE_CURRENT_LIST_DIR}/share/cmake/${PROJECT_NAME}/private")
list(APPEND CMAKE_MODULE_PATH ${PRIVATE_MODULES_DIR})

# Builtin modules
include(FetchContent)

# Internal modules
include(EnsureLinuxBuild)
include(EnsureOutOfSourceBuild)
include(Msg)
include(DownloadAndExtractLibtorch)
include(SetCudnnDirs)
include(SetCudaToolkitDirs)
include(InstallTorchani)
include(JitCompileModels)

ensure_out_of_source_build()  # fail with fatal error if out-of-source
ensure_linux_build()  # fail with fatal error if non-linux

# User options and flags
option(LOCAL_INSTALL "Install in ~/.local" ON)
option(CUSTOM_CUDNN "Explicitly specify the cuDNN location" ON)
option(CUSTOM_CUDA "Explicitly set CUDA root variable" ON)
option(CONDA_CUDNN "Use the cuDNN from the conda environment" ON)
option(CONDA_CUDA "Use CUDA Toolkit from conda environment" ON)
option(TORCHANI_INSTALL "Install Numpy, PyTorch and TorchANI" OFF)
option(DOWNLOAD_AND_EXTRACT_LIBTORCH "Download LibTorch library and extract locally" OFF)

set(CUDA_TOOLKIT_VERSION "11.8" CACHE STRING "CUDA Toolkit version to set CUDA vars")

# PyTorch backend versions
set(PYTORCH_VERSION "2.3.0" CACHE STRING "PyTorch version to use")
set(PYTORCH_CUDNN_VERSION "8.9.7" CACHE STRING "cuDNN version for PyTorch")
set(PYTORCH_CUDA_VERSION "11.8" CACHE STRING "CUDA Toolkit version for PyTorch")
set(PYTORCH_PYTHON_VERSION "3.10" CACHE STRING "Python version for PyTorch")

# Libtorch backend versions (only needed if using an external libtorch)
set(LIBTORCH_VERSION "2.3.0" CACHE STRING "LibTorch version to use")
set(LIBTORCH_CUDNN_VERSION "8.9.7" CACHE STRING "cuDNN version for LibTorch")
set(LIBTORCH_CUDA_VERSION "11.8" CACHE STRING "CUDA Toolkit version for LibTorch")
set(LIBTORCH_USE_CXX11ABI TRUE CACHE BOOL "Download a LibTorch version that exposes CXX11 ABI")

# cuDNN options
set(LIBTORCH_CUDNN_CUDA_VERSION "11.8" CACHE STRING "CUDA version cuDNN is compatible with")

# JIT options
option(JIT_DISABLE_OPTIMIZATIONS "Avoid TorchScript when compiling models optimizations" ON)
option(JIT_FORCE_RECOMPILE "Force JIT recompilation of all models" OFF)
option(JIT_COMPILE_MODELS "JIT-compile TorchANI models" ON)
option(JIT_STANDARD "JIT-compile standard models" ON)
option(JIT_TORCH_CELL_LIST "JIT-compile models with Torch cell list" ON)
option(JIT_EXTERNAL_CELL_LIST "JIT-compile models with external cell list" OFF)
option(JIT_CUAEV "JIT-compile models with cuAEV extension" ON)
option(JIT_CUAEV_TORCH_CELL_LIST "JIT-compile models with Torch cell list" ON)
option(JIT_CUAEV_EXTERNAL_CELL_LIST "JIT-compile models with external cell list and cuAEV extension" OFF)

# Use conda packages in current environment
list(APPEND CMAKE_PREFIX_PATH $ENV{CONDA_PREFIX})
# Make sure CMake can find conda torch
list(APPEND CMAKE_PREFIX_PATH $ENV{CONDA_PREFIX}/lib/python${PYTORCH_PYTHON_VERSION}/site-packages/torch)

if(DOWNLOAD_AND_EXTRACT_LIBTORCH)
    # If the requested version already exists it is not extracted
    # Also sets LIBTORCH_ROOT, LIBTORCH_LIBRARY_DIR, LIBTORCH_INCLUDE_DIR
    download_and_extract_libtorch(
      CXX11ABI ${LIBTORCH_USE_CXX11ABI}
      LIBRARY_VERSION ${LIBTORCH_VERSION}
      CUDA_VERSION ${LIBTORCH_CUDA_VERSION}
    )
else()
    msg_info("Skipped LibTorch download. Will use PyTorch directly (this is preferred)")
endif()
list(APPEND CMAKE_PREFIX_PATH ${LIBTORCH_ROOT})

# cuDNN may be linked by providing its location
# ("custom cuDNN"), or it may be directly extracted into the CUDA Toolkit dir,
# in which case it is not needed to specify the paths.
if (CUSTOM_CUDNN)
    # sets CUDNN_ROOT, CUDNN_LIBRARY, CUDNN_INCLUDE_DIR
    set_cudnn_dirs(
        CONDA_CUDNN ${CONDA_CUDNN}
        LIBRARY_VERSION ${LIBTORCH_CUDNN_VERSION}
        CUDA_VERSION ${LIBTORCH_CUDNN_CUDA_VERSION}
    )
    list(APPEND CMAKE_PREFIX_PATH "${CUDNN_LIBRARY}")
else()
    msg_warn("Skip setting cuDNN dirs, make sure cuDNN is present in CUDA Toolkit")
endif()
message(STATUS "CMake - prefix path: ${CMAKE_PREFIX_PATH}")

if(CUSTOM_CUDA)
    # sets CUDA_TOOLKIT_ROOT_DIR
    set_cuda_toolkit_dirs(
        CONDA_CUDA ${CONDA_CUDA}
        CUDA_VERSION ${CUDA_TOOLKIT_VERSION}
    )
else()
    msg_warn("Skip setting CUDA dirs, make sure cmake finds the correct CUDA Toolkit")
endif()

if(TORCHANI_INSTALL)
    install_torchani(
        PYTORCH_CUDNN ${PYTORCH_CUDNN_VERSION}
        PYTORCH_PYTHON ${PYTORCH_PYTHON_VERSION}
        PYTORCH_VERSION ${PYTORCH_VERSION}
        PYTORCH_CUDA_VERSION ${PYTORCH_CUDA_VERSION}
    )
else()
    msg_warn("Skipping python package installs, make sure JIT models are available")
endif()

if(JIT_COMPILE_MODELS)
    jit_compile_models(
        JIT_DISABLE_OPTIMIZATIONS ${JIT_DISABLE_OPTIMIZATIONS}
        JIT_STANDARD ${JIT_STANDARD}
        JIT_TORCH_CELL_LIST ${JIT_TORCH_CELL_LIST}
        JIT_EXTERNAL_CELL_LIST ${JIT_EXTERNAL_CELL_LIST}
        JIT_CUAEV ${JIT_CUAEV}
        JIT_CUAEV_TORCH_CELL_LIST ${JIT_CUAEV_TORCH_CELL_LIST}
        JIT_CUAEV_EXTERNAL_CELL_LIST ${JIT_CUAEV_EXTERNAL_CELL_LIST}
    )
else()
    msg_warn("Skipping JIT compilation of models, make sure JIT models are available")
endif()

# LibTorch is a required dependency
# sets TORCH_FOUND, TORCH_LIBRARIES, TORCH_INCLUDE_DIRS, TORCH_CXX_FLAGS
find_package(Torch ${LIBTORCH_VERSION} EXACT REQUIRED)

# The cuAEV extension lib is a required dep, and needs the Python headers and libraries
find_package(Python REQUIRED COMPONENTS Interpreter Development)

if(LOCAL_INSTALL)
    set(TORCHANI_LOCAL_LIB /home/$ENV{USER}/.local/lib/Torchani)
else()
    set(TORCHANI_LOCAL_LIB /usr/local/lib/Torchani)
endif()
set(TORCHANI_INSTALL_RPATH ${LIBTORCH_LIBRARY_DIR} ${CUDA_TOOLKIT_ROOT_DIR} ${TORCHANI_LOCAL_LIB})

# Now build both libcuaev and libtorchanis
#
# NOTE: In both cases the install rpath is hardcoded to be the same as the link path,
# to avoid issues if the correct CUDA is not installed system-wide. If this is
# not done then CMake tries to re-link the binaries at install-time, and
# prefers the system CUDA, which may be bad.
#
# NOTE: When doing this, both the build-tree and the install-tree binaries are
# still linked to the CUDA driver library from the system (libcuda.so) by
# default. This behavior seems to be a design choice of libtorch. I belive this
# is ok, since the cuda driver is in principle unrelated to the CUDA Toolkit,
# so I will leave it this way, but I'm documenting it here just in case.
#
# TODO: Torch libraries have to be force-linked since during runtime libnvrtc
# can't be found otherwise, it may only be needed to force-link libnvrtc (and
# maybe nvtoolsext?) but for now I force-link everything since it is easier
# althought a bit dirtier.

set(CUAEV_ROOT "${CMAKE_CURRENT_LIST_DIR}/submodules/torchani_sandbox/torchani/csrc")
# This is needed to safely use cub:
# https://github.com/pytorch/pytorch/pull/55292
# https://github.com/pytorch/pytorch/pull/66219
# TODO: These flags should not be global, but not sure how to do that
string(APPEND CMAKE_CUDA_FLAGS " -DCUB_NS_QUALIFIER=::cuaev::cub")
string(APPEND CMAKE_CUDA_FLAGS " -DCUB_NS_PREFIX='namespace cuaev {'")
string(APPEND CMAKE_CUDA_FLAGS " -DCUB_NS_POSTFIX=}")
add_library(
    cuaev
    SHARED
    "${CUAEV_ROOT}/cuaev.cpp"
    "${CUAEV_ROOT}/aev.cu"
    "${CUAEV_ROOT}/aev.h"
    "${CUAEV_ROOT}/cuaev_cub.cuh"
)
set_target_properties(
    cuaev
    PROPERTIES
        INSTALL_RPATH  "${TORCHANI_INSTALL_RPATH}"
        INSTALL_RPATH_USE_LINK_PATH TRUE
)
target_compile_features(cuaev PRIVATE cxx_std_17)
# TORCHANI_OPT and "-use_fast_math" make calculations faster in libcuaev.so.
# TORCHANI_OPT commands libcuaev to use cuda intrinsics for cos, sin, etc.
target_compile_definitions(cuaev PRIVATE TORCHANI_OPT)
target_compile_options(cuaev PRIVATE "-use_fast_math")
target_link_options(cuaev PRIVATE "LINKER:--no-as-needed")
target_link_directories(cuaev PRIVATE $ENV{CONDA_PREFIX}/lib)
target_link_libraries(cuaev PRIVATE ${TORCH_LIBRARIES} ${Python_LIBRARIES} nvrtc-builtins)
target_include_directories(cuaev PRIVATE ${CUAEV_ROOT} ${Python_INCLUDE_DIRS})

file(GLOB TORCHANI_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_LIST_DIR}/src/*.cpp")
add_library(torchani SHARED ${TORCHANI_SOURCES})
set_target_properties(
    torchani
    PROPERTIES
        INSTALL_RPATH  "${TORCHANI_INSTALL_RPATH}"
        INSTALL_RPATH_USE_LINK_PATH TRUE
)
# target_compile_options(torchani PRIVATE ${TORCH_CXX_FLAGS} $<$<CONFIG:Debug>:"-g">)
target_compile_options(
    torchani
    PRIVATE
        ${TORCH_CXX_FLAGS}
        -g
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
target_compile_features(torchani PRIVATE cxx_std_17)
target_compile_definitions(torchani PRIVATE $<$<CONFIG:Debug>:"DEBUG">)
target_link_options(torchani PRIVATE "LINKER:--no-as-needed")
target_link_directories(torchani PRIVATE $ENV{CONDA_PREFIX}/lib)
target_link_libraries(torchani PRIVATE ${TORCH_LIBRARIES} cuaev nvrtc-builtins)
target_include_directories(torchani PRIVATE "${CMAKE_CURRENT_LIST_DIR}/include" ${TORCH_INCLUDE_DIRS})

# Build Catch2 subproject (unit test framework) and Torchani unit tests
# TODO: Catch2 needs to be compiled with the same GLIBCXX_ABI
# as torch. Manually adding public "target_compile_options" to the Catch2
# liberaries is the dirty way I found to do this. There may be a better way to
# do it but I'm not sure how to otherwise propagate these flags to the
# subproject. Also, note that flags can't be set on alias targets (such as
# Catch2::Catch2WithMain), they have to be set on "Catch2WithMain" directly, and
# they have to be set on both, "Catch2" and "Catch2WithMain", since for some reason
# CMake doesn't propagate the flags between these libraries either.
include(CTest)
set(CATCH2_ROOT "${CMAKE_CURRENT_LIST_DIR}/submodules/Catch2")
add_subdirectory(${CATCH2_ROOT})
target_compile_options(Catch2WithMain PUBLIC ${TORCH_CXX_FLAGS})
target_compile_options(Catch2 PUBLIC ${TORCH_CXX_FLAGS})
# Make Catch2 provided cmake-modules available
list(APPEND CMAKE_MODULE_PATH "${CATCH2_ROOT}/extras" )
include(Catch)

add_executable(tests "${CMAKE_CURRENT_LIST_DIR}/tests/tests.cpp")
target_include_directories(tests PRIVATE "${CMAKE_CURRENT_LIST_DIR}/include")
target_compile_options(tests PUBLIC ${TORCH_CXX_FLAGS})
target_link_libraries(tests PRIVATE Catch2::Catch2WithMain torchani cuaev)
catch_discover_tests(tests)

# Execute target installation boilerplate

# Default CMake installation prefix for Linux is /usr/local. Its better to
# use ~/.local since it is typically user-writable in Linux systems (no sudo needed)
if(LOCAL_INSTALL)
    set(LOCAL_PREFIX /home/$ENV{USER}/.local)
else()
    set(LOCAL_PREFIX /usr/local)
endif()
set(CMAKE_INSTALL_PREFIX ${LOCAL_PREFIX} CACHE PATH "Installation prefix" FORCE)
message(STATUS "CMake - Install location initialized to: ${CMAKE_INSTALL_PREFIX}")

# Set dirs for project installation
# Note that by default in unix the rel_project(_install)_configdir are the same
set(REL_PROJECT_CONFIGDIR "share/cmake/${PROJECT_NAME}")  # local configdir
set(REL_PROJECT_INSTALL_CONFIGDIR "${CMAKE_INSTALL_DATAROOTDIR}/cmake/${PROJECT_NAME}")
set(REL_PROJECT_INSTALL_LIBDIR "${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME}")

# Set stems of generated and configured shareable installation files
set(PROJECT_CONFIG_STEM "${PROJECT_NAME}Config")
set(PROJECT_VERSION_STEM "${PROJECT_NAME}ConfigVersion")
set(PROJECT_TARGETS_STEM "${PROJECT_NAME}Targets")

# Set targets namespace
set(PROJECT_TARGETS_NAMESPACE "${PROJECT_NAME}::")

# Set PackageConfig file paths (input (source-tree) and output (binary-tree))
set(PROJECT_CONFIG_FNAME "${PROJECT_CONFIG_STEM}.cmake")
set(PROJECT_SOURCE_CONFIG_PATH "${CMAKE_SOURCE_DIR}/${REL_PROJECT_CONFIGDIR}/${PROJECT_CONFIG_FNAME}.in")
set(PROJECT_BINARY_CONFIG_PATH "${CMAKE_BINARY_DIR}/${REL_PROJECT_CONFIGDIR}/${PROJECT_CONFIG_FNAME}")

# Set PackageVersion file path (output (binary-tree) only, version file is generated on-the-fly)
set(PROJECT_VERSION_FNAME "${PROJECT_VERSION_STEM}.cmake")
set(PROJECT_BINARY_VERSION_PATH "${CMAKE_BINARY_DIR}/${REL_PROJECT_CONFIGDIR}/${PROJECT_VERSION_FNAME}")

# Set targets file path (output (binary-tree) only, targets file is generated on-the-fly)
set(PROJECT_TARGETS_FNAME "${PROJECT_TARGETS_STEM}.cmake")
set(PROJECT_BINARY_TARGETS_PATH "${CMAKE_BINARY_DIR}/${REL_PROJECT_INSTALL_CONFIGDIR}/${PROJECT_TARGETS_FNAME}")

# Create and install files for use by downstream packages
# - <PkgName>Config.cmake (configure_package_config_file)
# - <PkgName>ConfigVersion.cmake (write_basic_version_file)
# - <PkgName>Targets.cmake & <proj-name>Targets-*.cmake  (install(EXPORT ...))
# - <PkgName>Targets.cmake & <proj-name>Targets-*.cmake (build-tree) (export(EXPORT ...))

configure_package_config_file(
    "${PROJECT_SOURCE_CONFIG_PATH}"
    "${PROJECT_BINARY_CONFIG_PATH}"
    INSTALL_DESTINATION "${REL_PROJECT_INSTALL_CONFIGDIR}"
)
write_basic_package_version_file(
    "${PROJECT_BINARY_VERSION_PATH}"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY ExactVersion
)

install(
    EXPORT "${PROJECT_TARGETS_STEM}"
    FILE "${PROJECT_TARGETS_FNAME}"
    NAMESPACE "${PROJECT_TARGETS_NAMESPACE}"
    DESTINATION "${REL_PROJECT_INSTALL_CONFIGDIR}"
)

# Export is needed to consume the project from its build-tree, it will create a
# Targets file in the build-tree, not only in the install-prefix
export(
    EXPORT "${PROJECT_TARGETS_STEM}"
    FILE "${PROJECT_BINARY_TARGETS_PATH}"
    NAMESPACE "${PROJECT_TARGETS_NAMESPACE}"
)
install(
    FILES
        "${PROJECT_BINARY_CONFIG_PATH}"
        "${PROJECT_BINARY_VERSION_PATH}"
    DESTINATION "${REL_PROJECT_INSTALL_CONFIGDIR}"
)
install(
    TARGETS
        torchani
        cuaev
    EXPORT "${PROJECT_TARGETS_STEM}"
    LIBRARY
        DESTINATION "${REL_PROJECT_INSTALL_LIBDIR}"
)
