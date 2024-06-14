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

# Necessary CMake built-in modules
include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

# Dump build database into compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Make private internal modules accessible
set(PRIVATE_MODULES_DIR "${CMAKE_CURRENT_LIST_DIR}/share/cmake/${PROJECT_NAME}/private")
list(APPEND CMAKE_MODULE_PATH ${PRIVATE_MODULES_DIR})

# Internal modules
include(Msg)
include(LinuxBuild)
include(OutOfSourceBuild)
include(LibtorchHelper)
include(PytorchHelper)
include(CudnnHelper)
include(CudaToolkitHelper)
include(TorchScriptUtils)

OutOfSourceBuild_ensure()  # Fail with fatal error if out-of-source
LinuxBuild_ensure()  # Fail with fatal error if non-linux

# General: options and flags
option(DO_LOCAL_INSTALL "Install in ~/.local instead of /usr/local" ON)
set(COMPILE_WITH_CXX11ABI FALSE CACHE BOOL "Compile binaries with C++11 ABI. Requires 'USE_ACTIVE_CONDA_PYTORCH=OFF' and 'INSTALL_CUSTOM_LIBTORCH=ON'")

# TODO: These options are still confusing, it is probably better to use CMakeDependentOptions
# PyTorch/LibTorch and their backends: options and flags
set(REQUIRED_TORCH_VERSION "2.3" CACHE STRING "Request this version of PyTorch/LibTorch")
option(USE_ACTIVE_CONDA_PYTORCH "Use PyTorch from active conda environment" ON)
option(INSTALL_CUSTOM_LIBTORCH "Download LibTorch library and extract locally" OFF)
# The following options are only used if "USE_ACTIVE_CONDA_PYTORCH=OFF""
set(REQUIRED_TORCH_CUDA_VERSION "11.8" CACHE STRING "Request PyTorch/LibTorch binaries precompiled for this CUDA Toolkit version")
# The following options are only used if "USE_ACTIVE_CONDA_PYTORCH=OFF" and "INSTALL_CUSTOM_LIBTORCH=OFF"
set(CUSTOM_PYTORCH_CUDNN_VERSION "8.9.7" CACHE STRING "Request PyTorch binaries precompiled for this cuDNN version")
set(CUSTOM_PYTORCH_PYTHON_VERSION "3.10" CACHE STRING "Request PyTorch binaries precompiled for this Python version")
# Validate options
if(INSTALL_CUSTOM_LIBTORCH AND USE_ACTIVE_CONDA_PYTORCH)
    msg_error("'USE_ACTIVE_CONDA_PYTORCH=OFF' needed to install custom LibTorch")
endif()
if(COMPILE_WITH_CXX11ABI)
    if(USE_ACTIVE_CONDA_PYTORCH)
        msg_error("'USE_ACTIVE_CONDA_PYTORCH=OFF' needed to compile with the CXX11ABI")
    elseif(NOT INSTALL_CUSTOM_LIBTORCH)
        msg_error("'INSTALL_CUSTOM_LIBTORCH=ON' needed to compile with the CXX11ABI")
    endif()
else()
    message(STATUS "NOTE: All binaries will be compiled with the pre-C++11 ABI")
    message(STATUS "Linking to binaries compiled with the C++11 ABI will not be possible")
    message(STATUS "If you need this feature then reconfigure with the following options:")
    message(STATUS "-DUSE_ACTIVE_CONDA_PYTORCH=OFF -DINSTALL_CUSTOM_LIBTORCH=ON -DCOMPILE_WITH_CXX11ABI=ON")
endif()

# cuDNN: Options and flags
option(USE_ACTIVE_CONDA_CUDNN "Use cuDNN from active conda environment (if needed)" ON)
option(SELECT_CUSTOM_CUDNN "Explicitly specify a cuDNN library" OFF)
# The following options are only used if "USE_ACTIVE_CONDA_CUDNN=OFF" and "SELECT_CUSTOM_CUDNN=ON"
set(CUSTOM_CUDNN_VERSION "8.9.7" CACHE STRING "cuDNN version to download")
set(CUSTOM_CUDNN_CUDA_VERSION "11.8" CACHE STRING "CUDA version cuDNN is compatible with")
# Validate options
if(USE_ACTIVE_CONDA_CUDNN AND SELECT_CUSTOM_CUDNN)
    msg_error("'USE_ACTIVE_CONDA_CUDNN=OFF' needed to select a custom cuDNN")
endif()

# CUDA Toolkit: Options and flags
option(USE_ACTIVE_CONDA_CUDA_TOOLKIT "Use CUDA Toolkit from active conda environment" ON)
# The following options are only used if "USE_ACTIVE_CONDA_CUDA_TOOLKIT=OFF"
set(CUSTOM_CUDA_TOOLKIT_VERSION "11.8" CACHE STRING "CUDA Toolkit version used to set dirs")

# JIT: Options and flags
option(JIT_COMPILE_MODELS "JIT-compile all the TorchANI models" ON)
option(JIT_DISABLE_OPTIMIZATIONS "Disable TorchScript optimizations when JIT-compiling models" ON)
option(JIT_CUAEV "JIT-compile models with cuAEV extension" ON)
option(JIT_EXTERNAL_NEIGHBORLIST "JIT-compile models with external neighborlist" OFF)
option(JIT_FORCE_RECOMPILATION "Force JIT-recompilation of all models, even if jit-compiled *.pt files with the expected names are found" OFF)

# Use conda packages in current environment

# PyTorch/LibTorch setup
if(USE_ACTIVE_CONDA_PYTORCH)
    msg_info("Will use PyTorch from active Conda env")
    # Detect PyTorch's location using Python directly
    message(STATUS "Trying to import PyTorch and find its path")
    execute_process(
        COMMAND python -c "import torch; from pathlib import Path; print(Path(torch.__file__).resolve().parent, end='')"
        OUTPUT_VARIABLE ACTIVE_CONDA_TORCH_ROOT
        RESULT_VARIABLE CONDA_TORCH_FAILED
    )
    if(CONDA_TORCH_FAILED)
        msg_error("Could not find torch in the active Conda env")
    else()
        msg_success("Successfully found PyTorch in the active Conda env")
    endif()
    list(APPEND CMAKE_PREFIX_PATH $ENV{CONDA_PREFIX})
    list(APPEND CMAKE_PREFIX_PATH ${ACTIVE_CONDA_TORCH_ROOT})
elseif(INSTALL_CUSTOM_LIBTORCH)
    msg_info("Will download and install user-specified LibTorch")
    # NOTE: If the requested version already exists it is *not* extracted
    # Set vars LIBTORCH_ROOT, LIBTORCH_LIBRARY_DIR
    LibtorchHelper_download_and_install(
      CXX11ABI ${COMPILE_WITH_CXX11ABI}
      LIB_VERSION ${REQUIRED_TORCH_VERSION}
      CUDA_VERSION ${REQUIRED_TORCH_CUDA_VERSION}
    )
    list(APPEND CMAKE_PREFIX_PATH ${LIBTORCH_ROOT})
else()
    # Fallback to installing PyTorch with conda from cmake
    msg_info("Will download and install user-specified PyTorch using Conda")
    # Don't set any vars
    PytorchHelper_download_and_install(
        LIB_VERSION ${REQUIRED_TORCH_VERSION}
        PYTHON_VERSION ${CUSTOM_PYTORCH_PYTHON_VERSION}
        CUDNN_VERSION ${CUSTOM_PYTORCH_CUDNN_VERSION}
        CUDA_VERSION ${REQUIRED_TORCH_CUDA_VERSION}
    )
    # Detect PyTorch's location using Python directly
    execute_process(
        COMMAND python -c "import torch; from pathlib import Path; print(Path(torch.__file__).resolve().parent, end='')"
        OUTPUT_VARIABLE ACTIVE_CONDA_TORCH_ROOT
        RESULT_VARIABLE CONDA_TORCH_FAILED
    )
    if(CONDA_TORCH_FAILED)
        msg_error("Could not find torch in the active Conda env")
    endif()
    list(APPEND CMAKE_PREFIX_PATH $ENV{CONDA_PREFIX})
    list(APPEND CMAKE_PREFIX_PATH ${ACTIVE_CONDA_TORCH_ROOT})
endif()

# cuDNN setup
if(USE_ACTIVE_CONDA_CUDNN)
    msg_info("Will use cuDNN from active Conda env if needed")
    # Set vars CUDNN_ROOT, CUDNN_LIBRARY, CUDNN_INCLUDE_DIR
    CudnnHelper_set_conda_paths()
    list(APPEND CMAKE_PREFIX_PATH ${CUDNN_LIBRARY})
elseif(SELECT_CUSTOM_CUDNN)
    msg_info("Will use user-specified cuDNN if needed")
    msg_info("Will assume cuDNN is located in ./external/cudnn<version>-cuda<version>")
    # Set vars CUDNN_ROOT, CUDNN_LIBRARY, CUDNN_INCLUDE_DIR
    CudnnHelper_set_custom_paths(
        LIB_VERSION ${CUSTOM_CUDNN_VERSION}
        CUDA_VERSION ${CUSTOM_CUDNN_CUDA_VERSION}
    )
    list(APPEND CMAKE_PREFIX_PATH ${CUDNN_LIBRARY})
else()
    # cuDNN may be directly extracted into the CUDA Toolkit dir,
    # if a user has done this then they don't need to specify an external version
    msg_info("Will try to find a cuDNN bundled with the CUDA Toolkit if needed")
endif()

# CUDA Toolkit setup
if(USE_ACTIVE_CONDA_CUDA_TOOLKIT)
    msg_info("Will use CUDA Toolkit from active Conda env")
    # Set CUDA_TOOLKIT_ROOT_DIR and CMAKE_CUDA_COMPILER
    CudaToolkitHelper_set_conda_paths()
else()
    msg_info("Will use user-specified CUDA Toolkit")
    msg_info("Will assume CUDA Toolkit is located in /usr/local/cuda-<version>")
    # Set CUDA_TOOLKIT_ROOT_DIR and CMAKE_CUDA_COMPILER
    CudaToolkitHelper_set_custom_paths(TOOLKIT_VERSION ${CUDA_TOOLKIT_VERSION})
endif()

# TODO: Disabling JIT optimizations during scripting probably does nothing
if(JIT_COMPILE_MODELS)
    msg_info("Will attempt to JIT-compile ANI models")
    TorchScriptUtils_jit_compile_models(
        FORCE_RECOMPILATION ${JIT_FORCE_RECOMPILATION}
        DISABLE_OPTIMIZATIONS ${JIT_DISABLE_OPTIMIZATIONS}
        WITH_CUAEV ${JIT_COMPILE_WITH_CUAEV}
        WITH_EXTERNAL_NEIGHBORLIST ${JIT_COMPILE_WITH_EXTERNAL_NEIGHBORLIST}
    )
else()
    msg_warn("Skipping JIT-compilation of models, make sure to manually JIT-compile needed ANI models")
endif()

message(STATUS "The modified CMake prefix path is: ${CMAKE_PREFIX_PATH}")

# Default CMake installation prefix for Linux is /usr/local. Its better to
# use ~/.local since it is typically user-writable in Linux systems (no sudo needed)
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT AND DO_LOCAL_INSTALL)
    set(TORCHANI_LOCAL_PREFIX /home/$ENV{USER}/.local)
    set(CMAKE_INSTALL_PREFIX ${TORCHANI_LOCAL_PREFIX} CACHE PATH "Installation prefix" FORCE)
    # TODO: Is it necessary to set this again for local scope?
    set(CMAKE_INSTALL_PREFIX ${TORCHANI_LOCAL_PREFIX})
endif()
message(STATUS "Install location set to: ${CMAKE_INSTALL_PREFIX}")

# LibTorch is a required dependency
# Set TORCH_FOUND, TORCH_LIBRARIES, TORCH_INCLUDE_DIRS, TORCH_CXX_FLAGS
find_package(Torch ${REQUIRED_TORCH_VERSION} REQUIRED)

# The cuAEV extension lib is a required dep, and needs the Python headers and libraries
# Set Python_LIBRARIES, Python_INCLUDE_DIRS
find_package(Python REQUIRED COMPONENTS Interpreter Development)

# Now build both libcuaev and libtorchanis
#
# NOTE: In both cases the install rpath is hardcoded to be the same as the link path,
# to avoid issues if the correct CUDA is not installed system-wide. If this is
# not done then CMake tries to re-link the binaries at install-time, and
# prefers the system CUDA, which may be bad.
#
# NOTE: When doing this, both the build-tree and the install-tree binaries are
# still linked to the CUDA driver library from the system (libcuda.so) by
# default. This behavior seems to be a design choice of Conda, since envs don't
# have a "real" libcuda.so, just a "stub" libcuda.so. I belive this is ok,
# since the cuda driver is unrelated to the CUDA Toolkit, but I'm documenting
# it here for completeness.
#
# TODO: Torch libraries have to be force-linked since during runtime libnvrtc
# can't be found otherwise, it may only be needed to force-link libnvrtc (and
# maybe nvtoolsext?) but for now I force-link everything since it is easier
# althought a bit dirtier.
set(TORCHANI_LIBRARIES_INSTALL_RPATH
    ${LIBTORCH_LIBRARY_DIR}
    ${CUDA_TOOLKIT_ROOT_DIR}
    ${CMAKE_INSTALL_FULL_LIBDIR}/Torchani
)

set(CUAEV_ROOT "${CMAKE_CURRENT_LIST_DIR}/submodules/torchani_sandbox/torchani/csrc")
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
        INSTALL_RPATH  "${TORCHANI_LIBRARIES_INSTALL_RPATH}"
        INSTALL_RPATH_USE_LINK_PATH TRUE
)
target_compile_features(cuaev PRIVATE cxx_std_17)
# TORCHANI_OPT and "-use_fast_math" make calculations faster in libcuaev.so.
# TORCHANI_OPT commands libcuaev to use cuda intrinsics for cos, sin, etc.
# "-use_fast_math" commands nvcc to use the fast math library
# The wrapped cub namespace flags are needed to safely use cub:
# https://github.com/pytorch/pytorch/pull/55292
# https://github.com/pytorch/pytorch/pull/66219
set(CUAEV_CUB_NAMESPACE_DEFS "CUB_NS_QUALIFIER=::cuaev::cub" "CUB_NS_PREFIX=namespace cuaev {" "CUB_NS_POSTFIX=}")
target_compile_definitions(
    cuaev
    PUBLIC ${TORCH_CXX_FLAGS}
    PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:TORCHANI_OPT>
        $<$<COMPILE_LANGUAGE:CUDA>:${CUAEV_CUB_NAMESPACE_DEFS}>
)
target_compile_options(cuaev PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math>)
target_link_options(cuaev PRIVATE "LINKER:--no-as-needed")
target_link_directories(cuaev PRIVATE $ENV{CONDA_PREFIX}/lib)
target_link_libraries(cuaev PRIVATE ${TORCH_LIBRARIES} ${Python_LIBRARIES} nvrtc-builtins)
target_include_directories(cuaev PRIVATE ${CUAEV_ROOT} ${Python_INCLUDE_DIRS})

file(GLOB TORCHANI_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_LIST_DIR}/src/*.cpp")
add_library(torchani SHARED ${TORCHANI_SOURCES})
set_target_properties(
    torchani
    PROPERTIES
        INSTALL_RPATH  "${TORCHANI_LIBRARIES_INSTALL_RPATH}"
        INSTALL_RPATH_USE_LINK_PATH TRUE
)
target_compile_options(
    torchani
    PRIVATE
        $<$<CONFIG:Debug>:"-g">
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
target_compile_definitions(
    torchani
        PUBLIC ${TORCH_CXX_FLAGS}
        PRIVATE $<$<CONFIG:Debug>:"DEBUG">
)
target_link_options(torchani PRIVATE "LINKER:--no-as-needed")
target_link_directories(torchani PRIVATE $ENV{CONDA_PREFIX}/lib)
target_link_libraries(torchani PRIVATE ${TORCH_LIBRARIES} cuaev nvrtc-builtins)
target_include_directories(torchani PRIVATE "${CMAKE_CURRENT_LIST_DIR}/include" ${TORCH_INCLUDE_DIRS})

# Build Catch2 subproject (unit test framework) and Torchani unit tests
#
# NOTE: Catch2 needs to be compiled with the same GLIBCXX_ABI
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
target_compile_definitions(Catch2WithMain PUBLIC ${TORCH_CXX_FLAGS})
target_compile_definitions(Catch2 PUBLIC ${TORCH_CXX_FLAGS})
# Make Catch2 provided cmake-modules available
list(APPEND CMAKE_MODULE_PATH "${CATCH2_ROOT}/extras" )
include(Catch)

add_executable(tests "${CMAKE_CURRENT_LIST_DIR}/tests/tests.cpp")
target_include_directories(tests PRIVATE "${CMAKE_CURRENT_LIST_DIR}/include")
target_compile_definitions(tests PUBLIC ${TORCH_CXX_FLAGS})
target_link_libraries(tests PRIVATE Catch2::Catch2WithMain torchani)
catch_discover_tests(tests)

# Set boilerplate variables needed for target installation


# Set dirs for project installation
# Note that by default in unix the project_relative(_install)_configdir are the same
set(PROJECT_RELATIVE_CONFIGDIR "share/cmake/${PROJECT_NAME}")  # local configdir
set(PROJECT_RELATIVE_INSTALL_CONFIGDIR "${CMAKE_INSTALL_DATAROOTDIR}/cmake/${PROJECT_NAME}")
set(PROJECT_RELATIVE_INSTALL_LIBDIR "${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME}")

# Set stems of generated and configured shareable installation files
set(PROJECT_CONFIG_STEM "${PROJECT_NAME}Config")
set(PROJECT_VERSION_STEM "${PROJECT_NAME}ConfigVersion")
set(PROJECT_TARGETS_STEM "${PROJECT_NAME}Targets")

# Set targets namespace
set(PROJECT_TARGETS_NAMESPACE "${PROJECT_NAME}::")

# Set PackageConfig file paths (input (source-tree) and output (binary-tree))
set(PROJECT_CONFIG_FNAME "${PROJECT_CONFIG_STEM}.cmake")
set(PROJECT_SOURCE_CONFIG_PATH "${CMAKE_SOURCE_DIR}/${PROJECT_RELATIVE_CONFIGDIR}/${PROJECT_CONFIG_FNAME}.in")
set(PROJECT_BINARY_CONFIG_PATH "${CMAKE_BINARY_DIR}/${PROJECT_RELATIVE_CONFIGDIR}/${PROJECT_CONFIG_FNAME}")

# Set PackageVersion file path (output (binary-tree) only, version file is generated on-the-fly)
set(PROJECT_VERSION_FNAME "${PROJECT_VERSION_STEM}.cmake")
set(PROJECT_BINARY_VERSION_PATH "${CMAKE_BINARY_DIR}/${PROJECT_RELATIVE_CONFIGDIR}/${PROJECT_VERSION_FNAME}")

# Set targets file path (output (binary-tree) only, targets file is generated on-the-fly)
set(PROJECT_TARGETS_FNAME "${PROJECT_TARGETS_STEM}.cmake")
set(PROJECT_BINARY_TARGETS_PATH "${CMAKE_BINARY_DIR}/${PROJECT_RELATIVE_INSTALL_CONFIGDIR}/${PROJECT_TARGETS_FNAME}")

# Create and install files for use by downstream packages
# - <PkgName>Config.cmake (configure_package_config_file)
# - <PkgName>ConfigVersion.cmake (write_basic_version_file)
# - <PkgName>Targets.cmake & <proj-name>Targets-*.cmake  (install(EXPORT ...))
# - <PkgName>Targets.cmake & <proj-name>Targets-*.cmake (build-tree) (export(EXPORT ...))

configure_package_config_file(
    "${PROJECT_SOURCE_CONFIG_PATH}"
    "${PROJECT_BINARY_CONFIG_PATH}"
    INSTALL_DESTINATION "${PROJECT_RELATIVE_INSTALL_CONFIGDIR}"
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
    DESTINATION "${PROJECT_RELATIVE_INSTALL_CONFIGDIR}"
)

# Export is needed to consume the project from its build-tree, it creates
# Targets files in the build-tree, not only in the install-prefix
export(
    EXPORT "${PROJECT_TARGETS_STEM}"
    FILE "${PROJECT_BINARY_TARGETS_PATH}"
    NAMESPACE "${PROJECT_TARGETS_NAMESPACE}"
)
install(
    FILES
        "${PROJECT_BINARY_CONFIG_PATH}"
        "${PROJECT_BINARY_VERSION_PATH}"
    DESTINATION "${PROJECT_RELATIVE_INSTALL_CONFIGDIR}"
)
install(
    TARGETS
        torchani
        cuaev
    EXPORT "${PROJECT_TARGETS_STEM}"
    LIBRARY
        DESTINATION "${PROJECT_RELATIVE_INSTALL_LIBDIR}"
)
