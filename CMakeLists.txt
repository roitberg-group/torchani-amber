cmake_minimum_required(
    VERSION 3.24
    FATAL_ERROR
)

project(
    Torchani
    LANGUAGES C CXX
    VERSION 0.3
    DESCRIPTION "C++ wrapper to execute ANI-Style models from TorchANI"
    HOMEPAGE_URL "https://github.com/roitberg-group/torchani-amber.git"
)

# Necessary CMake built-in modules
include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

# Make private internal modules available
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

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
set(COMPILE_WITH_CXX11ABI FALSE CACHE BOOL "Compile binaries with C++11 ABI. Requires 'USE_ACTIVE_CONDA_PYTORCH=OFF' and 'INSTALL_CUSTOM_LIBTORCH=ON'")

# TODO: These options are still confusing, it is probably better to use CMakeDependentOptions
# PyTorch/LibTorch and their backends: options and flags
set(REQUIRED_TORCH_VERSION "2.5" CACHE STRING "Request this version of PyTorch/LibTorch")
option(USE_ACTIVE_CONDA_PYTORCH "Use PyTorch from active conda environment" ON)
option(INSTALL_CUSTOM_LIBTORCH "Download LibTorch library and extract locally" OFF)
option(SET_TORCHANI_LIBRARIES_INSTALL_RPATH "Hardcode the RPATH used during build-time for the installed libraries" ON)
# The following options are only used if "USE_ACTIVE_CONDA_PYTORCH=OFF""
set(REQUIRED_TORCH_CUDA_VERSION "11.8" CACHE STRING "Request PyTorch/LibTorch binaries precompiled for this CUDA Toolkit version")
# The following options are only used if "USE_ACTIVE_CONDA_PYTORCH=OFF" and "INSTALL_CUSTOM_LIBTORCH=OFF"
set(CUSTOM_PYTORCH_CUDNN_VERSION "8.7.0" CACHE STRING "Request PyTorch binaries precompiled for this cuDNN version")
set(CUSTOM_PYTORCH_PYTHON_VERSION "3.10" CACHE STRING "Request PyTorch binaries precompiled for this Python version")

# Validate options
if(INSTALL_CUSTOM_LIBTORCH AND USE_ACTIVE_CONDA_PYTORCH)
    msg_error("'USE_ACTIVE_CONDA_PYTORCH=OFF' needed to install custom LibTorch")
endif()

if(COMPILE_WITH_CXX11ABI AND (USE_ACTIVE_CONDA_PYTORCH OR (NOT INSTALL_CUSTOM_LIBTORCH)))
    msg_error("'INSTALL_CUSTOM_LIBTORCH=ON' and 'USE_ACTIVE_CONDA_PYTORCH=OFF' needed to compile with the CXX11ABI")
elseif(NOT COMPILE_WITH_CXX11ABI)
    message(STATUS "NOTE: All binaries will be compiled with the pre-C++11 ABI")
    message(STATUS "Linking to binaries compiled with the C++11 ABI will not be possible")
    message(STATUS "If you need this feature then reconfigure with the following options:")
    message(STATUS "-DUSE_ACTIVE_CONDA_PYTORCH=OFF -DINSTALL_CUSTOM_LIBTORCH=ON -DCOMPILE_WITH_CXX11ABI=ON")
endif()

# cuDNN: Options and flags
option(USE_ACTIVE_CONDA_CUDNN "Use cuDNN from active conda environment (if needed)" ON)
option(SELECT_CUSTOM_CUDNN "Explicitly specify a cuDNN library" OFF)
# The following options are only used if "USE_ACTIVE_CONDA_CUDNN=OFF" and "SELECT_CUSTOM_CUDNN=ON"
set(CUSTOM_CUDNN_VERSION "8.7.0" CACHE STRING "cuDNN version to download")
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
option(JIT_FORCE_RECOMPILATION "Force JIT-recompilation of all models, even if jit-compiled *.pt files with the expected names are found" OFF)

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
    )
else()
    msg_warn("Skipping JIT-compilation of models, make sure to manually JIT-compile needed ANI models")
endif()

message(STATUS "The modified CMake prefix path is: ${CMAKE_PREFIX_PATH}")

# LibTorch is a required dep
# Set TORCH_FOUND, TORCH_LIBRARIES, TORCH_INCLUDE_DIRS, TORCH_CXX_FLAGS
find_package(Torch ${REQUIRED_TORCH_VERSION} REQUIRED)

# The cuAEV extension lib is a required dep, and needs the Python headers and libraries
# Set Python_LIBRARIES, Python_INCLUDE_DIRS
find_package(Python REQUIRED COMPONENTS Interpreter Development)

# Build both libcuaev and libtorchani
set(CUAEV_ROOT "${CMAKE_CURRENT_LIST_DIR}/submodules/torchani_sandbox/torchani/csrc")
add_library(
    cuaev
    SHARED
    "${CUAEV_ROOT}/cuaev.cpp"
    "${CUAEV_ROOT}/aev.cu"
    "${CUAEV_ROOT}/aev.h"
    "${CUAEV_ROOT}/cuaev_cub.cuh"
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
target_link_options(torchani PRIVATE "LINKER:--no-as-needed" "LINKER:-force_load,$<TARGET_FILE:cuaev>")
target_link_directories(torchani PRIVATE $ENV{CONDA_PREFIX}/lib)
target_link_libraries(torchani PRIVATE ${TORCH_LIBRARIES} cuaev nvrtc-builtins)
target_include_directories(torchani PRIVATE "${CMAKE_CURRENT_LIST_DIR}/include" ${TORCH_INCLUDE_DIRS})

# For both libcuaev and libtorchani, the install RPATH is set by default to be
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
# TODO: Torch libraries have to be force-linked since during runtime libnvrtc
# can't be found otherwise, it may only be needed to force-link libnvrtc (and
# maybe nvtoolsext?) but for now I force-link everything since it is easier
# althought a bit dirtier.
if(SET_TORCHANI_LIBRARIES_INSTALL_RPATH)
    set(TORCHANI_LIBRARIES_INSTALL_RPATH
        ${LIBTORCH_LIBRARY_DIR}
        ${CUDA_TOOLKIT_ROOT_DIR}
        ${CMAKE_INSTALL_FULL_LIBDIR}/Torchani
    )
    set_target_properties(
        cuaev
        PROPERTIES
            INSTALL_RPATH  "${TORCHANI_LIBRARIES_INSTALL_RPATH}"
            INSTALL_RPATH_USE_LINK_PATH TRUE
    )
    set_target_properties(
        torchani
        PROPERTIES
            INSTALL_RPATH  "${TORCHANI_LIBRARIES_INSTALL_RPATH}"
            INSTALL_RPATH_USE_LINK_PATH TRUE
    )
endif()

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
target_link_options(tests PRIVATE "LINKER:--no-as-needed")
target_link_libraries(tests PRIVATE Catch2::Catch2WithMain torchani cuaev)
catch_discover_tests(tests)

# Create and install files for discovery and config by downstream pkgs
# - pkg-config-file: <Pkg>Config.cmake
# - pkg-ver-file: <Pkg>ConfigVersion.cmake
set(PKG_VERSION_FILE_BLDTREE_PATH "${CMAKE_BINARY_DIR}/cmake/${PROJECT_NAME}ConfigVersion.cmake")
write_basic_package_version_file(
    "${PKG_VERSION_FILE_BLDTREE_PATH}"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMinorVersion
)
set(PKG_CONFIG_FILE_BLDTREE_PATH "${CMAKE_BINARY_DIR}/cmake/${PROJECT_NAME}Config.cmake")

# *_INSTALL_DIR are all relative to the CMAKE_INSTALL_PREFIX
set(PKGDATA_INSTALL_DIR "${CMAKE_INSTALL_DATAROOTDIR}/cmake/${PROJECT_NAME}")
set(INCLUDE_INSTALL_DIR "${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}")
set(DATA_INSTALL_DIR "${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}")
set(LIBRARY_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME}")
set(FORTRAN_INTERFACE_INSTALL_SRC_FILE "${DATA_INSTALL_DIR}/Fortran/torchani.F90")
configure_package_config_file(
    "${CMAKE_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in"
    "${PKG_CONFIG_FILE_BLDTREE_PATH}"
    INSTALL_DESTINATION "${PKGDATA_INSTALL_DIR}"
    PATH_VARS
        INCLUDE_INSTALL_DIR
        DATA_INSTALL_DIR
        LIBRARY_INSTALL_DIR
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
set(PKG_EXPORT_FILESET_NAME "${PROJECT_NAME}Targets")
install(
    EXPORT ${PKG_EXPORT_FILESET_NAME}
    FILE "${PROJECT_NAME}Targets.cmake"  # Primary filename in the pkg-export-fileset
    NAMESPACE "${PROJECT_NAME}::"  # Exported targets namespace
    DESTINATION "${PKGDATA_INSTALL_DIR}"
)

# Install corresponding targets
install(
    TARGETS
        torchani
        cuaev
    EXPORT ${PKG_EXPORT_FILESET_NAME}  # Add ref to targs in the pkg-export-fileset
    LIBRARY
        DESTINATION "${LIBRARY_INSTALL_DIR}"
)
# Install C-interface include file
install(
    FILES
        "${CMAKE_SOURCE_DIR}/include/torchani.h"
    DESTINATION "${INCLUDE_INSTALL_DIR}"
)

# Install module file necessary for a portable fortran interface
install(
    FILES
        "${CMAKE_SOURCE_DIR}/src-fortran/torchani.F90"
    DESTINATION "${DATA_INSTALL_DIR}/Fortran"
)
# TODO Add model.jit.pt files
