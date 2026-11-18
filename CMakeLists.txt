cmake_minimum_required(
    VERSION 3.24
    FATAL_ERROR
)

project(
    Torchani
    LANGUAGES C CXX
    VERSION 0.7
    DESCRIPTION "Fortran, C and C++ interface to execute ANI-Style models built using the TorchANI library"
    HOMEPAGE_URL "https://github.com/roitberg-group/torchani-amber.git"
)

# Necessary CMake built-in modules
include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

# Make private internal modules available
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

# Internal modules
include(Torchani_Msg)
include(Torchani_LinuxBuild)
include(Torchani_OutOfSourceBuild)
include(Torchani_LibtorchHelper)
include(Torchani_PytorchHelper)
include(Torchani_CudnnHelper)
include(Torchani_CudaToolkitHelper)
include(Torchani_TorchScriptUtils)

Torchani_OutOfSourceBuild_ensure()  # Fail with fatal error if out-of-source
Torchani_LinuxBuild_ensure()  # Fail with fatal error if non-linux

# TODO: These options are still confusing, it is probably better to use CMakeDependentOptions
# PyTorch/LibTorch and their backends: options and flags

# NOTE: Libtorch installs require extra libraries to be present, that are *not* in the
# plain toolkit:
# libcudnn.so, libnccl.so, libcusparseLt.so, libnvshmem_host.so
# Without those installed, libtorch fails at runtime
# The pytorch path is simpler, since the pytorch wheel pulls the required nvidia
# wheels and toolkit libraries directly
# Neigher of these includes nvcc or CUDA headers, these are needed separately
# This means using libtorch is a pain, and is probably best avoided

# General: options and flags
option(TORCHANI_LINK_TO_PYTORCH_LIBS "Try to find and link to a local PyTorch installation" ON)
option(TORCHANI_SET_LIBRARIES_INSTALL_RPATH "Hardcode the RPATH used during build-time for the installed libs ($ORIGIN is always added)" ON)
option(TORCHANI_DOWNLOAD_AND_INSTALL_LIBTORCH "Download LibTorch library and extract to TORCHANI_LIBTORCH_INSTALL_DIR/libtorch<version>-cuda<version>" OFF)
set(TORCHANI_LIBTORCH_INSTALL_DIR "${CMAKE_CURRENT_LIST_DIR}/external" CACHE PATH "Dir where LibTorch is downloaded and extracted, if a download is requested")
option(TORCHANI_PREFER_CONDA_LIBS "Prefer libraries under $CONDA_PREFIX (the active conda env)" ON)
option(TORCHANI_USE_CUSTOM_CUDNN "Try to find cuDNN under ./external/cudnn<version>-cuda<version> if needed" OFF)
option(TORCHANI_USE_CUSTOM_CUDATK "Try to find CUDA Toolkit under /usr/local/cuda-<version> if needed" OFF)
option(TORCHANI_ALLOW_UNSUPPORTED_COMPILERS "Allow compiling with a compiler deemed not supported by nvcc" OFF)
option(TORCHANI_CHECK_TORCH_AND_CUDA_VERSIONS "Check the required versions for Torch and CUDA" OFF)

# Exact major.minor versions must match for these
# The libtorch patch version is required since the download url contains the patch
# version. It is not checked afterwards
set(TORCHANI_REQUIRED_TORCH_VERSION "2.13" CACHE STRING "Require this version of PyTorch/LibTorch")
set(TORCHANI_REQUIRED_CUDATK_VERSION "13.0" CACHE STRING "Require this version of the CUDA Toolkit")
set(TORCHANI_REQUIRED_CUDNN_VERSION "9" CACHE STRING "Require this version of the cuDNN library")
set(TORCHANI_REQUIRED_LIBTORCH_PATCH_VERSION "0" CACHE STRING "LibTorch patch version. Only used if a LibTorch download is requested")

# Validate options
if(TORCHANI_LINK_TO_PYTORCH_LIBS AND TORCHANI_DOWNLOAD_AND_INSTALL_LIBTORCH)
    Torchani_Msg_error("'TORCHANI_LINK_TO_PYTORCH_LIBS' is incompatible with 'TORCHANI_DOWNLOAD_AND_INSTALL_LIBTORCH'")
endif()

if("$ENV{CONDA_DEFAULT_ENV}" STREQUAL "base")
    Torchani_Msg_warn("You are building from the conda 'base' env")
endif()

set(USE_CONDA_LIBS OFF)
if(NOT "$ENV{CONDA_PREFIX}" STREQUAL "")
    if(TORCHANI_PREFER_CONDA_LIBS)
        Torchani_Msg_info("You are running inside a conda env with prefix $ENV{CONDA_PREFIX}")
        Torchani_Msg_info("Libs will be searched first under this prefix")
        Torchani_Msg_info("If you don't want this, reconfigure with -DTORCHANI_PREFER_CONDA_LIBS=OFF")
        list(APPEND CMAKE_PREFIX_PATH $ENV{CONDA_PREFIX})
        # Workaround since torch 2.8 can't find cuda installed in a conda env otherwise
        # TODO: x86_64-linux is hardcoded, on aarch64 conda uses targets/sbsa-linux
        set(ENV{CUDA_INC_PATH} $ENV{CONDA_PREFIX}/targets/x86_64-linux/include)
        set(USE_CONDA_LIBS ON)
    else()
        Torchani_Msg_warn("You seem to be installing inside a conda env but with TORCHANI_PREFER_CONDA_LIBS=OFF")
        Torchani_Msg_warn("This is probably not what you want, and it is *HIGHLY RECOMMENDED* to avoid this")
    endif()
endif()

if(USE_CONDA_LIBS AND (TORCHANI_USE_CUSTOM_CUDNN OR TORCHANI_USE_CUSTOM_CUDATK))
    Torchani_Msg_error("'TORCHANI_PREFER_CONDA_LIBS' is incompatible with TORCHANI_USE_CUSTOM_CUDNN or TORCHANI_USE_CUSTOM_CUDATK")
endif()

# Validate compilers
if(NOT TORCHANI_ALLOW_UNSUPPORTED_COMPILERS)
    set(COMPILER_IDS ${CMAKE_CXX_COMPILER_ID} ${CMAKE_C_COMPILER_ID} ${CMAKE_Fortran_COMPILER_ID})
    foreach(COMPILER_ID ${COMPILER_IDS})
        if(NOT (COMPILER_ID STREQUAL GNU))
            Torchani_Msg_error("TorchANI-Interface requires GNU compilers (g++, gcc, gfortran)")
        endif()
    endforeach()
    set(COMPILER_VERSIONS ${CMAKE_CXX_COMPILER_VERSION} ${CMAKE_C_COMPILER_VERSION} ${CMAKE_Fortran_COMPILER_VERSION})
    foreach(COMPILER_VERSION ${COMPILER_VERSIONS})
        if(COMPILER_VERSION VERSION_LESS 11.0 OR COMPILER_VERSION VERSION_GREATER_EQUAL 16.0)
            Torchani_Msg_error("TorchANI-Interface requires GNU compilers with version >=11 and <16")
        endif()
    endforeach()
endif()

# JIT: Options and flags
option(TORCHANI_JIT_COMPILE_MODELS "JIT-compile all the TorchANI models" ON)
option(TORCHANI_JIT_DISABLE_OPTIMIZATIONS "Disable TorchScript optimizations when JIT-compiling models" ON)
option(TORCHANI_JIT_FORCE_RECOMPILATION "Force JIT-recompilation of all models, even if jit-compiled *.pt files with the expected names are found" OFF)
option(TORCHANI_JIT_COMPILE_AIMNET2 "Also generate the AimNet2 models (jit/aimnet2) from the bundled weights, if not already there" ON)

if(TORCHANI_JIT_COMPILE_MODELS AND NOT TORCHANI_LINK_TO_PYTORCH_LIBS)
    Torchani_Msg_error("'TORCHANI_JIT_COMPILE_MODELS' requires 'TORCHANI_LINK_TO_PYTORCH_LIBS'")
endif()

# PyTorch/LibTorch setup
if(TORCHANI_LINK_TO_PYTORCH_LIBS)
    Torchani_Msg_info("Will try to find and link to an existing PyTorch installation")
    # Set var PYTORCH_ROOT
    Torchani_PytorchHelper_detect(
        CHECK_VERSIONS ${TORCHANI_CHECK_TORCH_AND_CUDA_VERSIONS}
        LIB_VERSION ${TORCHANI_REQUIRED_TORCH_VERSION}
        CUDATK_VERSION ${TORCHANI_REQUIRED_CUDATK_VERSION}
        CUDNN_VERSION ${TORCHANI_REQUIRED_CUDNN_VERSION}
    )
    list(APPEND CMAKE_PREFIX_PATH ${PYTORCH_ROOT})
elseif(TORCHANI_DOWNLOAD_AND_INSTALL_LIBTORCH)
    Torchani_Msg_info("Will download and install user-specified LibTorch")
    # NOTE: If the requested version already exists it is *not* extracted
    # Set vars LIBTORCH_ROOT, LIBTORCH_LIBRARY_DIR
    Torchani_LibtorchHelper_download_and_install(
        LIB_VERSION ${TORCHANI_REQUIRED_TORCH_VERSION}
        PATCH_VERSION ${TORCHANI_REQUIRED_LIBTORCH_PATCH_VERSION}
        CUDATK_VERSION ${TORCHANI_REQUIRED_CUDATK_VERSION}
        INSTALL_DIR "${TORCHANI_LIBTORCH_INSTALL_DIR}"
    )
    list(APPEND CMAKE_PREFIX_PATH ${LIBTORCH_ROOT})
endif()

# cuDNN setup
if(USE_CONDA_LIBS)
    Torchani_Msg_info("Will try to use cuDNN from active Conda env if needed")
    # Set vars CUDNN_ROOT, CUDNN_LIBRARY, CUDNN_INCLUDE_DIR
    Torchani_CudnnHelper_set_conda_paths()
    list(APPEND CMAKE_PREFIX_PATH ${CUDNN_ROOT})
elseif(TORCHANI_USE_CUSTOM_CUDNN)
    Torchani_Msg_info("Will manually set variables that point to cuDNN")
    Torchani_Msg_info("assuming it is located in ./external/cudnn<version>-cuda<version>")
    # Set vars CUDNN_ROOT, CUDNN_LIBRARY, CUDNN_INCLUDE_DIR
    Torchani_CudnnHelper_set_custom_paths(
        LIB_VERSION ${TORCHANI_REQUIRED_CUDNN_VERSION}
        CUDATK_VERSION ${TORCHANI_REQUIRED_CUDATK_VERSION}
    )
    list(APPEND CMAKE_PREFIX_PATH ${CUDNN_ROOT})
else()
    # cuDNN may be directly extracted into the CUDA Toolkit dir,
    # if a user has done this then they don't need to specify an external version
    Torchani_Msg_info("Will assume cuDNN, if needed, is located in the same dir as the CUDA Toolkit")
endif()

# CUDA Toolkit setup
if(USE_CONDA_LIBS)
    Torchani_Msg_info("Will try to use CUDA Toolkit from the active Conda env")
    # Set CUDA_TOOLKIT_ROOT_DIR and CMAKE_CUDA_COMPILER
    Torchani_CudaToolkitHelper_set_conda_paths()
    Torchani_Msg_banner(
        "TorchANI requires CUDA Toolkit dev version, which is *not* bundled with PyTorch.
        At this stage of the configuration, we don't check whether all necessary CUDA
        Toolkit features are present.
        Make sure you have the required headers, NVCC, and *all* CUDA Toolkit libs. To
        ensure this, for example, use the provided conda env, which has all
        dependencies, or directly download and install the CUDA Toolkit
        ${TORCHANI_REQUIRED_CUDATK_VERSION} from Nvidia's web page. Failures related to CUDA
        after this point may indicate you are missing a required part of the Toolkit."
    )
elseif(TORCHANI_USE_CUSTOM_CUDATK)
    Torchani_Msg_info("Will manually set variables that point to the CUDA Toolkit")
    Torchani_Msg_info("assuming it is located in /usr/local/cuda-<version>")
    # Set CUDA_TOOLKIT_ROOT_DIR and CMAKE_CUDA_COMPILER
    Torchani_CudaToolkitHelper_set_custom_paths(CUDATK_VERSION ${TORCHANI_REQUIRED_CUDATK_VERSION})
else()
    Torchani_Msg_info("Will follow LibTorch's default lookup procedure for the CUDA Toolkit")
    set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
endif()
enable_language(CUDA)

message(STATUS "The modified CMake prefix path is: ${CMAKE_PREFIX_PATH}")

# LibTorch is a required dep
# Set TORCH_FOUND, TORCH_LIBRARIES, TORCH_INCLUDE_DIRS
Torchani_Msg_banner(
    "CMake warnings about 'kineto not being found', or 'CMAKE_CUDA_ARCHITECTURES being
    incompatible with pytorch' are safe to ignore. Other warnings should not be ignored."
)
find_package(Torch ${TORCHANI_REQUIRED_TORCH_VERSION} REQUIRED)

# Build both libcuaev, libcell_list, and libtorchani
set(CELL_ROOT "${CMAKE_CURRENT_LIST_DIR}/submodules/torchani/torchani/csrc")
add_library(
    torchani_cell_list
    SHARED
    "${CELL_ROOT}/cell_list.cpp"
)
target_compile_features(torchani_cell_list PRIVATE cxx_std_17)
if(USE_CONDA_LIBS)
    target_link_directories(torchani_cell_list PRIVATE $ENV{CONDA_PREFIX}/lib)
endif()
target_link_libraries(torchani_cell_list PRIVATE ${TORCH_LIBRARIES})
target_include_directories(torchani_cell_list PRIVATE ${CELL_ROOT})

set(CUAEV_ROOT "${CMAKE_CURRENT_LIST_DIR}/submodules/torchani/torchani/csrc")
add_library(
    torchani_cuaev
    SHARED
    "${CUAEV_ROOT}/cuaev.cpp"
    "${CUAEV_ROOT}/aev.cu"
    "${CUAEV_ROOT}/aev_cpu.cpp"
    "${CUAEV_ROOT}/aev.h"
    "${CUAEV_ROOT}/cuaev_cub.cuh"
)
target_compile_features(torchani_cuaev PRIVATE cxx_std_17)
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
if(USE_CONDA_LIBS)
    target_link_directories(torchani_cuaev PRIVATE $ENV{CONDA_PREFIX}/lib)
endif()
target_link_libraries(torchani_cuaev PRIVATE ${TORCH_LIBRARIES})
target_include_directories(torchani_cuaev PRIVATE ${CUAEV_ROOT})

# The extensions are named as the TorchANI Python package expects them (cuaev.so,
# cell_list.so). After installation, the TorchANI Python package finds them in
# <prefix>/lib/Torchani, so they don't need to be built a second time by pip or
# "ani build-extensions"
set_target_properties(torchani_cuaev torchani_cell_list PROPERTIES PREFIX "")
set_target_properties(torchani_cuaev PROPERTIES OUTPUT_NAME cuaev EXPORT_NAME cuaev)
set_target_properties(torchani_cell_list PROPERTIES OUTPUT_NAME cell_list EXPORT_NAME cell_list)

file(GLOB TORCHANI_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_LIST_DIR}/src/*.cpp")
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
target_compile_features(torchani PRIVATE cxx_std_17)
target_compile_definitions(
    torchani
    PRIVATE
        $<$<CONFIG:Debug>:DEBUG>
)
# cuaev and cell_list must stay in DT_NEEDED even though libtorchani references none of
# their symbols: loading them is what registers their Torch ops. --no-as-needed is
# scoped to these two libs since conda's LDFLAGS append a global --as-needed
set(CMAKE_LINK_LIBRARY_USING_NO_AS_NEEDED
    "LINKER:--push-state,--no-as-needed"
    "<LINK_ITEM>"
    "LINKER:--pop-state"
)
set(CMAKE_LINK_LIBRARY_USING_NO_AS_NEEDED_SUPPORTED TRUE)
if(USE_CONDA_LIBS)
    target_link_directories(torchani PRIVATE $ENV{CONDA_PREFIX}/lib)
endif()
target_link_libraries(
    torchani
    PRIVATE
        ${TORCH_LIBRARIES}
        "$<LINK_LIBRARY:NO_AS_NEEDED,torchani_cuaev,torchani_cell_list>"
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
# (LIBTORCH_LIBRARY_DIR and the Torchani installation dir, which is $ORIGIN).
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
        ${LIBTORCH_LIBRARY_DIR}
        ${CUDART_LIBRARY_DIR}
        ${NVRTC_LIBRARY_DIR}
    )
    list(REMOVE_DUPLICATES TORCHANI_LIBRARIES_INSTALL_RPATH)
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
# NOTE: Disabling JIT optimizations during scripting has been confirmed to be a no-op
# (the saved *.pt files are identical), since these flags only affect how torch *runs*
# TorchScript graphs, not how it scripts or serializes them
# TODO: Remove this in the future
if(TORCHANI_JIT_COMPILE_MODELS)
    Torchani_Msg_info("Will attempt to JIT-compile ANI models after building the extensions")
    Torchani_TorchScriptUtils_add_jit_compile_models_target(
        FORCE_RECOMPILATION ${TORCHANI_JIT_FORCE_RECOMPILATION}
        DISABLE_OPTIMIZATIONS ${TORCHANI_JIT_DISABLE_OPTIMIZATIONS}
        AIMNET2 ${TORCHANI_JIT_COMPILE_AIMNET2}
        EXTENSIONS_DIR $<TARGET_FILE_DIR:torchani_cuaev>
    )
    add_dependencies(torchani_jit_compile_models torchani_cuaev torchani_cell_list)
else()
    Torchani_Msg_warn("Skipping JIT-compilation of models, make sure to manually JIT-compile needed ANI models")
endif()

# Build Catch2 subproject (unit test framework) and Torchani unit tests
include(CTest)
if(BUILD_TESTING)
    set(CATCH2_ROOT "${CMAKE_CURRENT_LIST_DIR}/submodules/Catch2")
    add_subdirectory(${CATCH2_ROOT})
    # Make Catch2 provided cmake-modules available
    list(APPEND CMAKE_MODULE_PATH "${CATCH2_ROOT}/extras")
    include(Catch)

    add_executable(torchani_tests "${CMAKE_CURRENT_LIST_DIR}/tests/tests.cpp")
    target_include_directories(torchani_tests PRIVATE "${CMAKE_CURRENT_LIST_DIR}/include")
    target_link_libraries(torchani_tests PRIVATE Catch2::Catch2WithMain torchani)
    catch_discover_tests(torchani_tests)
endif()

# Same names as the exported targets, for consumers that use add_subdirectory
add_library(${PROJECT_NAME}::torchani ALIAS torchani)
add_library(${PROJECT_NAME}::cuaev ALIAS torchani_cuaev)
add_library(${PROJECT_NAME}::cell_list ALIAS torchani_cell_list)

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

# *_INSTALL_DIR are all relative to the CMAKE_INSTALL_PREFIX
set(PKGDATA_INSTALL_DIR "${CMAKE_INSTALL_DATAROOTDIR}/cmake/${PROJECT_NAME}")
set(INCLUDE_INSTALL_DIR "${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}")
set(DATA_INSTALL_DIR "${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}")
set(LIBRARY_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME}")
set(FORTRAN_INTERFACE_INSTALL_SRC_FILE "${DATA_INSTALL_DIR}/Fortran/torchani.F90")
configure_package_config_file(
    "${CMAKE_CURRENT_LIST_DIR}/cmake/${PROJECT_NAME}Config.cmake.in"
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
        torchani_cuaev
        torchani_cell_list
    EXPORT ${PKG_EXPORT_FILESET_NAME}  # Add ref to targs in the pkg-export-fileset
    LIBRARY
        DESTINATION "${LIBRARY_INSTALL_DIR}"
)
# Install C-interface include file
install(
    FILES
        "${CMAKE_CURRENT_LIST_DIR}/include/torchani.h"
    DESTINATION "${INCLUDE_INSTALL_DIR}"
)

# Install module file necessary for a portable fortran interface
install(
    FILES
        "${CMAKE_CURRENT_LIST_DIR}/interface/torchani.F90"
    DESTINATION "${DATA_INSTALL_DIR}/Fortran"
)
# TODO Add model.jit.pt files
