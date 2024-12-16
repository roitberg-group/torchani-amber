cmake_minimum_required(
    VERSION 3.24
    FATAL_ERROR
)

project(
    Torchani
    LANGUAGES C CXX
    VERSION 0.5
    DESCRIPTION "Fortran, C and C++ interface to execute ANI-Style models built using the TorchANI library"
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

# TODO: These options are still confusing, it is probably better to use CMakeDependentOptions
# PyTorch/LibTorch and their backends: options and flags

# General: options and flags
option(LINK_TO_PYTORCH_LIBS "Try to find and link to a local PyTorch installation" ON)
option(SET_TORCHANI_LIBRARIES_INSTALL_RPATH "Hardcode the RPATH used during build-time for the installed libs" ON)
option(COMPILE_WITH_CXX11ABI "Compile binaries with C++11 ABI. Requires 'LINK_TO_PYTORCH_LIBS=OFF'" OFF)
option(DOWNLOAD_AND_INSTALL_LIBTORCH "Download LibTorch library and extract locally to ./external/libtorch<version>-cuda<version>" OFF)
option(PREFER_CONDA_LIBS "Prefer libraries under $CONDA_PREFIX (the active conda env)" ON)
option(USE_CUSTOM_CUDNN "Try to find cuDNN under ./external/cudnn<version>-cuda<version> if needed" OFF)
option(USE_CUSTOM_CUDATK "Try to find CUDA Toolkit under ./usr/local/cuda-<version> if needed" OFF)

# Exact major.minor versions must match for these
set(REQUIRED_TORCH_VERSION "2.5" CACHE STRING "Require this version of PyTorch/LibTorch")
set(REQUIRED_CUDATK_VERSION "12.4" CACHE STRING "Require this version of the CUDA Toolkit")
set(REQUIRED_CUDNN_VERSION "9.1" CACHE STRING "Require this version of the cuDNN library")

# Validate options
if(LINK_TO_PYTORCH_LIBS AND DOWNLOAD_AND_INSTALL_LIBTORCH)
    msg_error("'LINK_TO_PYTORCH_LIBS' is incompatible with 'DOWNLOAD_AND_INSTALL_LIBTORCH'")
endif()

if(PREFER_CONDA_LIBS AND (USE_CUSTOM_CUDNN OR USE_CUSTOM_CUDATK))
    msg_error("'PREFER_CONDA_LIBS' is incompatible with USE_CUSTOM_CUDNN or USE_CUSTOM_CUDATK")
endif()

if(COMPILE_WITH_CXX11ABI AND LINK_TO_PYTORCH_LIBS)
    msg_error("'LINK_TO_PYTORCH_LIBS=OFF' is needed to compile with the CXX11ABI")
elseif(NOT COMPILE_WITH_CXX11ABI)
    msg_banner(
        "NOTE: All binaries will be compiled *without* the C++11 ABI
        This means linking to other binaries compiled *with* the C++11 ABI will *not* be possible
        If you need this, you will need to link to LibTorch instead
        of PyTorch, to do this reconfigure with the following options:
        -DLINK_TO_PYTORCH_LIBS=OFF -DCOMPILE_WITH_CXX11ABI=ON -DDOWNLOAD_AND_INSTALL_LIBTORCH=ON"
    )
endif()

# JIT: Options and flags
option(JIT_COMPILE_MODELS "JIT-compile all the TorchANI models" ON)
option(JIT_DISABLE_OPTIMIZATIONS "Disable TorchScript optimizations when JIT-compiling models" ON)
option(JIT_FORCE_RECOMPILATION "Force JIT-recompilation of all models, even if jit-compiled *.pt files with the expected names are found" OFF)

if(NOT $ENV{CONDA_PREFIX} STREQUAL "" AND PREFER_CONDA_LIBS)
    msg_info("You are running inside a conda env with prefix $ENV{CONDA_PREFIX}")
    msg_info("Libs will be searched first under this prefix")
    msg_info("If you don't want this, reconfigure with -DPREFER_CONDA_LIBS=OFF")
    list(APPEND CMAKE_PREFIX_PATH $ENV{CONDA_PREFIX})
    set(USE_CONDA_CUDNN ON)
    set(USE_CONDA_CUDATK ON)
else()
    set(USE_CONDA_CUDNN OFF)
    set(USE_CONDA_CUDATK OFF)
endif()

# PyTorch/LibTorch setup
if(LINK_TO_PYTORCH_LIBS)
    # Detect PyTorch's location using Python directly
    msg_info("Will try to find and link to an existing PyTorch installation")
    message(STATUS "Trying to use python to import PyTorch and find its path and specs")
    # Note that the CUDA and torch versions are M.m.p and the cuDNN is Mmmpp
    execute_process(
        COMMAND python cmake/detect_pytorch.py
        OUTPUT_VARIABLE PYTORCH_SPECS
        RESULT_VARIABLE PYTORCH_DETECTION_FAILED
    )
    if(PYTORCH_DETECTION_FAILED)
        msg_error("Could not find a PyTorch installation compiled with CUDA and cuDNN support")
    else()
        list(GET PYTORCH_SPECS 0 DETECTED_TORCH_VERSION)
        list(GET PYTORCH_SPECS 1 DETECTED_CUDATK_VERSION)
        list(GET PYTORCH_SPECS 2 DETECTED_CUDNN_VERSION)
        list(GET PYTORCH_SPECS 3 DETECTED_TORCH_ROOT)
        msg_success("Successfully found a PyTorch installation compiled with CUDA and cuDNN support")
        message(STATUS "The PyTorch root is ${DETECTED_TORCH_ROOT}")
        message(STATUS "Validating the detected PyTorch installation")
        if(NOT DETECTED_TORCH_VERSION STREQUAL REQUIRED_TORCH_VERSION)
            msg_error("PyTorch version ${DETECTED_TORCH_VERSION} doesn't match required ${REQUIRED_TORCH_VERSION}")
        else()
            msg_success("PyTorch version ${DETECTED_TORCH_VERSION} is consistent with required")
        endif()
        if(NOT DETECTED_CUDATK_VERSION STREQUAL REQUIRED_CUDATK_VERSION)
            msg_error("CUDA-Toolkit PyTorch was compiled against has version: ${DETECTED_CUDATK_VERSION} doesn't match required ${REQUIRED_CUDATK_VERSION}")
        else()
            msg_success("CUDA-Toolkit PyTorch was compiled against has version: ${DETECTED_CUDATK_VERSION}, consistent with required")
        endif()
        if(NOT DETECTED_CUDNN_VERSION STREQUAL REQUIRED_CUDNN_VERSION)
            msg_error("cuDNN PyTorch was compiled against has version: ${DETECTED_CUDNN_VERSION} which doesn't match required ${REQUIRED_CUDNN_VERSION}")
        else()
            msg_success("cuDNN PyTorch was compiled against has version: ${DETECTED_CUDNN_VERSION}, consistent with required")
        endif()
    endif()
    # Print banner since currently there are no checks for the presence of the dev
    # version of the CUDAToolkit libs
    list(APPEND CMAKE_PREFIX_PATH ${DETECTED_TORCH_ROOT})
elseif(DOWNLOAD_AND_INSTALL_LIBTORCH)
    msg_info("Will download and install user-specified LibTorch")
    # NOTE: If the requested version already exists it is *not* extracted
    # Set vars LIBTORCH_ROOT, LIBTORCH_LIBRARY_DIR
    LibtorchHelper_download_and_install(
        CXX11ABI ${COMPILE_WITH_CXX11ABI}
        LIB_VERSION ${REQUIRED_TORCH_VERSION}
        CUDA_VERSION ${REQUIRED_CUDATK_VERSION}
    )
    list(APPEND CMAKE_PREFIX_PATH ${LIBTORCH_ROOT})
endif()

# cuDNN setup
if(USE_CONDA_CUDNN)
    msg_info("Will try to use cuDNN from active Conda env if needed")
    # Set vars CUDNN_ROOT, CUDNN_LIBRARY, CUDNN_INCLUDE_DIR
    CudnnHelper_set_conda_paths()
    list(APPEND CMAKE_PREFIX_PATH ${CUDNN_LIBRARY})
elseif(USE_CUSTOM_CUDNN)
    msg_info("Will manually set variables that point to cuDNN")
    msg_info("assuming it is located in ./external/cudnn<version>-cuda<version>")
    # Set vars CUDNN_ROOT, CUDNN_LIBRARY, CUDNN_INCLUDE_DIR
    CudnnHelper_set_custom_paths(
        LIB_VERSION ${REQUIRED_CUDNN_VERSION}
        CUDA_VERSION ${REQUIRED_CUDATK_VERSION}
    )
    list(APPEND CMAKE_PREFIX_PATH ${CUDNN_LIBRARY})
else()
    # cuDNN may be directly extracted into the CUDA Toolkit dir,
    # if a user has done this then they don't need to specify an external version
    msg_info("Will assume cuDNN, if needed, is located in the same dir as the CUDA Toolkit")
endif()

# CUDA Toolkit setup
if(USE_CONDA_CUDATK)
    msg_info("Will try to use CUDA Toolkit from the active Conda env")
    # Set CUDA_TOOLKIT_ROOT_DIR and CMAKE_CUDA_COMPILER
    CudaToolkitHelper_set_conda_paths()
    msg_banner(
        "TorchANI requires CUDA Toolkit dev version, which is *not* bundled with PyTorch,
        and is *not* contained in the conda 'nvidia::cudatoolkit' package.
        At this stage of the configuration, we don't check whether all necessary CUDA
        Toolkit features are present.
        Make sure you have the required headers, NVCC, and *all* CUDA Toolkit libs. To
        ensure this, for example, use the provided conda env, which has all
        dependencies, or directly download and install the CUDA Toolkit
        ${REQUIRED_CUDATK_VERSION} from Nvidia's web page. Failures related to CUDA
        after this point may indicate you are missing a required part of the Toolkit."
    )
elseif(USE_CUSTOM_CUDATK)
    msg_info("Will manually set variables that point to the CUDA Toolkit")
    msg_info("assuming it is located in /usr/local/cuda-<version>")
    # Set CUDA_TOOLKIT_ROOT_DIR and CMAKE_CUDA_COMPILER
    CudaToolkitHelper_set_custom_paths(CUDATK_VERSION ${REQUIRED_CUDATK_VERSION})
else()
    msg_info("Will follow LibTorch's default lookup procedure for the CUDA Toolkit")
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
target_compile_options(
    cuaev
    PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math>
)
target_link_directories(cuaev PRIVATE $ENV{CONDA_PREFIX}/lib)
target_link_libraries(cuaev PRIVATE ${TORCH_LIBRARIES} ${Python_LIBRARIES})
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
        PRIVATE
            $<$<CONFIG:Debug>:"DEBUG">
            $<$<CONFIG:Debug>:"TIMING">
)
target_link_options(torchani PRIVATE "LINKER:-force_load,$<TARGET_FILE:cuaev>")
target_link_directories(torchani PRIVATE $ENV{CONDA_PREFIX}/lib)
target_link_libraries(torchani PRIVATE ${TORCH_LIBRARIES} cuaev)
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
# TODO: cuAEV needs to be force-loaded due to an internal limitation of Torch extensions
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
# NOTE:
# Catch2 needs to be compiled with the same GLIBCXX_ABI as torch. Manually adding
# public "target_compile_options" to the Catch2 liberaries is the dirty way I found to
# do this. There may be a better way, but I'm not sure how to otherwise propagate these
# flags to the subproject.
#
# Also, note that flags can't be set on alias targets (such as Catch2::Catch2WithMain),
# they have to be set on "Catch2WithMain" directly, and they have to be set on both,
# "Catch2" and "Catch2WithMain", since CMake doesn't propagate the flags between these
# libraries either.
include(CTest)
set(CATCH2_ROOT "${CMAKE_CURRENT_LIST_DIR}/submodules/Catch2")
add_subdirectory(${CATCH2_ROOT})
target_compile_definitions(Catch2WithMain PUBLIC ${TORCH_CXX_FLAGS})
target_compile_definitions(Catch2 PUBLIC ${TORCH_CXX_FLAGS})
# Make Catch2 provided cmake-modules available
list(APPEND CMAKE_MODULE_PATH "${CATCH2_ROOT}/extras")
include(Catch)

add_executable(tests "${CMAKE_CURRENT_LIST_DIR}/tests/tests.cpp")
target_include_directories(tests PRIVATE "${CMAKE_CURRENT_LIST_DIR}/include")
target_compile_definitions(tests PUBLIC ${TORCH_CXX_FLAGS})
target_link_libraries(tests PRIVATE Catch2::Catch2WithMain torchani)
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
