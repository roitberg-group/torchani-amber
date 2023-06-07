cmake_minimum_required(
    VERSION 3.16
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

# Internal includes
include(FetchContent)
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
option(CUSTOM_CUDNN "Explicitly specify the cuDNN location" ON)
option(CUSTOM_CUDA "Explicitly set CUDA root variable" ON)
option(CONDA_CUDNN "Use the cuDNN from the conda environment" ON)
option(CONDA_CUDA "Use CUDA Toolkit from conda environment" ON)
option(TORCHANI_INSTALL "Install Numpy, PyTorch and TorchANI" OFF)

set(LIBTORCH_CUDNN_VERSION "8.3.2" CACHE STRING "cuDNN version to use for LibTorch")
set(PYTORCH_CUDNN_VERSION "8.3.2" CACHE STRING "cuDNN version PyTorch is compatible with")
set(CUDA_TOOLKIT_VERSION "11.6" CACHE STRING "CUDA Toolkit version to set CUDA vars")
set(LIBTORCH_CUDNN_CUDA_VERSION "11.6" CACHE STRING "CUDA version cuDNN is compatible with")
set(PYTORCH_CUDA_VERSION "11.6" CACHE STRING "CUDA version PyTorch is compatible with")
set(LIBTORCH_CUDA_VERSION "11.6" CACHE STRING "CUDA version LibTorch is compatible with")
set(PYTORCH_VERSION "1.13.1" CACHE STRING "PyTorch version to use for TorchANI")
set(LIBTORCH_VERSION "1.13.1" CACHE STRING "LibTorch version to use")

option(JIT_DISABLE_OPTIMIZATIONS "Avoid JIT optimizations" ON)
option(JIT_COMPILE_MODELS "JIT-compile all TorchANI models" ON)
option(JIT_TORCH_CELL_LIST "JIT-compile models with Torch neighbor List" ON)
option(JIT_EXTERNAL_CELL_LIST "JIT-compile models that use External Neighbor List" OFF)

set(PYTORCH_PYTHON_VERSION "3.8" CACHE STRING "Python version PyTorch is compatible with")
set(LIBTORCH_USE_CXX11ABI TRUE CACHE BOOL "Download a LibTorch version that exposes CXX11 ABI")

# Sets LIBTORCH_ROOT, LIBTORCH_LIBRARY_DIR, LIBTORCH_INCLUDE_DIR
# If the requested version already exists it is not extracted
download_and_extract_libtorch(
    CXX11ABI ${LIBTORCH_USE_CXX11ABI}
    LIBRARY_VERSION ${LIBTORCH_VERSION}
    CUDA_VERSION ${LIBTORCH_CUDA_VERSION}
)
list(APPEND CMAKE_PREFIX_PATH ${LIBTORCH_ROOT})

# cuDNN may be linked by LibTorch by providing it with the necessary paths
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
    msg_warn("Skip setting CUDA dirs, make sure correct CUDA Toolkit is used")
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
        JIT_EXTERNAL_CELL_LIST ${JIT_EXTERNAL_CELL_LIST}
        JIT_TORCH_CELL_LIST ${JIT_TORCH_CELL_LIST}
    )
else()
    msg_warn("Skipping JIT compilation of models, make sure JIT models are available")
endif()

# Generate Makefiles, build a release version of the shared library and run
# unit tests (CPU and CUDA if it is available)

file(GLOB TORCHANI_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_LIST_DIR}/src/*.cpp")
add_library(torchani SHARED ${TORCHANI_SOURCES})

# LibTorch is a required dependency
# sets TORCH_FOUND, TORCH_LIBRARIES, TORCH_INCLUDE_DIRS, TORCH_CXX_FLAGS
find_package(Torch ${LIBTORCH_VERSION} EXACT QUIET REQUIRED)
# Install rpath is hardcoded to be the same as the link path, to avoid Issues
# if the correct CUDA is not installed system-wide. If this is not done then
# CMake tries to re-link the binaries at install-time, and prefers the system
# CUDA, which may be bad.
#
# TODO: When doing this, both the build-tree and the install-tree binaries are
# still linked to the CUDA driver library from the system (libcuda.so) by
# default. This behavior seems to be a design choice of libtorch. I belive this
# is ok, since the cuda driver is in principle unrelated to the CUDA Toolkit,
# so I will leave it this way, but I'm documenting it here just in case.
#
# TODO: Both, manual setup of the RPATH **and** specifying automatic link-path
# usage are needed in order for this to properly work, I'm not 100 % sure why
set(TORCHANI_INSTALL_RPATH ${LIBTORCH_LIBRARY_DIR} ${CUDA_TOOLKIT_ROOT_DIR})
set_target_properties(
    torchani
    PROPERTIES
        INSTALL_RPATH  "${TORCHANI_INSTALL_RPATH}"
        INSTALL_RPATH_USE_LINK_PATH TRUE
)
target_compile_options(torchani PRIVATE ${TORCH_CXX_FLAGS} $<$<CONFIG:Debug>:"-g">)
target_compile_features(torchani PRIVATE cxx_std_14)
target_compile_definitions(torchani PRIVATE $<$<CONFIG:Debug>:"DEBUG">)
target_link_libraries(torchani PRIVATE ${TORCH_LIBRARIES})
# TODO: Torch libraries have to be force-linked since during runtime libnvrtc
# can't be found otherwise, it may only be needed to force-link libnvrtc (and
# maybe nvtoolsext?) but for now I force-link everything since it is easier
# althought a bit dirtier.
target_link_options(torchani PRIVATE "LINKER:--no-as-needed")
target_include_directories(torchani PRIVATE "${CMAKE_CURRENT_LIST_DIR}/include" ${TORCH_INCLUDE_DIRS})

# Build unit tests
FetchContent_Declare(
    Catch2
    PREFIX "${CMAKE_CURRENT_LIST_DIR}/external/Catch2/"
    BINARY_DIR "${CMAKE_CURRENT_LIST_DIR}/external/Catch2/build/"
    STAMP_DIR "${CMAKE_CURRENT_LIST_DIR}/external/Catch2/stamp/"
    TMP_DIR "${CMAKE_CURRENT_LIST_DIR}/external/Catch2/tmp/"
    DOWNLOAD_DIR "${CMAKE_CURRENT_LIST_DIR}/external/Catch2/Catch2/"
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG 3f0283de7a9c43200033da996ff9093be3ac84dc
)

FetchContent_MakeAvailable(Catch2)
add_executable(unit_tests "${CMAKE_CURRENT_LIST_DIR}/tests/tests.cpp")
target_include_directories(unit_tests PRIVATE "${CMAKE_CURRENT_LIST_DIR}/include")
target_link_libraries(unit_tests PRIVATE Catch2::Catch2WithMain torchani)

# Installation boilerplate for target

# Override CMake's default prefix for all install files if unset by user
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    # Default CMake installation prefix for Linux is /usr/local. Its better to
    # use ~/.local since it is typically user-writable in Linux systems (no sudo needed)
    # Expand tilde for cmake < 3.21
    get_filename_component(_INSTALL_PREFIX "~/.local" ABSOLUTE)  # resolve
    set(CMAKE_INSTALL_PREFIX "${_INSTALL_PREFIX}" CACHE PATH "Installation prefix" FORCE)
    message(STATUS "CMake - Install location is set to the default ~/.local")
endif()
message(STATUS "CMake - Install location initialized to: ${CMAKE_INSTALL_PREFIX}")

# Set dirs for project installation
# Note that by default in unix the rel_project(_install)_configdir are the same
set(REL_PROJECT_CONFIGDIR "share/cmake/${PROJECT_NAME}")  # local configdir
set(REL_PROJECT_INSTALL_CONFIGDIR "${CMAKE_INSTALL_DATAROOTDIR}/cmake/${PROJECT_NAME}")
set(REL_PROJECT_INSTALL_LIBDIR "${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME}")

# Set stems generated && configured shareable installation files
set(PROJECT_CONFIG_STEM "${PROJECT_NAME}Config")
set(PROJECT_VERSION_STEM "${PROJECT_NAME}ConfigVersion")
set(PROJECT_TARGETS_STEM "${PROJECT_NAME}Targets")

# Set targets namespace
set(PROJECT_TARGETS_NAMESPACE "${PROJECT_NAME}::")

# PackageConfig file paths (input (source-tree) and output (binary-tree))
set(PROJECT_CONFIG_FNAME "${PROJECT_CONFIG_STEM}.cmake")
set(PROJECT_SOURCE_CONFIG_PATH "${CMAKE_SOURCE_DIR}/${REL_PROJECT_CONFIGDIR}/${PROJECT_CONFIG_FNAME}.in")
set(PROJECT_BINARY_CONFIG_PATH "${CMAKE_BINARY_DIR}/${REL_PROJECT_CONFIGDIR}/${PROJECT_CONFIG_FNAME}")

# PackageVersion file path (output (binary-tree) only, version file is generated on-the-fly)
set(PROJECT_VERSION_FNAME "${PROJECT_VERSION_STEM}.cmake")
set(PROJECT_BINARY_VERSION_PATH "${CMAKE_BINARY_DIR}/${REL_PROJECT_CONFIGDIR}/${PROJECT_VERSION_FNAME}")

# Targets file (output (binary-tree) only, targets file is generated on-the-fly)
set(PROJECT_TARGETS_FNAME "${PROJECT_TARGETS_STEM}.cmake")
set(PROJECT_BINARY_TARGETS_PATH "${CMAKE_BINARY_DIR}/${REL_PROJECT_INSTALL_CONFIGDIR}/${PROJECT_TARGETS_FNAME}")

# Create and install files for use by downstream packages
# - <PkgName>Config.cmake (configure_package_config_file)
# - <PkgName>ConfigVersion.cmake (write_basic_version_file)
# ??
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
    TARGETS torchani
    EXPORT "${PROJECT_TARGETS_STEM}"
    LIBRARY
        DESTINATION "${REL_PROJECT_INSTALL_LIBDIR}"
)
