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
# CMake stdlib includes
include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

# Append internal modules path and include internal modules
set(PRIVATE_MODULES_DIR "${CMAKE_CURRENT_LIST_DIR}/share/cmake/${PROJECT_NAME}/private")
list(APPEND CMAKE_MODULE_PATH ${PRIVATE_MODULES_DIR})
include(Msg)
include(EnsureLinuxBuild)
include(EnsureOutOfSourceBuild)
include(DownloadAndExtractLibtorch)
include(SetCudnnDirs)
include(SetCudaToolkitDirs)
include(InstallTorchani)
include(JitCompileModels)

# Fail if building in-source
ensure_out_of_source_build()

# Fail if not building on linux
ensure_linux_build()

# User options and flags
option(CUSTOM_CUDNN "Explicitly specify the cuDNN location" ON)
option(CUSTOM_CUDA "Explicitly set CUDA root variable" ON)
option(CONDA_CUDNN "Use the cuDNN from the conda environment" OFF)
option(CONDA_CUDA "Use CUDA toolkit from conda environment" OFF)
option(TORCHANI_INSTALL "Install numpy, PyTorch and TorchANI" OFF)

set(LIBTORCH_CUDNN_VERSION "8.3.2" CACHE STRING "cuDNN version to use for LibTorch")
set(PYTORCH_CUDNN_VERSION "8.3.2" CACHE STRING "cuDNN version PyTorch is compatible with")

set(CUDA_TOOLKIT_VERSION "11.6" CACHE STRING "CUDA Toolkit version to set CUDA vars")
set(LIBTORCH_CUDNN_CUDA_VERSION "11.6" CACHE STRING "CUDA version cuDNN is compatible with")
set(PYTORCH_CUDA_VERSION "11.6" CACHE STRING "CUDA version PyTorch is compatible with")
set(LIBTORCH_CUDA_VERSION "11.6" CACHE STRING "CUDA verison LibTorch is compatible with")

set(PYTORCH_VERSION "1.13.1" CACHE STRING "PyTorch version to use for TorchANI")
set(LIBTORCH_VERSION "1.13.1" CACHE STRING "LibTorch version to use")

option(JIT_AVOID_OPTIMIZATIONS "Avoid JIT optimizations" ON)
option(JIT_COMPILE_MODELS "JIT-compile all TorchANI models" ON)
option(JIT_TORCH_CELL_LIST "JIT-compile models with torch neighbor list" ON)
option(JIT_EXTERNAL_CELL_LIST "JIT-compile models that use external neighbor list" OFF)

set(PYTORCH_PYTHON_VERSION "3.8" CACHE STRING "Python version PyTorch is compatible with")
set(LIBTORCH_USE_CXX11ABI TRUE CACHE BOOL "Whether LibTorch exposes CXX11 ABI")

# Sets LIBTORCH_ROOT, LIBTORCH_LIBRARY_DIR, LIBTORCH_INCLUDE_DIR
# If the requested version already exists it is not extracted
download_and_extract_libtorch(
    CXX11ABI ${LIBTORCH_USE_CXX11ABI}
    LIBRARY_VERSION ${LIBTORCH_VERSION}
    CUDA_VERSION ${LIBTORCH_CUDA_VERSION}
)
list(APPEND CMAKE_PREFIX_PATH ${LIBTORCH_ROOT})
list(APPEND CMAKE_INSTALL_RPATH ${LIBTORCH_LIBRARY_DIR})  # RPATH?

# CUDNN may be linked by LibTorch by providing it with the necessary paths
# ("custom cuDNN"), or it may be directly extracted into the cuda toolkit dir,
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
        JIT_AVOID_OPTIMIZATIONS ${JIT_AVOID_OPTIMIZATIONS}
        JIT_EXTERNAL_CELL_LIST ${JIT_EXTERNAL_CELL_LIST}
        JIT_TORCH_CELL_LIST ${JIT_TORCH_CELL_LIST}
    )
else()
    msg_warn("Skipping JIT compilation of models, make sure JIT models are available")
endif()

# Generate Makefiles, build a release version of the shared library and run
# unit tests (CPU and CUDA if it is available)

# LibTorch is a required dependency
# sets TORCH_FOUND, TORCH_LIBRARIES, TORCH_INCLUDE_DIRS, TORCH_CXX_FLAGS

file(GLOB TORCHANI_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_LIST_DIR}/src/*.cpp")
set(TORCHANI_INCLUDE_DIRS "${CMAKE_CURRENT_LIST_DIR}/include")
add_library(torchani SHARED ${TORCHANI_SOURCES})
set_target_properties(
    torchani
    PROPERTIES
        CXX_STANDARD 14
        CXX_STANDARD_REQUIRED TRUE
        INCLUDE_DIRECTORIES ${TORCHANI_INCLUDE_DIRS}
)
#target_link_libraries(torchani PRIVATE ${TORCH_LIBRARIES})
# Seems like the include dirs are necessary for exporting torchani, not sure why
# I think setting target compile options is better, I believe public is needed
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")  # TODO: this is dirty

# Append to these always
find_package(Torch ${LIBTORCH_VERSION} EXACT QUIET REQUIRED)
target_compile_options(torchani PRIVATE ${TORCH_CXX_FLAGS})
target_link_libraries(torchani PRIVATE ${TORCH_LIBRARIES})
target_include_directories(torchani PRIVATE ${TORCH_INCLUDE_DIRS})

# Append these in Debug config
target_compile_definitions(torchani PRIVATE $<$<CONFIG:Debug>:"DEBUG">)
target_compile_options(torchani PRIVATE $<$<CONFIG:Debug>:"-g">)

# Build unit tests
add_subdirectory(test)

# Installation boilerplate for target
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    # Default prefix for Linux is /usr/local
    message(STATUS "CMake - Install location is set to the default for the system")
endif()

# path prefix for all installation directories
message(STATUS "CMake - Install location initialized to: ${CMAKE_INSTALL_PREFIX}")

# Set dirs for project
set(REL_PROJECT_CONFIGDIR "share/cmake/${PROJECT_NAME}")  # local configdir
set(REL_PROJECT_INSTALL_CONFIGDIR "${CMAKE_INSTALL_DATAROOTDIR}/cmake/${PROJECT_NAME}")
set(REL_PROJECT_INSTALL_LIBDIR "${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME}")

# Set stems for files
set(PROJECT_CONFIG_STEM "${PROJECT_NAME}Config")
set(PROJECT_VERSION_STEM "${PROJECT_NAME}ConfigVersion")
set(PROJECT_TARGETS_STEM "${PROJECT_NAME}Targets")

# Set namespace
set(PROJECT_TARGETS_NAMESPACE "${PROJECT_NAME}::")

# Config file
set(PROJECT_CONFIG_FNAME "${PROJECT_CONFIG_STEM}.cmake")
set(PROJECT_SOURCE_CONFIG_PATH "${CMAKE_SOURCE_DIR}/${REL_PROJECT_CONFIGDIR}/${PROJECT_CONFIG_FNAME}.in")
set(PROJECT_BINARY_CONFIG_PATH "${CMAKE_BINARY_DIR}/${REL_PROJECT_CONFIGDIR}/${PROJECT_CONFIG_FNAME}")

# Version file
set(PROJECT_VERSION_FNAME "${PROJECT_VERSION_STEM}.cmake")
# Source version file does not exist, binary-tree version is created on the fly
set(PROJECT_BINARY_VERSION_PATH "${CMAKE_BINARY_DIR}/${REL_PROJECT_CONFIGDIR}/${PROJECT_VERSION_FNAME}")

# Targets file
set(PROJECT_TARGETS_FNAME "${PROJECT_TARGETS_STEM}.cmake")
# Source targets file does not exist, binary-tree targets is created on the fly
set(PROJECT_BINARY_TARGETS_PATH "${CMAKE_BINARY_DIR}/${REL_PROJECT_INSTALL_CONFIGDIR}/${PROJECT_TARGETS_FNAME}")

# Create
# - <proj-name>Config.cmake
# - <proj-name>ConfigVersion.cmake
# - <proj-name>Targets.cmake
# - <proj-name>Targets-*.cmake
# - <proj-name>Targets.cmake (build-tree)
# - <proj-name>Targets-*.cmake (build-tree)
# For use by downstream packages

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

# export is needed to consume the project from its build-tree, it will create
# a Targets file in the build-tree, not only in the install-prefix
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
