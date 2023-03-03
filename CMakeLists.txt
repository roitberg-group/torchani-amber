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
include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

# Append internal modules path and include internal modules
set(PUBLIC_MODULES_DIR "${CMAKE_CURRENT_LIST_DIR}/cmake")
list(APPEND CMAKE_MODULE_PATH ${PUBLIC_MODULES_DIR})
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
option(CONDA_CUDNN "Use the cuDNN from the conda environment" ON)
option(CONDA_CUDA "Use CUDA toolkit from conda environment" ON)
option(TORCHANI_INSTALL "Install numpy, PyTorch and TorchANI" OFF)

set(LIBTORCH_CUDNN_VERSION "8.3.2" CACHE STRING "cuDNN version to use for LibTorch")
set(PYTORCH_CUDNN_VERSION "8.3.2" CACHE STRING "cuDNN version PyTorch is compatible with")

set(CUDA_TOOLKIT_VERSION "11.6" CACHE STRING "CUDA Toolkit version to set CUDA vars")
set(LIBTORCH_CUDNN_CUDA_VERSION "11.6" CACHE STRING "CUDA version cuDNN is compatible with")
set(PYTORCH_CUDA_VERSION "11.6" CACHE STRING "CUDA version PyTorch is compatible with")
set(LIBTORCH_CUDA_VERSION "11.6" CACHE STRING "CUDA verison LibTorch is compatible with")

set(PYTORCH_VERSION "1.13.1" CACHE STRING "PyTorch version to use for TorchANI")
set(LIBTORCH_VERSION "1.13.1" CACHE STRING "LibTorch version to use")

option(JIT_AVOID_OPTIMIZATIONS "Avoid JIT optimizations" OFF)
option(JIT_COMPILE_MODELS "JIT-compile all TorchANI models" ON)
option(JIT_TORCH_CELL_LIST "JIT-compile models with torch neighbor list" ON)
option(JIT_EXTERNAL_CELL_LIST "JIT-compile models that use external neighbor list" OFF)

set(PYTORCH_PYTHON_VERSION "3.8" CACHE STRING "Python version PyTorch is compatible with")
set(LIBTORCH_USES_CXX11ABI TRUE CACHE BOOL "Whether LibTorch exposes CXX11 ABI")

# Sets LIBTORCH_ROOT, LIBTORCH_LIBRARY_DIR, LIBTORCH_INCLUDE_DIR
# If the requested version already exists it is not extracted
download_and_extract_libtorch(
    CXX11ABI ${LIBTORCH_USES_CXX11ABI}
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
find_package(Torch ${LIBTORCH_VERSION} EXACT QUIET REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

file(GLOB ANI_AMBER_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_LIST_DIR}/src/*.cpp")
add_library(torchani SHARED ${ANI_AMBER_SOURCES})
set_target_properties(torchani PROPERTIES CXX_STANDARD 14 CXX_STANDARD_REQUIRED TRUE)
target_link_libraries(torchani PRIVATE ${TORCH_LIBRARIES})
# Seems like the include dirs are necessary for exporting torchani, not sure why
target_include_directories(torchani PRIVATE ./include ${TORCH_INCLUDE_DIRS})
target_compile_definitions(torchani PRIVATE $<$<CONFIG:Debug>:"DEBUG">)
target_compile_options(torchani PRIVATE $<$<CONFIG:Debug>:"-g">)

# Build unit tests
add_subdirectory(test)

# Installation boilerplate for target
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    # default for unix is the /usr/local prefix
    message(STATUS "CMake - Install location is set to the default for the system")
endif()
# path prefix for all installation directories
message(STATUS "CMake - Install location initialized to: ${CMAKE_INSTALL_PREFIX}")

set(PROJECT_BINARY_VERSION_PATH "${CMAKE_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake")
set(PROJECT_BINARY_CONFIG_PATH "${CMAKE_BINARY_DIR}/${PROJECT_NAME}Config.cmake")
set(PROJECT_SOURCE_CONFIG_PATH "${CMAKE_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in")
set(PROJECT_TARGETS_PATH "${CMAKE_INSTALL_CONFIGDIR}/${PROJECT_NAME}Targets.cmake")
set(PROJECT_INSTALL_CONFIGDIR "share/cmake/${PROJECT_NAME}")
set(PROJECT_INSTALL_LIBDIR ${CMAKE_INSTALL_LIBDIR})
# Create
# - <proj-name>Config.cmake
# - <proj-name>ConfigVersion.cmake
# - <proj-name>Targets.cmake
# - <proj-name>Targets-release?.cmake
# For use by downstream packages

configure_package_config_file(
    ${PROJECT_SOURCE_CONFIG_PATH}
    ${PROJECT_BINARY_CONFIG_PATH}
    PATH_VARS PROJECT_TARGETS_PATH
    INSTALL_DESTINATION ${PROJECT_INSTALL_CONFIGDIR}
)
write_basic_package_version_file(
    ${PROJECT_BINARY_VERSION_PATH}
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY ExactVersion
)

install(
    FILES
        ${PROJECT_BINARY_CONFIG_PATH}
        ${PROJECT_BINARY_VERSION_PATH}
    DESTINATION ${PROJECT_INSTALL_CONFIGDIR}
)
install(
    EXPORT "${PROJECT_NAME}Targets"
    NAMESPACE "${PROJECT_NAME}::"
    DESTINATION ${PROJECT_INSTALL_CONFIGDIR}
)
install(
    TARGETS torchani
    EXPORT "${PROJECT_NAME}Targets"
    LIBRARY
    DESTINATION ${PROJECT_INSTALL_LIBDIR}
)
