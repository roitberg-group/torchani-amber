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
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(Msg)
include(EnsureLinuxBuild)
include(DownloadAndExtractLibtorch)
include(SetCustomCudnnPaths)
include(InstallTorchani)
include(JitCompileModels)
include(SetCudaToolkitVars)

# Fail if not building on linux
ensure_linux_build()

# User options and flags
option(CUSTOM_CUDNN "Explicitly specify the cuDNN location" ON)
set(CUSTOM_CUDNN_VERSION "8" CACHE STRING "cuDNN version to use for LibTorch")
set(CUSTOM_CUDNN_CUDA_VERSION "11.6" CACHE STRING "CUDA version cuDNN is compatible with")
option(TORCHANI_INSTALL "Install numpy, PyTorch and TorchANI" ON)
option(JIT_COMPILE_MODELS "JIT-compile all TorchANI models" ON)
option(JIT_AVOID_OPTIMIZATIONS "Avoid JIT optimizations" ON)
option(JIT_TORCH_CELL_LIST "JIT-compile models with torch neighbor list" ON)
option(JIT_EXTERNAL_CELL_LIST "JIT-compile models that use external neighbor list" ON)
option(CUDA_TOOLKIT_SET_VARS "Manually set cuda variables" ON)
set(CUDA_TOOLKIT_VERSION "11.6" CACHE STRING "CUDA Toolkit version to set CUDA vars")
set(PYTORCH_VERSION "1.13.1" CACHE STRING "PyTorch version to use for TorchANI")
set(PYTORCH_TORCHVISION_VERSION "0.14" CACHE STRING "Torchvision version to use for TorchANI")
set(PYTORCH_CUDA_VERSION "11.6" CACHE STRING "CUDA version PyTorch is compatible with")
set(LIBTORCH_VERSION "1.13.1" CACHE STRING "LibTorch version to use")
set(LIBTORCH_CUDA_VERSION "11.6" CACHE STRING "CUDA verison LibTorch is compatible with")
set(LIBTORCH_USES_CXX11ABI TRUE CACHE BOOL "Whether LibTorch exposes CXX11 ABI")

# Sets LIBTORCH_EXTRACTED_DIR, LIBTORCH_EXTRACTED_LIB_DIR
# If the requested version already exists it is not extracted
download_and_extract_libtorch(
    CXX11ABI ${LIBTORCH_USES_CXX11ABI}
    LIBRARY_VERSION ${LIBTORCH_VERSION}
    CUDA_VERSION ${LIBTORCH_CUDA_VERSION}
)
list(APPEND CMAKE_PREFIX_PATH ${LIBTORCH_EXTRACTED_DIR})
list(APPEND CMAKE_INSTALL_RPATH ${LIBTORCH_EXTRACTED_LIB_DIR})  # RPATH?

# CUDNN may be linked by LibTorch by providing it with the necessary paths
# ("custom cuDNN"), or it may be directly extracted into the cuda toolkit dir,
# in which case it is not needed to specify the paths.
if (CUSTOM_CUDNN)
    # sets CUDNN_LIBRARY_DIR, CUDNN_LIBRARY_PATH, CUDNN_INCLUDE_PATH
    set_custom_cudnn_paths(
        LIBRARY_VERSION ${CUSTOM_CUDNN_VERSION}
        CUDA_VERSION ${CUSTOM_CUDNN_CUDA_VERSION}
    )
    list(APPEND CMAKE_PREFIX_PATH "${CUDNN_LIBRARY_DIR}")
else()
    msg_warn("Skip setting cuDNN paths, make sure cuDNN is present in CUDA Toolkit")
endif()
message(STATUS "CMake - prefix path: ${CMAKE_PREFIX_PATH}")

if(CUDA_TOOLKIT_SET_VARS)
    # set CUDA_TOOLKIT_ROOT_DIR, CUDA_INCLUDE_DIRS, CUDA_TOOLKIT_INCLUDE,
    # CUDA_CUDART_LIBRARY, ENV{CUDA_BIN_DIR}
    set_cuda_toolkit_vars(CUDA_VERSION ${CUDA_TOOLKIT_VERSION})
else()
    msg_warn("Skip explicit set of CUDA vars, make sure correct CUDA Toolkit is used")
endif()

if(TORCHANI_INSTALL)
    install_torchani(
        PYTORCH_VERSION ${PYTORCH_VERSION}
        PYTORCH_CUDA_VERSION ${PYTORCH_CUDA_VERSION}
        TORCHVISION_VERSION ${PYTORCH_TORCHVISION_VERSION}
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
find_package(Torch REQUIRED)  # sets TORCH_LIBRARIES, TORCH_INCLUDE_DIRS

file(GLOB ANI_AMBER_SOURCES "./src/*.cpp")
add_library(ani_amber SHARED ${ANI_AMBER_SOURCES})
set_target_properties(ani_amber PROPERTIES CXX_STANDARD 14 CXX_STANDARD_REQUIRED TRUE)
target_link_libraries(ani_amber PRIVATE "${TORCH_LIBRARIES}")
# Seems like the include dirs are necessary for exporting ani_amber, not sure why
target_include_directories(ani_amber PRIVATE ./include "${TORCH_INCLUDE_DIRS}")
target_compile_definitions(ani_amber PRIVATE $<$<CONFIG:Debug>:"DEBUG">)
target_compile_options(ani_amber PRIVATE $<$<CONFIG:Debug>:"-g">)

# Build unit tests
add_subdirectory(test)

# Installation boilerplate for target
# Note that the CMAKE_INSTALL_PREFIX is prepended to all install directories
# For instance in unix systems(i.e. linux) /usr/local/ is prepended
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    message(STATUS "CMake - Install location is set to the default for the system")
endif()
message(STATUS "CMake - Install location initialized to: ${CMAKE_INSTALL_PREFIX}")

set(PROJECT_VERSION_FILE "${CMAKE_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake")
set(PROJECT_CONFIG_FILE "${CMAKE_BINARY_DIR}/${PROJECT_NAME}Config.cmake")
set(PROJECT_INSTALL_CONFIGDIR "share/cmake/${PROJECT_NAME}")
set(PROJECT_INSTALL_LIBDIR ${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME})
# Create the <pname>Config.cmake and <pname>ConfigVersion.cmake files
# For use by downstream packages
configure_package_config_file(
    "${CMAKE_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in"
    ${PROJECT_CONFIG_FILE}
    INSTALL_DESTINATION ${PROJECT_INSTALL_CONFIGDIR}
)
write_basic_package_version_file(
    ${PROJECT_VERSION_FILE}
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)
install(
    FILES
        ${PROJECT_CONFIG_FILE}
        ${PROJECT_VERSION_FILE}
    DESTINATION ${PROJECT_INSTALL_CONFIGDIR}
)
install(
    EXPORT "${PROJECT_NAME}Targets"
    NAMESPACE "${PROJECT_NAME}::"
    DESTINATION ${PROJECT_INSTALL_CONFIGDIR}
)
install(
    TARGETS ani_amber
    EXPORT "${PROJECT_NAME}Targets"
    LIBRARY DESTINATION ${PROJECT_INSTALL_LIBDIR}
)
