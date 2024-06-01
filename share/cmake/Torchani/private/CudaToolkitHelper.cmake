include_guard()
include(Msg)

# The two following functions set, in the PARENT scope:
# - CMAKE_CUDA_COMPILER
# - CUDA_TOOLKIT_ROOT_DIR
function(CudaToolkitHelper_set_custom_paths)
    set(options "")
    set(oneValueArgs TOOLKIT_VERSION)
    set(multiValueArgs "")
    cmake_parse_arguments(
        _FN
        "${options}"
        "${oneValueArgs}"
        "${multiValueArgs}"
        ${ARGN}
    )
    message(STATUS "CUDAToolkit - Setting custom paths")
    set(_CUDA_ROOT "/usr/local/cuda-${_FN_TOOLKIT_VERSION}")
    set(CUDA_TOOLKIT_ROOT_DIR ${_CUDA_ROOT} PARENT_SCOPE)
    set(CMAKE_CUDA_COMPILER "${_CUDA_ROOT}/bin/nvcc" PARENT_SCOPE)
    set(CUDA_TOOLKIT_ROOT_DIR ${_CUDA_ROOT})
    set(CMAKE_CUDA_COMPILER "${_CUDA_ROOT}/bin/nvcc")
    message(STATUS "CUDAToolkit - Root dir: ${CUDA_TOOLKIT_ROOT_DIR}")
    message(STATUS "CUDAToolkit - CMake CUDA compiler: ${CMAKE_CUDA_COMPILER}")
    msg_success("CUDAToolkit - Successfully set paths")
endfunction()

function(CudaToolkitHelper_set_conda_paths)
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs "")
    cmake_parse_arguments(
        _FN
        "${options}"
        "${oneValueArgs}"
        "${multiValueArgs}"
        ${ARGN}
    )
    message(STATUS "CUDAToolkit - Setting paths using Conda prefix")
    set(_CUDA_ROOT $ENV{CONDA_PREFIX})
    set(CUDA_TOOLKIT_ROOT_DIR ${_CUDA_ROOT} PARENT_SCOPE)
    set(CMAKE_CUDA_COMPILER "${_CUDA_ROOT}/bin/nvcc" PARENT_SCOPE)
    set(CUDA_TOOLKIT_ROOT_DIR ${_CUDA_ROOT})
    set(CMAKE_CUDA_COMPILER "${_CUDA_ROOT}/bin/nvcc")
    message(STATUS "CUDAToolkit - Root dir: ${CUDA_TOOLKIT_ROOT_DIR}")
    message(STATUS "CUDAToolkit - CMake CUDA compiler: ${CMAKE_CUDA_COMPILER}")
    msg_success("CUDAToolkit - Successfully set paths")
endfunction()
