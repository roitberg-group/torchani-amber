include_guard()

# The two following functions set, in the PARENT scope:
# - CMAKE_CUDA_COMPILER
# - CUDA_TOOLKIT_ROOT_DIR
function(CudaToolkitHelper_set_custom_dirs)
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
    message(STATUS "CUDAToolkit - Setting custom dirs")
    set(_CUDA_ROOT "/usr/local/cuda-${_FN_TOOLKIT_VERSION}")
    set(CUDA_TOOLKIT_ROOT_DIR ${_CUDA_ROOT} PARENT_SCOPE)
    set(CMAKE_CUDA_COMPILER "${_CUDA_ROOT}/bin/nvcc" PARENT_SCOPE)
endfunction()

function(CudaToolkitHelper_set_conda_dirs)
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
    message(STATUS "CUDAToolkit - Setting conda dirs")
    set(_CUDA_ROOT $ENV{CONDA_PREFIX})
    set(CUDA_TOOLKIT_ROOT_DIR ${_CUDA_ROOT} PARENT_SCOPE)
    set(CMAKE_CUDA_COMPILER "${_CUDA_ROOT}/bin/nvcc" PARENT_SCOPE)
endfunction()
