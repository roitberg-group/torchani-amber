include_guard()

function(set_cuda_toolkit_vars)
    set(options "")
    set(oneValueArgs CUDA_VERSION)
    set(multiValueArgs "")
    cmake_parse_arguments(
        _FN
        "${options}"
        "${oneValueArgs}"
        "${multiValueArgs}"
        ${ARGN}
    )
    message(STATUS "CUDA Toolkit - Setting include dirs for CUDA ${_FN_CUDA_VERSION}")
    set(_CUDA_PREFIX "/usr/local/cuda-${_FN_CUDA_VERSION}")
    set(ENV{CUDA_BIN_DIR} ${_CUDA_PREFIX})
    set(CUDA_TOOLKIT_ROOT_DIR "${_CUDA_PREFIX}" PARENT_SCOPE)
    set(CUDA_INCLUDE_DIRS "${_CUDA_PREFIX}/include" PARENT_SCOPE)
    set(CUDA_TOOLKIT_INCLUDE "${_CUDA_PREFIX}/include" PARENT_SCOPE)
    set(CUDA_CUDART_LIBRARY "${_CUDA_PREFIX}/lib64/libcudart.so" PARENT_SCOPE)
    message(STATUS "CUDA toolkit - Using version ${_FN_CUDA_VERSION}")
endfunction()
