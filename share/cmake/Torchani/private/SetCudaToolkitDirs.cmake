include_guard()

function(set_cuda_toolkit_dirs)
    set(options "")
    set(oneValueArgs CONDA_CUDA CUDA_VERSION)
    set(multiValueArgs "")
    cmake_parse_arguments(
        _FN
        "${options}"
        "${oneValueArgs}"
        "${multiValueArgs}"
        ${ARGN}
    )
    if (_FN_CONDA_CUDA)
        message(STATUS "CUDA Toolkit - Using conda CUDA")
        set(_CUDA_ROOT $ENV{CONDA_PREFIX})
    else()
        message(STATUS "CUDA Toolkit - Setting include dirs for CUDA ${_FN_CUDA_VERSION}")
        set(_CUDA_ROOT "/usr/local/cuda-${_FN_CUDA_VERSION}")
    endif()
    set(CUDA_TOOLKIT_ROOT_DIR "${_CUDA_ROOT}" PARENT_SCOPE)
    set(CMAKE_CUDA_COMPILER "${_CUDA_ROOT}/bin/nvcc" PARENT_SCOPE)
endfunction()
