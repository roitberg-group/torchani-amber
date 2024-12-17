include_guard()
include(Msg)

# The two following functions set, in the PARENT scope:
# - CUDNN_ROOT
# - CUDNN_LIBRARY
# - CUDNN_INCLUDE_DIR
function(CudnnHelper_set_custom_paths)
    set(options "")
    set(oneValueArgs LIB_VERSION CUDATK_VERSION)
    set(multiValueArgs "")
    cmake_parse_arguments(
        _FN
        "${options}"
        "${oneValueArgs}"
        "${multiValueArgs}"
        ${ARGN}
    )
    message(STATUS "cuDNN - Setting custom paths")
    if(_FN_CONDA_CUDNN)
        set(_CUDNN_ROOT $ENV{CONDA_PREFIX})
    else()
        set(_CUDNN_ROOT "${CMAKE_SOURCE_DIR}/external/cudnn${_FN_LIB_VERSION}-cuda${_FN_CUDATK_VERSION}")
    endif()

    set(_CUDNN_LIBRARY "${_CUDNN_ROOT}/lib")
    set(_CUDNN_INCLUDE_DIR "${_CUDNN_ROOT}/include")

    set(CUDNN_INCLUDE_DIR "${_CUDNN_INCLUDE_DIR}" PARENT_SCOPE)
    set(CUDNN_LIBRARY "${_CUDNN_LIBRARY}" PARENT_SCOPE)
    set(CUDNN_ROOT "${_CUDNN_ROOT}" PARENT_SCOPE)

    message(STATUS "cuDNN - Version: ${_FN_LIB_VERSION}")
    message(STATUS "cuDNN - Compiled for CUDA Toolkit version: ${_FN_CUDATK_VERSION}")
    message(STATUS "cuDNN - Manually set root to: ${_CUDNN_ROOT}")
    message(STATUS "cuDNN - Manually set include dir to: ${_CUDNN_INCLUDE_DIR}")
    message(STATUS "cuDNN - Manually set lib dir to: ${_CUDNN_LIBRARY}")
    msg_success("cuDNN - Successfully set paths")
endfunction()

function(CudnnHelper_set_conda_paths)
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
    message(STATUS "cuDNN - Setting paths using Conda prefix")
    set(_CUDNN_ROOT $ENV{CONDA_PREFIX})
    set(_CUDNN_LIBRARY "${_CUDNN_ROOT}/lib")
    set(_CUDNN_INCLUDE_DIR "${_CUDNN_ROOT}/include")
    set(CUDNN_INCLUDE_DIR "${_CUDNN_INCLUDE_DIR}" PARENT_SCOPE)
    set(CUDNN_LIBRARY "${_CUDNN_LIBRARY}" PARENT_SCOPE)
    set(CUDNN_ROOT "${_CUDNN_ROOT}" PARENT_SCOPE)
    message(STATUS "cuDNN - Manually set root to: ${_CUDNN_ROOT}")
    message(STATUS "cuDNN - Manually set include dir to: ${_CUDNN_INCLUDE_DIR}")
    message(STATUS "cuDNN - Manually set lib dir to: ${_CUDNN_LIBRARY}")
    msg_success("cuDNN - Successfully set paths")
endfunction()
