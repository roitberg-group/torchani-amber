include_guard()
include(Msg)

# This function sets
# - CUDNN_ROOT
# - CUDNN_LIBRARY
# - CUDNN_INCLUDE_DIR
function(set_cudnn_dirs)
    set(options "")
    set(oneValueArgs CONDA_CUDA LIBRARY_VERSION CUDA_VERSION)
    set(multiValueArgs "")
    cmake_parse_arguments(
        _FN
        "${options}"
        "${oneValueArgs}"
        "${multiValueArgs}"
        ${ARGN}
    )
    # Either find the highest cudnn version or use a specific one if requested
    message(STATUS "cuDNN - Setting custom paths")
    string(REPLACE "." "" _FN_CUDA_JOINT ${_FN_CUDA_VERSION})
    if(_FN_CONDA_CUDA)
        set(_CUDNN_ROOT $ENV{CONDA_PREFIX})
    elseif(_FN_LIBRARY_VERSION STREQUAL "any")
        set(_FN_LIBRARY_VERSION 1000)
        set(_CUDNN_ROOT "${CMAKE_CURRENT_LIST_DIR}/lib/cudnn${_FN_LIBRARY_VERSION}_cu${_FN_CUDA_JOINT}")
        while(NOT EXISTS "${_CUDNN_ROOT}/include")
            math(EXPR _FN_LIBRARY_VERSION "${_FN_LIBRARY_VERSION} - 1")
            set(_CUDNN_ROOT "${CMAKE_CURRENT_LIST_DIR}/lib/cudnn${_FN_LIBRARY_VERSION}_cu${_FN_CUDA_JOINT}")
            if (_FN_LIBRARY_VERSION LESS 100)
                msg_error("cuDNN - No version compatible with ${_FN_CUDA_VERSION} CUDA was found")
            endif()
        endwhile()
    else()
        set(_CUDNN_ROOT "${CMAKE_CURRENT_LIST_DIR}/lib/cudnn${_FN_LIBRARY_VERSION}_cu${_FN_CUDA_JOINT}")
    endif()

    set(_CUDNN_LIBRARY "${_CUDNN_ROOT}/lib64")
    set(_CUDNN_INCLUDE_DIR "${_CUDNN_ROOT}/include")

    set(CUDNN_INCLUDE_DIR ${_CUDNN_INCLUDE_DIR} PARENT_SCOPE)
    set(CUDNN_LIBRARY ${_CUDNN_LIBRARY_DIR} PARENT_SCOPE)
    set(CUDNN_ROOT ${_CUDNN_ROOT} PARENT_SCOPE)

    msg_success("cuDNN - Successfully set paths")
    message(STATUS "cuDNN - Using custom version: ${_FN_LIBRARY_VERSION}")
    message(STATUS "cuDNN - Compatible with CUDA: ${_FN_CUDA_VERSION}")
    message(STATUS "cuDNN - root dir: ${_CUDNN_ROOT}")
    message(STATUS "cuDNN - include dir: ${_CUDNN_INCLUDE_DIR}")
    message(STATUS "cuDNN - library dir: ${_CUDNN_LIBRARY}")

    # Probably not needed
    if(EXISTS "${_CUDNN_LIBARARY}/libcudnn.so.8")
        set(_CUDNN_LIBRARY_PATH "${_CUDNN_LIBRARY}/libcudnn.so.8")
    elseif(EXISTS "${_CUDNN_LIBARARY}/libcudnn.so.7")
        set(_CUDNN_LIBRARY_PATH "${_CUDNN_LIBRARY}/libcudnn.so.7")
    else()
        set(_CUDNN_LIBRARY_PATH "${_CUDNN_LIBRARY}/libcudnn.so")
    endif()

    # message(STATUS "cuDNN - library path: ${CUDNN_LIBRARY_DIR}")
    # message(STATUS "cuDNN - include path: ${CUDNN_LIBRARY_DIR}/${_CUDNN_LIBRARY_NAME}")
    # set(CUDNN_LIBRARY_PATH ${_CUDNN_LIBRARY_PATH} PARENT_SCOPE)
    # set(CUDNN_INCLUDE_PATH ${_CUDNN_INCLUDE_PATH} PARENT_SCOPE)
endfunction()
