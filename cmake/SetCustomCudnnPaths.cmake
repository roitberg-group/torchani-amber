include_guard()
include(Msg)

# This function sets
# - CUDNN_LIBRARY_DIR
# - CUDNN_LIBRARY_PATH
# - CUDNN_INCLUDE_PATH
function(set_custom_cudnn_paths)
    set(options "")
    set(oneValueArgs LIBRARY_VERSION CUDA_VERSION)
    set(multiValueArgs "")
    cmake_parse_arguments(
        _CUDNN
        "${options}"
        "${oneValueArgs}"
        "${multiValueArgs}"
        ${ARGN}
    )
    # Either find the highest cudnn version or use a specific one if requested
    message(STATUS "cuDNN - Setting custom paths")
    string(REPLACE "." "" _CUDNN_CUDA_JOINT ${_CUDNN_CUDA_VERSION})

    if(_CUDNN_LIBRARY_VERSION STREQUAL "any")
        set(_CUDNN_LIBRARY_VERSION 1000)
        set(_CUDNN_PREFIX "${CMAKE_CURRENT_LIST_DIR}/lib/cudnn${_CUDNN_LIBRARY_VERSION}_cu${_CUDNN_CUDA_JOINT}")
        while(NOT EXISTS "${_CUDNN_PREFIX}/include")
            math(EXPR _CUDNN_LIBRARY_VERSION "${_CUDNN_LIBRARY_VERSION} - 1")
            set(_CUDNN_PREFIX "${CMAKE_CURRENT_LIST_DIR}/lib/cudnn${_CUDNN_LIBRARY_VERSION}_cu${_CUDNN_CUDA_JOINT}")
            if (_CUDNN_LIBRARY_VERSION LESS 100)
                msg_error("cuDNN - No version compatible with ${_CUDNN_CUDA_VERSION} CUDA was found")
            endif()
        endwhile()
    else()
        set(_CUDNN_PREFIX "${CMAKE_CURRENT_LIST_DIR}/lib/cudnn${_CUDNN_LIBRARY_VERSION}_cu${_CUDNN_CUDA_JOINT}")
    endif()

    set(_CUDNN_LIBRARY_DIR "${_CUDNN_PREFIX}/lib64")
    if(EXISTS "${_CUDNN_LIBARARY_DIR}/libcudnn.so.8")
        set(_CUDNN_LIBRARY_NAME "libcudnn.so.8")
    elseif(EXISTS "${_CUDNN_LIBARARY_DIR}/libcudnn.so.7")
        set(_CUDNN_LIBRARY_NAME "libcudnn.so.7")
    else()
        set(_CUDNN_LIBRARY_NAME "libcudnn.so")
    endif()

    set(CUDNN_INCLUDE_PATH "${_CUDNN_PREFIX}/include" PARENT_SCOPE)
    set(CUDNN_LIBRARY_DIR "${_CUDNN_LIBRARY_DIR}" PARENT_SCOPE)
    set(CUDNN_LIBRARY_PATH "${_CUDNN_LIBRARY_DIR}/${_CUDNN_LIBRARY_NAME}" PARENT_SCOPE)
    msg_success("cuDNN - Successfully set paths")
    message(STATUS "cuDNN - Using custom version: ${_CUDNN_LIBRARY_VERSION}")
    message(STATUS "cuDNN - Compatible with CUDA: ${_CUDNN_CUDA_VERSION}")
    message(STATUS "cuDNN - include path: ${CUDNN_PREFIX}/include")
    message(STATUS "cuDNN - library dir: ${CUDNN_LIBRARY_DIR}")
    message(STATUS "cuDNN - library path: ${CUDNN_LIBRARY_DIR}/${_CUDNN_LIBRARY_NAME}")
endfunction()
