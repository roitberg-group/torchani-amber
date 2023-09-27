include_guard()
include(Msg)

# This function sets
# - CUDNN_ROOT
# - CUDNN_LIBRARY
# - CUDNN_INCLUDE_DIR
function(set_cudnn_dirs)
    set(options "")
    set(oneValueArgs CONDA_CUDNN LIBRARY_VERSION CUDA_VERSION)
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
    if(_FN_CONDA_CUDNN)
        set(_CUDNN_ROOT $ENV{CONDA_PREFIX})
    else()
        set(_CUDNN_ROOT "${CMAKE_SOURCE_DIR}/external/cudnn${_FN_LIBRARY_VERSION}-cuda${_FN_CUDA_VERSION}")
    endif()

    set(_CUDNN_LIBRARY "${_CUDNN_ROOT}/lib")
    set(_CUDNN_INCLUDE_DIR "${_CUDNN_ROOT}/include")

    set(CUDNN_INCLUDE_DIR "${_CUDNN_INCLUDE_DIR}" PARENT_SCOPE)
    set(CUDNN_LIBRARY "${_CUDNN_LIBRARY}" PARENT_SCOPE)
    set(CUDNN_ROOT "${_CUDNN_ROOT}" PARENT_SCOPE)

    message(STATUS "cuDNN - Using custom version: ${_FN_LIBRARY_VERSION}")
    message(STATUS "cuDNN - Compatible with CUDA: ${_FN_CUDA_VERSION}")
    message(STATUS "cuDNN - root dir: ${_CUDNN_ROOT}")
    message(STATUS "cuDNN - include dir: ${_CUDNN_INCLUDE_DIR}")
    message(STATUS "cuDNN - library dir: ${_CUDNN_LIBRARY}")
    msg_success("cuDNN - Successfully set paths")

    # Probably not needed
    # if(EXISTS "${_CUDNN_LIBARARY}/libcudnn.so.8")
        # set(_CUDNN_LIBRARY_PATH "${_CUDNN_LIBRARY}/libcudnn.so.8")
    # elseif(EXISTS "${_CUDNN_LIBARARY}/libcudnn.so.7")
        # set(_CUDNN_LIBRARY_PATH "${_CUDNN_LIBRARY}/libcudnn.so.7")
    # else()
        # set(_CUDNN_LIBRARY_PATH "${_CUDNN_LIBRARY}/libcudnn.so")
    # endif()
    # message(STATUS "cuDNN - library path: ${CUDNN_LIBRARY_DIR}")
    # message(STATUS "cuDNN - include path: ${CUDNN_LIBRARY_DIR}/${_CUDNN_LIBRARY_NAME}")
    # set(CUDNN_LIBRARY_PATH ${_CUDNN_LIBRARY_PATH} PARENT_SCOPE)
    # set(CUDNN_INCLUDE_PATH ${_CUDNN_INCLUDE_PATH} PARENT_SCOPE)
endfunction()
