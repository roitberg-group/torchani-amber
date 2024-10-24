include_guard()
include(Msg)

# The following function sets, in the PARENT scope:
# - LIBTORCH_LIBRARY_DIR
# - LIBTORCH_ROOT
#
# Download LibTorch from the pytorch website and extract it into
# ./external/LIBTORCH_ROOT_NAME, if the libtorch directory does not exist.
function(LibtorchHelper_download_and_install)
    set(options "")
    set(oneValueArgs COMPILE_WITH_CXX11ABI LIB_VERSION CUDA_VERSION)
    set(multiValueArgs "")
    cmake_parse_arguments(
        _FN
        "${options}"
        "${oneValueArgs}"
        "${multiValueArgs}"
        ${ARGN}
    )

    message(STATUS "LibTorch - Downloading and installing prebuilt binaries")
    string(REPLACE "." "" _LIBTORCH_CUDA_JOINT ${_FN_CUDA_VERSION})

    if(_FN_COMPILE_WITH_CXX11ABI)
        set(_CXX11ABI_STR "-cxx11-abi")
        set(_CXX11ABI_PATH "-cxx11-abi")
    else()
        set(_CXX11ABI_STR "")
        set(_CXX11ABI_PATH "-no-cxx11-abi")
    endif()

    # Parse the libtorch directory and download url from the options
    set(_LIBTORCH_ZIPFILE "libtorch${_CXX11ABI_STR}-shared-with-deps-${_FN_LIB_VERSION}%2bcu${_LIBTORCH_CUDA_JOINT}.zip")
    set(_LIBTORCH_URL "https://download.pytorch.org/libtorch/cu${_FN_CUDA_JOINT}/${_LIBTORCH_ZIPFILE}")
    set(_LIBTORCH_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/external/libtorch${_FN_LIB_VERSION}${_CXX11ABI_PATH}-cuda${_FN_CUDA_VERSION}")

    if(EXISTS ${_LIBTORCH_ROOT})
        # Skip download if the requested libtorch version exists
        message(STATUS "LibTorch - Download was skipped because ${_LIBTORCH_ROOT} already exists")
        message(STATUS "LibTorch - If you want to download LibTorch again delete this dir and reconfigure")
    else()
        # Download libtorch to a temporary file _LIBTORCH_TMP_FILE
        # and extract it into _LIBTORCH_ROOT
        set(_LIBTORCH_TMP_FILE "${CMAKE_CURRENT_LIST_DIR}/libtorch.zip")
        message(STATUS "LibTorch - Downloading")
        file(
            DOWNLOAD
                "${_LIBTORCH_URL}"
                "${_LIBTORCH_TMP_FILE}"
            SHOW_PROGRESS
        )

        message(STATUS "LibTorch - Extracting into ${_LIBTORCH_ROOT}")
        execute_process(
            COMMAND ${CMAKE_COMMAND} -E tar xfz "${_LIBTORCH_TMP_FILE}"
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/external"
        )
        if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/external/libtorch")
            msg_error("LibTorch - Download and/or installation failed")
        else()
            file(RENAME "${CMAKE_CURRENT_SOURCE_DIR}/external/libtorch" "${_LIBTORCH_ROOT}")
            message(STATUS "LibTorch - Downloaded and installeds")
        endif()
        file(REMOVE "${_LIBTORCH_TMP_FILE}")
    endif()
    set(_LIBTORCH_LIBRARY_DIR "${_LIBTORCH_ROOT}/lib")
    set(LIBTORCH_ROOT "${_LIBTORCH_ROOT}" PARENT_SCOPE)
    set(LIBTORCH_LIBRARY_DIR "${_LIBTORCH_LIBRARY_DIR}" PARENT_SCOPE)
    message(STATUS "LibTorch - Version: ${_FN_LIB_VERSION}")
    message(STATUS "LibTorch - Compiled for CUDA Toolkit version: ${_FN_CUDA_VERSION}")
    message(STATUS "LibTorch - Root dir: ${_LIBTORCH_ROOT}")
    message(STATUS "LibTorch - Lib dir: ${_LIBTORCH_LIBRARY_DIR}")
    msg_success("LibTorch - Done")
endfunction()
