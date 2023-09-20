include_guard()
include(Msg)

# Download LibTorch from the pytorch website and extract it into
# ./external/LIBTORCH_ROOT_NAME, if the libtorch directory does not exist.
# This function sets the variables LIBTORCH_LIBRARY_DIR, and LIBTORCH_ROOT
function(download_and_extract_libtorch)
    set(options "")
    set(oneValueArgs CXX11ABI LIBRARY_VERSION CUDA_VERSION)
    set(multiValueArgs "")
    cmake_parse_arguments(
        _LIBTORCH
        "${options}"
        "${oneValueArgs}"
        "${multiValueArgs}"
        ${ARGN}
    )

    message(STATUS "LibTorch - Download and extraction requested")
    string(REPLACE "." "" _LIBTORCH_CUDA_JOINT ${_LIBTORCH_CUDA_VERSION})

    if(_LIBTORCH_CXX11ABI)
        set(_CXX11ABI_STR "-cxx11-abi")
        set(_CXX11ABI_PATH "-cxx11-abi")
    else()
        set(_CXX11ABI_STR "")
        set(_CXX11ABI_PATH "-no-cxx11-abi")
    endif()

    # parse the libtorch directory and download url from the options
    set(_LIBTORCH_ZIPFILE "libtorch${_CXX11ABI_STR}-shared-with-deps-${_LIBTORCH_LIBRARY_VERSION}%2bcu${_LIBTORCH_CUDA_JOINT}.zip")

    set(_LIBTORCH_URL "https://download.pytorch.org/libtorch/cu${_LIBTORCH_CUDA_JOINT}/${_LIBTORCH_ZIPFILE}")
    set(_LIBTORCH_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/external/libtorch${_LIBTORCH_LIBRARY_VERSION}${_CXX11ABI_PATH}-cuda${_LIBTORCH_CUDA_VERSION}")

    if(EXISTS ${_LIBTORCH_ROOT})
        # Skip download if the requested libtorch version exists
        message(STATUS "LibTorch - Download was skipped because ${_LIBTORCH_ROOT} already exists.")
        message(STATUS "LibTorch - If you want to download LibTorch again delete dir and rerun.")
    else()
        # download libtorch to a temporary file _LIBTORCH_TMP_FILE
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
        file(REMOVE "${_LIBTORCH_TMP_FILE}")
    endif()
    set(_LIBTORCH_LIBRARY_DIR "${_LIBTORCH_ROOT}/lib")
    set(LIBTORCH_ROOT "${_LIBTORCH_ROOT}" PARENT_SCOPE)
    set(LIBTORCH_LIBRARY_DIR "${_LIBTORCH_LIBRARY_DIR}" PARENT_SCOPE)
    message(STATUS "LibTorch - Using version: ${_LIBTORCH_LIBRARY_VERSION}")
    message(STATUS "LibTorch - Compatible with CUDA: ${_LIBTORCH_CUDA_VERSION}")
    message(STATUS "LibTorch - Root dir: ${_LIBTORCH_ROOT}")
    message(STATUS "LibTorch - Lib dir: ${_LIBTORCH_LIBRARY_DIR}")
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/external/libtorch")
        msg_error("LibTorch - Could not be correctly downloaded and extracted")
    else()
        file(RENAME "${CMAKE_CURRENT_SOURCE_DIR}/external/libtorch" "${_LIBTORCH_ROOT}")
        msg_success("LibTorch - Downloaded and extracted")
    endif()
endfunction()
