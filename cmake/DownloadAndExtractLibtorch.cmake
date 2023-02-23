include_guard()
include(Msg)

# Download LibTorch from the pytorch website and extract it into
# ./lib/_LIBTORCH_DIR, if the libtorch directory does not exist.
# This function sets the variable LIBTORCH_EXTRACTED_DIR
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
        set(_CXX11ABI_PATH "cxx11-abi")
    else()
        set(_CXX11ABI_STR "")
        set(_CXX11ABI_PATH "no-cxx11-abi")
    endif()

    # parse the libtorch directory and download url from the options
    if(_LIBTORCH_LIBRARY_VERSION STREQUAL "latest") # Nightly libtorch has a slightly different file name
        set(_LIBTORCH_ZIPFILE "libtorch${_CXX11ABI_STR}-shared-with-deps-latest.zip")
        set(_START_PATH "libtorch/nightly")
    else()
        set(_LIBTORCH_ZIPFILE "libtorch${_CXX11ABI_STR}-shared-with-deps-${_LIBTORCH_LIBRARY_VERSION}%2bcu${_LIBTORCH_CUDA_JOINT}.zip")
        set(_START_PATH "libtorch")
    endif()

    set(_LIBTORCH_URL "https://download.pytorch.org/${_START_PATH}/cu${_LIBTORCH_CUDA_JOINT}/${_LIBTORCH_ZIPFILE}")
    set(_LIBTORCH_DIR "${CMAKE_CURRENT_SOURCE_DIR}/lib/libtorch-${_LIBTORCH_LIBRARY_VERSION}-${_CXX11ABI_PATH}-cuda-${_LIBTORCH_CUDA_VERSION}")

    if(EXISTS ${_LIBTORCH_DIR})
        # Skip download if the requested libtorch version exists
        message(STATUS "LibTorch - Installation was skipped because ${_LIBTORCH_DIR} already exists.")
        message(STATUS "LibTorch - If you want to install LibTorch again delete dir and rerun.")
    else()
        # download libtorch to a temporary file _LIBTORCH_TMP_FILE
        # and extract it into _LIBTORCH_DIR
        set(_LIBTORCH_TMP_FILE "${CMAKE_CURRENT_LIST_DIR}/libtorch.zip")
        message(STATUS "LibTorch - Downloading")
        file(
            DOWNLOAD
                ${_LIBTORCH_URL}
                ${_LIBTORCH_TMP_FILE}
            SHOW_PROGRESS
        )

        message(STATUS "LibTorch - Extracting into ${_LIBTORCH_DIR}")
        execute_process(
            COMMAND ${CMAKE_COMMAND} -E tar xfz ${_LIBTORCH_TMP_FILE}
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/lib"
        )
        file(REMOVE ${_LIBTORCH_TMP_FILE})

        if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/lib/libtorch")
            msg_error("LibTorch - Could not be correctly downloaded and extracted")
        else()
            file(RENAME "${CMAKE_CURRENT_SOURCE_DIR}/lib/libtorch" ${_LIBTORCH_DIR})
            set(LIBTORCH_EXTRACTED_DIR ${_LIBTORCH_DIR} PARENT_SCOPE)
            set(LIBTORCH_EXTRACTED_LIB_DIR "${_LIBTORCH_DIR}/lib" PARENT_SCOPE)
            msg_success("LibTorch - Successfully downloaded and extracted")
        endif()
    endif()
    message(STATUS "LibTorch - Using version: ${_LIBTORCH_LIBRARY_VERSION}")
    message(STATUS "LibTorch - Compatible with CUDA: ${_LIBTORCH_CUDA_VERSION}")
    message(STATUS "LibTorch - Extracted dir: ${_LIBTORCH_DIR}")
    message(STATUS "LibTorch - Extracted lib dir: ${_LIBTORCH_DIR}/lib")
endfunction()
