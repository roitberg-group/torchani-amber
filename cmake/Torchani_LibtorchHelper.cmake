include_guard()
include(Torchani_Msg)

# The following function sets, in the PARENT scope:
# - LIBTORCH_LIBRARY_DIR
# - LIBTORCH_ROOT
#
# Download LibTorch from the pytorch website and extract it into
# ./external/LIBTORCH_ROOT_NAME, if said dir doesn't exist
function(Torchani_LibtorchHelper_download_and_install)
    set(options "")
    set(oneValueArgs LIB_VERSION PATCH_VERSION CUDATK_VERSION)
    set(multiValueArgs "")
    cmake_parse_arguments(
        _FN
        "${options}"
        "${oneValueArgs}"
        "${multiValueArgs}"
        ${ARGN}
    )

    message(STATUS "LibTorch - Downloading and installing prebuilt binaries")
    string(REPLACE "." "" _CUDA_JOINT ${_FN_CUDATK_VERSION})

    # Parse the libtorch directory and download url from the options
    set(_LIBTORCH_ZIPFILE "libtorch-shared-with-deps-${_FN_LIB_VERSION}.${_FN_PATCH_VERSION}%2Bcu${_CUDA_JOINT}.zip")
    set(_LIBTORCH_URL "https://download.pytorch.org/libtorch/cu${_CUDA_JOINT}/${_LIBTORCH_ZIPFILE}")
    set(_LIBTORCH_ROOT "${PROJECT_SOURCE_DIR}/external/libtorch${_FN_LIB_VERSION}.${_FN_PATCH_VERSION}-cuda${_FN_CUDATK_VERSION}")

    if(EXISTS ${_LIBTORCH_ROOT})
        # Skip download if the requested libtorch version exists
        message(STATUS "LibTorch - Download was skipped because ${_LIBTORCH_ROOT} already exists")
        message(STATUS "LibTorch - If you want to download LibTorch again delete ${_LIBTORCH_ROOT} and reconfigure")
    else()
        # Download libtorch to a temporary file _LIBTORCH_TMP_FILE
        # and extract it into _LIBTORCH_ROOT
        set(_LIBTORCH_TMP_FILE "${PROJECT_BINARY_DIR}/libtorch.zip")
        message(STATUS "LibTorch - Downloading")
        file(
            DOWNLOAD
                "${_LIBTORCH_URL}"
                "${_LIBTORCH_TMP_FILE}"
            SHOW_PROGRESS
            STATUS _DOWNLOAD_STATUS
        )
        list(GET _DOWNLOAD_STATUS 0 _DOWNLOAD_CODE)
        list(GET _DOWNLOAD_STATUS 1 _DOWNLOAD_MSG)
        if(NOT _DOWNLOAD_CODE EQUAL 0)
            file(REMOVE "${_LIBTORCH_TMP_FILE}")
            Torchani_Msg_error("LibTorch - Download of ${_LIBTORCH_URL} failed: ${_DOWNLOAD_MSG}")
        endif()

        message(STATUS "LibTorch - Extracting into ${_LIBTORCH_ROOT}")
        execute_process(
            COMMAND ${CMAKE_COMMAND} -E tar xfz "${_LIBTORCH_TMP_FILE}"
            WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/external"
        )
        if(NOT EXISTS "${PROJECT_SOURCE_DIR}/external/libtorch")
            Torchani_Msg_error("LibTorch - Extraction failed")
        else()
            file(RENAME "${PROJECT_SOURCE_DIR}/external/libtorch" "${_LIBTORCH_ROOT}")
            message(STATUS "LibTorch - Downloaded and installed")
        endif()
        file(REMOVE "${_LIBTORCH_TMP_FILE}")
    endif()
    set(_LIBTORCH_LIBRARY_DIR "${_LIBTORCH_ROOT}/lib")
    set(LIBTORCH_ROOT "${_LIBTORCH_ROOT}" PARENT_SCOPE)
    set(LIBTORCH_LIBRARY_DIR "${_LIBTORCH_LIBRARY_DIR}" PARENT_SCOPE)
    message(STATUS "LibTorch - Version: ${_FN_LIB_VERSION}.${_FN_PATCH_VERSION}")
    message(STATUS "LibTorch - Compiled for CUDA Toolkit version: ${_FN_CUDATK_VERSION}")
    message(STATUS "LibTorch - Manually set root to: ${_LIBTORCH_ROOT}")
    message(STATUS "LibTorch - Manually set lib dir to: ${_LIBTORCH_LIBRARY_DIR}")
    Torchani_Msg_success("LibTorch - Done")
endfunction()
