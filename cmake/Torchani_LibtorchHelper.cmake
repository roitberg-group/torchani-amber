include_guard()
include(Torchani_Msg)

# The following function sets, in the PARENT scope:
# - LIBTORCH_LIBRARY_DIR
# - LIBTORCH_ROOT
#
# Download LibTorch from the pytorch website and extract it into
# INSTALL_DIR/LIBTORCH_ROOT_NAME, if said dir doesn't exist. If USE_CUDA is false, the
# CPU-only LibTorch is downloaded and CUDATK_VERSION is ignored
function(Torchani_LibtorchHelper_download_and_install)
    set(options "")
    set(oneValueArgs USE_CUDA LIB_VERSION PATCH_VERSION CUDATK_VERSION INSTALL_DIR)
    set(multiValueArgs "")
    cmake_parse_arguments(
        _FN
        "${options}"
        "${oneValueArgs}"
        "${multiValueArgs}"
        ${ARGN}
    )

    message(STATUS "LibTorch - Downloading and installing prebuilt binaries")
    if(_FN_USE_CUDA)
        string(REPLACE "." "" _CUDA_JOINT ${_FN_CUDATK_VERSION})
        set(_BACKEND "cu${_CUDA_JOINT}")
        set(_ROOT_SUFFIX "cuda${_FN_CUDATK_VERSION}")
    else()
        set(_BACKEND "cpu")
        set(_ROOT_SUFFIX "cpu")
    endif()

    # Parse the libtorch directory and download url from the options
    set(_LIBTORCH_ZIPFILE "libtorch-shared-with-deps-${_FN_LIB_VERSION}.${_FN_PATCH_VERSION}%2B${_BACKEND}.zip")
    set(_LIBTORCH_URL "https://download.pytorch.org/libtorch/${_BACKEND}/${_LIBTORCH_ZIPFILE}")
    set(_LIBTORCH_ROOT "${_FN_INSTALL_DIR}/libtorch${_FN_LIB_VERSION}.${_FN_PATCH_VERSION}-${_ROOT_SUFFIX}")

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
        file(ARCHIVE_EXTRACT INPUT "${_LIBTORCH_TMP_FILE}" DESTINATION "${_FN_INSTALL_DIR}")
        if(NOT EXISTS "${_FN_INSTALL_DIR}/libtorch")
            Torchani_Msg_error("LibTorch - Extracted archive has no top-level 'libtorch' dir")
        else()
            file(RENAME "${_FN_INSTALL_DIR}/libtorch" "${_LIBTORCH_ROOT}")
            message(STATUS "LibTorch - Downloaded and installed")
        endif()
        file(REMOVE "${_LIBTORCH_TMP_FILE}")
    endif()
    set(_LIBTORCH_LIBRARY_DIR "${_LIBTORCH_ROOT}/lib")
    set(LIBTORCH_ROOT "${_LIBTORCH_ROOT}" PARENT_SCOPE)
    set(LIBTORCH_LIBRARY_DIR "${_LIBTORCH_LIBRARY_DIR}" PARENT_SCOPE)
    message(STATUS "LibTorch - Version: ${_FN_LIB_VERSION}.${_FN_PATCH_VERSION}")
    if(_FN_USE_CUDA)
        message(STATUS "LibTorch - Compiled for CUDA Toolkit version: ${_FN_CUDATK_VERSION}")
    else()
        message(STATUS "LibTorch - CPU-only")
    endif()
    message(STATUS "LibTorch - Manually set root to: ${_LIBTORCH_ROOT}")
    message(STATUS "LibTorch - Manually set lib dir to: ${_LIBTORCH_LIBRARY_DIR}")
    Torchani_Msg_success("LibTorch - Done")
endfunction()
