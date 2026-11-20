include_guard()
include(Torchani_Msg)

# Download the linux x86_64 wheel PACKAGE==VERSION from PyPI, and copy the libs and
# headers under nvidia/SUBDIR into LIBTORCH_DIR/lib and LIBTORCH_DIR/include.
# Unversioned symlinks (e.g. libcudnn.so -> libcudnn.so.9) are created for the libs
function(_Torchani_LibtorchHelper_add_nvidia_wheel)
    set(options "")
    set(oneValueArgs PACKAGE VERSION SUBDIR LIBTORCH_DIR WORK_DIR)
    set(multiValueArgs "")
    cmake_parse_arguments(
        _FN
        "${options}"
        "${oneValueArgs}"
        "${multiValueArgs}"
        ${ARGN}
    )

    message(STATUS "LibTorch - Adding ${_FN_PACKAGE}==${_FN_VERSION}")
    set(_META_FILE "${_FN_WORK_DIR}/${_FN_PACKAGE}.json")
    file(
        DOWNLOAD
            "https://pypi.org/pypi/${_FN_PACKAGE}/${_FN_VERSION}/json"
            "${_META_FILE}"
        STATUS _DOWNLOAD_STATUS
    )
    list(GET _DOWNLOAD_STATUS 0 _DOWNLOAD_CODE)
    if(NOT _DOWNLOAD_CODE EQUAL 0)
        Torchani_Msg_error("LibTorch - Could not find ${_FN_PACKAGE}==${_FN_VERSION} in PyPI")
    endif()
    file(READ "${_META_FILE}" _META)

    set(_WHEEL_URL "")
    string(JSON _NUM_FILES LENGTH "${_META}" urls)
    math(EXPR _LAST_IDX "${_NUM_FILES} - 1")
    foreach(_IDX RANGE ${_LAST_IDX})
        string(JSON _FILENAME GET "${_META}" urls ${_IDX} filename)
        if(_FILENAME MATCHES "manylinux.*_x86_64\\.whl$")
            string(JSON _WHEEL_URL GET "${_META}" urls ${_IDX} url)
            string(JSON _WHEEL_SHA256 GET "${_META}" urls ${_IDX} digests sha256)
        endif()
    endforeach()
    if(_WHEEL_URL STREQUAL "")
        Torchani_Msg_error("LibTorch - No linux x86_64 wheel found for ${_FN_PACKAGE}==${_FN_VERSION}")
    endif()

    set(_WHEEL_FILE "${_FN_WORK_DIR}/${_FN_PACKAGE}.whl")
    file(
        DOWNLOAD
            "${_WHEEL_URL}"
            "${_WHEEL_FILE}"
        EXPECTED_HASH SHA256=${_WHEEL_SHA256}
        SHOW_PROGRESS
        STATUS _DOWNLOAD_STATUS
    )
    list(GET _DOWNLOAD_STATUS 0 _DOWNLOAD_CODE)
    list(GET _DOWNLOAD_STATUS 1 _DOWNLOAD_MSG)
    if(NOT _DOWNLOAD_CODE EQUAL 0)
        Torchani_Msg_error("LibTorch - Download of ${_WHEEL_URL} failed: ${_DOWNLOAD_MSG}")
    endif()

    set(_WHEEL_ROOT "${_FN_WORK_DIR}/nvidia/${_FN_SUBDIR}")
    file(
        ARCHIVE_EXTRACT
            INPUT "${_WHEEL_FILE}"
            DESTINATION "${_FN_WORK_DIR}"
            PATTERNS "nvidia/${_FN_SUBDIR}/lib/*.so*" "nvidia/${_FN_SUBDIR}/include"
    )
    file(REMOVE "${_WHEEL_FILE}")
    if(NOT EXISTS "${_WHEEL_ROOT}/lib" OR NOT EXISTS "${_WHEEL_ROOT}/include")
        Torchani_Msg_error("LibTorch - Wheel ${_FN_PACKAGE} has no nvidia/${_FN_SUBDIR}/{lib,include} dirs")
    endif()
    file(COPY "${_WHEEL_ROOT}/include/" DESTINATION "${_FN_LIBTORCH_DIR}/include")
    file(COPY "${_WHEEL_ROOT}/lib/" DESTINATION "${_FN_LIBTORCH_DIR}/lib")

    file(GLOB _LIB_PATHS RELATIVE "${_WHEEL_ROOT}/lib" "${_WHEEL_ROOT}/lib/lib*.so.*")
    foreach(_LIB_NAME IN LISTS _LIB_PATHS)
        string(REGEX REPLACE "\\.so\\..*$" ".so" _LINK_NAME "${_LIB_NAME}")
        file(CREATE_LINK "${_LIB_NAME}" "${_FN_LIBTORCH_DIR}/lib/${_LINK_NAME}" SYMBOLIC)
    endforeach()
endfunction()

# The following function sets, in the PARENT scope:
# - LIBTORCH_LIBRARY_DIR
# - LIBTORCH_ROOT
#
# Download LibTorch from the pytorch website and extract it into
# INSTALL_DIR/LIBTORCH_ROOT_NAME, if said dir doesn't exist. If USE_CUDA is false, the
# CPU-only LibTorch is downloaded and CUDATK_VERSION is ignored
#
# If USE_CUDA is true, the libs and headers of cuDNN, NCCL, cuSPARSELt and NVSHMEM,
# which LibTorch requires but are not in the CUDA Toolkit, are taken from the
# nvidia-*-cu<cuda-major> PyPI wheels with the *_WHEEL_VERSION versions, and added to
# LibTorch's lib and include dirs
function(Torchani_LibtorchHelper_download_and_install)
    set(options "")
    set(
        oneValueArgs
        USE_CUDA
        LIB_VERSION
        PATCH_VERSION
        CUDATK_VERSION
        INSTALL_DIR
        CUDNN_WHEEL_VERSION
        NCCL_WHEEL_VERSION
        CUSPARSELT_WHEEL_VERSION
        NVSHMEM_WHEEL_VERSION
    )
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
            if(_FN_USE_CUDA)
                string(REGEX REPLACE "\\..*$" "" _CUDA_MAJOR ${_FN_CUDATK_VERSION})
                set(_WHEEL_WORK_DIR "${PROJECT_BINARY_DIR}/nvidia-wheels")
                file(REMOVE_RECURSE "${_WHEEL_WORK_DIR}")
                file(MAKE_DIRECTORY "${_WHEEL_WORK_DIR}")
                foreach(_NAME cudnn nccl cusparselt nvshmem)
                    string(TOUPPER ${_NAME} _NAME_UPPER)
                    _Torchani_LibtorchHelper_add_nvidia_wheel(
                        PACKAGE "nvidia-${_NAME}-cu${_CUDA_MAJOR}"
                        VERSION ${_FN_${_NAME_UPPER}_WHEEL_VERSION}
                        SUBDIR ${_NAME}
                        LIBTORCH_DIR "${_FN_INSTALL_DIR}/libtorch"
                        WORK_DIR "${_WHEEL_WORK_DIR}"
                    )
                endforeach()
                file(REMOVE_RECURSE "${_WHEEL_WORK_DIR}")
            endif()
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
