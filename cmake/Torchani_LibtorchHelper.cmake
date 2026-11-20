include_guard()
include(Torchani_Msg)

# Find the linux x86_64 wheel PACKAGE==VERSION in PyPI, and set its url and sha256 in the
# PARENT scope, as OUT_URL and OUT_SHA256
function(_Torchani_LibtorchHelper_find_nvidia_wheel)
    set(options "")
    set(oneValueArgs PACKAGE VERSION WORK_DIR OUT_URL OUT_SHA256)
    set(multiValueArgs "")
    cmake_parse_arguments(
        _FN
        "${options}"
        "${oneValueArgs}"
        "${multiValueArgs}"
        ${ARGN}
    )

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
    message(STATUS "LibTorch - Found ${_FN_PACKAGE}==${_FN_VERSION}")
    set(${_FN_OUT_URL} "${_WHEEL_URL}" PARENT_SCOPE)
    set(${_FN_OUT_SHA256} "${_WHEEL_SHA256}" PARENT_SCOPE)
endfunction()

# Download the wheel at URL, and copy the libs and headers under nvidia/SUBDIR into
# LIBTORCH_DIR/lib and LIBTORCH_DIR/include.
# Unversioned symlinks (e.g. libcudnn.so -> libcudnn.so.9) are created for the libs
function(_Torchani_LibtorchHelper_add_nvidia_wheel)
    set(options "")
    set(oneValueArgs PACKAGE URL SHA256 SUBDIR LIBTORCH_DIR WORK_DIR)
    set(multiValueArgs "")
    cmake_parse_arguments(
        _FN
        "${options}"
        "${oneValueArgs}"
        "${multiValueArgs}"
        ${ARGN}
    )

    message(STATUS "LibTorch - Adding ${_FN_PACKAGE}")
    set(_WHEEL_FILE "${_FN_WORK_DIR}/${_FN_PACKAGE}.whl")
    file(
        DOWNLOAD
            "${_FN_URL}"
            "${_WHEEL_FILE}"
        EXPECTED_HASH SHA256=${_FN_SHA256}
        SHOW_PROGRESS
        STATUS _DOWNLOAD_STATUS
    )
    list(GET _DOWNLOAD_STATUS 0 _DOWNLOAD_CODE)
    list(GET _DOWNLOAD_STATUS 1 _DOWNLOAD_MSG)
    if(NOT _DOWNLOAD_CODE EQUAL 0)
        Torchani_Msg_error("LibTorch - Download of ${_FN_URL} failed: ${_DOWNLOAD_MSG}")
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

# Set OUT_PATCH in the PARENT scope to the largest patch version among ITEMS, captured by
# the first group of REGEX, or to "" if no item matches REGEX
function(_Torchani_LibtorchHelper_find_latest_patch)
    set(options "")
    set(oneValueArgs REGEX OUT_PATCH)
    set(multiValueArgs ITEMS)
    cmake_parse_arguments(
        _FN
        "${options}"
        "${oneValueArgs}"
        "${multiValueArgs}"
        ${ARGN}
    )
    set(_LATEST_PATCH "")
    foreach(_ITEM IN LISTS _FN_ITEMS)
        if(_ITEM MATCHES "${_FN_REGEX}")
            if(_LATEST_PATCH STREQUAL "" OR CMAKE_MATCH_1 GREATER _LATEST_PATCH)
                set(_LATEST_PATCH "${CMAKE_MATCH_1}")
            endif()
        endif()
    endforeach()
    set(${_FN_OUT_PATCH} "${_LATEST_PATCH}" PARENT_SCOPE)
endfunction()

# The following function sets, in the PARENT scope:
# - LIBTORCH_LIBRARY_DIR
# - LIBTORCH_ROOT
#
# Download LibTorch from the pytorch website and extract it into
# INSTALL_DIR/libtorch<M.m.p>-<cuda<CUDATK_VERSION>|cpu>, if said dir doesn't exist. If
# USE_CUDA is false, the CPU-only LibTorch is downloaded and CUDATK_VERSION is ignored
#
# If LIB_VERSION is M.m.p, that exact version is used. If it is M.m, the latest patch
# version already in INSTALL_DIR is used, and if there is none, the latest patch version
# listed in the LibTorch index is downloaded
#
# If USE_CUDA is true, the libs and headers of cuDNN, NCCL, cuSPARSELt and NVSHMEM,
# which LibTorch requires but are not in the CUDA Toolkit, are taken from the
# nvidia-*-cu<cuda-major> PyPI wheels, and added to LibTorch's lib and include dirs. The
# wheel versions are the ones pinned by the torch wheel with the same torch and CUDA
# versions, read from its metadata in the PyTorch index
function(Torchani_LibtorchHelper_download_and_install)
    set(options "")
    set(
        oneValueArgs
        USE_CUDA
        LIB_VERSION
        CUDATK_VERSION
        INSTALL_DIR
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

    if(_FN_LIB_VERSION MATCHES "^[0-9]+\\.[0-9]+$")
        file(
            GLOB _ROOT_NAMES
            RELATIVE "${_FN_INSTALL_DIR}"
            LIST_DIRECTORIES TRUE
            "${_FN_INSTALL_DIR}/libtorch${_FN_LIB_VERSION}.*-${_ROOT_SUFFIX}"
        )
        _Torchani_LibtorchHelper_find_latest_patch(
            ITEMS ${_ROOT_NAMES}
            REGEX "^libtorch${_FN_LIB_VERSION}\\.([0-9]+)-${_ROOT_SUFFIX}$"
            OUT_PATCH _PATCH
        )
        if(_PATCH STREQUAL "")
            set(_FULL_VERSION "")
        else()
            set(_FULL_VERSION "${_FN_LIB_VERSION}.${_PATCH}")
        endif()
    else()
        set(_FULL_VERSION "${_FN_LIB_VERSION}")
    endif()
    set(_LIBTORCH_ROOT "${_FN_INSTALL_DIR}/libtorch${_FULL_VERSION}-${_ROOT_SUFFIX}")
    set(_LIBTORCH_INDEX_URL "https://download.pytorch.org/libtorch/${_BACKEND}/")

    if(NOT _FULL_VERSION STREQUAL "" AND EXISTS "${_LIBTORCH_ROOT}")
        # Skip download if the requested libtorch version exists
        message(STATUS "LibTorch - Download was skipped because ${_LIBTORCH_ROOT} already exists")
        message(STATUS "LibTorch - If you want to download LibTorch again delete ${_LIBTORCH_ROOT} and reconfigure")
    else()
        # All metadata is fetched and all urls are resolved before any large download
        set(_WORK_DIR "${PROJECT_BINARY_DIR}/libtorch-download")
        file(REMOVE_RECURSE "${_WORK_DIR}")
        file(MAKE_DIRECTORY "${_WORK_DIR}")

        set(_LIBTORCH_INDEX_FILE "${_WORK_DIR}/libtorch-index.html")
        file(DOWNLOAD "${_LIBTORCH_INDEX_URL}" "${_LIBTORCH_INDEX_FILE}" STATUS _DOWNLOAD_STATUS)
        list(GET _DOWNLOAD_STATUS 0 _DOWNLOAD_CODE)
        if(NOT _DOWNLOAD_CODE EQUAL 0)
            Torchani_Msg_error("LibTorch - Download of ${_LIBTORCH_INDEX_URL} failed")
        endif()
        file(READ "${_LIBTORCH_INDEX_FILE}" _LIBTORCH_INDEX)
        if(_FULL_VERSION STREQUAL "")
            string(
                REGEX MATCHALL
                    "libtorch-shared-with-deps-${_FN_LIB_VERSION}\\.[0-9]+%2B${_BACKEND}\\.zip"
                    _INDEX_ZIPFILES
                    "${_LIBTORCH_INDEX}"
            )
            _Torchani_LibtorchHelper_find_latest_patch(
                ITEMS ${_INDEX_ZIPFILES}
                REGEX "^libtorch-shared-with-deps-${_FN_LIB_VERSION}\\.([0-9]+)%2B"
                OUT_PATCH _PATCH
            )
            if(_PATCH STREQUAL "")
                Torchani_Msg_error("LibTorch - No ${_FN_LIB_VERSION}.* version is listed in ${_LIBTORCH_INDEX_URL}")
            endif()
            set(_FULL_VERSION "${_FN_LIB_VERSION}.${_PATCH}")
            set(_LIBTORCH_ROOT "${_FN_INSTALL_DIR}/libtorch${_FULL_VERSION}-${_ROOT_SUFFIX}")
            message(STATUS "LibTorch - Latest ${_FN_LIB_VERSION}.* version in the index is ${_FULL_VERSION}")
        endif()
        set(_LIBTORCH_ZIPFILE "libtorch-shared-with-deps-${_FULL_VERSION}%2B${_BACKEND}.zip")
        set(_LIBTORCH_URL "${_LIBTORCH_INDEX_URL}${_LIBTORCH_ZIPFILE}")
        string(FIND "${_LIBTORCH_INDEX}" "${_LIBTORCH_ZIPFILE}" _ZIPFILE_POS)
        if(_ZIPFILE_POS EQUAL -1)
            Torchani_Msg_error("LibTorch - ${_LIBTORCH_ZIPFILE} is not listed in ${_LIBTORCH_INDEX_URL}")
        endif()

        set(_NVIDIA_NAMES cudnn nccl cusparselt nvshmem)
        if(_FN_USE_CUDA)
            string(REGEX REPLACE "\\..*$" "" _CUDA_MAJOR ${_FN_CUDATK_VERSION})
            # NOTE: The CPython tag (cp312) and the manylinux tag of the torch wheel
            # are hardcoded. The CPython version should be irrelevant to the pinned
            # nvidia wheels, which contain no Python-specific libs
            set(_TORCH_WHEEL_NAME "torch-${_FULL_VERSION}%2B${_BACKEND}-cp312-cp312-manylinux_2_28_x86_64.whl")
            set(_TORCH_META_URL "https://download.pytorch.org/whl/${_BACKEND}/${_TORCH_WHEEL_NAME}.metadata")
            set(_TORCH_META_FILE "${_WORK_DIR}/torch.metadata")
            file(DOWNLOAD "${_TORCH_META_URL}" "${_TORCH_META_FILE}" STATUS _DOWNLOAD_STATUS)
            list(GET _DOWNLOAD_STATUS 0 _DOWNLOAD_CODE)
            if(NOT _DOWNLOAD_CODE EQUAL 0)
                Torchani_Msg_error("LibTorch - Download of ${_TORCH_META_URL} failed")
            endif()
            file(READ "${_TORCH_META_FILE}" _TORCH_META)

            foreach(_NAME IN LISTS _NVIDIA_NAMES)
                set(_PACKAGE "nvidia-${_NAME}-cu${_CUDA_MAJOR}")
                if(NOT _TORCH_META MATCHES "Requires-Dist: ${_PACKAGE}==([^ ;\r\n]+)")
                    Torchani_Msg_error("LibTorch - The torch wheel metadata doesn't pin ${_PACKAGE}")
                endif()
                _Torchani_LibtorchHelper_find_nvidia_wheel(
                    PACKAGE ${_PACKAGE}
                    VERSION ${CMAKE_MATCH_1}
                    WORK_DIR "${_WORK_DIR}"
                    OUT_URL _WHEEL_URL_${_NAME}
                    OUT_SHA256 _WHEEL_SHA256_${_NAME}
                )
            endforeach()
        endif()

        # Download libtorch to a temporary file _LIBTORCH_TMP_FILE
        # and extract it into _LIBTORCH_ROOT
        set(_LIBTORCH_TMP_FILE "${_WORK_DIR}/libtorch.zip")
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
            file(REMOVE_RECURSE "${_WORK_DIR}")
            Torchani_Msg_error("LibTorch - Download of ${_LIBTORCH_URL} failed: ${_DOWNLOAD_MSG}")
        endif()

        message(STATUS "LibTorch - Extracting into ${_LIBTORCH_ROOT}")
        file(ARCHIVE_EXTRACT INPUT "${_LIBTORCH_TMP_FILE}" DESTINATION "${_FN_INSTALL_DIR}")
        file(REMOVE "${_LIBTORCH_TMP_FILE}")
        if(NOT EXISTS "${_FN_INSTALL_DIR}/libtorch")
            Torchani_Msg_error("LibTorch - Extracted archive has no top-level 'libtorch' dir")
        endif()
        if(_FN_USE_CUDA)
            foreach(_NAME IN LISTS _NVIDIA_NAMES)
                _Torchani_LibtorchHelper_add_nvidia_wheel(
                    PACKAGE "nvidia-${_NAME}-cu${_CUDA_MAJOR}"
                    URL "${_WHEEL_URL_${_NAME}}"
                    SHA256 "${_WHEEL_SHA256_${_NAME}}"
                    SUBDIR ${_NAME}
                    LIBTORCH_DIR "${_FN_INSTALL_DIR}/libtorch"
                    WORK_DIR "${_WORK_DIR}"
                )
            endforeach()
        endif()
        file(RENAME "${_FN_INSTALL_DIR}/libtorch" "${_LIBTORCH_ROOT}")
        file(REMOVE_RECURSE "${_WORK_DIR}")
        message(STATUS "LibTorch - Downloaded and installed")
    endif()
    set(_LIBTORCH_LIBRARY_DIR "${_LIBTORCH_ROOT}/lib")
    set(LIBTORCH_ROOT "${_LIBTORCH_ROOT}" PARENT_SCOPE)
    set(LIBTORCH_LIBRARY_DIR "${_LIBTORCH_LIBRARY_DIR}" PARENT_SCOPE)
    message(STATUS "LibTorch - Version: ${_FULL_VERSION}")
    if(_FN_USE_CUDA)
        message(STATUS "LibTorch - Compiled for CUDA Toolkit version: ${_FN_CUDATK_VERSION}")
    else()
        message(STATUS "LibTorch - CPU-only")
    endif()
    message(STATUS "LibTorch - Manually set root to: ${_LIBTORCH_ROOT}")
    message(STATUS "LibTorch - Manually set lib dir to: ${_LIBTORCH_LIBRARY_DIR}")
    Torchani_Msg_success("LibTorch - Done")
endfunction()
