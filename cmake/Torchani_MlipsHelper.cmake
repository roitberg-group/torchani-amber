include_guard()
include(Torchani_Msg)
include(Torchani_DownloadHelper)

# Download the *.pt files under REPO_DIR in the Hugging Face model repo REPO (at
# REVISION) into CACHE_DIR, keeping their paths relative to REPO_DIR, and copy them into
# DESTINATION. Files in CACHE_DIR are not downloaded again if their SHA256 matches the
# one listed in the repo. Only the listed files are copied, so cached files that are no
# longer in the repo are not. If the repo listing can't be fetched (e.g. no network), all
# the cached files are copied, if there are any
function(_Torchani_MlipsHelper_fetch_repo_dir)
    set(options "")
    set(oneValueArgs REPO REVISION REPO_DIR CACHE_DIR DESTINATION)
    set(multiValueArgs "")
    cmake_parse_arguments(
        _FN
        "${options}"
        "${oneValueArgs}"
        "${multiValueArgs}"
        ${ARGN}
    )

    set(_LISTING_URL "https://huggingface.co/api/models/${_FN_REPO}/tree/${_FN_REVISION}/${_FN_REPO_DIR}?recursive=true")
    set(_LISTING_FILE "${PROJECT_BINARY_DIR}/mlips-listing.json")
    file(MAKE_DIRECTORY "${_FN_CACHE_DIR}")

    set(_REL_PATHS "")
    message(STATUS "MLIPs - Fetching ${_FN_REPO_DIR} from ${_FN_REPO}")
    file(DOWNLOAD "${_LISTING_URL}" "${_LISTING_FILE}" STATUS _DOWNLOAD_STATUS)
    list(GET _DOWNLOAD_STATUS 0 _DOWNLOAD_CODE)
    list(GET _DOWNLOAD_STATUS 1 _DOWNLOAD_MSG)
    if(_DOWNLOAD_CODE EQUAL 0)
        file(READ "${_LISTING_FILE}" _LISTING)
        string(JSON _NUM_ENTRIES LENGTH "${_LISTING}")
        if(_NUM_ENTRIES EQUAL 0)
            Torchani_Msg_error("MLIPs - ${_LISTING_URL} lists no files")
        endif()
        math(EXPR _LAST_IDX "${_NUM_ENTRIES} - 1")
        foreach(_IDX RANGE ${_LAST_IDX})
            string(JSON _PATH GET "${_LISTING}" ${_IDX} path)
            if(NOT _PATH MATCHES "\\.pt$")
                continue()
            endif()
            # The SHA256 of a file is only listed if it is stored with Git LFS, which is how
            # Hugging Face currently stores *.pt files. The fallback is insurance in case
            # this changes in the future
            string(JSON _SHA256 ERROR_VARIABLE _SHA256_ERROR GET "${_LISTING}" ${_IDX} lfs oid)
            if(_SHA256_ERROR)
                set(_SHA256 "")
                Torchani_Msg_warn("MLIPs - ${_PATH} has no SHA256 in the repo listing, it will be downloaded without checking its integrity")
            endif()
            file(RELATIVE_PATH _REL_PATH "/${_FN_REPO_DIR}" "/${_PATH}")
            get_filename_component(_REL_DIR "${_REL_PATH}" DIRECTORY)
            file(MAKE_DIRECTORY "${_FN_CACHE_DIR}/${_REL_DIR}")
            message(STATUS "MLIPs - ${_REL_PATH}")
            Torchani_DownloadHelper_download_with_progress(
                URL "https://huggingface.co/${_FN_REPO}/resolve/${_FN_REVISION}/${_PATH}"
                FILE "${_FN_CACHE_DIR}/${_REL_PATH}"
                SHA256 ${_SHA256}
                OUT_ERROR _DOWNLOAD_ERROR
            )
            if(NOT _DOWNLOAD_ERROR STREQUAL "")
                Torchani_Msg_error("MLIPs - Download of ${_PATH} failed: ${_DOWNLOAD_ERROR}")
            endif()
            list(APPEND _REL_PATHS "${_REL_PATH}")
        endforeach()
    endif()
    file(REMOVE "${_LISTING_FILE}")

    if(NOT _DOWNLOAD_CODE EQUAL 0)
        file(GLOB_RECURSE _REL_PATHS RELATIVE "${_FN_CACHE_DIR}" "${_FN_CACHE_DIR}/*.pt")
        if(NOT _REL_PATHS)
            Torchani_Msg_error(
                "MLIPs - Could not fetch ${_LISTING_URL} (${_DOWNLOAD_MSG}), and there are no cached MLIP models in ${_FN_CACHE_DIR}. Either there is no network access, or ${_FN_REPO_DIR} doesn't exist in ${_FN_REPO}"
            )
        endif()
        Torchani_Msg_warn("MLIPs - Could not fetch ${_LISTING_URL} (${_DOWNLOAD_MSG}), using the cached MLIP models in ${_FN_CACHE_DIR}")
    endif()

    foreach(_REL_PATH IN LISTS _REL_PATHS)
        get_filename_component(_REL_DIR "${_REL_PATH}" DIRECTORY)
        file(COPY "${_FN_CACHE_DIR}/${_REL_PATH}" DESTINATION "${_FN_DESTINATION}/${_REL_DIR}")
    endforeach()
endfunction()

# Download the JIT-compiled MLIPs in the Hugging Face model repo REPO (at REVISION)
# and copy them into DESTINATION, which is emptied first. Two dirs of the repo are
# fetched, and their contents are merged into DESTINATION:
# - jit/<M.m>, where <M.m> is the major.minor of TORCH_VERSION, into
#   CACHE_DIR/mlips/jit/<M.m>
# - portable, which holds models that don't depend on the TorchANI extensions, into
#   CACHE_DIR/mlips/portable
function(Torchani_MlipsHelper_fetch_jit_models)
    set(options "")
    set(oneValueArgs REPO REVISION TORCH_VERSION CACHE_DIR DESTINATION)
    set(multiValueArgs "")
    cmake_parse_arguments(
        _FN
        "${options}"
        "${oneValueArgs}"
        "${multiValueArgs}"
        ${ARGN}
    )

    if(NOT _FN_TORCH_VERSION MATCHES "^([0-9]+\\.[0-9]+)")
        Torchani_Msg_error("MLIPs - Unexpected PyTorch/LibTorch version format '${_FN_TORCH_VERSION}'")
    endif()
    set(_TORCH_MINOR_VERSION "${CMAKE_MATCH_1}")
    set(_JIT_CACHE_DIR "${_FN_CACHE_DIR}/mlips/jit/${_TORCH_MINOR_VERSION}")
    set(_PORTABLE_CACHE_DIR "${_FN_CACHE_DIR}/mlips/portable")
    file(REMOVE_RECURSE "${_FN_DESTINATION}")
    _Torchani_MlipsHelper_fetch_repo_dir(
        REPO "${_FN_REPO}"
        REVISION "${_FN_REVISION}"
        REPO_DIR "jit/${_TORCH_MINOR_VERSION}"
        CACHE_DIR "${_JIT_CACHE_DIR}"
        DESTINATION "${_FN_DESTINATION}"
    )
    _Torchani_MlipsHelper_fetch_repo_dir(
        REPO "${_FN_REPO}"
        REVISION "${_FN_REVISION}"
        REPO_DIR "portable"
        CACHE_DIR "${_PORTABLE_CACHE_DIR}"
        DESTINATION "${_FN_DESTINATION}"
    )
    Torchani_Msg_success("MLIPs - Copied JIT-compiled MLIP models into ${_FN_DESTINATION}")
endfunction()
