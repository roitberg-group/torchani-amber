include_guard()

# Download URL into FILE, showing a progress bar. curl is used if found, since CMake's
# progress output prints one line per percent. If SHA256 is given, the download is
# skipped if FILE already exists with that hash, and the hash of the downloaded FILE is
# checked. OUT_ERROR is set in the PARENT scope to an error message, or to "" on success
function(Torchani_DownloadHelper_download_with_progress)
    set(options "")
    set(oneValueArgs URL FILE SHA256 OUT_ERROR)
    set(multiValueArgs "")
    cmake_parse_arguments(
        _FN
        "${options}"
        "${oneValueArgs}"
        "${multiValueArgs}"
        ${ARGN}
    )

    set(${_FN_OUT_ERROR} "" PARENT_SCOPE)
    if(_FN_SHA256 AND EXISTS "${_FN_FILE}")
        file(SHA256 "${_FN_FILE}" _EXISTING_SHA256)
        if(_EXISTING_SHA256 STREQUAL _FN_SHA256)
            return()
        endif()
    endif()

    find_program(TORCHANI_CURL_EXECUTABLE curl)
    if(TORCHANI_CURL_EXECUTABLE)
        execute_process(
            COMMAND
                "${TORCHANI_CURL_EXECUTABLE}"
                --fail --location --progress-bar
                --output "${_FN_FILE}"
                "${_FN_URL}"
            RESULT_VARIABLE _CURL_CODE
        )
        if(NOT _CURL_CODE EQUAL 0)
            file(REMOVE "${_FN_FILE}")
            set(${_FN_OUT_ERROR} "curl exited with code ${_CURL_CODE}" PARENT_SCOPE)
            return()
        endif()
    else()
        file(DOWNLOAD "${_FN_URL}" "${_FN_FILE}" SHOW_PROGRESS STATUS _DOWNLOAD_STATUS)
        list(GET _DOWNLOAD_STATUS 0 _DOWNLOAD_CODE)
        list(GET _DOWNLOAD_STATUS 1 _DOWNLOAD_MSG)
        if(NOT _DOWNLOAD_CODE EQUAL 0)
            set(${_FN_OUT_ERROR} "${_DOWNLOAD_MSG}" PARENT_SCOPE)
            return()
        endif()
    endif()

    if(_FN_SHA256)
        file(SHA256 "${_FN_FILE}" _DOWNLOADED_SHA256)
        if(NOT _DOWNLOADED_SHA256 STREQUAL _FN_SHA256)
            file(REMOVE "${_FN_FILE}")
            set(${_FN_OUT_ERROR} "SHA256 mismatch, expected ${_FN_SHA256}, got ${_DOWNLOADED_SHA256}" PARENT_SCOPE)
        endif()
    endif()
endfunction()
