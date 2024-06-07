include_guard()
include(Msg)
# Fail if trying an in-source build
function(OutOfSourceBuild_ensure)
    if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
        msg_error("OutOfSourceBuild - In source builds are not allowed, you must create a build directory, please remove ./CMakeFiles/ and ./CMakeCache.txt")
    endif()
endfunction()
