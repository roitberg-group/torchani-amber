include_guard()
include(Msg)
# Fail if trying an in-source build
function(ensure_out_of_source_build)
    if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
        msg_error("InSource - In source builds are not allowed, you must create a build directory, please remove CMakeFiles/ and CMakeCache.txt.")
    endif()
endfunction()
