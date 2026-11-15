include_guard()
include(Torchani_Msg)
# Fail if trying an in-source build
function(Torchani_OutOfSourceBuild_ensure)
    if("${PROJECT_SOURCE_DIR}" STREQUAL "${PROJECT_BINARY_DIR}")
        Torchani_Msg_error("OutOfSourceBuild - In source builds are not allowed, you must create a build directory, please remove ./CMakeFiles/ and ./CMakeCache.txt")
    endif()
endfunction()
