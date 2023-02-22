include_guard()
include(Msg)
# Fail if trying to build in a non-linux system
function(ensure_linux_build)
    if(NOT "${CMAKE_SYSTEM_NAME}" STREQUAL "Linux")
        msg_error("EnsureLinux - This project must be built on Linux, attempting to build on ${CMAKE_SYSTEM}")
    endif()
endfunction()
