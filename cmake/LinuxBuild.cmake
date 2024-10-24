include_guard()
include(Msg)
# Fail if trying to build in a non-linux system
function(LinuxBuild_ensure)
    if(NOT "${CMAKE_SYSTEM_NAME}" STREQUAL "Linux")
        msg_error("LinuxBuild - This project must be built on Linux, but system is ${CMAKE_SYSTEM}")
    endif()
endfunction()
