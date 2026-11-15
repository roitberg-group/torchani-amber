include_guard()
include(Torchani_Msg)
# Fail if trying to build in a non-linux system
function(Torchani_LinuxBuild_ensure)
    if(NOT "${CMAKE_SYSTEM_NAME}" STREQUAL "Linux")
        Torchani_Msg_error("LinuxBuild - This project must be built on Linux, but system is ${CMAKE_SYSTEM}")
    endif()
endfunction()
