include_guard()
include(Color)

function(msg_error ARG)
    message(FATAL_ERROR "${ColorRed}${ARG}${ColorReset}")
endfunction()


function(msg_warn ARG)
    message(STATUS "${ColorYellow}${ARG}${ColorReset}")
endfunction()


function(msg_success ARG)
    message(STATUS "${ColorGreen}${ARG}${ColorReset}")
endfunction()
