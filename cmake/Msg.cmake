include_guard()
# Allow colors in cmake messages
# http://stackoverflow.com/a/19578320
if(NOT WIN32)
  string(ASCII 27 EscapeKey)
  set(ColorReset "${EscapeKey}[m")
  set(ColorBold "${EscapeKey}[1m")
  set(ColorRed "${EscapeKey}[31m")
  set(ColorCyan "${EscapeKey}[36m")
  set(ColorGreen "${EscapeKey}[32m")
  set(ColorYellow "${EscapeKey}[33m")
  set(ColorBoldRed "${EscapeKey}[1;31m")
  set(ColorBoldGreen "${EscapeKey}[1;32m")
  set(ColorBoldYellow "${EscapeKey}[1;33m")
endif()

function(msg_error ARG)
    message(FATAL_ERROR "${ColorRed}${ARG}${ColorReset}")
endfunction()


function(msg_warn ARG)
    message(STATUS "${ColorYellow}${ARG}${ColorReset}")
endfunction()

function(msg_info ARG)
    message(STATUS "${ColorCyan}${ARG}${ColorReset}")
endfunction()

function(msg_success ARG)
    message(STATUS "${ColorGreen}${ARG}${ColorReset}")
endfunction()

function(msg_banner ARG)
    msg_info("")
    msg_info("")
    msg_info("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    msg_info("")
    string(REPLACE "\n" ";" ARG_LS ${ARG})
    foreach(el ${ARG_LS})
        string(STRIP ${el} el_strip)
        msg_info(${el_strip})
    endforeach()
    msg_info("")
    msg_info("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    msg_info("")
    msg_info("")
endfunction()
