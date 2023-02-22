include_guard()
# Allow colors in cmake messages
# http://stackoverflow.com/a/19578320
if(NOT WIN32)
  string(ASCII 27 EscapeKey)
  set(ColorReset "${EscapeKey}[m")
  set(ColorBold "${EscapeKey}[1m")
  set(ColorRed "${EscapeKey}[31m")
  set(ColorGreen "${EscapeKey}[32m")
  set(ColorYellow "${EscapeKey}[33m")
  set(ColorBoldRed "${EscapeKey}[1;31m")
  set(ColorBoldGreen "${EscapeKey}[1;32m")
  set(ColorBoldYellow "${EscapeKey}[1;33m")
endif()
