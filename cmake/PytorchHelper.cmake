include_guard()
include(Msg)

# TODO refactor so that pytorch detection and validation happens here
# The following function should set LIBTORCH_ROOT
function(PytorchHelper_detect)
    set(options "")
    set(oneValueArgs LIB_VERSION CUDATK_VERSION CUDNN_VERSION)
    set(multiValueArgs "")
    cmake_parse_arguments(
        _FN
        "${options}"
        "${oneValueArgs}"
        "${multiValueArgs}"
        ${ARGN}
    )
endfunction()
