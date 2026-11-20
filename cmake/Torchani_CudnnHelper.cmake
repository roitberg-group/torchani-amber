include_guard()
include(Torchani_Msg)

# The following function sets, in the PARENT scope:
# - CUDNN_ROOT
# - CUDNN_LIBRARY
# - CUDNN_INCLUDE_DIR
function(Torchani_CudnnHelper_set_conda_paths)
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs "")
    cmake_parse_arguments(
        _FN
        "${options}"
        "${oneValueArgs}"
        "${multiValueArgs}"
        ${ARGN}
    )
    message(STATUS "cuDNN - Setting paths using Conda prefix")
    set(_CUDNN_ROOT $ENV{CONDA_PREFIX})
    set(_CUDNN_LIBRARY "${_CUDNN_ROOT}/lib")
    set(_CUDNN_INCLUDE_DIR "${_CUDNN_ROOT}/include")
    set(CUDNN_INCLUDE_DIR "${_CUDNN_INCLUDE_DIR}" PARENT_SCOPE)
    set(CUDNN_LIBRARY "${_CUDNN_LIBRARY}" PARENT_SCOPE)
    set(CUDNN_ROOT "${_CUDNN_ROOT}" PARENT_SCOPE)
    message(STATUS "cuDNN - Manually set root to: ${_CUDNN_ROOT}")
    message(STATUS "cuDNN - Manually set include dir to: ${_CUDNN_INCLUDE_DIR}")
    message(STATUS "cuDNN - Manually set lib dir to: ${_CUDNN_LIBRARY}")
    Torchani_Msg_success("cuDNN - Successfully set paths")
endfunction()
