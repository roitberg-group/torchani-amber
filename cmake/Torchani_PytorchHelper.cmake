include_guard()
include(Torchani_Msg)

# The following function sets, in the PARENT scope:
# - PYTORCH_ROOT
#
# Detect the PyTorch installation importable by Python, and if CHECK_VERSIONS is true,
# validate its versions and the versions of the CUDA Toolkit and cuDNN it was compiled
# against. If USE_CUDA is false, a CPU-only PyTorch installation is required instead
function(Torchani_PytorchHelper_detect)
    set(options "")
    set(oneValueArgs USE_CUDA CHECK_VERSIONS LIB_VERSION CUDATK_VERSION CUDNN_VERSION)
    set(multiValueArgs "")
    cmake_parse_arguments(
        _FN
        "${options}"
        "${oneValueArgs}"
        "${multiValueArgs}"
        ${ARGN}
    )
    message(STATUS "PyTorch - Trying to use python to import PyTorch and find its path and specs")
    find_package(Python REQUIRED COMPONENTS Interpreter)
    # Note that the CUDA and torch versions are M.m.p and the cuDNN is Mmmpp
    execute_process(
        COMMAND "${Python_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/cmake/detect_pytorch.py"
        OUTPUT_VARIABLE _PYTORCH_SPECS
        RESULT_VARIABLE _PYTORCH_DETECTION_FAILED
    )
    if(_PYTORCH_DETECTION_FAILED)
        Torchani_Msg_error("PyTorch - Could not find a PyTorch installation")
    endif()
    list(GET _PYTORCH_SPECS 0 _TORCH_VERSION)
    list(GET _PYTORCH_SPECS 1 _CUDATK_VERSION)
    list(GET _PYTORCH_SPECS 2 _CUDNN_VERSION)
    list(GET _PYTORCH_SPECS 3 _PYTORCH_ROOT)
    if(_FN_USE_CUDA AND _CUDATK_VERSION STREQUAL "none")
        Torchani_Msg_error("PyTorch - Found a CPU-only PyTorch installation, reconfigure with -DTORCHANI_BUILD_CUDA=OFF to use it")
    endif()
    if(NOT _FN_USE_CUDA AND NOT _CUDATK_VERSION STREQUAL "none")
        Torchani_Msg_error("PyTorch - TORCHANI_BUILD_CUDA=OFF requires a CPU-only PyTorch installation, but found one compiled with CUDA ${_CUDATK_VERSION}")
    endif()
    set(PYTORCH_ROOT "${_PYTORCH_ROOT}" PARENT_SCOPE)
    if(_FN_USE_CUDA)
        Torchani_Msg_success("PyTorch - Successfully found a PyTorch installation compiled with CUDA and cuDNN support")
    else()
        Torchani_Msg_success("PyTorch - Successfully found a CPU-only PyTorch installation")
    endif()
    message(STATUS "PyTorch - Root: ${_PYTORCH_ROOT}")

    if(NOT _FN_CHECK_VERSIONS)
        return()
    endif()
    message(STATUS "PyTorch - Validating the detected PyTorch installation")
    if(NOT _TORCH_VERSION STREQUAL _FN_LIB_VERSION)
        Torchani_Msg_error("PyTorch - Version ${_TORCH_VERSION} doesn't match required ${_FN_LIB_VERSION}")
    else()
        Torchani_Msg_success("PyTorch - Version ${_TORCH_VERSION} is consistent with required")
    endif()
    if(NOT _FN_USE_CUDA)
        return()
    endif()
    if(NOT _CUDATK_VERSION STREQUAL _FN_CUDATK_VERSION)
        Torchani_Msg_error("PyTorch - Compiled against CUDA Toolkit version: ${_CUDATK_VERSION}, which doesn't match required ${_FN_CUDATK_VERSION}")
    else()
        Torchani_Msg_success("PyTorch - Compiled against CUDA Toolkit version: ${_CUDATK_VERSION}, consistent with required")
    endif()
    if(NOT _CUDNN_VERSION STREQUAL _FN_CUDNN_VERSION)
        Torchani_Msg_error("PyTorch - Compiled against cuDNN version: ${_CUDNN_VERSION}, which doesn't match required ${_FN_CUDNN_VERSION}")
    else()
        Torchani_Msg_success("PyTorch - Compiled against cuDNN version: ${_CUDNN_VERSION}, consistent with required")
    endif()
endfunction()
