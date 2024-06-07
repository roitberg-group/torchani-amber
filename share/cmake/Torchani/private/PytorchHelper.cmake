include_guard()
include(Msg)

# The following function doesn't set any vars
function(PytorchHelper_download_and_install)
    set(options "")
    set(oneValueArgs LIB_VERSION CUDA_VERSION CUDNN_VERSION PYTHON_VERSION)
    set(multiValueArgs "")
    cmake_parse_arguments(
        _FN
        "${options}"
        "${oneValueArgs}"
        "${multiValueArgs}"
        ${ARGN}
    )
    message(STATUS "PyTorch - Downloading and installing prebuilt binaries, with conda")
    execute_process(
        COMMAND which conda
        RESULT_VARIABLE CONDA_NOT_FOUND
        OUTPUT_VARIABLE CONDA_LOCATION
    )
    if(CONDA_NOT_FOUND)
        msg_error("PyTorch - conda is required to install a user-specified pytorch, but 'conda' command could not be found. Maybe run 'conda activate' before cmake?")
    else()
        message(STATUS "PyTorch - Using conda located in ${CONDA_LOCATION}")
    endif()
    # A specific build-string is requested for reproducibility
    set(_PYTORCH_BUILD_STRING "=${_FN_LIB_VERSION}=py${_FN_PYTHON_VERSION}_cuda${_FN_CUDA_VERSION}_cudnn${_FN_CUDNN_VERSION}_0")
    message(STATUS "PyTorch - Using channels: pytorch, nvidia")
    message(STATUS "PyTorch - Version:${_BUILD_STRING}")
    message(STATUS "PyTorch - Package build-string: ${_PYTORCH_BUILD_STRING}")
    message(STATUS "PyTorch - Compiled for Python version: ${_FN_PYTHON_VERSION}")
    message(STATUS "PyTorch - Compiled for CUDA Toolkit version: ${_FN_CUDA_VERSION}")
    message(STATUS "PyTorch - Compiled for cuDNN version: ${_FN_CUDNN_VERSION}")
    execute_process(
        COMMAND
            conda
            install
            -y
            -c pytorch
            -c nvidia
            pytorch=${_FN_LIB_VERSION}=${_BUILD_STRING}
            pytorch-cuda=${_FN_CUDA_VERSION}
        RESULT_VARIABLE INSTALL_FAILED
    )
    if(INSTALL_FAILED)
        msg_error("PyTorch - Download and/or installation failed")
    else()
        msg_success("PyTorch - Done")
    endif()
endfunction()
