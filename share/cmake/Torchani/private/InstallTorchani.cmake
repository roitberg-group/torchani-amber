include_guard()
function(install_torchani)
    set(options "")
    set(oneValueArgs PYTORCH_VERSION PYTORCH_CUDA PYTORCH_CUDNN PYTORCH_PYTHON)
    set(multiValueArgs "")
    cmake_parse_arguments(
        _FN
        "${options}"
        "${oneValueArgs}"
        "${multiValueArgs}"
        ${ARGN}
    )
    # This has to be run inside a conda environment
    # numpy
    message(STATUS "Numpy - Installing")
    execute_process(COMMAND conda install numpy -y)

    # torch
    message(STATUS "PyTorch - Installing")
    # select a specific build-string for reproducibility
    set(_BUILD_STRING "=${_FN_PYTORCH_VERSION}=py${_FN_PYTORCH_PYTHON}_cuda${_FN_PYTORCH_CUDA}_cudnn${_FN_PYTORCH_CUDNN}_0")
    execute_process(COMMAND conda install pytorch=${_FN_PYTORCH_VERSION}=${_BUILD_STRING} cudatoolkit=${_FN_PYTORCH_CUDA} -c pytorch -c nvidia -y)
    message(STATUS "PyTorch - Using version ${_FN_PYTORCH_VERSION}")
    message(STATUS "PyTorch - Compatible with CUDA: ${_FN_PYTORCH_CUDA}")
    message(STATUS "PyTorch - For python: ${_FN_PYTORCH_PYTHON}")
    message(STATUS "PyTorch - Compatible with cuDNN: ${_FN_PYTORCH_CUDNN}")

    # torchani
    message(STATUS "TorchANI - Installing")
    execute_process(COMMAND pip install -e . WORKING_DIRECTORY "./submodules/torchani")
endfunction()
