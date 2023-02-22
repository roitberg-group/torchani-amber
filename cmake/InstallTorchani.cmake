include_guard()
function(install_torchani)
    set(options "")
    set(oneValueArgs TORCHVISION_VERSION PYTORCH_VERSION PYTORCH_CUDA)
    set(multiValueArgs "")
    cmake_parse_arguments(
        _FN
        "${options}"
        "${oneValueArgs}"
        "${multiValueArgs}"
        ${ARGN}
    )
    # numpy
    message(STATUS "Numpy - Installing")
    execute_process(COMMAND conda install numpy -y)

    # torch
    message(STATUS "PyTorch - Installing")
    if(_FN_PYTORCH_VERSION STREQUAL "nightly" )
        set(_CHANNEL "pytorch-nightly")
        set(_EQUALS_VERSION "")
    else()
        set(_CHANNEL "pytorch")
        set(_EQUALS_VERSION "==${_FN_PYTORCH_VERSION}")
    endif()
    execute_process(COMMAND conda install pytorch${_EQUALS_VERSION} torchvision==${_FN_TORCHVISION_VERSION} cudatoolkit=${_FN_PYTORCH_CUDA} -c ${_CHANNEL} -c nvidia -y)
    message(STATUS "PyTorch - Using version ${_FN_PYTORCH_VERSION}")
    message(STATUS "PyTorch - Compatible with CUDA: ${_FN_PYTORCH_CUDA}")
    message(STATUS "PyTorch - Using Torchvision version ${_FN_TORCHVISION_VERSION}")

    # torchani
    message(STATUS "TorchANI - Installing")
    execute_process(COMMAND pip install -e . WORKING_DIRECTORY "./submodules/torchani")
endfunction()
