include_guard()
include(Msg)
# JIT compile all Torchani models
# (requires torchani and torch to be in PYTHONPATH or inside a conda environment)
function(TorchScriptUtils_jit_compile_models)
    set(options "")
    set(
        oneValueArgs
        DISABLE_OPTIMIZATIONS
        FORCE_RECOMPILATION
        _FN_WITH_EXTERNAL_NEIGHBORLIST
        WITH_CUAEV
    )
    set(multiValueArgs "")
    cmake_parse_arguments(
        _FN
        "${options}"
        "${oneValueArgs}"
        "${multiValueArgs}"
        ${ARGN}
    )
    find_program(HAS_PYTHON python REQUIRED)
    set(CMD python "${CMAKE_SOURCE_DIR}/jit/jit.py")
    if(_FN_DISABLE_OPTIMIZATIONS)
        list(APPEND CMD "--disable-optimizations")
    endif()
    if(_FN_FORCE_RECOMPILATION)
        list(APPEND CMD "--force")
    endif()
    if(_FN_WITH_EXTERNAL_NEIGHBORLIST)
        list(APPEND CMD "--external-neighborlist")
    endif()
    if(_FN_WITH_CUAEV)
        list(APPEND CMD "--cuaev")
    endif()
    message(STATUS "TorchScript - JIT Compiling models")
    execute_process(COMMAND ${CMD})
    msg_success("TorchScript - Finished JIT compilation")
endfunction()
