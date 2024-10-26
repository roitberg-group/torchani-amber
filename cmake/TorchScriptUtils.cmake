include_guard()
# JIT compile all Torchani models
# (requires torchani and torch to be in PYTHONPATH or inside a conda environment)
# All messages are delegated to the underlying python process
function(TorchScriptUtils_jit_compile_models)
    set(options "")
    set(
        oneValueArgs
        DISABLE_OPTIMIZATIONS
        FORCE_RECOMPILATION
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
    execute_process(COMMAND ${CMD})
endfunction()
