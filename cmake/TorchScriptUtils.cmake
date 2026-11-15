include_guard()
# Add a target, built by default, that JIT compiles all Torchani models
# (requires torchani and torch to be in PYTHONPATH or inside a conda environment)
# All messages are delegated to the underlying python process
function(TorchScriptUtils_add_jit_compile_models_target)
    set(options "")
    set(
        oneValueArgs
        DISABLE_OPTIMIZATIONS
        FORCE_RECOMPILATION
        AIMNET2
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
    if(_FN_AIMNET2)
        list(APPEND CMD "--aimnet2")
    endif()
    add_custom_target(jit_compile_models ALL COMMAND ${CMD} VERBATIM)
endfunction()
