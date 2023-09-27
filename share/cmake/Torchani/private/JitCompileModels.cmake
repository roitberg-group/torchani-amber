include_guard()
include(Msg)
# JIT compile all Torchani models
# (requires torchani and torch to be in PYTHONPATH or inside a conda environment)
function(jit_compile_models)
    set(options "")
    set(
        oneValueArgs
        JIT_DISABLE_OPTIMIZATIONS
        JIT_FORCE_RECOMPILE
        JIT_STANDARD
        JIT_EXTERNAL_CELL_LIST
        JIT_TORCH_CELL_LIST
        JIT_CUAEV
        JIT_CUAEV_TORCH_CELL_LIST
        JIT_CUAEV_EXTERNAL_CELL_LIST
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
    if(_FN_JIT_DISABLE_OPTIMIZATIONS)
        list(APPEND CMD "--disable-optimizations")
    endif()
    if(_FN_JIT_FORCE_RECOMPILE)
        list(APPEND CMD "--force-recompile")
    endif()
    if(_FN_JIT_STANDARD)
        list(APPEND CMD "--standard")
    endif()
    if(_FN_JIT_TORCH_CELL_LIST)
        list(APPEND CMD "--torch-cell-list")
    endif()
    if(_FN_JIT_EXTERNAL_CELL_LIST)
        list(APPEND CMD "--external-cell-list")
    endif()
    if(_FN_JIT_CUAEV)
        list(APPEND CMD "--cuaev")
    endif()
    if(_FN_JIT_CUAEV_TORCH_CELL_LIST)
        list(APPEND CMD "--cuaev-torch-cell-list")
    endif()
    if(_FN_JIT_CUAEV_EXTERNAL_CELL_LIST)
        list(APPEND CMD "--cuaev-external-cell-list")
    endif()
    message(STATUS "JIT - Compiling models")
    execute_process(COMMAND ${CMD})
    msg_success("JIT - Finished compilation")
endfunction()
