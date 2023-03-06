include_guard()
# JIT compile all Torchani models
# (requires torchani and torch to be in PYTHONPATH or inside a conda environment)
function(jit_compile_models)
    set(options "")
    set(oneValueArgs JIT_AVOID_OPTIMIZATIONS JIT_EXTERNAL_CELL_LIST JIT_TORCH_CELL_LIST)
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
    if(_FN_JIT_AVOID_OPTIMIZATIONS)
        list(APPEND CMD "--avoid-optimizations")
    endif()
    if(_FN_JIT_EXTERNAL_CELL_LIST)
        list(APPEND CMD "--external-cell-list")
    endif()
    if(_FN_JIT_TORCH_CELL_LIST)
        list(APPEND CMD "--torch-cell-list")
    endif()
    message(STATUS "JIT - Compiling models")
    execute_process(COMMAND ${CMD})
endfunction()
