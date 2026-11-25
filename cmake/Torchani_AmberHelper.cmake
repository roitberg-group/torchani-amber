include_guard()
include(Torchani_Msg)

# Amber finds its own CUDA Toolkit, and both must match to avoid loading two
# different CUDA runtimes into the same process
function(Torchani_AmberHelper_check_nvcc_matches)
    set(options "")
    set(oneValueArgs TORCHANI_NVCC AMBER_NVCC)
    set(multiValueArgs "")
    cmake_parse_arguments(
        _FN
        "${options}"
        "${oneValueArgs}"
        "${multiValueArgs}"
        ${ARGN}
    )
    file(REAL_PATH "${_FN_TORCHANI_NVCC}" _TORCHANI_NVCC_REALPATH)
    file(REAL_PATH "${_FN_AMBER_NVCC}" _AMBER_NVCC_REALPATH)
    if(NOT _TORCHANI_NVCC_REALPATH STREQUAL _AMBER_NVCC_REALPATH)
        set(_NVCC_MISMATCH_MSG "The CUDA compiler used by the MLIP interface (${_TORCHANI_NVCC_REALPATH})")
        set(_NVCC_MISMATCH_MSG "${_NVCC_MISMATCH_MSG} differs from the one used by Amber (${_AMBER_NVCC_REALPATH})")
        Torchani_Msg_error("${_NVCC_MISMATCH_MSG}")
    endif()
endfunction()

# NOTE: Very Very Ugly workaround: Amber compiles its .cu files with FindCUDA and
# never calls enable_language(CUDA). Enabling CUDA here (and in Torch's config)
# makes CMake also try to compile Amber's .cu files which fails. This modifies
# properties of Amber's sources: marking them HEADER_FILE_ONLY after Amber has
# created its targets disables this compilation and keeps the old FindCUDA commands.
# New Amber FindCUDA targets must be added manually to AMBER_FINDCUDA_TARGETS.
#
# The modification is deferred to the end of the configuration of AMBER_SOURCE_DIR
function(Torchani_AmberHelper_hide_amber_cuda_sources)
    set(options "")
    set(oneValueArgs AMBER_SOURCE_DIR)
    set(multiValueArgs "")
    cmake_parse_arguments(
        _FN
        "${options}"
        "${oneValueArgs}"
        "${multiValueArgs}"
        ${ARGN}
    )
    cmake_language(DEFER DIRECTORY "${_FN_AMBER_SOURCE_DIR}" CALL _Torchani_AmberHelper_hide_amber_cuda_sources_now)
endfunction()

function(_Torchani_AmberHelper_hide_amber_cuda_sources_now)
    set(AMBER_FINDCUDA_TARGETS
        pbsa.cuda
        rism_cuda
        rism_cuda_double
        mdgx.cuda
        gpuP2PCheck
        pmemd_xray_cuda
        pmemd_cuda_SPFP
        pmemd_cuda_DPFP
        pmemd_cuda_SPFP_mpi
        pmemd_cuda_DPFP_mpi
        cpptraj_cuda_routines
        xc_cuda
        quick_cuda_kernels
        quick_cuda_kernels_mpi
    )
    foreach(target IN LISTS AMBER_FINDCUDA_TARGETS)
        if(NOT TARGET ${target})
            continue()
        endif()
        get_target_property(target_dir ${target} SOURCE_DIR)
        get_target_property(cu_sources ${target} SOURCES)
        list(FILTER cu_sources INCLUDE REGEX "\\.cu$")
        list(TRANSFORM cu_sources PREPEND "${target_dir}/" REGEX "^[^/]")
        set_source_files_properties(${cu_sources} DIRECTORY "${target_dir}" PROPERTIES HEADER_FILE_ONLY TRUE)
    endforeach()
endfunction()

# NOTE: Ugly Workaround: Torch's and Amber's MKL CMake configs both have issues and interact
# badly when MKL is not found. An empty caffe2::mkl makes Torch skip its MKL setup.
# Must be called before find_package(Torch)
function(Torchani_AmberHelper_disable_torch_mkl_if_not_found)
    set(options "")
    set(oneValueArgs MKL_FOUND)
    set(multiValueArgs "")
    cmake_parse_arguments(
        _FN
        "${options}"
        "${oneValueArgs}"
        "${multiValueArgs}"
        ${ARGN}
    )
    if(NOT _FN_MKL_FOUND AND NOT TARGET caffe2::mkl)
        add_library(caffe2::mkl INTERFACE IMPORTED)
    endif()
endfunction()

# Without a GPU at configure time Torch falls back to its "common" arch list, which
# includes very old archs. Amber's CUDA_NVCC_FLAGS are applied to cuaev anyway
# so forward the archs to Torch too, from AMBER_NVCC_FLAGS
# A user-set TORCH_CUDA_ARCH_LIST (variable or env) is respected.
# Must be called before find_package(Torch)
function(Torchani_AmberHelper_set_torch_cuda_arch_list)
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs AMBER_NVCC_FLAGS)
    cmake_parse_arguments(
        _FN
        "${options}"
        "${oneValueArgs}"
        "${multiValueArgs}"
        ${ARGN}
    )
    if(DEFINED TORCH_CUDA_ARCH_LIST OR DEFINED ENV{TORCH_CUDA_ARCH_LIST})
        return()
    endif()
    string(REGEX MATCHALL "code=sm_[0-9]+" _SM_CODES "${_FN_AMBER_NVCC_FLAGS}")
    set(_ARCH_LIST "")
    foreach(_SM_CODE IN LISTS _SM_CODES)
        string(REGEX REPLACE "code=sm_([0-9]+)([0-9])" "\\1.\\2" _ARCH "${_SM_CODE}")
        list(APPEND _ARCH_LIST "${_ARCH}")
    endforeach()
    list(REMOVE_DUPLICATES _ARCH_LIST)
    if(_ARCH_LIST)
        Torchani_Msg_info("Using Amber's CUDA archs for LibTorch: ${_ARCH_LIST}")
        set(TORCH_CUDA_ARCH_LIST "${_ARCH_LIST}" PARENT_SCOPE)
    endif()
endfunction()
