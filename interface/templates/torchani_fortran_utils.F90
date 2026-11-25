! Hand-written part of the torchani module, from
! interface/templates/torchani_fortran_utils.F90.
!
! This is not a standalone file. interface/templates/torchani.F90.jinja includes
! it verbatim inside the "contains" section of module torchani, and the result
! is interface/torchani.F90. This file holds the hand-written code. Things to
! keep in mind:
! - It relies on the module header in the template: "use iso_c_binding" and
!   "implicit none" are not repeated here.
! - ifmlip_initialize calls internal_initialize, the bind(c) interface
!   generated from interface/interface.yaml. If the args of ifmlip_initialize
!   change in the yaml, that call must be updated.
! - The generated Fortran wrappers call fbool_to_cbool, defined here.
! - The ifmlip_store_*_tensor wrappers call the bind(c) interfaces
!   internal_store_*_tensor, generated from interface/interface.yaml.
! - Routines here are made public in the "public" list of the template, so a
!   new public routine must also be added there.

subroutine ifmlip_initialize( &
    atomic_nums, &
    config_path, &
    read_mlmm_config, &
    validate_amber_config, &
    print_config_info &
)
    integer(c_int), contiguous, intent(in) :: atomic_nums(:)
    character(len=*, kind=kind("a")), intent(in) :: config_path
    logical, intent(in) :: read_mlmm_config
    logical, optional, intent(in) :: validate_amber_config
    logical, optional, intent(in) :: print_config_info

    character(len=len_trim(config_path) + 1, kind=c_char) :: c_config_path
    logical :: do_validate_amber_config
    logical :: do_print_config_info
    ! Copying to a temporary string with kind=c_char "casts" the kind to c_char
    c_config_path = trim(config_path) // c_null_char
    do_validate_amber_config = .false.
    if (present(validate_amber_config)) then
        do_validate_amber_config = validate_amber_config
    endif
    do_print_config_info = .false.
    if (present(print_config_info)) then
        do_print_config_info = print_config_info
    endif
    call internal_initialize( &
        size(atomic_nums), &
        atomic_nums, &
        c_config_path, &
        fbool_to_cbool(read_mlmm_config), &
        fbool_to_cbool(do_validate_amber_config), &
        fbool_to_cbool(do_print_config_info) &
    )
endsubroutine

! Coordinates must be mapped to central cell in sander!!!

! TODO: Check which of these arrays store img_idx and which store atom_idx
! (for both pmemd)

! TODO: There seems to be a serious problem with the shifts, the shifts are not the same for sander and for ani, setting shifts to 0
! fixes this but avoids pbc. I believe there may be a similar issue with the pmemd shifts

! Convert the amber-neighborlist to "ANI fmt", which is a 2-dim array. Each column
! in the array holds the index of one of the two atoms in each pair. The array's shape
! is not known at compile-time, so it is dynamically created *each time this function is called*
subroutine convert_sander_neighborlist_to_ani_fmt( &
    img_idx_to_atom_idx, &
    atom_idx_to_neighbors_num, &
    translation_vector, &
    sander_neighborlist, &
    ani_neighborlist, &
    ani_shifts &
)
    integer, intent(in) :: img_idx_to_atom_idx(:)
    integer, intent(in) :: atom_idx_to_neighbors_num(:)
    ! Sander's list is assumed-shape, so (:) can't be used
    integer, intent(in) :: sander_neighborlist(*)
    double precision, intent(in) :: translation_vector(1:3, 1:18)
    integer, allocatable, intent(out) :: ani_neighborlist(:, :)
    double precision, allocatable, intent(out) :: ani_shifts(:, :)

    integer :: num_atoms
    integer :: img_idx
    integer :: offset
    integer :: pair_idx
    integer :: img_neighbors_num

    ! Sander's list containes img_idx of the neighbors of each atom.
    ! They are ordered in *the img-order*
    ! (see ew_force for details, iteration happens over k)

    ! The number of neighbors for a given img_idx is given by
    ! atom_idx_to_neighbors_num(img_idx_to_atom_idx(img_idx))
    ! atom_idx_to_neighbors_num can be obtained from numvdw + numhbnd within sander
    ! (numhbnd should be zero for modern FF)
    num_atoms = size(img_idx_to_atom_idx, dim=1)
    allocate(ani_neighborlist(sum(atom_idx_to_neighbors_num), 2))
    allocate(ani_shifts(3, sum(atom_idx_to_neighbors_num)))
    pair_idx = 0
    do img_idx = 1, num_atoms
        img_neighbors_num = atom_idx_to_neighbors_num(img_idx_to_atom_idx(img_idx))
        do offset = 1, img_neighbors_num
            pair_idx = pair_idx + 1
            ! The order in which atoms are added into the ani_neighborlist
            ! is irrelevant, the important thing is that the atom_idx are added, and
            ! not the img_idx. Most straightforward is to fill using the img-order
            ani_neighborlist(pair_idx, 1) = img_idx_to_atom_idx(img_idx)
            ! The translation vec index and the image idx
            ! are encoded by sander into the same elem. iand and ishft decode them
            ani_neighborlist(pair_idx, 2) &
                = img_idx_to_atom_idx(iand(sander_neighborlist(pair_idx), 2**27 - 1))
            ani_shifts(:, pair_idx) &
                = translation_vector(:, ishft(sander_neighborlist(pair_idx), -27))
        end do
    end do
    ! ANI neighborlist has idxs 1 less than Sander, so subtract 1 fromm all idxs
    ani_neighborlist = ani_neighborlist - 1
    ani_shifts = -ani_shifts
end subroutine

! Convert the pmemd-neighborlist to "ANI fmt", which is a 2-dim array. Each column
! in the array holds the index of one of the two atoms in each pair. The array's shape
! is not known at compile-time, so it is dynamically created *each time this function is called*
subroutine convert_pmemd_neighborlist_to_ani_fmt( &
    image_idx_to_atom_idx, &
    translation_vector, &
    pmemd_neighborlist, &
    ani_neighborlist, &
    ani_shifts &
)
    integer, intent(in) :: image_idx_to_atom_idx(:)
    integer, intent(in) :: pmemd_neighborlist(:)
    double precision, intent(in) :: translation_vector(1:3, 0:17)
    integer, allocatable, intent(out) :: ani_neighborlist(:, :)
    double precision, allocatable, intent(out) :: ani_shifts(:, :)

    integer :: num_atoms
    integer :: img_idx, j
    integer :: image_idx
    integer :: pmemd_idx
    integer :: ani_idx
    integer :: translation_vector_idx
    integer, allocatable :: img_neighbors_num(:)
    logical :: has_no_translation
    ! The ANI neighborlist is constructed naively. First do a pass over
    ! the Pmemd neighborlist to determine the shape of the array, then allocate it,
    ! and finally perform a second pass to fill the array.
    num_atoms = size(image_idx_to_atom_idx, dim=1)
    allocate(img_neighbors_num(num_atoms))
    call get_pmemd_img_neighbors_num(pmemd_neighborlist, img_neighbors_num)
    allocate(ani_neighborlist(sum(img_neighbors_num), 2))
    allocate(ani_shifts(3, sum(img_neighbors_num)))

    ! Atom indices for pmemd and ani
    pmemd_idx = 1
    ani_idx = 0
    do img_idx = 1, num_atoms
        do j = 1, img_neighbors_num(img_idx)
            ani_neighborlist(ani_idx + j, 1) = image_idx_to_atom_idx(img_idx) - 1
        end do

        has_no_translation = (pmemd_neighborlist(pmemd_idx) == 1)
        pmemd_idx = pmemd_idx + 2
        if (has_no_translation) then
            do j = 1, img_neighbors_num(img_idx)
                ! If no translation is present the tranvec_idx is hardcoded to 13
                image_idx = pmemd_neighborlist(pmemd_idx + j)
                translation_vector_idx = 13
                ! ani neighborlist has indices 1 less than amber due to indices in python starting from zero
                ani_neighborlist(ani_idx + j, 2) = image_idx_to_atom_idx(image_idx) - 1
                ani_shifts(:, ani_idx + j) = translation_vector(:, translation_vector_idx)
            end do
        else
            do j = 1, img_neighbors_num(img_idx)
                ! If translation is present the translation vector index and the image idx
                ! are encoded by pmemd into the same idx. To decode them, use iand and ishft
                image_idx = iand(pmemd_neighborlist(pmemd_idx + j), Z"07FFFFFF")
                translation_vector_idx = ishft(pmemd_neighborlist(pmemd_idx + j), -27)
                ani_neighborlist(ani_idx + j, 2) = image_idx_to_atom_idx(image_idx) - 1
                ani_shifts(:, ani_idx + j) = translation_vector(:, translation_vector_idx)
            end do
        end if
        ani_idx = ani_idx + img_neighbors_num(img_idx)
        pmemd_idx = pmemd_idx + img_neighbors_num(img_idx) + 1
    end do
    ! ANI shifts are shifted with respect to the amber shifts, thus the negative sign
    ani_shifts = -ani_shifts
end subroutine

! Obtain the number of neighbors of each atom in the neighborlist
subroutine get_pmemd_img_neighbors_num(pmemd_neighborlist, img_neighbors_num)
    integer, intent(in) :: pmemd_neighborlist(:)
    integer, intent(out) :: img_neighbors_num(:)

    integer :: idx
    integer :: total_neighbors
    integer :: i
    idx = 1
    do i=1, size(img_neighbors_num, 1)
        ! Add "electrostatic neighbors" and "VDW neighbors"
        total_neighbors = pmemd_neighborlist(idx + 1)  + pmemd_neighborlist(idx + 2)
        img_neighbors_num(i) = total_neighbors
        ! Move the index forward to jump over the read atoms
        idx = idx + 2 + total_neighbors + 1
    end do
end subroutine

logical(c_bool) function fbool_to_cbool(fbool) result(ret)
    logical, intent(in) :: fbool
    if (fbool) then
        ret = .true._c_bool
    else
        ret = .false._c_bool
    endif
endfunction

subroutine ifmlip_set_no_eval_atoms_mask(values)
    integer(c_int), contiguous, intent(in) :: values(:)
    call internal_set_no_eval_atoms_mask(int(size(values), c_int), values)
endsubroutine

! Wrapper needed to take arrays of any rank, and convert between fortran strings and
! null-terminated C strings
subroutine ifmlip_store_float_tensor(name, values)
    character(len=*, kind=kind("a")), intent(in) :: name
    real(c_double), contiguous, target, intent(in) :: values(..)
    character(len=len_trim(name) + 1, kind=c_char) :: c_name
    real(c_double), pointer :: flat_values(:)
    real(c_double), target :: no_values(0)
    c_name = trim(name) // c_null_char
    flat_values => no_values
    if (size(values) > 0) then
        call c_f_pointer(c_loc(values), flat_values, [size(values)])
    endif
    call internal_store_float_tensor( &
        c_name, &
        flat_values, &
        c_order_dims(shape(values)), &
        int(rank(values), c_int) &
    )
endsubroutine

! Wrapper needed to take arrays of any rank, and convert between fortran strings and
! null-terminated C strings
subroutine ifmlip_store_int_tensor(name, values)
    character(len=*, kind=kind("a")), intent(in) :: name
    integer(c_int), contiguous, target, intent(in) :: values(..)
    character(len=len_trim(name) + 1, kind=c_char) :: c_name
    integer(c_int), pointer :: flat_values(:)
    integer(c_int), target :: no_values(0)
    c_name = trim(name) // c_null_char
    flat_values => no_values
    if (size(values) > 0) then
        call c_f_pointer(c_loc(values), flat_values, [size(values)])
    endif
    call internal_store_int_tensor( &
        c_name, &
        flat_values, &
        c_order_dims(shape(values)), &
        int(rank(values), c_int) &
    )
endsubroutine

! Wrapper needed to take arrays of any rank, and convert between fortran strings and
! null-terminated C strings
subroutine ifmlip_store_mask_tensor(name, values)
    character(len=*, kind=kind("a")), intent(in) :: name
    integer(c_int), contiguous, target, intent(in) :: values(..)
    character(len=len_trim(name) + 1, kind=c_char) :: c_name
    integer(c_int), pointer :: flat_values(:)
    integer(c_int), target :: no_values(0)
    c_name = trim(name) // c_null_char
    flat_values => no_values
    if (size(values) > 0) then
        call c_f_pointer(c_loc(values), flat_values, [size(values)])
    endif
    call internal_store_mask_tensor( &
        c_name, &
        flat_values, &
        c_order_dims(shape(values)), &
        int(rank(values), c_int) &
    )
endsubroutine

! Dims in C (row-major) order, from the shape of a Fortran (column-major) array
pure function c_order_dims(fortran_shape) result(dims)
    integer, intent(in) :: fortran_shape(:)
    integer(c_int) :: dims(size(fortran_shape))
    dims = int(fortran_shape(size(fortran_shape):1:-1), c_int)
endfunction

subroutine ifmlip_dump_walltime()
    ! State, clock count rate is assumed to be constant between calls
    logical, save :: first_call = .true.
    character(len=1), save :: need_walltime = "0"  ! Only set once
    integer(kind=8), save :: clock_count_rate = 1  ! Only set once
    double precision, save :: prev_time_ms = 0.0D0  ! Reset every call
    integer, parameter :: walltime_fd = 280000
    ! TODO: It is gfortran specific that kind=8 is int64, which makes the clock precise
    integer(kind=8) :: clock_count
    double precision :: time_ms

    clock_count = 0_8
    time_ms = 0.0D0

    if (first_call) then
        call get_environment_variable("TORCHANI_AMBER_WALLTIME", need_walltime)
        if (need_walltime /= "1") then
            return
        endif
        open(unit=walltime_fd, file='walltime_ms.dat', status='REPLACE')
        close(walltime_fd)
        call system_clock(clock_count, clock_count_rate)
        prev_time_ms = dble(clock_count) / dble(clock_count_rate) * 1000.0D0
        first_call = .false.
    else
        if (need_walltime /= "1") then
            return
        endif
        ! NOTE: possible wraparound in system clock is disregarded since time
        ! between calls is extremely short
        open(unit=walltime_fd, file='walltime_ms.dat', status='OLD', position='APPEND')
        call system_clock(clock_count)
        time_ms = dble(clock_count) / dble(clock_count_rate) * 1000.0D0
        write(walltime_fd, "(es30.15E4)") time_ms - prev_time_ms
        flush(unit=walltime_fd)
        prev_time_ms = time_ms
        close(unit=walltime_fd)
    endif
endsubroutine
