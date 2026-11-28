! Hand-written part of the torchani module, from
! interface/templates/torchani_fortran_utils.F90.
!
! This is not a standalone file. interface/templates/torchani.F90.jinja includes
! it verbatim inside the "contains" section of module torchani, and the result
! is interface/torchani.F90. This file holds the hand-written code. Things to
! keep in mind:
! - It relies on the module header in the template: "use iso_c_binding" and
!   "implicit none" are not repeated here.
! - mlip_initialize calls internal_initialize, the bind(c) interface
!   generated from interface/interface.yaml. If the args of mlip_initialize
!   change in the yaml, that call must be updated.
! - mlip_calc_energy_force_assumed_size and
!   mlip_calc_energy_force_from_external_neighbors call internal_calc_energy_force and
!   internal_calc_energy_force_from_external_neighbors,
!   generated from interface/interface.yaml. If their args change in the yaml, those
!   calls must be updated.
! - mlip_calc_energy_force_mlmm calls internal_calc_energy_force_mlmm or
!   internal_calc_energy_force_mlmm_qbc, generated from interface/interface.yaml. If
!   their args change in the yaml, those calls must be updated.
! - The generated Fortran wrappers call fbool_to_cbool, defined here.
! - mlip_set_output_unit calls internal_set_output_unit, generated from
!   interface/interface.yaml, and passes it mlip_write_line_to_unit. The args of
!   mlip_write_line_to_unit must match mlip_write_line_fn in include/torchani.h.
! - The mlip_store_*_tensor wrappers call the bind(c) interfaces
!   internal_store_*_tensor, generated from interface/interface.yaml.
! - Routines here are made public in the "public" list of the template, so a
!   new public routine must also be added there.

subroutine mlip_initialize( &
    config_path, &
    from_namelist, &
    calling_program &
)
    character(len=*, kind=kind("a")), intent(in) :: config_path
    character(len=*, kind=kind("a")), optional, intent(in) :: from_namelist
    character(len=*, kind=kind("a")), optional, intent(in) :: calling_program

    character(len=len_trim(config_path) + 1, kind=c_char) :: c_config_path
    character(len=:, kind=c_char), allocatable :: c_from_namelist
    character(len=:, kind=c_char), allocatable :: c_calling_program
    ! Copying to a temporary string with kind=c_char "casts" the kind to c_char
    c_config_path = trim(config_path) // c_null_char
    c_from_namelist = "mlip" // c_null_char
    if (present(from_namelist)) then
        c_from_namelist = trim(from_namelist) // c_null_char
    endif
    c_calling_program = "none" // c_null_char
    if (present(calling_program)) then
        c_calling_program = trim(calling_program) // c_null_char
    endif
    call internal_initialize( &
        c_config_path, &
        c_from_namelist, &
        c_calling_program &
    )
endsubroutine

! The output printed by libtorchani is written to unit instead of stdout. Must be called
! after each mlip_initialize
subroutine mlip_set_output_unit(unit)
    integer, intent(in) :: unit
    call internal_set_output_unit(int(unit, c_int), c_funloc(mlip_write_line_to_unit))
endsubroutine

! Called by libtorchani, once per line
subroutine mlip_write_line_to_unit(unit, line, line_len) bind(c)
    integer(c_int), value, intent(in) :: unit
    integer(c_int), value, intent(in) :: line_len
    character(len=1, kind=c_char), intent(in) :: line(line_len)
    write(unit, '(*(a))') line
endsubroutine

! PBC are used only if cell is present. molecule_idxs and calc_only_bonded must be both
! present or both absent, if absent calc_only_bonded is false
subroutine mlip_calc_energy_force( &
    coords, &
    molecule_idxs, &
    calc_only_bonded, &
    out_forces, &
    out_pot_energy, &
    cell &
)
    real(c_double), contiguous, intent(in) :: coords(:, :)  ! Shape (3, atoms)
    integer(c_int), optional, contiguous, intent(in) :: molecule_idxs(:)  ! Shape (atoms,)
    logical, optional, intent(in) :: calc_only_bonded
    ! Outputs
    real(c_double), contiguous, intent(out) :: out_forces(:, :)  ! Shape (3, atoms)
    real(c_double), intent(out) :: out_pot_energy
    real(c_double), optional, intent(in) :: cell(3, 3)

    if (size(coords, 1) /= 3) then
        call exit_with_error("mlip_calc_energy_force: coords must have shape (3, atoms)")
    endif
    if (any(shape(out_forces) /= shape(coords))) then
        call exit_with_error("mlip_calc_energy_force: out_forces and coords shapes differ")
    endif
    if (present(molecule_idxs)) then
        if (size(molecule_idxs) /= size(coords, 2)) then
            call exit_with_error("mlip_calc_energy_force: molecule_idxs and coords sizes differ")
        endif
    endif
    call mlip_calc_energy_force_assumed_size( &
        size(coords, 2), &
        coords, &
        molecule_idxs, &
        calc_only_bonded, &
        out_forces, &
        out_pot_energy, &
        cell &
    )
endsubroutine

! Same as mlip_calc_energy_force, for arrays whose size is not known (e.g. assumed-size)
subroutine mlip_calc_energy_force_assumed_size( &
    num_atoms, &
    coords, &
    molecule_idxs, &
    calc_only_bonded, &
    out_forces, &
    out_pot_energy, &
    cell &
)
    integer(c_int), intent(in) :: num_atoms
    real(c_double), intent(in) :: coords(*)
    integer(c_int), optional, intent(in) :: molecule_idxs(*)
    logical, optional, intent(in) :: calc_only_bonded
    ! Outputs
    real(c_double), intent(out) :: out_forces(*)
    real(c_double), intent(out) :: out_pot_energy
    real(c_double), optional, intent(in) :: cell(3, 3)

    real(c_double) :: pbc_cell(3, 3)
    integer(c_int) :: used_molecule_idxs(num_atoms)
    logical :: do_calc_only_bonded
    if (present(molecule_idxs) .neqv. present(calc_only_bonded)) then
        call exit_with_error( &
            "mlip_calc_energy_force: molecule_idxs and calc_only_bonded must be both" &
            // " present or both absent" &
        )
    endif
    used_molecule_idxs = 0
    do_calc_only_bonded = .false.
    if (present(molecule_idxs)) then
        used_molecule_idxs = molecule_idxs(1:num_atoms)
        do_calc_only_bonded = calc_only_bonded
    endif
    pbc_cell = 0.0d0
    if (present(cell)) then
        pbc_cell = cell
    endif
    call internal_calc_energy_force( &
        num_atoms, &
        coords, &
        pbc_cell, &
        fbool_to_cbool(present(cell)), &
        used_molecule_idxs, &
        fbool_to_cbool(do_calc_only_bonded), &
        out_forces, &
        out_pot_energy &
    )
endsubroutine

! molecule_idxs and calc_only_bonded must be both present or both absent, if absent
! calc_only_bonded is false
subroutine mlip_calc_energy_force_from_external_neighbors( &
    coords, &
    neighborlist, &
    shifts, &
    molecule_idxs, &
    calc_only_bonded, &
    out_forces, &
    out_pot_energy &
)
    real(c_double), contiguous, intent(in) :: coords(:, :)  ! Shape (3, atoms)
    integer(c_int), contiguous, intent(in) :: neighborlist(:, :)  ! Shape (neighbors, 2)
    real(c_double), contiguous, intent(in) :: shifts(:, :)  ! Shape (3, neighbors)
    integer(c_int), optional, contiguous, intent(in) :: molecule_idxs(:)  ! Shape (atoms,)
    logical, optional, intent(in) :: calc_only_bonded
    ! Outputs
    real(c_double), intent(out) :: out_forces(*)  ! 3 * atoms values
    real(c_double), intent(out) :: out_pot_energy

    integer(c_int) :: used_molecule_idxs(size(coords, 2))
    logical :: do_calc_only_bonded
    if (size(coords, 1) /= 3 .or. size(shifts, 1) /= 3 .or. size(neighborlist, 2) /= 2) then
        call exit_with_error( &
            "mlip_calc_energy_force_from_external_neighbors: coords, shifts or" &
            // " neighborlist have the wrong shape" &
        )
    endif
    if (size(shifts, 2) /= size(neighborlist, 1)) then
        call exit_with_error( &
            "mlip_calc_energy_force_from_external_neighbors: shifts and neighborlist" &
            // " sizes differ" &
        )
    endif
    if (present(molecule_idxs) .neqv. present(calc_only_bonded)) then
        call exit_with_error( &
            "mlip_calc_energy_force_from_external_neighbors: molecule_idxs and" &
            // " calc_only_bonded must be both present or both absent" &
        )
    endif
    used_molecule_idxs = 0
    do_calc_only_bonded = .false.
    if (present(molecule_idxs)) then
        if (size(molecule_idxs) /= size(coords, 2)) then
            call exit_with_error( &
                "mlip_calc_energy_force_from_external_neighbors: molecule_idxs and coords" &
                // " sizes differ" &
            )
        endif
        used_molecule_idxs = molecule_idxs
        do_calc_only_bonded = calc_only_bonded
    endif
    call internal_calc_energy_force_from_external_neighbors( &
        size(coords, 2), &
        size(neighborlist, 1), &
        coords, &
        neighborlist, &
        shifts, &
        used_molecule_idxs, &
        fbool_to_cbool(do_calc_only_bonded), &
        out_forces, &
        out_pot_energy &
    )
endsubroutine

! If calc_qbc_and_qbc_grad, calls mlip_calc_energy_force_mlmm_qbc instead of
! mlip_calc_energy_force_mlmm
! If return_grads_instead_of_forces, out_ml_forces and out_mm_forces hold the energy grads
! PBC are used only if cell is present
subroutine mlip_calc_energy_force_mlmm( &
    ml_coords, &
    mm_coords, &
    mm_atomic_charges, &
    cell, &
    calc_qbc_and_qbc_grad, &
    return_grads_instead_of_forces, &
    verbose, &
    out_ml_forces, &
    out_mm_forces, &
    out_pot_energy &
)
    real(c_double), contiguous, intent(in) :: ml_coords(:, :)  ! Shape (3, ML)
    real(c_double), contiguous, intent(in) :: mm_coords(:, :)  ! Shape (3, MM)
    real(c_double), contiguous, intent(in) :: mm_atomic_charges(:)  ! Shape (MM,)
    real(c_double), optional, intent(in) :: cell(3, 3)
    logical, optional, intent(in) :: calc_qbc_and_qbc_grad
    logical, optional, intent(in) :: return_grads_instead_of_forces
    logical, intent(in) :: verbose
    ! Outputs
    real(c_double), contiguous, intent(out) :: out_ml_forces(:, :)  ! Shape (3, ML)
    real(c_double), contiguous, intent(out) :: out_mm_forces(:, :)  ! Shape (3, MM)
    real(c_double), intent(out) :: out_pot_energy

    real(c_double) :: pbc_cell(3, 3)
    logical :: do_calc_qbc_and_qbc_grad
    if (size(ml_coords, 1) /= 3) then
        call exit_with_error("mlip_calc_energy_force_mlmm: ml_coords must have shape (3, ML)")
    endif
    if (size(mm_coords, 1) /= 3) then
        call exit_with_error("mlip_calc_energy_force_mlmm: mm_coords must have shape (3, MM)")
    endif
    if (any(shape(out_ml_forces) /= shape(ml_coords))) then
        call exit_with_error("mlip_calc_energy_force_mlmm: out_ml_forces and ml_coords shapes differ")
    endif
    if (any(shape(out_mm_forces) /= shape(mm_coords))) then
        call exit_with_error("mlip_calc_energy_force_mlmm: out_mm_forces and mm_coords shapes differ")
    endif
    if (size(mm_atomic_charges) /= size(mm_coords, 2)) then
        call exit_with_error( &
            "mlip_calc_energy_force_mlmm: mm_atomic_charges and mm_coords sizes differ" &
        )
    endif
    pbc_cell = 0.0d0
    if (present(cell)) then
        pbc_cell = cell
    endif
    do_calc_qbc_and_qbc_grad = .false.
    if (present(calc_qbc_and_qbc_grad)) then
        do_calc_qbc_and_qbc_grad = calc_qbc_and_qbc_grad
    endif
    if (do_calc_qbc_and_qbc_grad) then
        call internal_calc_energy_force_mlmm_qbc( &
            size(ml_coords, 2), &
            size(mm_coords, 2), &
            ml_coords, &
            mm_coords, &
            mm_atomic_charges, &
            pbc_cell, &
            fbool_to_cbool(present(cell)), &
            fbool_to_cbool(verbose), &
            out_ml_forces, &
            out_mm_forces, &
            out_pot_energy &
        )
    else
        call internal_calc_energy_force_mlmm( &
            size(ml_coords, 2), &
            size(mm_coords, 2), &
            ml_coords, &
            mm_coords, &
            mm_atomic_charges, &
            pbc_cell, &
            fbool_to_cbool(present(cell)), &
            fbool_to_cbool(verbose), &
            out_ml_forces, &
            out_mm_forces, &
            out_pot_energy &
        )
    endif
    if (present(return_grads_instead_of_forces)) then
        if (return_grads_instead_of_forces) then
            out_ml_forces = -out_ml_forces
            out_mm_forces = -out_mm_forces
        endif
    endif
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
    ! Sander's list is assumed-size, so (:) can't be used
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

! Same format and exit code as the errors of libtorchani
subroutine exit_with_error(message)
    character(len=*), intent(in) :: message
    write(error_unit, "(a)") "ERROR (libtorchani)"
    write(error_unit, "(a)") message
    error stop 2
endsubroutine

logical(c_bool) function fbool_to_cbool(fbool) result(ret)
    logical, intent(in) :: fbool
    if (fbool) then
        ret = .true._c_bool
    else
        ret = .false._c_bool
    endif
endfunction

subroutine mlip_set_atomic_numbers(atomic_nums, print_info)
    integer(c_int), contiguous, intent(in) :: atomic_nums(:)
    logical, optional, intent(in) :: print_info
    logical :: do_print_info
    do_print_info = .false.
    if (present(print_info)) then
        do_print_info = print_info
    endif
    call internal_set_atomic_numbers( &
        int(size(atomic_nums), c_int), &
        atomic_nums, &
        fbool_to_cbool(do_print_info) &
    )
endsubroutine

subroutine mlip_set_no_eval_atoms_mask(values)
    integer(c_int), contiguous, intent(in) :: values(:)
    call internal_set_no_eval_atoms_mask(int(size(values), c_int), values)
endsubroutine

! Wrapper needed to take arrays of any rank, and convert between fortran strings and
! null-terminated C strings
subroutine mlip_store_float_tensor(name, values)
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
subroutine mlip_store_int_tensor(name, values)
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
subroutine mlip_store_mask_tensor(name, values)
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

! Wrapper needed to take arrays of any rank, and convert between fortran strings and
! null-terminated C strings
subroutine mlip_fetch_result(name, values)
    character(len=*, kind=kind("a")), intent(in) :: name
    real(c_double), contiguous, target, intent(out) :: values(..)
    character(len=len_trim(name) + 1, kind=c_char) :: c_name
    real(c_double), pointer :: flat_values(:)
    real(c_double), target :: no_values(0)
    c_name = trim(name) // c_null_char
    flat_values => no_values
    if (size(values) > 0) then
        call c_f_pointer(c_loc(values), flat_values, [size(values)])
    endif
    call internal_fetch_result( &
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

subroutine mlip_dump_walltime()
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
