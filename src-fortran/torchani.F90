module torchani
use, intrinsic :: iso_fortran_env
use, intrinsic :: iso_c_binding
implicit none (type, external)
! Plan for the interface
public :: &
    torchani_init_model, &
    torchani_energy_force, &
    torchani_energy_force_from_external_neighbors, &
    torchani_energy_force_pbc, &
    torchani_energy_force_qbc, &
    torchani_energy_force_atomic_charges, &
    torchani_energy_force_atomic_charges_with_derivatives, &
    torchani_data_for_monitored_mlmm, &
    convert_sander_neighborlist_to_ani_fmt, &
    convert_pmemd_neighborlist_to_ani_fmt

interface
subroutine internal_init_model( &
    num_atoms, &
    atomic_nums, &
    model_type, &
    device_index, &
    network_index, &
    use_double_precision, &
    use_cuda_device, &
    use_cuaev &
) bind(c, name="torchani_init_model")
    use, intrinsic :: iso_c_binding
    integer(c_int), value, intent(in) :: num_atoms
    integer(c_int), intent(in) :: atomic_nums(*)
    character(len=1, kind=c_char), intent(in) :: model_type(*)
    integer(c_int), value, intent(in) :: device_index
    integer(c_int), value, intent(in) :: network_index
    ! Flags
    logical(c_bool), value, intent(in) :: use_double_precision
    logical(c_bool), value, intent(in) :: use_cuda_device
    logical(c_bool), value, intent(in) :: use_cuaev
endsubroutine

subroutine torchani_energy_force_atomic_charges( &
    num_atoms, &
    coords, &
    forces, &
    atomic_charges, &
    potential_energy &
) bind(c, name="torchani_energy_force_atomic_charges")
    use, intrinsic :: iso_c_binding
    integer(c_int), value, intent(in) :: num_atoms
    real(c_double), intent(in) :: coords(*)
    ! Outputs
    real(c_double), intent(out) :: forces(*)
    real(c_double), intent(out) :: atomic_charges(*)
    real(c_double), intent(out) :: potential_energy
endsubroutine

subroutine torchani_energy_force_atomic_charges_with_derivatives( &
    num_atoms, &
    coords, &
    forces, &
    atomic_charges, &
    atomic_charges_grad, &
    potential_energy &
) bind(c, name="torchani_energy_force_atomic_charges_with_derivatives")
    use, intrinsic :: iso_c_binding
    integer(c_int), value, intent(in) :: num_atoms
    real(c_double), intent(in) :: coords(*)
    ! Outputs
    real(c_double), intent(out) :: forces(*)
    real(c_double), intent(out) :: atomic_charges(*)
    real(c_double), intent(out) :: atomic_charges_grad(*)
    real(c_double), intent(out) :: potential_energy
endsubroutine

subroutine torchani_energy_force( &
    num_atoms, &
    coords, &
    forces, &
    potential_energy &
) bind(c, name="torchani_energy_force")
    use, intrinsic :: iso_c_binding
    integer(c_int), value, intent(in) :: num_atoms
    real(c_double), intent(in) :: coords(*)
    ! Outputs
    real(c_double), intent(out) :: forces(*)
    real(c_double), intent(out) :: potential_energy
endsubroutine

subroutine torchani_energy_force_qbc( &
    num_atoms, &
    coords, &
    forces, &
    qbc, &
    potential_energy &
) bind(c, name="torchani_energy_force_qbc")
    use, intrinsic :: iso_c_binding
    integer(c_int), value, intent(in) :: num_atoms
    real(c_double), intent(in) :: coords(*)
    ! Outputs
    real(c_double), intent(out) :: forces(*)
    real(c_double), intent(out) :: qbc
    real(c_double), intent(out) :: potential_energy
endsubroutine

subroutine torchani_energy_force_pbc( &
    num_atoms, &
    coords, &
    cell, &
    forces, &
    potential_energy &
) bind(c, name="torchani_energy_force_pbc")
    use, intrinsic :: iso_c_binding
    integer(c_int), value, intent(in) :: num_atoms
    real(c_double), intent(in) :: coords(*)
    real(c_double), intent(in) :: cell(*)
    ! Outputs
    real(c_double), intent(out) :: forces(*)
    real(c_double), intent(out) :: potential_energy
endsubroutine

subroutine torchani_data_for_monitored_mlmm( &
    num_atoms, &
    coords, &
    forces, &
    atomic_charges, &
    atomic_charges_grad, &
    potential_energy, &
    qbc, &
    qbc_grad &
) bind(c, name="torchani_data_for_monitored_mlmm")
    use, intrinsic :: iso_c_binding
    integer(c_int), value, intent(in) :: num_atoms
    real(c_double), intent(in) :: coords(*)
    ! Outputs
    real(c_double), intent(out) :: forces(*)
    real(c_double), intent(out) :: atomic_charges(*)
    real(c_double), intent(out) :: atomic_charges_grad(*)
    real(c_double), intent(out) :: qbc
    real(c_double), intent(out) :: qbc_grad(*)
    real(c_double), intent(out) :: potential_energy
endsubroutine

subroutine torchani_energy_force_from_external_neighbors( &
    num_atoms, &
    num_neighbors, &
    coords, &
    neighborlist, &
    shifts, &
    forces, &
    potential_energy &
) bind(c, name="torchani_energy_force_from_external_neighbors")
    use, intrinsic :: iso_c_binding
    integer(c_int), value, intent(in) :: num_atoms
    integer(c_int), value, intent(in) :: num_neighbors
    real(c_double), intent(in) :: coords(*)
    integer(c_int), intent(in) :: neighborlist(*)
    real(c_double), intent(in) :: shifts(*)
    ! Outputs
    real(c_double), intent(out) :: forces(*)
    real(c_double), intent(out) :: potential_energy
endsubroutine
endinterface

contains

! Wrapper over this routine is needed in case fbool don't have the same binary
! representation as c_bools.
subroutine torchani_init_model( &
    atomic_nums, &
    device_index, &
    model_type, &
    network_index, &
    use_double_precision, &
    use_cuda_device, &
    use_cuaev &
)
    integer(c_int), contiguous, intent(in) :: atomic_nums(:)
    integer(c_int), value, intent(in) :: device_index
    character(len=*, kind=kind("a")), intent(in) :: model_type
    integer(c_int), intent(in) :: network_index
    ! Flags
    logical, intent(in) :: use_double_precision
    logical, intent(in) :: use_cuda_device
    logical, intent(in) :: use_cuaev

    character(len=len_trim(model_type) + 1, kind=c_char) :: c_model_type
    ! Copying to a temporary string with kind=c_char "casts" the kind to c_char
    c_model_type = trim(model_type) // c_null_char
    call internal_init_model( &
        size(atomic_nums), &
        atomic_nums, &
        c_model_type, &
        device_index, &
        network_index, &
        fbool_to_cbool(use_double_precision), &
        fbool_to_cbool(use_cuda_device), &
        fbool_to_cbool(use_cuaev) &
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
endmodule
