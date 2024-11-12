module torchani
use, intrinsic :: iso_fortran_env
use, intrinsic :: iso_c_binding
implicit none (type, external)
! Plan for the interface
public :: &
    torchani_init_atom_types, &
    torchani_energy_force, &
    torchani_energy_force_external_neighborlist, &
    torchani_energy_force_pbc, &
    torchani_energy_force_qbc, &
    torchani_energy_force_atomic_charges, &
    torchani_energy_force_atomic_charges_with_derivatives, &
    torchani_data_for_monitored_mlmm

interface
subroutine internal_init_atom_types( &
    atomic_nums, &
    num_atoms, &
    device_index, &
    model_type, &
    network_index, &
    use_double_precision, &
    use_cuda_device, &
    use_cuaev &
) bind(c, name="torchani_init_atom_types")
    use, intrinsic :: iso_c_binding
    integer(c_int), intent(in) :: atomic_nums(*)
    integer(c_int), intent(in) :: num_atoms
    integer(c_int), intent(in) :: device_index
    character(len=1, kind=c_char), intent(in) :: model_type(*)
    integer(c_int), intent(in) :: network_index
    ! Flags
    logical(c_bool), intent(in) :: use_double_precision
    logical(c_bool), intent(in) :: use_cuda_device
    logical(c_bool), intent(in) :: use_cuaev
endsubroutine

subroutine torchani_energy_force( &
    num_atoms, &
    coords, &
    forces, &
    potential_energy &
) bind(c, name="torchani_energy_force")
    use, intrinsic :: iso_c_binding
    integer(c_int), intent(in) :: num_atoms
    real(c_double), intent(in) :: coords(*)
    ! Outputs
    real(c_double), intent(out) :: forces(*)
    real(c_double), intent(out) :: potential_energy
endsubroutine

subroutine torchani_energy_force_qbc( &
    num_atoms, &
    coords, &
    forces, &
    potential_energy, &
    qbc &
) bind(c, name="torchani_energy_force_qbc")
    use, intrinsic :: iso_c_binding
    integer(c_int), intent(in) :: num_atoms
    real(c_double), intent(in) :: coords(*)
    ! Outputs
    real(c_double), intent(out) :: forces(*)
    real(c_double), intent(out) :: potential_energy
    real(c_double), intent(out) :: qbc(*)
endsubroutine

subroutine torchani_energy_force_pbc( &
    num_atoms, &
    coords, &
    forces, &
    pbc_box, &
    potential_energy &
) bind(c, name="torchani_energy_force_pbc")
    use, intrinsic :: iso_c_binding
    integer(c_int), intent(in) :: num_atoms
    real(c_double), intent(in) :: coords(*)
    real(c_double), intent(out) :: pbc_box(*)
    ! Outputs
    real(c_double), intent(out) :: forces(*)
    real(c_double), intent(out) :: potential_energy
endsubroutine

subroutine torchani_energy_force_atomic_charges( &
    num_atoms, &
    coords, &
    forces, &
    atomic_charges, &
    potential_energy &
) bind(c, name="torchani_energy_force_atomic_charges")
    use, intrinsic :: iso_c_binding
    integer(c_int), intent(in) :: num_atoms
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
    integer(c_int), intent(in) :: num_atoms
    real(c_double), intent(in) :: coords(*)
    ! Outputs
    real(c_double), intent(out) :: forces(*)
    real(c_double), intent(out) :: atomic_charges(*)
    real(c_double), intent(out) :: atomic_charges_grad(*)
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
    integer(c_int), intent(in) :: num_atoms
    real(c_double), intent(in) :: coords(*)
    ! Outputs
    real(c_double), intent(out) :: forces(*)
    real(c_double), intent(out) :: atomic_charges(*)
    real(c_double), intent(out) :: atomic_charges_grad(*)
    real(c_double), intent(out) :: qbc(*)
    real(c_double), intent(out) :: qbc_grad(*)
    real(c_double), intent(out) :: potential_energy
endsubroutine

subroutine torchani_energy_force_external_neighborlist( &
    num_atoms, &
    coords, &
    num_neighbors, &
    neighborlist, &
    shifts, &
    forces, &
    potential_energy &
) bind(c, name="torchani_energy_force_external_neighborlist")
    use, intrinsic :: iso_c_binding
    integer(c_int), intent(in) :: num_atoms
    integer(c_int), intent(in) :: num_neighbors
    real(c_double), intent(in) :: coords(*)
    real(c_double), intent(in) :: shifts(*)
    integer(c_int), intent(in) :: neighborlist(*)
    ! Outputs
    real(c_double), intent(out) :: forces(*)
    real(c_double), intent(out) :: potential_energy(*)
endsubroutine
endinterface

contains

! Wrapper over this routine is needed in case fbool don't have the same binary
! representation as c_bools.
subroutine torchani_init_atom_types( &
    atomic_nums, &
    num_atoms, &
    device_index, &
    model_type, &
    network_index, &
    use_double_precision, &
    use_cuda_device, &
    use_cuaev &
)
    integer(c_int), intent(in) :: atomic_nums(*)
    integer(c_int), intent(in) :: num_atoms
    integer(c_int), intent(in) :: device_index
    character(len=1, kind=c_char), intent(in) :: model_type(*)
    integer(c_int), intent(in) :: network_index
    ! Flags
    logical, intent(in) :: use_double_precision
    logical, intent(in) :: use_cuda_device
    logical, intent(in) :: use_cuaev
    call internal_init_atom_types( &
        atomic_nums, &
        num_atoms, &
        device_index, &
        model_type, &
        network_index, &
        fbool_to_cbool(use_double_precision), &
        fbool_to_cbool(use_cuda_device), &
        fbool_to_cbool(use_cuaev) &
    )
endsubroutine

logical(c_bool) function fbool_to_cbool(fbool) result(ret)
    logical, intent(in) :: fbool
    if (fbool) then
        ret = .true._c_bool
    else
        ret = .false._c_bool
    endif
endfunction
endmodule
