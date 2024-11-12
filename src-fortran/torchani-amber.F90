module torchani
use, intrinsic :: iso_fortran_env
use, intrinsic :: iso_c_binding
implicit none (type, external)
! Plan for the interface
public :: &
    torchani_init_atom_types, &
    torchani_energy_force, &
    torchani_energy_force_external_neighborlist_, &
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
    torchani_model_index, &
    network_index, &
    use_double_precision, &
    use_cuda_device, &
    use_torch_cell_list, &
    use_external_neighborlist, &
    use_cuaev &
) bind(c, name="torchani_init_atom_types")
    use, intrinsic :: iso_c_binding
    integer(c_int), intent(in) :: num_models
    integer(c_int), intent(in) :: err_status
    character(kind=c_char, len=1), intent(in) :: err_msg(*)
endsubroutine
endinterface
contains

subroutine internal_init_atom_types( &
    atomic_nums, &
    num_atoms, &
    device_index, &
    torchani_model_index, &
    network_index, &
    use_double_precision, &
    use_cuda_device, &
    use_torch_cell_list, &
    use_external_neighborlist, &
    use_cuaev &
) bind(c, name="torchani_init_atom_types")
endmodule
