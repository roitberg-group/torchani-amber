module torchani
use, intrinsic :: iso_fortran_env
use, intrinsic :: iso_c_binding
implicit none (type, external)
! Plan for the interface
public :: &
    torchani_init, &
    torchani_init_model, &
    torchani_setup_atomic_nums, &
    torchani_single_point, &
    torchani_single_point_from_external_neighbors, &
    torchani_fetch_float64_result, &
    torchani_finalize_model, &
    torchani_finalize

interface
subroutine torchani_init(num_models, err_status, err_msg) bind(c, name="torchani_init")
    use, intrinsic :: iso_c_binding
    integer(c_int), intent(in) :: num_models
    integer(c_int), intent(in) :: err_status
    character(kind=c_char, len=1), intent(in) :: err_msg(*)
endsubroutine

subroutine torchani_init_model(id, path, device_idx, double, cuda, err_status, err_msg) bind(c, name="torchani_init_model")
    use, intrinsic :: iso_c_binding
    integer(c_int), intent(in) :: id
    character(kind=c_char, len=1), intent(in) :: path(*)
    integer(c_int), intent(in) :: device_idx
    integer(c_int), intent(in) :: cuda  ! 0 or 1 (bool)
    integer(c_int), intent(in) :: double  ! 0 or 1 (bool)
    integer(c_int), intent(in) :: err_status
    character(kind=c_char, len=1), intent(in) :: err_msg(*)
endsubroutine

subroutine torchani_setup_atomic_nums(id, atomic_nums, err_status, err_msg) bind(c, name="torchani_setup_atomic_nums")
    use, intrinsic :: iso_c_binding
    integer(c_int), intent(in) :: id
    integer(c_int), intent(in) :: atomic_nums(*)
    integer(c_int), intent(in) :: err_status
    character(kind=c_char, len=1), intent(in) :: err_msg(*)
endsubroutine

subroutine torchani_single_point(&
    id, &
    atomic_nums, &
    coords, &
    cell, &
    pbc, &
    properties, &
    err_status, &
    err_msg &
) bind(c, name="torchani_single_point")
    use, intrinsic :: iso_c_binding
    integer(c_int), intent(in) :: id
    integer(c_int), intent(in) :: atomic_nums(*)
    real(c_double), intent(in) :: coords(*)
    real(c_double), intent(in) :: cell(*)
    integer(c_int), intent(in) :: pbc  ! 1 or 2
    character(kind=c_char, len=1), intent(in) :: properties(*)
    integer(c_int), intent(in) :: err_status
    character(kind=c_char, len=1), intent(in) :: err_msg(*)
endsubroutine

subroutine torchani_single_point_from_external_neighbors(&
    id, &
    atomic_nums, &
    coords, &
    cell, &
    pbc, &
    properties, &
    err_status, &
    err_msg &
) bind(c, name="torchani_single_point_from_external_neighbors")
    use, intrinsic :: iso_c_binding
    integer(c_int), intent(in) :: id
    integer(c_int), intent(in) :: atomic_nums(*)
    real(c_double), intent(in) :: coords(*)
    real(c_double), intent(in) :: cell(*)
    integer(c_int), intent(in) :: pbc  ! 1 or 2
    character(kind=c_char, len=1), intent(in) :: properties(*)
    integer(c_int), intent(in) :: err_status
    character(kind=c_char, len=1), intent(in) :: err_msg(*)
endsubroutine

! Results are always obtained as double precision arrays
subroutine torchani_fetch_float64_result(id, property, array, err_status, err_msg) bind(c, name="torchani_single_point")
    use, intrinsic :: iso_c_binding
    integer(c_int), intent(in) :: id
    character(kind=c_char, len=1), intent(in) :: property(*)
    real(c_double), intent(in) :: array(*)
    integer(c_int), intent(in) :: err_status
    character(kind=c_char, len=1), intent(in) :: err_msg(*)
endsubroutine

! Results are always obtained as double precision arrays
subroutine torchani_finalize_model(id, err_status, err_msg) bind(c, name="torchani_single_point")
    use, intrinsic :: iso_c_binding
    integer(c_int), intent(in) :: id
    integer(c_int), intent(in) :: err_status
    character(kind=c_char, len=1), intent(in) :: err_msg(*)
endsubroutine

subroutine torchani_finalize(err_status, err_msg) bind(c, name="torchani_finalize")
    use, intrinsic :: iso_c_binding
    integer(c_int), intent(in) :: err_status
    character(kind=c_char, len=1), intent(in) :: err_msg(*)
    ! No-op
endsubroutine
endinterface
endmodule
