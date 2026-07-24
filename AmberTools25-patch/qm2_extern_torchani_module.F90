#include "../include/dprec.fh"
#include "../include/assert.fh"

! ----------------------------------------------------------------
! Interface for torchani based QM and QM/MM MD
!
! Currently supports:
! pure QM and QM/MM
!
! Authors: Jonathan Semelak, Ignacio Pickering
!
! Based on qm2_extern_orc_module.F90 and qm2_extern_lio_module.F90
! Uses torchani-amber interface (Author: Ignacio Pickering)
! Date: March 2023
! ----------------------------------------------------------------
module qm2_extern_torchani_module
#ifdef TORCHANI
use qmmm_module, only: qmmm_nml, qmmm_struct
use qmmm_nml_module, only: qmmm_nml_type
use torchani, only : &
    torchani_initialize, &
    torchani_calc_energy_force, &
    torchani_calc_energy_force_qbc, &
    torchani_calc_data_for_monitored_mlmm, &
    torchani_calc_energy_force_with_coupling, &
    torchani_energy_force_atomic_charges, &
    torchani_energy_force_atomic_charges_with_derivatives
implicit none

private
public :: get_torchani_forces, torchani_get_qmmm_int5_pol_settings

! TODO: Double check issues with AU and angstrom
double precision :: au_to_ang3 = 0.1481847d0
double precision :: amber_charge_units_to_au = 18.2223d0
double precision :: dummy_ucell(3,3) = 0.0d0

integer, parameter :: INFER_ML_SYSTEM_CHARGE = -9999999

! Dummy variables, allocated on first call to get_torchani_forces
integer, allocatable :: no_eval_atoms(:)
integer, allocatable :: pme_external_molecule_idx(:)  ! Read-only

type ani_nml_type
    ! General TorchANI-Amber config
    character(len=256) :: model_type
    logical :: use_cuda_device
    logical :: use_double_precision
    logical :: use_amber_neighborlist  ! not supported
    logical :: use_cuaev
    ! Advanced TorchANI-Amber config
    integer :: model_index
    integer :: cuda_device_index

    ! Coupling and charges config
    integer :: mlmm_coupling
    logical :: use_torchani_charges
    logical :: use_torchani_volumes
    double precision :: inv_pol_dielectric
    double precision :: pol_cutoff
    logical :: use_torch_coupling
    logical :: use_charges_derivatives  ! Dev only, not user facing
    logical :: use_numerical_qmmm_forces  ! Dev only, not user facing

    ! Output related config
    logical :: write_xyz
    logical :: write_forces
    logical :: write_charges
    logical :: write_charges_grad
    logical :: write_volumes
    logical :: write_volumes_grad

    ! All the following options are either experimental or deprecated
    ! Mixed QM+ANI forces config, 'Switching', experimental
    character(len=256) :: switching_program
    logical :: use_switching_function
    double precision :: qlow
    double precision :: qhigh
    ! External QM/MM coupling config, experimental
    logical :: use_extcoupling
    character(len=256) :: extcoupling_program
end type ani_nml_type

type(ani_nml_type), save :: cached_ani_nml
double precision, save :: cached_elem_alphas(18) = -1.0d0
logical, save :: cached_use_internal_opts = .false.
logical, save :: cached_ani_nml_ready = .false.

character(len=2) :: elem_symbols(18) = [&
    "H ", "He", "Li", "Be" ,&
    "B ", "C ", "N ", "O ",&
    "F ", "Ne", "Na", "Mg",&
    "Al", "Si", "P ", "S ",&
    "Cl", "Ar"&
]
contains

! ----------------------------------------------------------------
! Get QM energy and forces from torchani
! ----------------------------------------------------------------
subroutine get_torchani_forces(&
    nstep,&
    ntpr_internal,&
    nqmatoms,&
    qmcoords,&
    nclatoms,&
    clcoords,&
    escf,&
    dxyzqm,&
    dxyzcl,&
    id&
)
    use file_io_dat
    use qm2_dftb_module
    integer, intent(in) :: nstep                ! Current dynamics step
    integer, intent(in) :: ntpr_internal        ! Interval for disk output
    character(len=3), intent(in) :: id          ! ID number for PIMD or REMD (unused)
    integer, intent(in) :: nqmatoms             ! Number of QM atoms
    double precision,  intent(in) :: qmcoords(3, nqmatoms) ! QM atom coordinates, shape (3, qm-atoms)
    integer, intent(in) :: nclatoms             ! Number of MM atoms
    double precision,  intent(in) :: clcoords(4, nclatoms) ! MM atom coords and charges, shape (4, cl-atoms)
    double precision, intent(inout) :: escf                 ! torchani energy
    double precision, intent(inout) :: dxyzqm(3, nqmatoms)   ! forces on QM atoms, shape (3, qm-atoms)
    double precision, intent(inout) :: dxyzcl(3, nclatoms)   ! forces on MM atoms, shape (3, cl-atoms)
    integer :: qm_atomic_nums(nqmatoms)  ! QM atomic numbers
    double precision :: dxyzqm_qmmm(3, nqmatoms)  ! torchani QMMM forces on QM atoms
    double precision :: alpha(nqmatoms)                  ! polarizabilities on QM atoms
    double precision :: ext_escf 
    double precision :: qbc                                  ! energy qbcs
    ! Experimental, Switching
    double precision :: qmcharges(nqmatoms)                  ! QM atom charges
    double precision dqmcharges(3, nqmatoms, nqmatoms)      ! QM atom charges forces
    double precision :: qmvolumes(nqmatoms)                  ! QM atom volumes
    double precision dqmvolumes(3, nqmatoms, nqmatoms)      ! QM atom volume derivatives
    ! Note that atomic_charge_derivatives is an array of shape [3, num_atoms, num_atoms]
    ! where the element [i, j, k] is the derivative of the **charge on
    ! k-th atom** with respect to the **i-th position of the j-th atom**
    logical :: use_internal_opts
    logical :: write_to_mdout_this_step
    logical :: polarizable_atom_mask(nqmatoms)

    ! Subroutine state
    ! Atomic polarizabilities for elem H-Ar. Negative value signals uninit
    double precision, save :: elem_alphas(18) = -1.0d0
    logical, save :: first_call = .true.
    type(ani_nml_type), save :: ani_nml

    qm_atomic_nums = qmmm_struct%iqm_atomic_numbers
    ! Initialization that happens the first time this subroutine is called
    if (first_call) then
        write (6,'(/,a,/)') '  >>> RUNNING CALCULATIONS WITH TORCHANI <<<'
        ! Initialize global state from namelist (or defaults):
        ! elem_alphas array and ani_nml struct
        call ani_nml_get_cached(ani_nml, elem_alphas, use_internal_opts)
        call ani_nml_print(ani_nml, qmmm_nml%qmmm_int, use_internal_opts)
        call ani_nml_validate(ani_nml, qmmm_nml%qmmm_int, qmmm_nml%qmcharge)
        ! Extra DFTB setup if 'use_extcoupling'
        if (ani_nml%use_extcoupling .and. trim(ani_nml%extcoupling_program) == 'amber-dftb') then
            call torchani_amber_dftb_setup(qmmm_nml)
        end if
        allocate(pme_external_molecule_idx(nqmatoms))
        allocate(no_eval_atoms(nqmatoms))
        pme_external_molecule_idx = 0
        no_eval_atoms = 0
        call torchani_initialize( &
            qm_atomic_nums, &
            no_eval_atoms, &
            ani_nml%cuda_device_index, &
            ani_nml%model_type, &
            ani_nml%model_index, &
            ani_nml%use_double_precision, &
            ani_nml%use_cuda_device, &
            ani_nml%use_cuaev &
        )
        write(6,*) "TORCHANI: Energies are printed in kcal/mol"
    end if

    alpha = 0.0d0
    polarizable_atom_mask = .true.
    if (ani_nml%mlmm_coupling == 1) then
        call qm_alphas_fill(elem_alphas, qm_atomic_nums, alpha)
        if (qmmm_nml%qmmm_int == 5 .and. qmmm_struct%nquant < nqmatoms) then
            polarizable_atom_mask(qmmm_struct%nquant + 1:nqmatoms) = .false.
            alpha(qmmm_struct%nquant + 1:nqmatoms) = 0.0d0
        endif
        if (first_call .and. (.not. ani_nml%use_torchani_volumes)) then
            call qm_alphas_print_in_use(elem_alphas, qm_atomic_nums, alpha)
        endif
    endif

    write_to_mdout_this_step = (ntpr_internal > 0 .and. mod(nstep + 1, ntpr_internal) == 0)

    if (write_to_mdout_this_step) then
        write(6,*) ""
        write(6,*) "------------------------------------------------------------------------------"
    end if

    dxyzqm = 0.0d0
    dxyzcl = 0.0d0
    dxyzqm_qmmm = 0.0d0
    ! Charge grad is init to 0, if the qs are geom-dependent it is modified in-place
    dqmcharges = 0.0d0
    dqmvolumes = 0.0d0
    qmvolumes = 0.0d0
    ! QM charges are initialized as the FF charges, but may be modified by torchani.
    ! Link atoms do not have topology charges, so keep them neutral unless a model
    ! predicts charges.
    qmcharges = 0.0d0
    qmcharges(1:qmmm_struct%nquant) = &
        qmmm_struct%qm_resp_charges(1:qmmm_struct%nquant) / amber_charge_units_to_au
    if (ani_nml%use_torch_coupling) then
        ! In this case torch handles both the in-vacuo energy calculation and the ML/MM coupling,
        ! No need to retrieve the in-vacuo energies separately
        continue
    elseif (ani_nml%use_switching_function) then
        ! This branch is experimental and untested, it does not support torch coupling,
        ! but it supports both mlmm_coupling = 1 and mlmm_coupling = 2
        call get_vacuum_energies_and_forces_switching(&
            nstep, &
            ntpr_internal, &
            nqmatoms, &
            nclatoms, &
            escf, &
            ext_escf, &
            dxyzqm, &
            dxyzcl, &
            qmcoords, &
            clcoords, &
            qm_atomic_nums, &
            qmmm_nml%qmcharge, &
            qmmm_nml%spin, &
            id, &
            ani_nml%switching_program, &
            ani_nml%qlow, &
            ani_nml%qhigh, &
            qbc, &
            ani_nml%use_torchani_charges, &
            qmcharges, &
            dqmcharges &
        )
    else
        ! Garanteed to have net charge == 0. This branch is only for debugging
        if (ani_nml%use_torchani_charges) then
            if (ani_nml%use_charges_derivatives) then
                call torchani_energy_force_atomic_charges_with_derivatives( &
                    size(qmcoords, 2), &
                    qmcoords, &
                    ! Outputs:
                    dxyzqm, &
                    qmcharges, &  ! Override the QM charges
                    dqmcharges, &  ! Override the QM charge-derivatives
                    escf &
                )
            else
                call torchani_energy_force_atomic_charges( &
                    size(qmcoords, 2), &
                    qmcoords, &
                    ! Outputs:
                    dxyzqm, &
                    qmcharges, &  ! Override the QM charges
                    escf &
                )
            endif
        else
            call torchani_calc_energy_force( &
                size(qmcoords, 2), &
                qmcoords, &
                dummy_ucell, &
                .false., &  ! Use pbc
                pme_external_molecule_idx, &
                .false., &  ! Use all_amber_nonbond
                0, & ! Net charge, force 0
                dxyzqm, &
                escf &
            )
        end if
        ! Invert dxyzqm because torchani returns forces, and dxyzqm expects grad
        dxyzqm = -dxyzqm
    end if

    ! Intermediate outputs if not using torch_coupling
    if (write_to_mdout_this_step .and. (ani_nml%use_switching_function)) then
        write(6,'(1x,"QM: IN VACUO ENERGY = ",f14.4)') ext_escf
    end if
    if (write_to_mdout_this_step .and. (.not. ani_nml%use_torch_coupling)) then
        write(6,'(1x,"TORCHANI: IN VACUO ENERGY = ",f14.4)') escf
    end if

    if (qmmm_nml%qmmm_int == 1) then
        if (ani_nml%use_torch_coupling) then
            call get_vacuum_and_coupling_energies_and_forces_through_torchani( &
                write_to_mdout_this_step, &
                qmcoords, &
                alpha, &
                clcoords, &
                ani_nml%inv_pol_dielectric, &
                ani_nml%use_torchani_charges, &
                ani_nml%use_torchani_volumes, &
                ani_nml%mlmm_coupling, &
                .true., &
                polarizable_atom_mask, &
                ani_nml%use_charges_derivatives, &
                qmmm_nml%qmcharge, &
                ani_nml%write_charges, &
                ani_nml%write_charges_grad, &
                ani_nml%write_volumes, &
                ani_nml%write_volumes_grad, &
                qmcharges, &  ! Override QM charges, grad is calc and used internally
                dqmcharges, &
                qmvolumes, &
                dqmvolumes, &
                dxyzqm, &
                dxyzcl, &
                escf &
            )
        else if (ani_nml%use_extcoupling) then
            ! External coupling. forwards call to get_*_forces
            ! which in turn calls a helper QM program for the ML/MM coupling energies
            ! This requires two calls to the external program (vacuum and in charge field)
            ! So it is quite inefficient
            call get_coupling_energies_and_forces_through_qm_helper( &
                nstep, &
                ntpr_internal, &
                nqmatoms, &
                nclatoms, &
                escf, &
                dxyzqm, &
                dxyzcl, &
                qmcoords, &
                clcoords, &
                qm_atomic_nums, &
                qmmm_nml%qmcharge, &
                qmmm_nml%spin, &
                id, &
                ani_nml%extcoupling_program, &
                write_to_mdout_this_step &
            )
        else
            ! This branch is only for debugging and for the switching branch
            call get_coupling_energies_and_forces_fortran( &
                nstep, &
                ntpr_internal, &
                nqmatoms, &
                nclatoms, &
                dxyzqm, &
                qmcoords, &
                clcoords, &
                dxyzcl, &
                dxyzqm_qmmm, &
                ani_nml%use_torchani_charges, &
                qmcharges, &  ! Read qmcharges
                dqmcharges, &  ! Read qm charge derivatives
                ani_nml%mlmm_coupling, &
                ani_nml%inv_pol_dielectric, &
                alpha, &
                ani_nml%use_numerical_qmmm_forces, &
                escf, &
                ani_nml%use_charges_derivatives &
            )
        endif
    elseif (qmmm_nml%qmmm_int == 5) then
        if (ani_nml%use_torch_coupling .and. ani_nml%mlmm_coupling == 1) then
            call get_vacuum_and_coupling_energies_and_forces_through_torchani( &
                write_to_mdout_this_step, &
                qmcoords, &
                alpha, &
                clcoords, &
                ani_nml%inv_pol_dielectric, &
                ani_nml%use_torchani_charges, &
                ani_nml%use_torchani_volumes, &
                ani_nml%mlmm_coupling, &
                .false., &
                polarizable_atom_mask, &
                ani_nml%use_charges_derivatives, &
                qmmm_nml%qmcharge, &
                ani_nml%write_charges, &
                ani_nml%write_charges_grad, &
                ani_nml%write_volumes, &
                ani_nml%write_volumes_grad, &
                qmcharges, &
                dqmcharges, &
                qmvolumes, &
                dqmvolumes, &
                dxyzqm, &
                dxyzcl, &
                escf &
            )
        endif
    endif
    if (first_call) then
        call create_output_files(&
            (size(dxyzcl, dim=2) > 0),&
            ani_nml%write_xyz,&
            ani_nml%write_forces,&
            ani_nml%write_charges,&
            ani_nml%write_charges_grad,&
            ani_nml%write_volumes,&
            ani_nml%write_volumes_grad&
        )
    endif
    call write_output_files(&
        qm_atomic_nums,&
        qmcoords,&
        qmcharges,&
        qmvolumes,&
        dxyzqm,&
        dqmcharges,&
        dqmvolumes,&
        dxyzqm_qmmm,&
        dxyzcl,&
        ani_nml%write_xyz,&
        ani_nml%write_forces,&
        ani_nml%write_charges,&
        ani_nml%write_charges_grad,&
        ani_nml%write_volumes,&
        ani_nml%write_volumes_grad&
    )
    if (write_to_mdout_this_step) then
        write(6,*) "------------------------------------------------------------------------------"
    end if
    ! First call of the subroutine ends here
    if (first_call) then
        first_call = .false.
    endif
endsubroutine

! Extra DFTB setup that needs to be performed on the main subroutine initialization
subroutine torchani_amber_dftb_setup(qmmm_nml)
    type(qmmm_nml_type), intent(inout) :: qmmm_nml
    ! Need to set qm-theory this way so DFTB arrays are allocated
    qmmm_nml%qmtheory%DFTB=.true.
    qmmm_nml%qmtheory%EXTERN=.false.
    qmmm_nml%qmmm_int=1
    call qm2_load_params_and_allocate(.false.)  ! TODO: Explicitly import routine
    ! qmmm_nml%qmmm_int=2 TODO: The default used to be 2, maybe this is not needed anymore?
endsubroutine

subroutine create_output_files(&
    has_mm_region, &
    write_xyz, &
    write_forces, &
    write_charges, &
    write_charges_grad, &
    write_volumes, &
    write_volumes_grad &
)
    logical, intent(in) :: has_mm_region
    logical, intent(in) :: write_xyz
    logical, intent(in) :: write_forces
    logical, intent(in) :: write_charges
    logical, intent(in) :: write_charges_grad
    logical, intent(in) :: write_volumes
    logical, intent(in) :: write_volumes_grad
    if (write_forces) then
        open(unit=271920, file='forces_qm_region.dat', status='REPLACE')
        close(271920)
        if (has_mm_region) then
            open(unit=271927, file='forces_qmmm_qm_region.dat', status='REPLACE')
            close(271927)
            open(unit=271923, file='forces_qmmm_mm_region.dat', status='REPLACE')
            close(271923)
        endif
    endif
    if (write_charges_grad) then
        open(unit=271929, file='charges_grad_qm_region.dat', status='REPLACE')
        close(271929)
    endif
    if (write_charges) then
        open(unit=271921, file='charges_qm_region.dat', status='REPLACE')
        close(271921)
    endif
    if (write_volumes_grad) then
        open(unit=271931, file='volumes_grad_qm_region.dat', status='REPLACE')
        close(271931)
    endif
    if (write_volumes) then
        open(unit=271930, file='volumes_qm_region.dat', status='REPLACE')
        close(271930)
    endif
    if (write_xyz) then
        open(unit=271922, file='qm_region.xyz', status='REPLACE')
        close(271922)
    endif
endsubroutine

! Dump forces, charges and coordinates of the QM region to files (ASCII-formatted)
subroutine write_output_files(&
    qm_atomic_nums,&
    qm_coords,&
    qm_charges,&
    qm_volumes,&
    qm_grads,&
    qm_charges_grad,&
    qm_volumes_grad,&
    qm_qmmm_grads,&
    mm_qmmm_grads,&
    write_xyz,&
    write_forces,&
    write_charges,&
    write_charges_grad,&
    write_volumes,&
    write_volumes_grad&
)
    integer, intent(in) :: qm_atomic_nums(:)
    double precision, intent(in) :: qm_coords(:, :)
    double precision, intent(in) :: qm_charges(:)
    double precision, intent(in) :: qm_volumes(:)
    double precision, intent(in) :: qm_grads(:, :)
    double precision, intent(in) :: qm_charges_grad(:, :, :)
    double precision, intent(in) :: qm_volumes_grad(:, :, :)
    double precision, intent(in) :: qm_qmmm_grads(:, :)
    double precision, intent(in) :: mm_qmmm_grads(:, :)
    logical, intent(in) :: write_xyz
    logical, intent(in) :: write_forces
    logical, intent(in) :: write_charges
    logical, intent(in) :: write_charges_grad
    logical, intent(in) :: write_volumes
    logical, intent(in) :: write_volumes_grad
    integer :: i, j
    integer :: num_qm_atoms
    integer :: num_mm_atoms
    num_qm_atoms = size(qm_atomic_nums)
    num_mm_atoms = size(mm_qmmm_grads, dim=2)
    ! TODO: What is this unit=271921?
    if (write_charges) then
        open(unit=271921, file='charges_qm_region.dat', status='OLD', position='APPEND')
        do i = 1, num_qm_atoms
            write(271921, '(F15.7)') qm_charges(i)
        end do
        write(271921, *)
        close(271921)
    endif

    if (write_charges_grad) then
        open(unit=271929, file='charges_grad_qm_region.dat', status='OLD', position='APPEND')
        do i = 1, num_qm_atoms
            do j = 1, num_qm_atoms
                write(271929, '(3F15.7)') qm_charges_grad(1, j, i), qm_charges_grad(2, j, i), qm_charges_grad(3, j, i)
            end do
        end do
        write(271929, *)
        close(271929)
    endif

    if (write_volumes) then
        open(unit=271930, file='volumes_qm_region.dat', status='OLD', position='APPEND')
        do i = 1, num_qm_atoms
            write(271930, '(F15.7)') qm_volumes(i)
        end do
        write(271930, *)
        close(271930)
    endif

    if (write_volumes_grad) then
        open(unit=271931, file='volumes_grad_qm_region.dat', status='OLD', position='APPEND')
        do i = 1, num_qm_atoms
            do j = 1, num_qm_atoms
                write(271931, '(3F15.7)') qm_volumes_grad(1, j, i), qm_volumes_grad(2, j, i), qm_volumes_grad(3, j, i)
            end do
        end do
        write(271931, *)
        close(271931)
    endif

    if (write_forces) then
        open(unit=271920, file='forces_qm_region.dat', status='OLD', position='APPEND')
        do i = 1, num_qm_atoms
            write(271920, '(3F15.7)') -qm_grads(1, i), -qm_grads(2, i), -qm_grads(3, i)
        end do
        write(271920, *)
        close(271920)
        if (num_mm_atoms >= 1) then
            open(unit=271923, file='forces_qmmm_mm_region.dat', status='OLD', position='APPEND')
            do i = 1, num_mm_atoms
                write(271923, '(3F15.7)') -mm_qmmm_grads(1, i), -mm_qmmm_grads(2, i), -mm_qmmm_grads(3, i)
            end do
            write(271923, *)
            close(271923)
            open(unit=271927, file='forces_qmmm_qm_region.dat', status='OLD', position='APPEND')
            do i = 1, num_qm_atoms
               write(271927, '(3F15.7)') -qm_qmmm_grads(1, i), -qm_qmmm_grads(2, i), -qm_qmmm_grads(3, i)
            end do
            write(271927, *)
            close(271927)
        end if
    endif

    if (write_xyz) then
        open(unit=271922, file='qm_region.xyz', status='OLD', position='APPEND')
        write(271922,'(I8)') num_qm_atoms
        write(271922,*) "TORCHANI: .xyz trajectory file of QM-region"
        do i = 1, num_qm_atoms
            write(271922, '(I3,3F15.7)') qm_atomic_nums(i), qm_coords(1, i), qm_coords(2, i), qm_coords(3, i)
        end do
        close(271922)
    endif
endsubroutine

subroutine ani_nml_get_cached(ani_nml, elem_alphas, use_internal_opts)
    type(ani_nml_type), intent(out) :: ani_nml
    double precision, intent(out) :: elem_alphas(:)
    logical, intent(out) :: use_internal_opts

    if (.not. cached_ani_nml_ready) then
        call ani_nml_init(qmmm_nml%qmmm_int, cached_ani_nml, &
                          cached_elem_alphas, cached_use_internal_opts)
        cached_ani_nml_ready = .true.
    endif

    ani_nml = cached_ani_nml
    elem_alphas = cached_elem_alphas
    use_internal_opts = cached_use_internal_opts
endsubroutine

subroutine torchani_get_qmmm_int5_pol_settings(enabled, pol_cutoff)
    logical, intent(out) :: enabled
    double precision, intent(out) :: pol_cutoff
    type(ani_nml_type) :: ani_nml
    double precision :: elem_alphas(18)
    logical :: use_internal_opts

    call ani_nml_get_cached(ani_nml, elem_alphas, use_internal_opts)

    enabled = .false.
    pol_cutoff = ani_nml%pol_cutoff
    if (qmmm_nml%qmmm_int == 5 .and. &
        ani_nml%use_torch_coupling .and. ani_nml%mlmm_coupling == 1) then
        if (ani_nml%pol_cutoff <= 0.0d0) then
            call ani_nml_error('qmmm_int = 5 polarization coupling requires pol_cutoff > 0 in &ani')
        endif
        enabled = .true.
    endif
endsubroutine

! Read torchani namelist values from file mdin,
! set default values if no user specified value exists for a given var
subroutine ani_nml_init(qmmm_int, ani_nml, elem_alphas, use_internal_opts)
    integer, intent(in) :: qmmm_int
    type(ani_nml_type), intent(out) :: ani_nml
    double precision, intent(out) :: elem_alphas(:)
    ! Detect if internal options are being used and warn accordingly
    logical, intent(out) :: use_internal_opts
    ! General TorchANI config
    character(len=256) :: model_type
    logical :: use_cuda_device
    logical :: use_double_precision
    logical :: use_amber_neighborlist
    logical :: use_cuaev
    ! Advanced TorchANI config
    integer :: model_index
    integer :: cuda_device_index
    ! Output related
    logical :: write_xyz
    logical :: write_forces
    logical :: write_charges
    logical :: write_charges_grad
    logical :: write_volumes
    logical :: write_volumes_grad
    ! mlmm_coupling is only used if qmmm_int = 1, frontend option for users
    integer :: mlmm_coupling
    logical :: use_torchani_charges
    logical :: use_torchani_volumes
    logical :: allow_untested_protocols
    logical :: use_torch_coupling
    double precision :: inv_pol_dielectric
    double precision :: pol_cutoff
    ! Polarizabilities for elements 1-18
    double precision :: pol_H, pol_He, pol_Li, pol_Be
    double precision :: pol_B, pol_C, pol_N, pol_O
    double precision :: pol_F, pol_Ne, pol_Na, pol_Mg
    double precision :: pol_Al, pol_Si, pol_P, pol_S
    double precision :: pol_Cl, pol_Ar
    ! Mixed QM-ANI forces config, 'Switching', experimental
    logical :: use_switching_function
    character(len=256) :: switching_program
    double precision :: qlow
    double precision :: qhigh
    ! External ML/MM coupling config, experimental
    logical :: use_extcoupling
    character(len=256) :: extcoupling_program
    ! Deprecated options and debugging options, do *NOT* use in new scripts
    logical :: use_charges_derivatives
    logical :: use_numerical_qmmm_forces
    namelist /ani/&
        model_type,&
        model_index,&
        mlmm_coupling,&
        allow_untested_protocols,&
        use_double_precision,&
        use_cuda_device,&
        use_cuaev,&
        use_amber_neighborlist,&
        cuda_device_index,&
        use_torchani_charges,&
        use_torchani_volumes,&
        use_torch_coupling,&
        inv_pol_dielectric,&
        pol_cutoff,&
        use_extcoupling,&
        extcoupling_program,&
        switching_program,&
        use_switching_function,&
        qlow,&
        qhigh,&
        use_numerical_qmmm_forces,&
        write_xyz,&
        write_forces,&
        write_charges,&
        write_charges_grad,&
        write_volumes,&
        write_volumes_grad,&
        use_charges_derivatives,&
        pol_H, pol_He, pol_Li, pol_Be,&
        pol_B, pol_C, pol_N, pol_O,&
        pol_F, pol_Ne, pol_Na, pol_Mg,&
        pol_Al, pol_Si, pol_P, pol_S,&
        pol_Cl, pol_Ar
    integer :: iostat
    iostat = 0

    ! Defaults
    ! ANI general
    model_type = 'ani1x'
    use_cuaev = .true.
    use_cuda_device = .true.
    use_double_precision = .true.
    use_amber_neighborlist = .false.
    model_index = -1
    cuda_device_index = 0
    ! ML/MM specific
    inv_pol_dielectric = 0.5d0
    pol_cutoff = -1.0d0
    use_torch_coupling = .true.
    mlmm_coupling = 0
    use_torchani_charges = .false.
    use_torchani_volumes = .false.
    allow_untested_protocols = .false.
    write_xyz = .false.
    write_forces = .false.
    write_charges = .false.
    write_charges_grad = .false.
    write_volumes = .false.
    write_volumes_grad = .false.
    ! Mixed QM-ANI forces config, 'Switching', experimental
    use_switching_function = .false.
    switching_program = 'none'
    qlow  = 0.2d0
    qhigh = 0.3d0
    ! External ML/MM coupling config, experimental
    use_extcoupling = .false.
    extcoupling_program = 'none'
    ! Deprecated and debug options, do *NOT* use in new scripts
    use_numerical_qmmm_forces = .false.
    use_charges_derivatives = .true.

    ! Default atomic polarizabilities
    ! H taken from Litman, J. M., Liu, C., and Ren, P. (2021).
    ! Journal of Chemical Information and Modeling, 62(1), 79-87.
    ! Corresponds to the average of all hydrogen types
    pol_H = 3.08d0
    ! Rest taken from Schwerdtfeger, P., and Nagle, J. K. (2019). Molecular Physics, 117(9-12), 1200-1225.
    pol_He = 1.38375d0
    pol_Li = 164.1125d0
    pol_Be = 37.74d0
    pol_B  = 20.5d0
    pol_C  = 11.3d0
    pol_N  = 7.4d0
    pol_O  = 5.3d0
    pol_F  = 3.74d0
    pol_Ne = 2.6611d0
    pol_Na = 162.7d0
    pol_Mg = 71.2d0
    pol_Al = 57.8d0
    pol_Si = 37.3d0
    pol_P  = 25.0d0
    pol_S  = 19.4d0
    pol_Cl = 14.6d0
    pol_Ar = 11.083d0

    rewind(5)
    read(5, nml=ani, iostat=iostat)
    if (iostat /= 0) then
        call ani_nml_error(&
            'Could not read the "ani" namelist,'&
            //' please check that your mdin file has the correct format'&
        )
    endif

    if (.not. allow_untested_protocols) then
        if (mlmm_coupling == 0 .and. use_torchani_charges) then
            write(6, fmt=*) "Coulombic-only coupling is *UNTESTED* with use_torchani_charges"
            write(6, fmt=*) "If you really mean to run this simulation set allow_untested_protocols = .true."
            call mexit(6,1)  ! TODO: Explicitly import routine
        endif
        if (qmmm_int /= 1 .and. qmmm_int /= 5) then
            write(6, fmt=*) "qmmm_int = 1 and qmmm_int = 5 are the supported options for TorchANI-Amber"
            write(6, fmt=*) "If you really mean to run this simulation set allow_untested_protocols = .true."
            call mexit(6,1)  ! TODO: Explicitly import routine
        endif
    endif

    ! Look for experimental, deprecated or debug opts in config, and warn accordingly
    use_internal_opts = (&
        use_switching_function&
        .or. (switching_program /= "none")&
        .or. (qlow /= 0.2d0)&
        .or. (qhigh /= 0.3d0)&
        .or. use_extcoupling&
        .or. (extcoupling_program /= "none")&
        .or. use_numerical_qmmm_forces&
        .or. (.not. use_charges_derivatives)&
    )

    if (use_internal_opts) then
         write(6,*) ''
         write(6,*) '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
         write(6,*) 'TORCHANI WARNING: You are using experimental or deprecated options'
         write(6,*) 'TORCHANI WARNING: Your code will break in future releases'
         write(6,*) 'TORCHANI WARNING: Results may be wrong, use under your own risk'
         write(6,*) 'TORCHANI WARNING: Consult README.md of torchani-amber for usage'
         write(6,*) '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
         write(6,*) ''
    end if
    ! These options are disregarded from the namelist, only mlmm_coupling is actually used
    ani_nml%mlmm_coupling=mlmm_coupling
    ani_nml%model_type = model_type
    ani_nml%model_index = model_index
    ani_nml%use_double_precision = use_double_precision
    ani_nml%use_cuda_device = use_cuda_device
    ani_nml%cuda_device_index = cuda_device_index
    ani_nml%use_torchani_charges = use_torchani_charges
    ani_nml%use_torchani_volumes = use_torchani_volumes
    ani_nml%use_torch_coupling = use_torch_coupling
    ani_nml%inv_pol_dielectric = inv_pol_dielectric
    ani_nml%pol_cutoff = pol_cutoff
    ani_nml%use_extcoupling = use_extcoupling
    ani_nml%extcoupling_program = extcoupling_program
    ani_nml%switching_program = switching_program
    ani_nml%use_switching_function = use_switching_function
    ani_nml%qlow = qlow
    ani_nml%qhigh = qhigh
    ani_nml%use_numerical_qmmm_forces = use_numerical_qmmm_forces
    ani_nml%write_xyz = write_xyz
    ani_nml%write_charges = write_charges
    ani_nml%write_charges_grad = write_charges_grad
    ani_nml%write_volumes = write_volumes
    ani_nml%write_volumes_grad = write_volumes_grad
    ani_nml%write_forces = write_forces
    ani_nml%use_charges_derivatives = use_charges_derivatives
    ani_nml%use_cuaev = use_cuaev

    elem_alphas(1) = pol_H
    elem_alphas(2) = pol_He
    elem_alphas(3) = pol_Li
    elem_alphas(4) = pol_Be
    elem_alphas(5) = pol_B
    elem_alphas(6) = pol_C
    elem_alphas(7) = pol_N
    elem_alphas(8) = pol_O
    elem_alphas(9) = pol_F
    elem_alphas(10) = pol_Ne
    elem_alphas(11) = pol_Na
    elem_alphas(12) = pol_Mg
    elem_alphas(13) = pol_Al
    elem_alphas(14) = pol_Si
    elem_alphas(15) = pol_P
    elem_alphas(16) = pol_S
    elem_alphas(17) = pol_Cl
    elem_alphas(18) = pol_Ar
endsubroutine

subroutine ani_nml_error(msg)
    character(len=*), intent(in) :: msg
    write(6, *) 'Error in qm2_extern_torchani_module, encountered in the "ani" namelist'
    write(6, *) msg
    call mexit(6,1)  ! TODO: Explicitly import routine
endsubroutine

! Check that all the options passed to the ani_nml are sound.
! If strange combinations with deprecated or experimental options are passed
! Then emit the corresponding warnings
subroutine ani_nml_validate(ani_nml, qmmm_int, ml_system_charge)
    type(ani_nml_type), intent(in) :: ani_nml
    integer, intent(in) :: qmmm_int
    integer, intent(in) :: ml_system_charge
    if (ani_nml%use_amber_neighborlist) then
        call ani_nml_error('Amber neighborlist is not supported for ML/MM simulations')
    endif

    if ((ml_system_charge /= 0) .and. (.not. ani_nml%use_torch_coupling)) then
        call ani_nml_error('Charged systems only supported with use_torch_coupling=True')
    endif

    if ((ani_nml%use_switching_function) .and. ani_nml%use_torch_coupling) then
        call ani_nml_error('Switching is not supported with torch coupling')
    endif

    if (ani_nml%use_torchani_volumes) then
        if (qmmm_int /= 1 .and. qmmm_int /= 5) then
            call ani_nml_error('use_torchani_volumes requires qmmm_int = 1 or qmmm_int = 5')
        endif
        if (ani_nml%mlmm_coupling /= 1) then
            call ani_nml_error('use_torchani_volumes requires mlmm_coupling = 1')
        endif
        if (.not. ani_nml%use_torch_coupling) then
            call ani_nml_error('use_torchani_volumes requires use_torch_coupling = .true.')
        endif
        if (trim(ani_nml%model_type) /= 'animbisv') then
            call ani_nml_error('use_torchani_volumes requires model_type = "animbisv"')
        endif
    endif
    if ((ani_nml%write_volumes .or. ani_nml%write_volumes_grad) .and. (.not. ani_nml%use_torchani_volumes)) then
        call ani_nml_error('write_volumes and write_volumes_grad require use_torchani_volumes = .true.')
    endif

    if (ani_nml%pol_cutoff > 0.0d0 .and. ani_nml%mlmm_coupling /= 1) then
        call ani_nml_error('pol_cutoff only has an effect with mlmm_coupling = 1')
    endif

    ! Check that polarization_dielectric was not modified
    if (ani_nml%inv_pol_dielectric /= 0.5d0) then
        write(6,*) "TORCHANI: *THE RECOMMENDED VALUE IS FOR THE INVERSE POLARIZATION DIELECTRIC IS 1/2*"
        write(6,*) "     Custom inv_pol_dielectric (1/epsilon) specified, with value:", ani_nml%inv_pol_dielectric
        if (ani_nml%mlmm_coupling /= 1) then
            write(6, *) "*ERROR! THE POLARIZATION DIELECTRIC ONLY HAS AN EFFECT WHEN mlmm_coupling=1*"
            call mexit(6, 1)
        endif
    endif

    ! External ML/MM coupling config, experimental
    if (ani_nml%use_extcoupling) then
        write(6,*) "TORCHANI: This is an EXPERIMENTAL feature, don't use in production code!"
        write(6,*) "     External coupling functionality of torchani is ON"
        write(6,*) "     The selected program for external coupling is: ", trim(ani_nml%extcoupling_program)
        if (&
              trim(ani_nml%extcoupling_program) /= 'orca'&
              .and. trim(ani_nml%extcoupling_program) /= 'lio'&
              .and. trim(ani_nml%extcoupling_program) /= 'gau'&
              .and. trim(ani_nml%extcoupling_program) /= 'amber-dftb'&
        ) then
            call ani_nml_error('Only amber-dftb, orca, gaussian and lio are valid external coupling programs for torchani')
        end if
    end if

    ! Mixed QM-ANI forces config, 'Switching', experimental
    if (ani_nml%use_switching_function) then
        write(6,*) "TORCHANI: This is an EXPERIMENTAL feature, don't use in production code!"
        write(6,*) "     Switching functionality of torchani is ON"
        write(6,*) "     The selected switching program is: ", trim(ani_nml%switching_program)
        if (&
            trim(ani_nml%switching_program) /= 'orca'&
            .and. trim(ani_nml%switching_program) /= 'lio'&
            .and. trim(ani_nml%switching_program) /= 'gau'&
        ) then
            call ani_nml_error('Only orca, gaussian and lio are valid switching programs for torchani')
        end if
    end if

    write(6,*) "TORCHANI ELECTROSTATIC COUPLING AND CHARGE CONFIG:"
    if (qmmm_int == 0) then
        ! No ML/MM coupling
        write(6,*) "TORCHANI WARNING: *YOU ARE USING AN UNTESTED PROTOCOL. MAKE SURE THIS IS WHAT YOU INTEND*"
        write(6,*) "     QM/MM electrostatic coupling is turned OFF"
        write(6,*) "     You probably want qmmm_int = 1, not 0"
        if (ani_nml%mlmm_coupling /= 0) then 
            call ani_nml_error('"mlmm_coupling" is not supported if "qmmm_int = 0"')
        endif
    elseif (qmmm_int == 5) then
        ! Sander manages the fixed-charge QM/MM coupling.
        write(6,*) "TORCHANI WARNING: *YOU ARE USING AN UNTESTED PROTOCOL. MAKE SURE THIS IS WHAT YOU INTEND*"
        write(6,*) "     QM/MM fixed-charge Coulomb and VDW terms will be calculated by SANDER"
        write(6,*) "     The mechanical embedding (ME) approximation will be used"
        write(6,*) "     Fixed QM-region charges will be read from the topology file"
        if (ani_nml%use_torchani_charges) then
            call ani_nml_error('qmmm_int = 5 requires use_torchani_charges = .false.; SANDER computes Coulomb from FF charges')
        endif
        if (ani_nml%mlmm_coupling == 0) then
            if (ani_nml%use_torch_coupling) then
                call ani_nml_error('qmmm_int = 5 with mlmm_coupling = 0 requires use_torch_coupling = .false.')
            endif
        elseif (ani_nml%mlmm_coupling == 1) then
            if (.not. ani_nml%use_torch_coupling) then
                call ani_nml_error('qmmm_int = 5 polarization coupling requires use_torch_coupling = .true.')
            endif
            if (ani_nml%pol_cutoff <= 0.0d0) then
                call ani_nml_error('qmmm_int = 5 polarization coupling requires pol_cutoff > 0 in &ani')
            endif
            write(6,*) "     TorchANI will add polarization/distortion using MM charges inside pol_cutoff"
            write(6,*) "     TorchANI will not compute a QM/MM Coulomb term"
            if (ani_nml%use_torchani_volumes) then
                write(6,*) "TORCHANI: Will use ANImbisv-predicted atomic volumes for QM-region polarizabilities"
            else
                write(6,*) "TORCHANI: Will use fixed element polarizabilities for the QM-region"
            endif
        else
            call ani_nml_error('Valid mlmm_coupling values for qmmm_int = 5 are 0 and 1')
        endif
    ! TorchANI manages QM/MM coupling (user-facing setting)
    elseif (qmmm_int == 1) then
        write(6,*) "TORCHANI: QM/MM electrostatic coupling will be calculated by torchani"
        ! External QM/MM coupling config, experimental
        ! TODO: This should probably be mlmm_coupling = 3?
        if (ani_nml%use_extcoupling) then
            write(6,*) "TORCHANI: This is an EXPERIMENTAL feature, don't use in production code!"
            write(6,*) "     Will delegate QM/MM electrostatic coupling to ", trim(ani_nml%extcoupling_program)
            write(6,*) "     The electrostatic coupling approximation will be used"
            write(6,*) "     The QM energies and forces will be also corrected"
        else
            ! Check coupling kind. Options are: 0 (coulombic), or 1 (simple polarizable)
            if (ani_nml%mlmm_coupling == 0) then
                write(6,*) "TORCHANI: Will use a coulombic-only QM/MM interaction energy, with no polarization corrections"
            elseif (ani_nml%mlmm_coupling == 1) then
                write(6,*) "TORCHANI: Will use the simple polarizable approximation for the QM/MM interaction energy"
                write(6,*) "     Thole-inspired correction terms will be added to the coulombic interaction"
                write(6,*) "     This partially accounts for QM-region electrostatic polarization and distortion"
                write(6,*) "     If you use this setting please cite 'https://doi.org/10.1021/acs.jcim.4c00478'"
            else
                call ani_nml_error('Valid mlmm_coupling values are 0 (coulombic) and 1 (simple polarizable)')
            endif
            ! Check charges kind. Options are: torchani-charges topology-charges
            if (ani_nml%use_torchani_charges) then
                write(6,*) "TORCHANI: Will use position-dependent NN-predicted charges for the QM-region, taking into account dq/dr"
                if (.not. ani_nml%use_charges_derivatives) then
                    write(6,*) "TORCHANI: *THE FOLLOWING IS MOST LIKELY AN ERROR, PLEASE CHECK YOUR INPUT!*"
                    write(6,*) "     Disregarding charge derivatives w.r.t coords"
                endif
                if (ani_nml%mlmm_coupling == 0) then
                    write(6,*) "TORCHANI WARNING: *YOU ARE USING AN UNTESTED PROTOCOL. MAKE SURE THIS IS WHAT YOU INTEND*"
                    write(6,*) "     The selected 'coulombic-only coupling' is untested with NN-predicted charges"
                endif
            else
                if (qmmm_struct%nlink > 0) then
                    call ani_nml_error(&
                        'qmmm_int = 1 with link atoms requires use_torchani_charges = .true.; '&
                        //'fixed topology charges cannot be assigned safely to link atoms'&
                    )
                endif
                write(6,*) "TORCHANI: Will use fixed charges read from the topology file for the QM-region"
                if (ani_nml%mlmm_coupling == 1 .and. (.not. ani_nml%use_torchani_volumes)) then
                    write(6,*) "TORCHANI WARNING: *YOU ARE USING AN UNTESTED PROTOCOL. MAKE SURE THIS IS WHAT YOU INTEND*"
                    write(6,*) "     The selected 'simple polarizable coupling' approx is only tested with NN-predicted charges"
                endif
            endif
            if (ani_nml%mlmm_coupling == 1) then
                if (ani_nml%use_torchani_volumes) then
                    write(6,*) "TORCHANI: Will use ANImbisv-predicted atomic volumes for QM-region polarizabilities"
                    if (ani_nml%use_torchani_charges) then
                        write(6,*) "     Using kz constants fitted for NN-predicted charges"
                    else
                        write(6,*) "     Using kz constants fitted for fixed force-field charges"
                    endif
                else
                    write(6,*) "TORCHANI: Will use fixed element polarizabilities for the QM-region"
                endif
            endif
        end if
    else
        call ani_nml_error('TorchANI-Amber requires qmmm_int = 1 (0 and 5 are internal options for debugging only)')
    end if
endsubroutine

subroutine ani_nml_print(ani_nml, qmmm_int, use_internal_opts)
    type(ani_nml_type), intent(in) :: ani_nml
    integer, intent(in) :: qmmm_int
    logical, intent(in) :: use_internal_opts
    write(6,*) 'TORCHANI:------------------TORCHANI config----------------'
    write(6,*) 'TORCHANI:  model_type                   ', ani_nml%model_type
    write(6,*) 'TORCHANI:  model_index                  ', ani_nml%model_index
    write(6,*) 'TORCHANI:  use_cuda_device              ', ani_nml%use_cuda_device
    write(6,*) 'TORCHANI:  use_double_precision         ', ani_nml%use_double_precision
    write(6,*) 'TORCHANI:  use_cuaev                    ', ani_nml%use_cuaev
    write(6,*) 'TORCHANI:  cuda_device_index            ', ani_nml%cuda_device_index
    write(6,*) 'TORCHANI:  write_xyz                    ', ani_nml%write_xyz
    write(6,*) 'TORCHANI:  write_forces                 ', ani_nml%write_forces
    write(6,*) 'TORCHANI:  write_charges                ', ani_nml%write_charges
    write(6,*) 'TORCHANI:  write_charges_grad           ', ani_nml%write_charges_grad
    write(6,*) 'TORCHANI:  write_volumes                ', ani_nml%write_volumes
    write(6,*) 'TORCHANI:  write_volumes_grad           ', ani_nml%write_volumes_grad
    write(6,*) 'TORCHANI:  use_torchani_charges         ', ani_nml%use_torchani_charges
    write(6,*) 'TORCHANI:  use_torchani_volumes         ', ani_nml%use_torchani_volumes
    write(6,*) 'TORCHANI:  pol_cutoff                   ', ani_nml%pol_cutoff
    if (qmmm_int == 1 .or. qmmm_int == 5) then
        if (ani_nml%mlmm_coupling == 0) then
            write(6,*) 'TORCHANI:  mlmm_coupling                ', 0, "(coulombic)"
        elseif (ani_nml%mlmm_coupling == 1) then
            write(6,*) 'TORCHANI:  mlmm_coupling                ', 1, "(simple polarizable)"
        else
            call internal_error()
        endif
    endif
    if (use_internal_opts) then
        write(6,*) 'TORCHANI:------WARNING! INTERNAL OPTIONS, DO NOT USE------'
        write(6,*) 'TORCHANI:  use_numerical_qmmm_forces    ', ani_nml%use_numerical_qmmm_forces
        write(6,*) 'TORCHANI:  use_charges_derivatives      ', ani_nml%use_charges_derivatives
        write(6,*) 'TORCHANI:  switching_program            ', ani_nml%switching_program
        write(6,*) 'TORCHANI:  use_switching_function       ', ani_nml%use_switching_function
        write(6,'(1x,A,F2.1)') 'TORCHANI:  qlow             ', ani_nml%qlow
        write(6,'(1x,A,F2.1)') 'TORCHANI:  qhigh            ', ani_nml%qhigh
        write(6,*) 'TORCHANI:  inv_pol_dielectric           ', ani_nml%inv_pol_dielectric
        write(6,*) 'TORCHANI:  use_extcoupling              ', ani_nml%use_extcoupling
        write(6,*) 'TORCHANI:  extcoupling_program          ', ani_nml%extcoupling_program
    endif
    write(6,*) 'TORCHANI:--------------end config-------------------'
endsubroutine

subroutine internal_error()
    write(6, *) 'Internal error in qm2_extern_torchani_module. Please report a bug'
    call mexit(6,1)  ! TODO: Explicitly import routine
endsubroutine

subroutine get_vacuum_and_coupling_energies_and_forces_through_torchani( &
    write_to_mdout_this_step, &
    qmcoords, &
    alpha, &
    clcoords, &
    inv_pol_dielectric, &
    use_torchani_charges, &
    use_torchani_volumes, &
    mlmm_coupling, &
    compute_coulomb, &
    polarizable_atom_mask, &
    use_charges_derivatives, &
    ml_system_charge, &
    write_charges, &
    write_charges_grad, &
    write_volumes, &
    write_volumes_grad, &
    qmcharges, &
    dqmcharges, &
    qmvolumes, &
    dqmvolumes, &
    dxyzqm, &
    dxyzcl, &
    escf &
)
    logical, intent(in) :: write_to_mdout_this_step
    logical, intent(in) :: use_torchani_charges
    logical, intent(in) :: use_torchani_volumes
    logical, intent(in) :: compute_coulomb
    logical, intent(in) :: polarizable_atom_mask(:)
    logical, intent(in) :: use_charges_derivatives
    logical, intent(in) :: write_charges
    logical, intent(in) :: write_charges_grad
    logical, intent(in) :: write_volumes
    logical, intent(in) :: write_volumes_grad
    integer, intent(in) :: mlmm_coupling
    integer, intent(in) :: ml_system_charge
    double precision, intent(in) :: qmcoords(:, :)
    double precision, intent(in) :: alpha(:)
    double precision, intent(in) :: clcoords(:, :)
    double precision, intent(in) :: inv_pol_dielectric
    double precision, intent(inout) :: qmcharges(:)
    double precision, intent(inout) :: dqmcharges(:, :, :)
    double precision, intent(inout) :: qmvolumes(:)
    double precision, intent(inout) :: dqmvolumes(:, :, :)
    double precision, intent(inout) :: dxyzqm(:, :)
    double precision, intent(inout) :: dxyzcl(:, :)
    double precision, intent(inout) :: escf
    double precision :: ene_pot_invacuo
    double precision :: ene_pot_embed_pol
    double precision :: ene_pot_embed_dist
    double precision :: ene_pot_embed_coulomb
    double precision :: ene_pot_embed_total
    ene_pot_invacuo= 0.0D0
    ene_pot_embed_pol = 0.0D0
    ene_pot_embed_dist = 0.0D0
    ene_pot_embed_coulomb = 0.0D0
    ene_pot_embed_total = 0.0D0

    if (mlmm_coupling /= 0 .and. mlmm_coupling /= 1) then
        write(6, fmt=*) "Torch coupling can only handle mlmm coupling = 0 and 1"
        call mexit(6,1)  ! TODO: Explicitly import routine
    endif

    call torchani_calc_energy_force_with_coupling( &
        size(qmcoords, 2), &
        size(clcoords, 2), &
        inv_pol_dielectric, &
        qmcoords, &
        alpha, &
        clcoords(:3, :), &
        clcoords(4, :), &
        compute_coulomb, &
        polarizable_atom_mask, &
        use_torchani_charges, &
        mlmm_coupling == 1, &
        use_charges_derivatives, &
        use_torchani_volumes, &
        ml_system_charge, &
        write_charges, &
        write_charges_grad, &
        write_volumes, &
        write_volumes_grad, &
        ! Outputs
        dxyzqm, &
        dxyzcl, &
        qmcharges, &
        dqmcharges, &
        qmvolumes, &
        dqmvolumes, &
        ene_pot_invacuo, &
        ene_pot_embed_pol, &
        ene_pot_embed_dist, &
        ene_pot_embed_coulomb, &
        escf &
    )

    ene_pot_embed_total = ene_pot_embed_dist + ene_pot_embed_pol + ene_pot_embed_coulomb
    if (write_to_mdout_this_step) then
        write(6,'(1x,"TORCHANI: IN VACUO ENERGY = ",f14.4)') ene_pot_invacuo
        if (compute_coulomb) then
            write(6,'(1x,"TORCHANI: QM/MM ENERGY (COULOMBIC) =",f14.4)') ene_pot_embed_coulomb
        endif
        write(6,'(1x,"TORCHANI: QM/MM ENERGY (DISTORTION) =",f14.4)') ene_pot_embed_dist
        write(6,'(1x,"TORCHANI: QM/MM ENERGY (POLARIZATION) =",f14.4)') ene_pot_embed_pol
        write(6,'(1x,"TORCHANI: QM/MM ENERGY (TOTAL) =",f14.4)') ene_pot_embed_total
    endif
    ! Invert dxyzqm and cl because torchani returns forces, and sander expects grad
    dxyzqm = -dxyzqm
    dxyzcl = -dxyzcl
endsubroutine

! Calculate the QMMM forces in the QM and the MM regions
! Dispatches to either a numerical or analytical branch depending on the value of
! ani_nml%use_numerical_forces
subroutine get_coupling_energies_and_forces_fortran( &
    nstep, &
    ntpr_internal, &
    nqmatoms, &
    nclatoms, &
    dxyzqm, &
    qmcoords, &
    clcoords, &
    dxyzcl, &
    dxyzqm_qmmm, &
    use_torchani_charges, &
    qmcharges, &
    dqmcharges, &
    mlmm_coupling, &
    inv_pol_dielectric, &
    alpha, &
    qmmm_use_numerical_forces, &
    escf, &
    use_charges_derivatives &
)
    use constants, only: CODATA08_AU_TO_KCAL, CODATA08_A_TO_BOHRS
    integer, intent(in) :: nstep                          ! Number of step
    integer, intent(in) :: ntpr_internal                  ! Printing frequency
    integer, intent(in) :: nqmatoms                       ! Number of QM atoms
    integer, intent(in) :: nclatoms                       ! Number of MM atoms
    double precision, intent(inout) :: dxyzqm(3,nqmatoms) ! QM atoms forces
    double precision,  intent(in) :: qmcoords(3, nqmatoms) ! QM atom coordinates
    double precision,  intent(in) :: clcoords(4, nclatoms) ! MM atom coordinates and charges in au
    double precision, intent(inout) :: dxyzcl(3,nclatoms) ! MM atoms qmmm forces
    double precision, intent(inout) :: dxyzqm_qmmm(3,nqmatoms) ! QM atom eqmmm forces
    logical, intent(in) :: use_torchani_charges           ! Use torchani charges
    double precision, intent(in):: qmcharges(nqmatoms)    ! QM atoms charges
    ! NOTE: atomic_charge_derivatives is  an array of shape [3, num_atoms, num_atoms]
    ! where the element [i, j, k] is the derivative of the **charge on
    ! k-th atom** with respect to the **i-th position of the j-th atom**
    double precision, intent(in) :: dqmcharges(3,nqmatoms,nqmatoms) ! QM atoms charges forces
    integer, intent(in) :: mlmm_coupling                  ! Kind of MLMM coupling (0 = coulombic, 1 = simple-polarizable)
    double precision, intent(in) :: inv_pol_dielectric ! Polarization dielectric
    double precision, intent(in) :: alpha(nqmatoms)       ! QM atoms atomic polarizabilities
    logical, intent(in) :: qmmm_use_numerical_forces      ! Calculate qmmm elec forces numerically
    double precision, intent(inout) :: escf               ! QM scf energy
    logical, intent(in) :: use_charges_derivatives        ! Consider or not that QM charges depend on QM coordinates

    double precision distij(nqmatoms,nclatoms)            ! QM/MM distance matrix
    double precision efield(3, nqmatoms)                  ! Electric field on QM atoms
    double precision eqmmm_pol                            ! QM/MM polariziton energy (on QM charges)
    double precision eqmmm_dist                           ! QM distortion energy (on QM system)
    double precision eqmmm_coulomb                        ! QM/MM coulombic electrostatic contribution
    double precision eqmmm_total                          ! QM/MM coulombic electrostatic contribution
    integer :: i

    ! Calculate QM/MM distance matrix
    call get_distij_matrix(qmcoords, clcoords, distij)

    ! Calculate QM/MM coulomb interaction energy
    eqmmm_coulomb = 0.d0
    do i=1,nqmatoms
        eqmmm_coulomb = eqmmm_coulomb + sum(qmcharges(i) * clcoords(4, :) / distij(i, :))
    end do

    ! Calculate QM/MM polarization and distortion energy (only for the QM-region)
    eqmmm_pol = 0.d0
    eqmmm_dist = 0d0
    if (mlmm_coupling == 1) then
        call get_efield(qmcoords, clcoords, distij, efield)
        eqmmm_pol = -inv_pol_dielectric * sum(alpha * sum(efield**2, dim=1))
        eqmmm_dist = -0.5d0 * eqmmm_pol
    end if

    ! Convert QM/MM coupling energy components to kcal
    ! TODO: Are these conversion factors correct?
    eqmmm_coulomb = eqmmm_coulomb * CODATA08_AU_TO_KCAL / CODATA08_A_TO_BOHRS
    eqmmm_pol = eqmmm_pol * CODATA08_AU_TO_KCAL / CODATA08_A_TO_BOHRS
    eqmmm_dist = eqmmm_dist * CODATA08_AU_TO_KCAL / CODATA08_A_TO_BOHRS 
    eqmmm_total = eqmmm_coulomb + eqmmm_dist + eqmmm_pol
    escf = escf + eqmmm_total

    if (ntpr_internal > 0 .and. mod(nstep+1, ntpr_internal) == 0) then
        write(6,'(1x,"TORCHANI: QM/MM ENERGY (COULOMBIC) =",f14.4)') eqmmm_coulomb
        if (mlmm_coupling == 1) then
            write(6,'(1x,"TORCHANI: QM/MM ENERGY (DISTORTION) =",f14.4)') eqmmm_dist
            write(6,'(1x,"TORCHANI: QM/MM ENERGY (POLARIZATION) =",f14.4)') eqmmm_pol
        endif
        write(6,'(1x,"TORCHANI: QM/MM ENERGY (TOTAL) =",f14.4)') eqmmm_total
    end if

    ! Calculates QM/MM interaction forces
    ! Either coulombic (mlmm_coupling = 0) or simple-polarizable (mlmm_coupling = 1)
    if (qmmm_use_numerical_forces) then
        call get_qmmm_forces_numerical(&
            nqmatoms,nclatoms,dxyzqm,qmcoords,clcoords,dxyzcl,dxyzqm_qmmm,&
            use_torchani_charges,qmcharges,mlmm_coupling,&
            inv_pol_dielectric,distij,efield,alpha,use_charges_derivatives&
        )
    else
        call get_qmmm_forces_analytical( &
            nqmatoms,nclatoms,dxyzqm,qmcoords,clcoords,dxyzcl,dxyzqm_qmmm, &
            qmcharges,dqmcharges, mlmm_coupling, &
            inv_pol_dielectric,distij,efield,alpha,use_charges_derivatives, &
            use_torchani_charges &
        )
    endif
endsubroutine

! Calculate the ||r_i - r_j|| dist matrix, where 'i in QM-region' and 'j in MM-region'
subroutine get_distij_matrix(qm_coords, mm_coords, distij)
    double precision, intent(in) :: qm_coords(:, :)
    double precision, intent(in) :: mm_coords(:, :)  ! Charges in last dim (4, mm-atoms)
    ! Dists between coords of QM-atoms and MM-atoms, shape (qm-atoms, mm-atoms)
    double precision, intent(out) :: distij(:, :)
    integer :: i, j
    do i = 1, size(qm_coords, dim=2)
        do j = 1, size(mm_coords, dim=2)
            distij(i, j) = dsqrt(sum((qm_coords(:, i) - mm_coords(:3, j))**2, dim=1))
        end do
    end do
endsubroutine

! Calculate electric field on the QM-atoms, generated by the MM-atoms
subroutine get_efield(qm_coords, mm_coords, distij, efield)
    double precision, intent(in) :: qm_coords(:, :)  ! Shape (3, qm-atoms)
    double precision, intent(in) :: mm_coords(:, :)  ! Charges in last dim (4, mm-atoms)
    ! Dists between coords of QM-atoms and MM-atoms, shape (qm-atoms, mm-atoms)
    double precision, intent(in) :: distij(:, :)
    ! Electric field on the QM-atoms, generated by the MM-atoms, shape (3, qm-atoms)
    double precision, intent(out):: efield(:, :)
    integer :: i, j
    efield = 0.0d0
    do i = 1, size(qm_coords, dim=2)
        do j = 1, size(mm_coords, dim=2)
            efield(:, i) = efield(:, i)&
                + mm_coords(4, j) * (mm_coords(:3, j) - qm_coords(:, i)) / (distij(i, j)**3)
        end do
    end do
endsubroutine

! Fill the "qm_alphas" array, which holds the QM-atoms polarizabilities,
! according to their element, from element-specific polarizabilities
subroutine qm_alphas_fill(elem_alphas, qm_atomic_nums, qm_alphas)
    double precision, intent(in):: elem_alphas(:)  ! shape (supported-elem-alphas)
    integer, intent(in) :: qm_atomic_nums(:)  ! shape (qm-atoms)
    double precision, intent(out) :: qm_alphas(:)  ! shape (qm-atoms)
    integer :: i
    do i = 1, size(qm_alphas)
        qm_alphas(i) = elem_alphas(qm_atomic_nums(i))
    end do
    qm_alphas = qm_alphas * au_to_ang3
endsubroutine

! Print to stdout all the atomic polarizabilities currently being used
subroutine qm_alphas_print_in_use(elem_alphas, qm_atomic_nums, qm_alphas)
    double precision, intent(in):: elem_alphas(:)  ! shape (supported-elem-alphas)
    integer, intent(in) :: qm_atomic_nums(:)  ! shape (qm-atoms)
    double precision, intent(in) :: qm_alphas(:)  ! shape (qm-atoms)
    integer :: i, j
    write(6, *) "TORCHANI: The following atomic polarizabilities (alpha) are in use:"
    write(6, *) "Element Symbol, Alpha(AU)"
    do i = 1, size(elem_alphas)
        do j = 1, size(qm_alphas)
            if(qm_atomic_nums(j) == i) then
                write(6,'(A, F10.4)')&
                    ' ' // elem_symbols(i) // ', ', elem_alphas(i) / au_to_ang3
                exit
            endif
        end do
    end do
endsubroutine

! Analytical version of the QM/MM forces. Dispatched by get_qmmm_energy_and_forces
subroutine get_qmmm_forces_analytical(nqmatoms,nclatoms,dxyzqm,qmcoords,clcoords,dxyzcl,dxyzqm_qmmm,&
                           qmcharges,dqmcharges, mlmm_coupling,inv_pol_dielectric,&
                           distij,efield,alpha,use_charges_derivatives, use_torchani_charges)
  use constants, only: CODATA08_AU_TO_KCAL, CODATA08_A_TO_BOHRS
  integer, intent(in) :: nqmatoms                       ! Number of QM atoms
  double precision,  intent(in) :: qmcoords(3,nqmatoms) ! QM atom coordinates
  integer, intent(in) :: nclatoms                       ! Number of MM atoms
  double precision,  intent(in) :: clcoords(4,nclatoms) ! MM atom coordinates and charges in au
  double precision, intent(in):: qmcharges(nqmatoms)    ! QM atoms charges
  double precision, intent(in) :: dqmcharges(3,nqmatoms,nqmatoms) ! QM atoms charges forces
  ! Element [i, j, k] is the derivative of the **charge on
  ! k-th atom** with respect to the **i-th position of the j-th atom**
  double precision, intent(inout) :: dxyzqm(3,nqmatoms) ! QM atom forces
  double precision, intent(inout) :: dxyzcl(3,nclatoms) ! MM atom qmmm forces
  double precision, intent(inout) :: dxyzqm_qmmm(3,nqmatoms) ! QM atom eqmmm forces
  integer, intent(in) :: mlmm_coupling
  logical, intent(in) :: use_torchani_charges  ! Whether the charges depend on positions
  logical, intent(in) :: use_charges_derivatives ! Whether to consider QM charges are funcion of QM coordinates
  double precision, intent(in) :: inv_pol_dielectric

  double precision :: distij(nqmatoms,nclatoms) ! QM/MM distance matrix
  double precision :: alpha(nqmatoms) ! QM atoms atomic polarizabilities
  double precision :: efield(3, nqmatoms) ! Electric field on QM atoms
  ! Partial derivatives of the electric field
  ! TODO: Change these to 3 x 3 x atoms arrays
  double precision :: dexdxi(nqmatoms), dexdyi(nqmatoms), dexdzi(nqmatoms)
  double precision :: deydxi(nqmatoms), deydyi(nqmatoms), deydzi(nqmatoms)
  double precision :: dezdxi(nqmatoms), dezdyi(nqmatoms), dezdzi(nqmatoms)
  double precision :: dexdxj(nclatoms), dexdyj(nclatoms), dexdzj(nclatoms)
  double precision :: deydxj(nclatoms), deydyj(nclatoms), deydzj(nclatoms)
  double precision :: dezdxj(nclatoms), dezdyj(nclatoms), dezdzj(nclatoms)
  double precision :: pair_grad(3)
  double precision :: grad_qm_pol(3, nqmatoms)
  double precision :: grad_mm_pol(3, nclatoms)
  integer :: i, j, k

  ! Initialize
  dexdxi = 0.d0
  dexdyi = 0.d0
  dexdzi = 0.d0
  deydxi = 0.d0
  deydyi = 0.d0
  deydzi = 0.d0
  dezdxi = 0.d0
  dezdyi = 0.d0
  dezdzi = 0.d0
  dexdxj = 0.d0
  dexdyj = 0.d0
  dexdzj = 0.d0
  deydxj = 0.d0
  deydyj = 0.d0
  deydzj = 0.d0
  dezdxj = 0.d0
  dezdyj = 0.d0
  dezdzj = 0.d0
  grad_qm_pol = 0.0d0
  grad_mm_pol = 0.0d0
  pair_grad = 0.0d0
  dxyzqm_qmmm=dxyzqm  ! Saves initial forces in QM region
  dxyzqm=dxyzqm*CODATA08_A_TO_BOHRS/CODATA08_AU_TO_KCAL

  ! Calculate coulombic force contribution on MM atoms
  dxyzcl=0.d0
  do j=1,nclatoms
     do i=1,nqmatoms
        pair_grad = (clcoords(4,j) * qmcharges(i) * (qmcoords(:, i) - clcoords(:3, j)) / distij(i,j)**3)
        ! Add coulombic grad contribution using Newton's third law
        dxyzcl(:, j) = dxyzcl(:, j) + pair_grad
        dxyzqm(:, i) = dxyzqm(:, i) - pair_grad
     end do
  end do

  ! This loop is *extremely expensive*
  ! Calculates additive term to forces on QM atoms due to QM charges
  ! being dependent on qmcoords
  if (use_charges_derivatives .and. use_torchani_charges) then
    ! Element [i, j, k] is the derivative of the **charge on
    ! k-th atom** with respect to the **i-th position of the j-th atom**
    do j=1,nclatoms
      do i=1,nqmatoms
        do k=1,nqmatoms
            dxyzqm(:, k) = dxyzqm(:, k) + clcoords(4,j) * dqmcharges(:,k,i) / distij(i,j)
          end do
        end do
      end do
  end if

  ! TODO: This loop can probably be written more clearly using fortran intrinsics.
  ! Also, there seem to be quite a number of repeated calculations
  ! Calculates QM/MM polarization energy
  ! Only polarization on the QM atoms is accounted
  if (mlmm_coupling == 1) then
    !here ex,ey,ez are the electricfield components
    !dexdx represents the partial derivative d(ex)/dx
    !xi,yi,zi refer to the coordinates of the ith QM atom
    !xj,yj,zj refer to the coordinates of the jth MM atom

    ! Forces due to polarization, on QM atoms
    do i=1,nqmatoms
      do j=1,nclatoms
        dexdxi(i)=dexdxi(i)+(clcoords(4,j)/distij(i,j)**5)*((3*(clcoords(1,j)-qmcoords(1,i))**2)-distij(i,j)**2)
        deydyi(i)=deydyi(i)+(clcoords(4,j)/distij(i,j)**5)*((3*(clcoords(2,j)-qmcoords(2,i))**2)-distij(i,j)**2)
        dezdzi(i)=dezdzi(i)+(clcoords(4,j)/distij(i,j)**5)*((3*(clcoords(3,j)-qmcoords(3,i))**2)-distij(i,j)**2)
        deydxi(i)=deydxi(i)+(clcoords(4,j)/distij(i,j)**5)*3*(clcoords(2,j)-qmcoords(2,i))*(clcoords(1,j)-qmcoords(1,i))
        dezdxi(i)=dezdxi(i)+(clcoords(4,j)/distij(i,j)**5)*3*(clcoords(3,j)-qmcoords(3,i))*(clcoords(1,j)-qmcoords(1,i))
        dezdyi(i)=dezdyi(i)+(clcoords(4,j)/distij(i,j)**5)*3*(clcoords(3,j)-qmcoords(3,i))*(clcoords(2,j)-qmcoords(2,i))
      end do
      dexdyi(i)=deydxi(i)
      dexdzi(i)=dezdxi(i)
      deydzi(i)=dezdyi(i)
      grad_qm_pol(1, i)=-alpha(i)*(efield(1,i)*dexdxi(i)+efield(2,i)*deydxi(i)+efield(3,i)*dezdxi(i))
      grad_qm_pol(2, i)=-alpha(i)*(efield(1,i)*dexdyi(i)+efield(2,i)*deydyi(i)+efield(3,i)*dezdyi(i))
      grad_qm_pol(3, i)=-alpha(i)*(efield(1,i)*dexdzi(i)+efield(2,i)*deydzi(i)+efield(3,i)*dezdzi(i))
    end do

    ! Forces due to polarization, on MM atoms
    do j=1,nclatoms
      do i=1,nqmatoms
        dexdxj(j)=(clcoords(4,j)/distij(i,j)**5)*(distij(i,j)**2-(3*(clcoords(1,j)-qmcoords(1,i))**2))
        deydyj(j)=(clcoords(4,j)/distij(i,j)**5)*(distij(i,j)**2-(3*(clcoords(2,j)-qmcoords(2,i))**2))
        dezdzj(j)=(clcoords(4,j)/distij(i,j)**5)*(distij(i,j)**2-(3*(clcoords(3,j)-qmcoords(3,i))**2))
        deydxj(j)=-(clcoords(4,j)/distij(i,j)**5)*3*(clcoords(2,j)-qmcoords(2,i))*(clcoords(1,j)-qmcoords(1,i))
        dezdxj(j)=-(clcoords(4,j)/distij(i,j)**5)*3*(clcoords(3,j)-qmcoords(3,i))*(clcoords(1,j)-qmcoords(1,i))
        dezdyj(j)=-(clcoords(4,j)/distij(i,j)**5)*3*(clcoords(3,j)-qmcoords(3,i))*(clcoords(2,j)-qmcoords(2,i))
        dexdyj(j)=deydxj(j)
        dexdzj(j)=dezdxj(j)
        deydzj(j)=dezdyj(j)
        grad_mm_pol(1, j) = grad_mm_pol(1, j)-alpha(i)*(efield(1,i)*dexdxj(j)+efield(2,i)*deydxj(j)+efield(3,i)*dezdxj(j))
        grad_mm_pol(2, j) = grad_mm_pol(2, j)-alpha(i)*(efield(1,i)*dexdyj(j)+efield(2,i)*deydyj(j)+efield(3,i)*dezdyj(j))
        grad_mm_pol(3, j) = grad_mm_pol(3, j)-alpha(i)*(efield(1,i)*dexdzj(j)+efield(2,i)*deydzj(j)+efield(3,i)*dezdzj(j))
      end do
    end do
    ! NOTE: Here I assume the calculation of the derivatives done before somehow includes a factor
    ! of 1/2 that got cancelled out, since it was coded when we thought that energy term had a factor of 1/2.
    ! Multiplying by 2/eps is equivalent to multiplying by 1 unless eps /= 2
    grad_qm_pol = inv_pol_dielectric * 2.0d0 * grad_qm_pol
    grad_mm_pol = inv_pol_dielectric * 2.0d0 * grad_mm_pol
    ! Include the polarization forces into the net forces
    ! Due to the distortion correction, only 1/2 is added
    dxyzqm = dxyzqm + 0.5d0 * grad_qm_pol
    dxyzcl = dxyzcl + 0.5d0 * grad_mm_pol
  end if
  dxyzcl = dxyzcl * CODATA08_AU_TO_KCAL/CODATA08_A_TO_BOHRS
  dxyzqm = dxyzqm * CODATA08_AU_TO_KCAL/CODATA08_A_TO_BOHRS
  dxyzqm_qmmm = dxyzqm-dxyzqm_qmmm  !saves the qmmm forces in the qm region
endsubroutine

! NOTE: All functions from this point onwards correspond to either experimental
! or debug-only features

! Numerical branch of the QM/MM forces. Dispatched by get_qmmm_energy_and_forces
subroutine get_qmmm_forces_numerical(nqmatoms,nclatoms,dxyzqm,qmcoords,clcoords,dxyzcl,dxyzqm_qmmm,&
                               use_torchani_charges,qmcharges,mlmm_coupling,&
                               inv_pol_dielectric, distij, efield,alpha,use_charges_derivatives)
  use constants, only: CODATA08_AU_TO_KCAL, CODATA08_A_TO_BOHRS
  integer, intent(in) :: nqmatoms                       ! Number of QM atoms
  double precision,  intent(in) :: qmcoords(3,nqmatoms) ! QM atom coordinates
  integer, intent(in) :: nclatoms                       ! Number of MM atoms
  double precision,  intent(in) :: clcoords(4,nclatoms) ! MM atom coordinates and charges in au
  double precision, intent(in):: qmcharges(nqmatoms)    ! QM atoms charges
  double precision, intent(inout) :: dxyzqm(3,nqmatoms) ! QM atom forces
  double precision, intent(inout) :: dxyzcl(3,nclatoms) ! MM atom qmmm forces
  double precision, intent(out) :: dxyzqm_qmmm(3,nqmatoms) ! QM atom eqmmm forces
  integer, intent(in) :: mlmm_coupling
  logical, intent(in) :: use_torchani_charges ! Use torchani charges or not
  logical, intent(in) :: use_charges_derivatives ! Use derivatives of torchani charges
  double precision, intent(in) :: inv_pol_dielectric
  double precision epol, edist ! Polarization and distortion energies
  double precision pert_escf, eqmmmtorchani_plusdelta, eqmmmtorchani_minusdelta
  double precision eqmmmtorchani
  double precision distij(nqmatoms,nclatoms) ! QM/MM distance matrix
  double precision, intent(in) :: alpha(nqmatoms) ! QM atoms atomic polarizabilities
  double precision efield(3,nqmatoms) ! Electric field on QM atoms
  double precision :: pert_qmcoords(3,nqmatoms) ! perturbed QM atom coordinates
  double precision :: pert_clcoords(4,nclatoms) ! perturbed MM atom coordinates and unperturbed charges in au
  double precision :: pert_qmcharges(nqmatoms)    ! perturbed QM atoms charges
  double precision :: pert_dxyzqm(3,nqmatoms) ! perturbed QM atom eqmmm forces
  double precision :: force ! force on a specific coordinate
  double precision :: delta ! coordinates perturbation factor
  integer k, m, i, j

  dxyzcl=0.d0
  dxyzqm_qmmm=dxyzqm  !saves initial forces in qm region
  pert_qmcharges=qmcharges

  delta=0.005
  do k=1,nqmatoms
    do m=1,3
      ! Summs delta to a temporary (pert_qmcoords)  array, in dimension m
      pert_qmcoords(:,:)=qmcoords(:,:)
      pert_qmcoords(m,k)=pert_qmcoords(m,k)+delta

      ! If necesary, calculates perturbed qm charges (and other things that we don't actually need...)
      if (use_torchani_charges .and. use_charges_derivatives) then
         call torchani_energy_force_atomic_charges( &
            size(pert_qmcoords, 2), &
            pert_qmcoords, &
            ! Outputs:
            pert_dxyzqm, &
            pert_qmcharges, &
            pert_escf &
        )
      endif

      ! Calculates the QM/MM energy
      ! Calculates QM/MM distance matrix
      call get_distij_matrix(pert_qmcoords, clcoords, distij)

      ! Calculates QM/MM coumlomb interaction energy
      eqmmmtorchani=0.d0
      do i=1,nqmatoms
        do j=1,nclatoms
           eqmmmtorchani = eqmmmtorchani + pert_qmcharges(i)*clcoords(4,j)/distij(i,j)
        end do
      end do

      ! Calculate QM/MM polarization and distortion energies
      ! Only polarization on the QM atoms is accounted for
      epol=0.d0
      edist=0.d0
      if (mlmm_coupling == 1) then
        call get_efield(pert_qmcoords, clcoords, distij, efield)
        do i=1,nqmatoms
          epol=epol+alpha(i)*(efield(1,i)**2+efield(2,i)**2+efield(3,i)**2)
        end do
        epol=-inv_pol_dielectric*epol
        edist=-0.5d0*epol
      end if

      eqmmmtorchani_plusdelta = eqmmmtorchani + epol + edist
      eqmmmtorchani_plusdelta = eqmmmtorchani_plusdelta*CODATA08_AU_TO_KCAL/CODATA08_A_TO_BOHRS !kcal/mol

      ! Repeats but now substracting delta
      pert_qmcoords(:,:)=qmcoords(:,:)
      pert_qmcoords(m,k)=pert_qmcoords(m,k)-delta

      ! Calculates perturbed qm charges (and other things that we don't actually need...)
      if (use_torchani_charges .and. use_charges_derivatives) then
         call torchani_energy_force_atomic_charges( &
            size(pert_qmcoords, 2), &
            pert_qmcoords, &
            ! Outputs:
            pert_dxyzqm, &
            pert_qmcharges, &
            pert_escf &
        )
      endif

      ! Calculates QM/MM distance matrix
      call get_distij_matrix(pert_qmcoords, clcoords, distij)

      ! Calculates QM/MM coumlomb interaction energy
      eqmmmtorchani=0.d0
      do i=1,nqmatoms
        do j=1,nclatoms
          eqmmmtorchani = eqmmmtorchani + pert_qmcharges(i)*clcoords(4,j)/distij(i,j)
        end do
      end do

      ! Calculates QM/MM polarization energy
      ! Calculates QM distortion energy
      epol=0.d0
      edist=0.d0
      ! Only polarization on the QM atoms is accounted
      if (mlmm_coupling == 1) then
        call get_efield(pert_qmcoords, clcoords, distij, efield)
        do i=1,nqmatoms
          epol=epol+alpha(i)*(efield(1,i)**2+efield(2,i)**2+efield(3,i)**2)
        end do
        epol=-inv_pol_dielectric*epol
        edist=-0.5d0*epol
      end if

      eqmmmtorchani_minusdelta = eqmmmtorchani + epol + edist
      eqmmmtorchani_minusdelta = eqmmmtorchani_minusdelta*CODATA08_AU_TO_KCAL/CODATA08_A_TO_BOHRS !kcal/mol

      force = (eqmmmtorchani_plusdelta - eqmmmtorchani_minusdelta)/(2.d0*delta) !kcal/(mol Bohr)
      dxyzqm(m,k) = dxyzqm(m,k) + force
    end do
 end do

 ! Does the same but for the MM atoms

 do k=1,nclatoms
   do m=1,3
     ! Summs delta to a temporary (pert_qmcoords)  array, in dimension m
     pert_clcoords(:,:)=clcoords(:,:)
     pert_clcoords(m,k)=pert_clcoords(m,k)+delta

     ! Calculates the QM/MM energy
     ! Calculates QM/MM distance matrix
     call get_distij_matrix(qmcoords, pert_clcoords, distij)

     ! Calculates QM/MM coumlomb interaction energy
     eqmmmtorchani=0.d0
     do i=1,nqmatoms
       do j=1,nclatoms
          eqmmmtorchani = eqmmmtorchani + qmcharges(i)*pert_clcoords(4,j)/distij(i,j)
       end do
     end do

     ! Calculates QM/MM polarization energy
     ! Calculates QM distortion energy
     epol=0.d0
     edist=0.d0
     ! Only polarization on the QM atoms is accounted
     if (mlmm_coupling == 1) then
       call get_efield(qmcoords, pert_clcoords, distij, efield)
       do i=1,nqmatoms
         epol=epol+alpha(i)*(efield(1,i)**2+efield(2,i)**2+efield(3,i)**2)
       end do
       epol=-inv_pol_dielectric*epol
       edist=-0.5d0*epol
     end if

     eqmmmtorchani_plusdelta = eqmmmtorchani + epol + edist
     eqmmmtorchani_plusdelta = eqmmmtorchani_plusdelta*CODATA08_AU_TO_KCAL/CODATA08_A_TO_BOHRS !kcal/mol

     ! Repeats but now substracting delta
     pert_clcoords(:,:)=clcoords(:,:)
     pert_clcoords(m,k)=pert_clcoords(m,k)-delta

     ! Calculates QM/MM distance matrix
     call get_distij_matrix(qmcoords, pert_clcoords, distij)

     ! Calculates QM/MM coumlomb interaction energy
     eqmmmtorchani=0.d0
     do i=1,nqmatoms
       do j=1,nclatoms
         eqmmmtorchani = eqmmmtorchani + qmcharges(i)*pert_clcoords(4,j)/distij(i,j)
       end do
     end do

     ! Calculates QM/MM polarization energy
     epol=0.d0
     edist=0.d0
     ! Only polarization on the QM atoms is accounted
     ! Calculates QM distortion energy
     if (mlmm_coupling == 1) then
       call get_efield(qmcoords, pert_clcoords, distij, efield)
       do i=1,nqmatoms
         epol=epol+alpha(i)*(efield(1,i)**2+efield(2,i)**2+efield(3,i)**2)
       end do
       epol=-inv_pol_dielectric*epol
       edist=-0.5d0*epol
     end if

     eqmmmtorchani_minusdelta = eqmmmtorchani + epol + edist
     eqmmmtorchani_minusdelta = eqmmmtorchani_minusdelta*CODATA08_AU_TO_KCAL/CODATA08_A_TO_BOHRS !kcal/mol

     force = (eqmmmtorchani_plusdelta - eqmmmtorchani_minusdelta)/(2.d0*delta)
     dxyzcl(m,k) = dxyzcl(m,k) + force
   end do
 end do
 dxyzqm_qmmm=dxyzqm-dxyzqm_qmmm  !saves the qmmm forces in the qm region
endsubroutine

! Use a helper QM program to calculate the QM/MM interaction
subroutine get_coupling_energies_and_forces_through_qm_helper( &
    nstep, &
    ntpr_internal, &
    nqmatoms, &
    nclatoms, &
    escf, &
    dxyzqm, &
    dxyzcl, &
    qmcoords, &
    clcoords, &
    qmtypes, &
    charge, &
    spinmult, &
    id, &
    extcoupling_program, &
    write_to_mdout_this_step &
)
    use qm2_extern_gau_module, only: get_gau_forces
    use qm2_extern_orc_module, only: get_orc_forces
#ifdef LIO
    use qm2_extern_lio_module, only: get_lio_forces
#endif
    integer, intent(in) :: nstep                         ! MDIN input
    integer, intent(in) :: ntpr_internal                         ! MDIN input
    integer, intent(in) :: nqmatoms                         ! Number of QM atoms
    integer, intent(in) :: nclatoms                         ! Number of MM atoms
    double precision, intent(in) :: qmcoords(3,nqmatoms)     ! QM atom coordinates
    double precision, intent(in) :: clcoords(3,nclatoms)     ! mm atom coordinates
    integer, intent(in) :: qmtypes(nqmatoms)                ! QM atoms atomic numbers
    character(len=3), intent(in) :: id                    ! ID number for PIMD or REMD
    double precision, intent(inout) :: dxyzqm(3,nqmatoms) ! QM atom eqmmm forces
    double precision, intent(inout) :: dxyzcl(3,nclatoms) ! MM atom qmmm forces
    double precision, intent(inout) :: escf               ! ESCF
    integer, intent(in) :: spinmult                      ! SPIN multiplicity
    integer, intent(in) :: charge                        ! QM charge
    character(len=256), intent(in) :: extcoupling_program ! External program

    double precision :: ext_dxyzqm(3,nqmatoms)              ! External QM atom eqmmm forces
    double precision :: ext_dxyzcl(3,nclatoms)              ! External MM atom qmmm forces
    double precision :: ext_escf                            ! External SCF
    double precision :: extvac_dxyzqm(3,nqmatoms)           ! External QM atom eqmmm forces in vacuo
    double precision :: extvac_dxyzcl(3,nclatoms)           ! External MM atom qmmm forces in vacuo
    double precision :: extvac_escf                         ! External SCF in vacup
    logical :: write_to_mdout_this_step

    if (trim(extcoupling_program) == 'orca') then
        call get_orc_forces(.true., nstep, ntpr_internal, id, nqmatoms, qmcoords,&
            qmtypes, nclatoms, clcoords, ext_escf, ext_dxyzqm, ext_dxyzcl,&
            charge, spinmult)
        call get_orc_forces(.true., nstep, ntpr_internal, id, nqmatoms, qmcoords,&
            qmtypes, 0, clcoords, extvac_escf, extvac_dxyzqm, extvac_dxyzcl,&
            charge, spinmult)
#ifdef LIO
    elseif (trim(extcoupling_program) == 'lio') then
        call get_lio_forces(nqmatoms, qmcoords,nclatoms, clcoords, ext_escf,&
            ext_dxyzqm,ext_dxyzcl,1)
        call get_lio_forces(nqmatoms, qmcoords,0, clcoords, extvac_escf,&
            extvac_dxyzqm,extvac_dxyzcl)
#endif
    elseif (trim(extcoupling_program) == 'gaussian') then
        call get_gau_forces(.true., nstep, ntpr_internal, id,&
                  nqmatoms, qmcoords, qmtypes, nclatoms, clcoords,&
                  ext_escf, ext_dxyzqm, ext_dxyzcl, charge, spinmult)
        call get_gau_forces(.true., nstep, ntpr_internal, id,&
                  nqmatoms, qmcoords, qmtypes, 0, clcoords,&
                  extvac_escf, extvac_dxyzqm, extvac_dxyzcl, charge, spinmult)
    elseif (trim(extcoupling_program) == 'amber-dftb') then
        ! TODO: calling amber dftb subroutines twice stacks information on the ener structure from the state module. This does not
        ! affect any aspect of the simulation, but the energies printed in the mdout should not be considered for any kind of
        ! analysis (EXTERNSCF and EPtot). To do so, one should consider the energies printed in the TORCHANI lines. It would be
        ! great to modify the printing of this variables to the mdout but that would require to modify subroutines that are external
        ! to the QM/MM ones.
        call get_amber_dftb_forces(nqmatoms,nclatoms,ext_escf,ext_dxyzqm,ext_dxyzcl,.false.)
        call get_amber_dftb_forces(nqmatoms,nclatoms,extvac_escf,extvac_dxyzqm,extvac_dxyzcl,.true.)
    else
        call internal_error()
    end if
    if (write_to_mdout_this_step) then
        write(6,'(1x,"TORCHANI: EXTERNAL SCF ENERGY = ",f14.4)') ext_escf
        write(6,'(1x,"TORCHANI: EXTERNAL SCF IN VACUO ENERGY            = ",f14.4)') extvac_escf
        ! TODO: What is this "Correction"?
        write(6,'(1x,"TORCHANI: EXTERNAL QM/MM ELEC + CORRECTION ENERGY = ",f14.4)') ext_escf-extvac_escf
    end if
    escf = escf + (ext_escf - extvac_escf)
    dxyzcl = ext_dxyzcl  ! TODO: Is this a bug? Why not dxyzcl + (ext_... - extvac_...)
    dxyzqm = dxyzqm + (ext_dxyzqm - extvac_dxyzqm)
endsubroutine

! If the QBC is large, use a helper QM program (different than TORCHANI) to calculate the QM energy
subroutine get_vacuum_energies_and_forces_switching( &
    nstep, &
    ntpr_internal, &
    nqmatoms, &
    nclatoms, &
    escf, &
    ext_escf, &
    dxyzqm, &
    dxyzcl, &
    qmcoords, &
    clcoords, &
    qmtypes, &
    charge, &
    spinmult, &
    id, &
    switching_program, &
    qlow, &
    qhigh, &
    qbc, &
    use_torchani_charges, &
    qmcharges, &
    dqmcharges &
)
    use qm2_extern_gau_module, only: get_gau_forces
    use qm2_extern_orc_module, only: get_orc_forces
#ifdef LIO
    use qm2_extern_lio_module, only: get_lio_forces
#endif
    integer, intent(in) :: nstep                            ! MDIN input
    integer, intent(in) :: ntpr_internal                    ! MDIN input
    integer, intent(in) :: nqmatoms                         ! Number of QM atoms
    integer, intent(in) :: nclatoms                         ! Number of MM atoms
    double precision, intent(out) :: qbc                    ! QBC estimator
    double precision, intent(in) :: qmcoords(3,nqmatoms)    ! QM atom coordinates
    double precision, intent(in) :: clcoords(3,nclatoms)    ! mm atom coordinates
    integer, intent(in) :: qmtypes(nqmatoms)                ! QM atoms atomic numbers
    character(len=3), intent(in) :: id                      ! id
    double precision, intent(inout) :: dxyzqm(3,nqmatoms)   ! QM atoms forces
    double precision, intent(inout) :: dxyzcl(2,nclatoms)   ! MM atoms forces
    double precision, intent(inout) :: escf                 ! ESCF
    double precision, intent(inout) :: ext_escf             ! External QM QM/MM
    ! Mixed QM-ANI forces config, 'Switching', experimental
    double precision, intent(in) :: qlow,qhigh              ! switching function parameters
    ! double precision ext_eqmmm                            ! External QMMM
    integer, intent(in) :: spinmult                         ! SPIN multiplicity
    integer, intent(in) :: charge                           ! QM charge
    character(len=256), intent(in) :: switching_program      ! External program
    double precision :: ext_dxyzqm(3,nqmatoms)              ! External QM atom eqmmm forces in vacuo
    double precision :: ext_dxyzcl(3,nclatoms)              ! External MM atom qmmm forces in vacuo
    double precision, intent(inout)              :: qmcharges(nqmatoms)  
    double precision, intent(inout)              :: dqmcharges(3,nqmatoms,nqmatoms) ! QM atom charges forces
    double precision              :: dqbc(3,nqmatoms)        ! QBC derivatives 
    logical :: use_torchani_charges

    if (use_torchani_charges) then
        call torchani_calc_data_for_monitored_mlmm( &
            size(qmcoords, 2), &
            qmcoords, &
            charge, &
            ! Outputs:
            dxyzqm, &
            qmcharges, &
            dqmcharges, &
            qbc, &
            dqbc, &
            escf &
        )
    else
        call torchani_calc_energy_force_qbc( &
            size(qmcoords, 2), &
            qmcoords, &
            charge, &
            ! Outputs:
            dxyzqm, &
            qbc, &
            dqbc, &
            escf &
        )
    endif

    ! Invert dxyzqm because torchani returns forces, and dxyzqm expects grad
    dxyzqm = -dxyzqm
    write(6,*) "TORCHANI: QBC                               = ", qbc
    write(6,*) "TORCHANI: qlow                              = ", qlow
    write(6,*) "TORCHANI: qhigh                             = ", qhigh
    write(6,*) "TORCHANI: PREDICTED ENERGY                  = ", escf

    if (qbc .ge. qhigh) then
        write(6,*) "TORCHANI: QBC is larger than qhigh "
    elseif ((qbc .ge. qlow) .and. (qbc .lt. qhigh)) then
        write(6,*) "TORCHANI: QBC is between qlow and qhigh"
    end if

    if (qbc >= qlow) then
        write(6,*) "TORCHANI: Running QM calculation..."
        if (trim(switching_program) == 'orca') then
            call get_orc_forces(.true., nstep, ntpr_internal, id, nqmatoms, qmcoords,&
                qmtypes, nclatoms, clcoords, ext_escf, ext_dxyzqm, ext_dxyzcl,&
                charge, spinmult)
#ifdef LIO
        elseif (trim(switching_program) == 'lio') then
            call get_lio_forces(nqmatoms, qmcoords, nclatoms, clcoords, ext_escf,&
                ext_dxyzqm,ext_dxyzcl)
#endif
        elseif (trim(switching_program) == 'gau') then
            call get_gau_forces(.true., nstep, ntpr_internal, id,&
                nqmatoms, qmcoords, qmtypes, nclatoms, clcoords,&
                ext_escf, ext_dxyzqm, ext_dxyzcl, charge, spinmult)
        else
            call internal_error()
        end if
    endif
    call switching_function( &
        nqmatoms, &
        nclatoms, &
        qbc, &
        dqbc, &
        qlow, &
        qhigh, &
        dxyzqm, &
        dxyzcl, &
        escf, &
        ext_dxyzqm, &
        ext_dxyzcl, &
        ext_escf &
)
endsubroutine

! Modifies escf by mixing in a portion of the externally calculated QM energy
subroutine switching_function( &
    nqmatoms, &
    nclatoms, &
    qbc, &
    dqbc, &
    qlow, &
    qhigh, &
    dxyzqm, &
    dxyzcl, &
    escf, &
    ext_dxyzqm, &
    ext_dxyzcl, &
    ext_escf &
)
    integer, intent(in)             :: nqmatoms                  ! Number of QM atoms
    integer, intent(in)             :: nclatoms                  ! Number of MM atoms
    double precision, intent(in)    :: qbc                       ! QBC estimator
    double precision, intent(in)    :: dqbc(3,nqmatoms)          ! QBC derivatives 
    double precision, intent(in)    :: qlow,qhigh                ! switching function parameters
    double precision, intent(inout) :: dxyzqm(3,nqmatoms)        ! Forces QM + QMMM on QM atoms (given by ANI)
    double precision, intent(inout) :: dxyzcl(3,nclatoms)        ! Forces on MM atoms
    double precision, intent(inout) :: escf                      ! Energy QM (given by ANI, vac)
    double precision, intent(in)    :: ext_escf                  ! QM + QMMM solv (given by external program)
    double precision  ext_dxyzqm(3,nqmatoms)    ! Forces QM + QMMM on QM atoms (given by extrenal program)
    double precision  ext_dxyzcl(3,nclatoms)
    double precision                :: x, alpha                  ! Functions of QBC             

    if (qhigh <= qbc) then
        write(6,*) "TORCHANI: 'qhigh <= QBC'"
        write(6,*) "TORCHANI: PES is full QM"
        write(6,*) "TORCHANI: EXTERNAL QM + QMMM SCF ENERGY (kcal/mol)  = ", ext_escf
        write(6,*) "TORCHANI: ABS DELTA SCF ENERGY (kcal/mol)          = ", abs(ext_escf-escf)
        escf = ext_escf
        dxyzqm = ext_dxyzqm
        dxyzcl = ext_dxyzcl
    elseif ((qlow <= qbc) .and. (qbc < qhigh)) then
        write(6,*) "TORCHANI: 'qlow <= QBC < qhigh'"
        write(6,*) "TORCHANI: PES is mixed ML and QM"
        write(6,*) "TORCHANI: EXTERNAL IN VACUO SCF ENERGY (kcal/mol)  = ", ext_escf
        write(6,*) "TORCHANI: ABS DELTA SCF ENERGY (kcal/mol)          = ", abs(ext_escf-escf)
        x = (qbc - qlow) / (qhigh - qlow)
        alpha = 3.d0 * x**2 - 2.d0 * x**3
        dxyzqm = alpha * ext_dxyzqm + (1.d0 - alpha) * dxyzqm
        ! Add d(qbc)/d(r) term
        dxyzqm = dxyzqm + (dqbc * (escf - ext_escf) * 6.d0 * (x - x**2) / (qhigh - qlow))
        escf = alpha * ext_escf + (1.d0 - alpha) * escf
        write(6,'(1x,A,F4.1)') "TORCHANI: %TORCHANI Energy = ", (1.d0 - alpha) * 100.d0
        write(6,'(1x,A,F4.1)') "TORCHANI: %QM  Energy     = ", alpha * 100.d0
        write(6,*) "TORCHANI: Forces contribution are propagated from mixed PES"
    elseif ((0 <= qbc) .and. (qbc < qlow)) then
        write(6,*) "TORCHANI: '0 <= QBC < qlow'"
        write(6,*) "TORCHANI: PES is full ML"
    else
        call internal_error()
    end if
endsubroutine

subroutine get_amber_dftb_forces(nqmatoms,nclatoms,escf,ext_dxyzqm,ext_dxyzcl,vacuum)
    use qmmm_module, only : qmmm_struct, qmmm_nml, qmmm_scratch
    use memory_module, only: natom  ! Isn't natom equal to nqmatoms + nclatoms?
    integer, intent(in) :: nqmatoms                         ! Number of QM atoms
    integer, intent(in) :: nclatoms                         ! Number of MM atoms in pairlist
    double precision, intent(inout) :: escf               ! ESCF
    logical, intent(in)  :: vacuum
    double precision scf_mchg(qmmm_struct%nquant_nlink)
    double precision ext_dxyzqm(3,nqmatoms) ! External QM atom qmmm forces
    double precision ext_dxyzcl(3,nclatoms) ! External MM atom qmmm forces
    integer iqmp

    qmmm_nml%qmtheory%DFTB=.true.
    qmmm_nml%qmtheory%EXTERN=.false.

    if (vacuum) then
      qmmm_nml%qmmm_int=0
    else
      qmmm_nml%qmmm_int=1
    end if

    ! TODO: What is happening here?
    ! What is the difference between qmmm_struct%qm_coords and qmcoords?
    ! What is the difference between qmmm_struct%qm_coords and qmmm_struct%qm_xcrd?
    ! TODO: GCC warns of temporaries creation here, probably not super important
    ! since the QM region is small, but something to keep in mind

    ! Calculate RIJ and many related equations here.
    ! Necessary memory allocation is done inside the routine.
    ! Store the results in memory to save time later.
    call qm2_calc_rij_and_eqns(&
        qmmm_struct%qm_coords,&
        qmmm_struct%nquant_nlink,&
        qmmm_struct%qm_xcrd,&
        natom,&
        qmmm_struct%qm_mm_pairs&
    )  ! TODO: Explicitly import routine

    if (qmmm_nml%qmgb == 2) then !Move this above
      call sander_bomb('qmgb is not available for torchani based simulations')  ! TODO: Explicitly import routine
    end if

    ! calculates scf energy
    call qm2_dftb_energy(escf,scf_mchg)  ! TODO: Explicitly import routine

    ! calculates forces on qm region
    ext_dxyzqm=0.0d0
    call qm2_dftb_get_qm_forces(ext_dxyzqm)  ! TODO: Explicitly import routine

    ! if not ME, calculates qmmmm electrostatic interaction
    if (qmmm_nml%qmmm_int > 0 .and. (qmmm_nml%qmmm_int /= 5) ) then
        ext_dxyzcl=0.0d0
        iqmp = qmmm_struct%qm_mm_pairs
        ! TODO: GCC warns of temporaries creation here, probably not super important
        ! since the QM region is small, but something to keep in mind
        call qm2_dftb_get_qmmm_forces(&
            ext_dxyzcl,&
            ext_dxyzqm,&
            qmmm_scratch%qm_real_scratch,&
            qmmm_scratch%qm_real_scratch(natom+1:natom+iqmp),&
            qmmm_scratch%qm_real_scratch(2*natom+1:2*natom+iqmp),&
            qmmm_scratch%qm_real_scratch(3*natom+1:3*natom+iqmp)&
        )  ! TODO: Explicitly import routine
    end if
    ! Print some extra information if verbosity level is > 0
    call qm2_print_energy(qmmm_nml%verbosity, qmmm_nml%qmtheory, escf, qmmm_struct)  ! TODO: Explicitly import routine
    qmmm_nml%qmmm_int=1
    qmmm_nml%qmtheory%DFTB=.false.
    qmmm_nml%qmtheory%EXTERN=.true.
endsubroutine
#endif  /* TORCHANI */
endmodule
