#include "copyright.h"
#include "../include/dprec.fh"
#include "def_time.h"
!------------------------------------------------------------------------------
! Principal code for calculating QM potential for QMMM simulations.
!
! Major Authors of current code: Ross Walker, Mike Crowley, Andreas Goetz
!
! Please send all comments or queries to: ross@rosswalker.co.uk
!------------------------------------------------------------------------------

!------------------------------------------------------------------------------
! qm_mm: the main driver function for QM/MM molecular dynamics
!
! Arguments:
!   coords:            Cartesian coordinates for all atoms in the system, both
!                      QM and MM (natom*3 _REAL_ elements)
!   natom:             the total number of REAL atoms
!   scaled_mm_charges: Atomic partial charges, scaled to proton units by the
!                      inverse of 18.2223 (sodium cation would read +1.0)
!   f:                 Atomic forces (natom*3 _REAL_ elements)
!   escf:              The calculated SCF energy, heat of formation from QM
!   periodic:          flag to
!   born_radii:        The calculated Generalize Born radii (for QM with GB,
!                      qm_gb == 2), natom _REAL_ elements
!   one_born_radii:    The inverse of the Generalized Born radii, natom _REAL_
!                      elements
!   intdiel:           Internal dielectric constant
!   extdiel:           External dielectric constant
!   Arad:              Atomic radii
!   mmcut2:            The squared cutoff on real-space classical interactions,
!                      cut as seen in the &cntrl namelist
!   scf_mchg:          The array of SCF Mulliken charges (there are nquant
!                      _REAL_ elements, see nquant defined in qmmm_struct_type
!                      as the total number of REAL, not link, quantum atoms)
!   ntype:             The number of classical atom symbols (i.e. CT, H1)
!   atom_name:         The atom names (an array of 4-character objects that is
!                      natom elements long)
!   atom_type:         The atom symbols (an array of objects that are the size
!                      of four characters, but in fact each atom symbol is only
!                      two characters )
!   atomic_mass:       The atomic masses in Daltons
!   atom_type_index:   The atom type indices as found in the prmtop (i.e. CT=1,
!                      H1=2, etc.)
!   nstep:             The current step number
!------------------------------------------------------------------------------
subroutine qm_mm(coords, natom, scaled_mm_charges, f, escf, periodic, &
                 born_radii, one_born_radii, intdiel, extdiel, Arad, mmcut2, &
                 scf_mchg, ntype, atom_name, atom_type, atomic_mass, &
                 atom_type_index, nstep)

  use qmmm_module, only : qmmm_nml, qmmm_struct, qm2_struct, qmewald, qm_gb, &
                          qmmm_mpi, qmmm_scratch, qmmm_opnq, get_atomic_number
#ifndef API
  use qmmm_module, only : qmmm_vsolv
#endif
  use constants, only : EV_TO_KCAL, KCAL_TO_EV, zero, one, alpb_alpha

  use parms, only : cn1, cn2, nttyp
  use nblist,only: alpha,beta,gamma
  use qm2_extern_module, only: qm2_extern_get_qm_forces, qm2_extern_using_ani
#ifdef TORCHANI
  use qm2_extern_torchani_module, only: torchani_get_qmmm_int5_pol_settings
#endif
  use abfqmmm_module, only: abfqmmm_param
  use quick_module, only: get_quick_qmmm_forces
  use tcpb_module, only: get_tcpb_qmmm_forces
  use xtb_module, only: get_xtb_qmmm_forces
  use dftbplus_module, only: get_dftbplus_qmmm_forces
  use neb_vars, only: ineb
  use pimd_vars, only: ipimd
  use full_pimd_vars, only: mybeadid
#if defined(MPI)
  use remd, only : rem
#endif

  ! ASM
#ifdef MPI
  use asm_mod, only : asm
#endif
#ifdef CEW
  use cewmod, only: use_cew
#endif

  implicit none

#include "../include/assert.fh"

#if defined(MPI)
#   include "parallel.h"
  include 'mpif.h'
#endif

  ! Passed in
  integer, intent(in) :: natom,periodic,nstep, ntype
  _REAL_ , intent(inout)  :: coords(natom*3) ! Amber array - note that this
                                             ! is adjusted for link atoms
  _REAL_ , intent(in)  :: scaled_mm_charges(natom)
  _REAL_ , intent(inout) :: f(natom*3)
  _REAL_ , intent(out) :: escf
  _REAL_ , intent(in) :: born_radii(natom), one_born_radii(natom)
  _REAL_ , intent(in) :: intdiel, extdiel, Arad, mmcut2
  _REAL_ , intent(inout) :: scf_mchg(qmmm_struct%nquant_nlink)
  character(len=4), intent(in) :: atom_name(natom)
  character(len=4), intent(in) :: atom_type(natom)
  integer, intent(in) :: atom_type_index(natom)
  _REAL_ , intent(in) :: atomic_mass(natom)

  ! Locals
  _REAL_ :: alpb_beta, temp
  logical::somethingWrong
  logical :: ani_qmmm_int5_pol, qm_mm_list_include_link_pair_atoms
  _REAL_ :: ani_pol_cutoff, qm_mm_list_cutoff

  integer :: ier
  integer i, j, n, m, offset, qm_no
  character(len=3) :: id

  ! Locals for link atoms
  _REAL_ :: forcemod(3)
  integer :: lnk_no, mm_no
  integer :: nclatoms

  ! START OF QMMM SETUP: allocate list memory
  ier = 0
  call timer_start(TIME_QMMMSETUP)

  ! Increment the counter of how many times qm_mm routine has been called.
  qmmm_struct%num_qmmm_calls = qmmm_struct%num_qmmm_calls + 1

  ! If this is the first call to the routine, do some initial allocation
  ! that has not been done elsewhere.
  if (abfqmmm_param%abfqmmm == 1) then
    qmmm_struct%qm_mm_first_call = .true.
  endif
  if (qmmm_struct%qm_mm_first_call) then

    ! Stores the REAL and link atom qm coordinates.  qm_coords is
    ! later deallocated by delete in qmmm_struct_module with a call
    ! from deallocate_qmmm.
    allocate (qmmm_struct%qm_coords(3, qmmm_struct%nquant + &
                                    qmmm_struct%nlink), stat=ier)
    REQUIRE(ier == 0)

    ! Do some initial setup for qm_ewald if in use
    if (qmmm_nml%qm_ewald>0) then
      call timer_start(TIME_QMMMEWALDSETUP)

      ! Specify that we haven't done any qm_ewald stuff before this point.
      ! Essentially that this is the first MD step.
      qmewald%ewald_startup = .true.

      ! QM ewald needs access to natom in some deep QM routines where it
      ! would not normally be available.  This also allocates kvec memory
      ! (totkq reals) and the KVec table (6 lots of [natom, totkq] reals).
      ! Finally, this partitions kvectors between cpus.
      qmewald%natom = natom
      call qm_ewald_setup(qmewald%totkq, qmmm_nml%kappa, qmmm_nml%kmaxqx, &
                          qmmm_nml%kmaxqy, qmmm_nml%kmaxqz, &
                          qmmm_nml%ksqmaxq, natom, qmmm_struct%nquant, &
                          qmmm_struct%nlink)

      ! If diagnostics are on we can print some info here.
      qmewald%kappa=qmmm_nml%kappa
      if (qmmm_mpi%commqmmm_master .AND. qmmm_nml%verbosity > 2) then
        write(6,*) ''
        write(6,*) 'QMMM: Ewald - kmaxq(x,y,z) = ', qmmm_nml%kmaxqx, ',', &
                   qmmm_nml%kmaxqy, ',', qmmm_nml%kmaxqz
        write(6,*) 'QMMM: Ewald -      ksqmaxq = ', qmmm_nml%ksqmaxq
        write(6,*) 'QMMM: Ewald - Total number of k vectors = ', &
                   qmewald%totkq
        write(6,*) 'QMMM: Ewald - Kappa = ',  qmewald%kappa
        write(6,*) ''
      end if

      ! If we are NOT updating the qm image atom charges on every SCF step
      ! for qm_ewald then we initially need to zero the scf_mchg array since
      ! in this situation on the first step we allow the QM image charges to
      ! vary with the SCF. It is only on steps 2 -> N that we keep them
      ! fixed during the SCF.  This avoids the need for an explicit test of
      ! the first call condition in the qm_ewald_calc_mm_pot code.
      if (qmmm_nml%qm_ewald==2) then
        scf_mchg(1:qmmm_struct%nquant_nlink) = zero
      end if

      call timer_stop(TIME_QMMMEWALDSETUP)
    else
      qmewald%ewald_startup = .false.
    end if

    ! Allocation for QM_GB (qmgb==2)
    if (qmmm_nml%qmgb == 2) then

      ! Calculate dielectric factor
      if (qm_gb%alpb_on) then
        alpb_beta=alpb_alpha*(intdiel/extdiel)
        qm_gb%intdieli = one/(intdiel*(one + alpb_beta))
        qm_gb%extdieli = one/(extdiel*(one + alpb_beta))
        qm_gb%one_Arad_beta = alpb_beta/Arad
      else
        qm_gb%intdieli = 1.0d0/intdiel
        qm_gb%extdieli = 1.0d0/extdiel
      end if
      qm_gb%mmcut2 = mmcut2
    end if
  end if
  ! End if (qmmm_struct%qm_mm_first_call)

  call timer_stop(TIME_QMMMSETUP)

  ani_qmmm_int5_pol = .false.
  ani_pol_cutoff = zero
#ifdef TORCHANI
  if (qmmm_nml%qmtheory%EXTERN .and. qmmm_nml%qmmm_int == 5) then
    if (qm2_extern_using_ani()) then
      call torchani_get_qmmm_int5_pol_settings(ani_qmmm_int5_pol, ani_pol_cutoff)
    end if
  end if
#endif

  qm_mm_list_cutoff = qmmm_nml%qmcut
  qm_mm_list_include_link_pair_atoms = .false.
  if (ani_qmmm_int5_pol) then
    qm_mm_list_cutoff = ani_pol_cutoff
    qm_mm_list_include_link_pair_atoms = .true.
  end if

  ! Build the non-bonded list for the QM region.  Check whether the system
  ! is periodic and and run some additional diagnostics.  This section also
  ! starts and stops timers relevant to QM setup.
  if (periodic == 1) then
    call timer_start(TIME_QMMMCOORDSX)
    call qm_fill_qm_xcrd_periodic(coords, natom, qmmm_struct%iqmatoms, &
                                  scaled_mm_charges, &
                                  qmmm_scratch%qm_real_scratch, &
                                  qm_mm_list_cutoff, &
                                  qm_mm_list_include_link_pair_atoms)

    ! QMMMCOORDSX was stopped and QMMMLISTBUILD was started
    ! in qm_fill_qm_xcrd_periodic.
    call timer_stop(TIME_QMMMLISTBUILD)
  else
    call timer_start(TIME_QMMMLISTBUILD)

    ! In a non-periodic system, the cutoff will be based on distance from
    ! the QM region as a whole.  nb_list also fills qm_xcrd array and
    ! extracts qm atoms.
    call qm_fill_qm_xcrd( coords, natom, scaled_mm_charges, &
                          qm_mm_list_cutoff, &
                          qm_mm_list_include_link_pair_atoms)
    call timer_stop(TIME_QMMMLISTBUILD)
  endif

  call timer_start(TIME_QMMMCOORDSX)

  ! If verbosity is on print the number of QMMM pairs
  if (qmmm_mpi%commqmmm_master .AND. qmmm_nml%verbosity > 1) then
    write(6,*) 'QMMM: No. QMMM Pairs per QM atom: ', qmmm_struct%qm_mm_pairs
  end if

  ! Finally we need to position the link atoms for this step
  ! We base this on the imaged coordinates so we don't need
  ! to worry about periodic boundaries.  This will write the
  ! link atoms to the end of the qmmm_struct%qm_coords array.
  if ( qmmm_struct%nlink > 0 ) then
    call position_link_atoms(coords)
  end if

#ifndef API
  if (qmmm_mpi%commqmmm_master .AND. &
      (qmmm_nml%verbosity>1 .OR. qmmm_struct%qm_mm_first_call)) then
    call print_link_atom_info( qmmm_struct%qm_coords,atom_type )
  end if
#endif  /* ifndef API */

  ! Begin the rest of the QM/MM setup.  Print the
  ! initial QM region coordinates if appropriate.
  call timer_stop_start(TIME_QMMMCOORDSX,TIME_QMMMSETUP)
#ifndef API
  if(qmmm_struct%qm_mm_first_call) then
    if (qmmm_mpi%commqmmm_master) then
      call qm_print_coords(nstep,.true.)
      if (abfqmmm_param%abfqmmm /= 1) then
         write(6,'(/80("-")/"  3.1 QM CALCULATION INFO",/80("-"))')
      end if
    end if
  end if

  ! Also, if we are doing nearest solvent and the QM region
  ! may have changed, reprint things.
  if (qmmm_mpi%commqmmm_master      .and. &
      qmmm_nml%vsolv > 0            .and. &
      qmmm_nml%verbosity > 0        .and. &
      .not. qmmm_struct%qm_mm_first_call) then
    if (mod(nstep,qmmm_vsolv%nearest_qm_solvent_fq)==0) then
      call qm_print_coords(nstep,.false.)
    end if
  end if
#endif /* ifndef API */

  ! Setup for QM Ewald: if we are doing QM Ewald, then we need to
  ! calculate the K vectors.  We do this on every step since the
  ! box dimensions may change, and the K vectors values depend on
  ! the box dimensions.
  if (qmmm_nml%qm_ewald>0) then
    call timer_stop_start(TIME_QMMMSETUP,TIME_QMMMEWALDKTABLE)

    !Parallel
    call qm_ewald_calc_kvec(qmewald%kvec, qmewald%dkvec, qmewald%dmkv, &
                            qmmm_nml%kmaxqx, qmmm_nml%kmaxqy, &
                            qmmm_nml%kmaxqz, qmmm_nml%ksqmaxq)

    ! Next we calculate the KTABLE, which is an array of exponentials (in
    ! complex sin, cos notation) in the form of 1->6 by 1->NKvectors wide by
    ! 1->Natom (all atoms) long.  Note, this routine is very memory intensive
    ! and very time consuming.  Although if we are using Walker and Crowley
    ! PME implementation then we only do the QM-QM table which is fast.
    ! Note this routine uses the unimaged coordinates from AMBER coords array
    ! so here the MMlink atom must have had its coordinates replaced with the
    ! link atom.
    if (qmmm_nml%ifqnt) call adj_mm_link_pair_crd(coords)

    ! Parallel
    ! This will fill the kvector tables with the exponential values.
    ! Memory for ktable_x_cos... should already have been allocated.
    call qm_ewald_calc_ktable(natom, qmmm_struct%nquant, qmmm_struct%nlink, &
                              coords, qmewald%dmkv)

    if (qmmm_nml%ifqnt) then
      call rst_mm_link_pair_crd(coords)
    end if
    call timer_stop_start(TIME_QMMMEWALDKTABLE,TIME_QMMMSETUP)
  end if
  ! End setup for QM Ewald

  ! Setup OPNQ stuff (Taisung Lee, Rutgers, 2011)
  if (qmmm_struct%qm_mm_first_call .and. qmmm_opnq%useOPNQ) then
    qmmm_opnq%OPNQCorrection=zero
    qmmm_opnq%vdWCorrection=zero
    qmmm_opnq%NB_cutoff=qmmm_nml%qmcut
    qmmm_opnq%switch_cutoff1=min(5.0D0, qmmm_nml%qmcut*0.6)
    qmmm_opnq%switch_cutoff2=min(6.5D0, qmmm_nml%qmcut*0.8)
    if (qmmm_mpi%commqmmm_master .AND. qmmm_nml%verbosity > 1) then
      write(6,'("QMMM: OPNQ correction is turned on")')
      write(6,'("QMMM: OPNQ Switching cutoff1=",f18.8)') &
            qmmm_opnq%switch_cutoff1
      write(6,'("QMMM: OPNQ Switching cutoff2=",f18.8)') &
            qmmm_opnq%switch_cutoff2
    end if

    ! Allocate space for pointer components of type qmmm_opnq_structure--
    ! qmmm_opnq is the sole structure of that type.  This is the sole
    ! allocation, i.e. the pointers are not associated.  Where is the
    ! deallocation ?? srb 5/2015
    allocate(qmmm_opnq%MM_atomType(natom), stat=ier )
    REQUIRE(ier == 0)
    ! store the mm type info
    qmmm_opnq%MM_atomType=atom_type_index
    n = floor(sqrt(nttyp*2.D0))
    REQUIRE(n==ntype)
    allocate(qmmm_opnq%supported(n), qmmm_opnq%LJ_r(n), &
             qmmm_opnq%LJ_epsilon(n), qmmm_opnq%atomic_number(n), stat=ier )
    REQUIRE(ier == 0)

    ! Get the atomic number for each atom
    do i = 1,natom
      qmmm_opnq%atomic_number(atom_type_index(i)) = 0
      if (atomic_mass(i) >= 0.01) then
        somethingWrong = .false.
        call get_atomic_number(atom_type(i), atomic_mass(i), j, &
                               somethingWrong)
        if (.not. somethingWrong) then
          qmmm_opnq%atomic_number(atom_type_index(i)) = j
        end if
      else
        qmmm_opnq%atomic_number(atom_type_index(i)) = 0
      endif
    end do

    ! store the mm LJ parameters
    temp=2.d0**(-5.d0/6.d0)
    do i = 1,n
      j = i*(i+1)/2
      if ((abs(cn1(j)) <1.0D-5) .and. (abs(cn2(j))<1.0D-5)) then
        qmmm_opnq%supported(i) = .false.
        qmmm_opnq%LJ_r(i) = 0.d0
        qmmm_opnq%LJ_epsilon(i) = 0.d0
      else
        qmmm_opnq%supported(i) = .true.
        qmmm_opnq%LJ_r(i) = temp*(  ( (cn1(j)/cn2(j)) )**(1.d0/6.d0) )
        qmmm_opnq%LJ_epsilon(i) = 0.25d0*cn2(j)*cn2(j)/cn1(j)
      end if
    end do
  end if
  ! End setup OPNQ stuff

  ! Setup for PM3MMX interface
  if (qmmm_struct%PM3MMX_INTERFACE) then

    ! qm_mm_pairs is known, so allocate the qm_mm_pair_atom_numbers array.
    ! This array will be qm_mm_pairs long and store the atomic Z-numbers of
    ! MM atoms in the qm_mm_pairs pair list (i.e. carbon = 6, oxygen = 8).
    ! This uses less memory since qm_mm_pairs <= (natom-nquant+1).
    !
    ! qm_mm_pair_atom_numbers is nullified in type qmmm_struct_type and it
    ! can be allocated elsewhere, so check before allocating; I don't know
    ! whether there can be conflicts so just emit a warning.  srb 5/2015
    !
    ! qm_mm_pair_atom_numbers is deallocated by delete in qmmm_struct_module
    ! which is called by deallocate_qmmm.
    if (associated(qmmm_struct%qm_mm_pair_atom_numbers)) then
      deallocate( qmmm_struct%qm_mm_pair_atom_numbers, stat=ier)
      REQUIRE(ier == 0)
    end if
    allocate (qmmm_struct%qm_mm_pair_atom_numbers(qmmm_struct%qm_mm_pairs), &
              stat=ier)
    REQUIRE(ier == 0)
    do i = 1,qmmm_struct%qm_mm_pairs

      ! The atom_name and atomic_mass arrays run over the list of all MM
      ! atoms in the topology, but the qm_mm_pair_atom_numbers array runs
      ! only over the MM atoms which make pairs with QM atoms of the system.
      ! (That can be a good chunk of the MM atoms, but not necessarily all.)
      ! The qm_mm_pair_list stores the number indices of MM atoms in the MM
      ! topology, so to get the atomic Z numbers of the MM atoms in the
      ! qm_mm_pair_list we must take indices and cross-reference.
      j = qmmm_struct%qm_mm_pair_list(i)
      qmmm_struct%qm_mm_pair_atom_numbers(i) = 0
      if (atomic_mass(j) >= 0.01d0) then ! dont fail on TIP4P EP particles, for example
         call get_atomic_number(atom_name(j), atomic_mass(j), &
                                qmmm_struct%qm_mm_pair_atom_numbers(i))
      end if
    end do
  end if
  ! End setup for PM3MMX interface

  ! This is the end of QM/MM setup--stop the timer
  call timer_stop(TIME_QMMMSETUP)

  ! Calculate Forces
  nclatoms = qmmm_struct%qm_mm_pairs
  if (qmmm_nml%qmmm_int == 5 .and. (.not. ani_qmmm_int5_pol)) then
    nclatoms = 0
  end if
  ! Set the mm atom types, otherwise the code gives a segfault
  if (associated(qmmm_struct%qm_mm_pair_atom_numbers)) then
      deallocate( qmmm_struct%qm_mm_pair_atom_numbers, stat=ier)
      REQUIRE(ier == 0)
    end if
    allocate (qmmm_struct%qm_mm_pair_atom_numbers(qmmm_struct%qm_mm_pairs), &
              stat=ier)
    REQUIRE(ier == 0)
    do i = 1,qmmm_struct%qm_mm_pairs

      ! The atom_name and atomic_mass arrays run over the list of all MM
      ! atoms in the topology, but the qm_mm_pair_atom_numbers array runs
      ! only over the MM atoms which make pairs with QM atoms of the system.
      ! (That can be a good chunk of the MM atoms, but not necessarily all.)
      ! The qm_mm_pair_list stores the number indices of MM atoms in the MM
      ! topology, so to get the atomic Z numbers of the MM atoms in the
      ! qm_mm_pair_list we must take indices and cross-reference.
      j = qmmm_struct%qm_mm_pair_list(i)
      qmmm_struct%qm_mm_pair_atom_numbers(i) = 0
      if (atomic_mass(j) >= 0.01d0) then ! dont fail on TIP4P EP particles, for example
         call get_atomic_number(atom_name(j), atomic_mass(j), &
                                qmmm_struct%qm_mm_pair_atom_numbers(i))
      end if
    end do
  
  ! Determine id
  id = ''
  if ( (ipimd > 0) .or. (ineb >0) ) then
    ! Add number in case of parallel PIMD runs
    write (id,'(i3.3)') mybeadid
  end if
#if defined(MPI)
  if ((rem > 0) .or. (qmmm_nml%vsolv > 1) .or. asm == 1) then
      ! Add rank for parallel REMD run
      write (id,'(i3.3)') masterrank
  end if
#endif /* MPI */
 
  if (qmmm_nml%qmtheory%EXTERN) then
    call qm2_extern_get_qm_forces(nstep, qmmm_struct%nquant_nlink, &
                                  qmmm_struct%qm_coords, &
                                  qmmm_struct%iqm_atomic_numbers,scf_mchg, &
                                  nclatoms, qmmm_struct%qm_xcrd, &
                                  qmmm_struct%qm_mm_pair_atom_numbers, escf, &
                                  qmmm_struct%dxyzqm, qmmm_struct%dxyzcl, id)
  else if (qmmm_nml%qmtheory%ISQUICK) then
    call get_quick_qmmm_forces(qmmm_struct%nquant_nlink, &
                               qmmm_struct%qm_coords, &
                               qmmm_struct%iqm_atomic_numbers, &
                               nclatoms, qmmm_struct%qm_xcrd, escf, &
#ifdef CEW
                               qmmm_struct%dxyzqm, qmmm_struct%dxyzcl, use_cew)
#else
                               qmmm_struct%dxyzqm, qmmm_struct%dxyzcl, .false.)
#endif
  else if (qmmm_nml%qmtheory%ISTCPB) then
    call get_tcpb_qmmm_forces(nstep, qmmm_struct%nquant_nlink, &
                               qmmm_struct%qm_coords, &
                               qmmm_struct%iqm_atomic_numbers, &
                               nclatoms, qmmm_struct%qm_xcrd, escf, &
                               qmmm_struct%dxyzqm, qmmm_struct%dxyzcl, id)
  else if (qmmm_nml%qmtheory%ISXTB) then
     call get_xtb_qmmm_forces( &
          & natom, scaled_mm_charges, &
          & qmmm_struct%nquant_nlink, &
          & qmmm_struct%iqm_atomic_numbers, &
          & qmmm_struct%qm_coords, nclatoms, &
          & qmmm_struct%qm_xcrd, escf, &
          & qmmm_struct%dxyzqm, qmmm_struct%dxyzcl, scf_mchg )
  else if (qmmm_nml%qmtheory%ISDFTBPLUS) then
     call get_dftbplus_qmmm_forces( &
          & natom, scaled_mm_charges, &
          & qmmm_struct%nquant_nlink, &
          & qmmm_struct%qm_ntypes, &
          & qmmm_struct%qm_type_id, &
          & qmmm_struct%qm_coords, nclatoms, &
          & qmmm_struct%qm_xcrd, escf, &
          & qmmm_struct%dxyzqm, qmmm_struct%dxyzcl, scf_mchg )
  else
    call get_qm2_forces(qmmm_mpi%commqmmm_master, qm2_struct%calc_mchg_scf, &
                        natom, born_radii, one_born_radii, coords, &
                        scaled_mm_charges, qmmm_nml, qmmm_struct, &
                        qmmm_scratch, scf_mchg, escf)
  end if

  ! If we are doing qm_ewald then we need to calculate the gradients due to
  ! the ewald potential here.  This is only available analytically: no
  ! numerical gradients are calculated.  With qm_pme the forces are calculated
  ! in ew_recip using the mulliken charges.
  if (qmmm_nml%qm_ewald>0) then
    call timer_start(TIME_QMMMFQMEWALD)

    ! Parallel
    call qm_ewald_get_forces(qmmm_struct%qm_xcrd, qmmm_struct%qm_coords,&
                             natom, scf_mchg, qmmm_nml%qmmmrij_incore, &
                             qmmm_struct%dxyzqm, qmmm_struct%dxyzcl, &
                             qmewald%dkvec, scaled_mm_charges)
    call timer_stop(TIME_QMMMFQMEWALD)
  end if

  ! Calculation of forces due to QM components is now complete.  The
  ! QM forces can now be put into a sander array for inclusion in the
  ! rest of the simulation.
  call timer_start(TIME_QMMMCOLLATEF)
  do i=1,qmmm_struct%nquant
    m = qmmm_struct%iqmatoms(i)
    m = (m-1)*3
    f(m+1) = f(m+1) - qmmm_struct%dxyzqm(1,i)
    f(m+2) = f(m+2) - qmmm_struct%dxyzqm(2,i)
    f(m+3) = f(m+3) - qmmm_struct%dxyzqm(3,i)
  enddo

  ! If there is something other than mechanical embedding (qmmm_int == 0 or
  ! qmmm_int == 5) then we must include the effects of the QM region on the
  ! system's classical (MM) atoms.  Only the MM atoms that are in the list
  ! qm_mm_pair_list need to be considered.
  if (qmmm_nml%qmmm_int > 0 .and. &
      ((qmmm_nml%qmmm_int /= 5) .or. ani_qmmm_int5_pol) ) then
    do i = 1,qmmm_struct%qm_mm_pairs
      m = (qmmm_struct%qm_mm_pair_list(i)-1)*3
      f(m+1) = f(m+1) - qmmm_struct%dxyzcl(1,i)
      f(m+2) = f(m+2) - qmmm_struct%dxyzcl(2,i)
      f(m+3) = f(m+3) - qmmm_struct%dxyzcl(3,i)
    end do
  end if

  ! If we are doing QM Ewald, then we have a set of forces on ALL MM atoms
  ! (not just those in the qm_mm_pair_list) that we need to put in the main
  ! force array.  For qm_pme this is done later in ew_recip, in the
  ! subroutine force (see force.F90).  Parallel division in the Ewald code
  ! occurs over K vectors, so all threads have part of the forces on every
  ! atom.
  if (qmmm_nml%qm_ewald>0 .and. .not. qmmm_nml%qm_pme) then
    do i = 1, natom
      offset = (3*i) - 2
      f(offset)   = f(offset)   - qmewald%d_ewald_mm(1,i)
      f(offset+1) = f(offset+1) - qmewald%d_ewald_mm(2,i)
      f(offset+2) = f(offset+2) - qmewald%d_ewald_mm(3,i)
    end do
  end if

  ! We need to divide the force on the link atom
  ! up between the QM and MM atoms it connects.
  do i=1,qmmm_struct%nlink

    ! The location of the MM atom in coordinate array coords: the x
    ! coordinate will be at mm_no, y and mm_no+1, z at mm_no+2.
    ! The mm_no index can also be a location in the force array f.
    mm_no = 3*qmmm_struct%link_pairs(1,i)-2

    ! Get the index of the link atom in the list of QM atoms.  The
    ! array qmmm_struct%iqmatoms is a list of indices of the QM
    ! atoms into the array of all system atoms in the prmtop, so
    ! one index leads to another which gets us the index of the
    ! x coordinate of the QM atom in this link pair within the
    ! array of system coordinates.  qm_no also applies to the force
    ! array f.
    lnk_no = qmmm_struct%link_pairs(2,i)
    qm_no = 3*qmmm_struct%iqmatoms(lnk_no)-2

    ! Note that this routine uses the flink in the form -flink.
    call distribute_lnk_f(forcemod,&
                          qmmm_struct%dxyzqm(1:3, qmmm_struct%nquant+i), &
                          coords(mm_no), coords(qm_no), qmmm_nml%lnk_dis)

    ! NOTE: forces are reversed in QM calc with respect to the amber
    ! force array so we subtract forcemod from MM atom and add it to
    ! QM atom.  MM atom's new force = FMM(x,y,z) - FORCEMOD(x,y,z).
    f(mm_no) = f(mm_no) - forcemod(1)
    f(mm_no+1) = f(mm_no+1) - forcemod(2)
    f(mm_no+2) = f(mm_no+2) - forcemod(3)

    ! QM atom's new force = FQM(x,y,z) - Flink(x,y,z) + FORCEMOD(x,y,z)
    ! Note QM forces should be subtracted from sander F array to leave
    ! the total force.
    f(qm_no) = f(qm_no) - qmmm_struct%dxyzqm(1,qmmm_struct%nquant+i) + &
               forcemod(1)
    f(qm_no+1) = f(qm_no+1) - qmmm_struct%dxyzqm(2,qmmm_struct%nquant+i) + &
                 forcemod(2)
    f(qm_no+2) = f(qm_no+2) - qmmm_struct%dxyzqm(3,qmmm_struct%nquant+i) + &
                 forcemod(3)
  end do

  call timer_stop(TIME_QMMMCOLLATEF)

  ! This is no longer the first call, and Ewald startup
  ! should be done if it is needed at all.  Still, there is
  ! the issue of whether this is part of abfqmmm.
  qmmm_struct%qm_mm_first_call = .false.
  qmewald%ewald_startup = .false.
  if (abfqmmm_param%abfqmmm == 1) then
    qmmm_struct%qm_mm_first_call = .true.
    qmewald%ewald_startup = .true.
  end if

end subroutine qm_mm

!------------------------------------------------------------------------------
! qm_fill_qm_xcrd: Build a non-periodic list, fill qm_xcrd and extract
!                  qm_coords.  This routine calculates a qm_mm_pair_list which
!                  is of the form [ listtype atom atom atom ... ] for each
!                  atom, excluding any link atoms (the number is given by
!                  qmmm_struct%nquant).  The list of MM atoms that interact
!                  with each QM atom is based on the cutoff distance.  In then
!                  fills qm_xcrd and extracts the qm coordinates from the Amber
!                  x array.  Each QM atom will use the same list, since to be
!                  included an MM atom only need to be within cut of any QM
!                  atom.  The qm_mm_pair_list resides in the global qmmm_struct
!                  imported from qmmm_module.
!
! Arguments:
!   x:                 The coordinates (3*natom _REAL_ numbers)
!   natom:             The number of atoms
!   scaled_mm_charges: Molecular mechanics charges in proton units, Na+ = +1.0
!------------------------------------------------------------------------------
subroutine qm_fill_qm_xcrd(x, natom, scaled_mm_charges, pair_cutoff, &
                           include_link_pair_atoms)

  use qmmm_module, only : qmmm_nml,qmmm_struct, qmmm_scratch
  implicit none

  !Passed in
  integer natom
  _REAL_ , intent(in) ,dimension(3,natom) :: x
  _REAL_, intent(in), dimension(natom) :: scaled_mm_charges
  _REAL_, intent(in) :: pair_cutoff
  logical, intent(in) :: include_link_pair_atoms

  !Local Variables!
  integer j,m,i,n1
  _REAL_ , dimension(6) :: bxbnd
  _REAL_ x_qm, y_qm, z_qm, dx2, xbnd0, xbnd1, ybnd0, ybnd1, zbnd0, zbnd1
  _REAL_ pair_cutoff2
  logical include_atom

  ! Find the bounding box limits of the QM region, then find all atoms
  ! inside the box + cutoff before calculating or testing distances.
  m = qmmm_struct%iqmatoms(1)
  xbnd0 = x(1,m)
  xbnd1 = x(1,m)
  ybnd0 = x(2,m)
  ybnd1 = x(2,m)
  zbnd0 = x(3,m)
  zbnd1 = x(3,m)
  do j = 2,qmmm_struct%nquant
    m = qmmm_struct%iqmatoms(j)
    xbnd0 = min(xbnd0,x(1,m))
    xbnd1 = max(xbnd1,x(1,m))
    ybnd0 = min(ybnd0,x(2,m))
    ybnd1 = max(ybnd1,x(2,m))
    zbnd0 = min(zbnd0,x(3,m))
    zbnd1 = max(zbnd1,x(3,m))
  enddo
  pair_cutoff2 = pair_cutoff * pair_cutoff
  bxbnd(1) = xbnd0-pair_cutoff
  bxbnd(2) = xbnd1+pair_cutoff
  bxbnd(3) = ybnd0-pair_cutoff
  bxbnd(4) = ybnd1+pair_cutoff
  bxbnd(5) = zbnd0-pair_cutoff
  bxbnd(6) = zbnd1+pair_cutoff

  include_atom = .false.

  ! Make a first pass to mask QM/MM atom pairs
  qmmm_scratch%qm_int_scratch(1:natom)=0 !Used for a mask
  do m = 1,natom

    ! There is no short circuit evaluation in fortran, so having
    ! a series of separate if statements here should be faster.
    if ( x(1,m) <= bxbnd(1) ) cycle
    if ( x(1,m) >= bxbnd(2) ) cycle
    if ( x(2,m) <= bxbnd(3) ) cycle
    if ( x(2,m) >= bxbnd(4) ) cycle
    if ( x(3,m) <= bxbnd(5) ) cycle
    if ( x(3,m) >= bxbnd(6) ) cycle
    qmmm_scratch%qm_int_scratch(m) = 1
  enddo

  ! Set QM atoms in qm_int_scratch to zero so they don't get counted as pairs.
  qmmm_scratch%qm_int_scratch(qmmm_struct%iqmatoms(1:qmmm_struct%nquant)) = 0

  ! The index n1 is a count of the number of MM atoms thus far included
  ! in the pair list for the QM atom region.  A second pass to map the
  ! QM/MM atom pairs, looping over all atoms (excluding QM atoms, which
  ! were filtered out in the first pass) now commences.
  n1 = 0
  do m = 1,natom
    if ( qmmm_scratch%qm_int_scratch(m) == 1 ) then
      check_cut: do j=1,qmmm_struct%nquant

        ! This is the atom number of the current QM atom, stored in i.
        ! Find all MM atoms that are within CUT of this QM atom
        i = qmmm_struct%iqmatoms(j)

        ! Get coordinates of the QM atom
        x_qm = x(1,i)
        y_qm = x(2,i)
        z_qm = x(3,i)

        ! Calculate the distance between this QM atom and the current MM
        ! atom (index m), see if it is within the cutoff, and include it
        ! UNLESS the pair would also be a QM/MM link pair.
        dx2 = ((x_qm - x(1,m)) * (x_qm - x(1,m)) + &
               (y_qm - x(2,m)) * (y_qm - x(2,m)) + &
               (z_qm - x(3,m)) * (z_qm - x(3,m)))
        if ( dx2 < pair_cutoff2 ) then
          if (qmmm_struct%mm_link_mask(m) .and. &
              (.not. include_link_pair_atoms)) then
            exit check_cut
          end if
          include_atom = .true.
          exit check_cut
        end if
      end do check_cut

      ! Include this mm atom in the list
      if ( include_atom ) then
        n1 = n1+1
        qmmm_struct%qm_mm_pair_list( n1 ) = m
        qmmm_struct%qm_xcrd(1:3,n1) = x(1:3,m)
        qmmm_struct%qm_xcrd(4,n1) = scaled_mm_charges(m)
        include_atom=.false.
      end if
    end if
  end do
  qmmm_struct%qm_mm_pairs = n1

  ! Extract QM atoms from x into qmcoords array.
  do m = 1,qmmm_struct%nquant
    i = qmmm_struct%iqmatoms(m)
    qmmm_struct%qm_coords(1:3,m) = x(1:3,i)
  enddo

end subroutine qm_fill_qm_xcrd

!------------------------------------------------------------------------------
! qm_fill_qm_xcrd_periodic: a variant of the preceding qm_fill_qm_xcrd function
!                           for periodic systems.
!
! Arguments:
!   x:                The coordinates (3*natom _REAL_ numbers)
!   natom:            The number of atoms
!   iqmatoms:         The index array of REAL QM atoms
!   scaled_mm_chrgs:  Molecular mechanics charges in proton units, Na+ = +1.0
!   real_scratch:     Scratch space of _REAL_ numbers
!------------------------------------------------------------------------------
subroutine qm_fill_qm_xcrd_periodic(x, natom, iqmatoms, scaled_mm_chrgs, &
                                    real_scratch, pair_cutoff, &
                                    include_link_pair_atoms)

  use qmmm_module, only: qmmm_nml,qmmm_struct, qmmm_scratch
  use nblist, only: a,b,c,alpha,beta,gamma,ucell,recip,sphere
  use constants, only : zero, one, half, two, TWOPI

  implicit none
  integer , intent(in) :: natom
  _REAL_ , intent(in), dimension(3,natom) :: x
  integer, intent(in), dimension(qmmm_struct%nquant) :: iqmatoms
  _REAL_, intent(in), dimension(natom) :: scaled_mm_chrgs
  _REAL_, intent(out), dimension(3,natom) :: real_scratch
  _REAL_, intent(in) :: pair_cutoff
  logical, intent(in) :: include_link_pair_atoms

  _REAL_, parameter :: box_buffer = 1.0d-12

  integer j,jqmatom
  _REAL_ :: xbnd0,xbnd1,ybnd0,ybnd1,zbnd0,zbnd1
  _REAL_ :: offset(3), frac(3), xx,yy,zz,xtmp,ytmp,ztmp,one_nquant
  _REAL_ , dimension(6) :: bxbnd

  integer :: m,n, n1
  _REAL_  :: dx2, pair_cutoff2
! _REAL_  :: fbndx0, fbndx1, fbndy0, fbndy1, fbndz0, fbndz1
  logical :: include_atom
  _REAL_ :: xcos,xsin,ycos,ysin,zcos,zsin

  ! do i=1,qmmm_struct%nquant
  !    iqmatom = iqmatoms(i)
  !    write(6,'(a,i3,i6,3f9.3)')"QMCRD ",i,iqmatom,x(1:3,iqmatom)
  ! end do


  !
  ! The code lines commented below were from:
  !
  ! commit 5ee8b91678cd543855247501053012823817e661
  ! Author: Ross Walker <ross@rosswalker.co.uk>
  ! Date:   Tue May 11 20:47:39 2010 -0700
  !
  !    RCW: Bugfix.35 from AMBER 10. This is bugfix.1 for AMBER 11.
  !    It fixes an imaging problem that can occur in QMMM simulations
  !    with small boxes.
  !
  ! The commit was made in response to this bug report on the
  ! Amber-Users mailing list:
  !   http://archive.ambermd.org/201004/0464.html
  !
  ! In brief, the user had 256 methane molecules in a box
  ! with dimensions 22.293, 22.418, 22.344 and cut=qmcut=4.0
  ! 32 of the 256 molecules were QM.
  ! The range of the QM region in the z-direction (from end-to-end)
  ! slightly exceeded 1/2 of the box length.
  ! The first QM atom was the lower bound in the z-direction
  ! of the box enclosing the QM atoms.
  ! Therefore, if the QM atoms were wrapped around the first
  ! QM atom, then the total length of the QM region cannot
  ! exceed 1/2 the box length or else some of the atoms
  ! get imaged incorrectly (one box length away from where one
  ! expects it to be).
  !
  ! Alternatively, if the QM atoms were imaged around the
  ! central-most-point of the QM region, then the QM region
  ! on either side of the central point can be at most 1/2
  ! the box length. In other words, the length of the QM
  ! region could then be longer than 1/2 the box length.
  ! Therefore, the quick-fix proposed by Ross Walker was to
  ! calculate the mean position of the QM atoms and wrap
  ! the QM atoms around the mean.  As stated by Prof. Walker
  ! in his follow-up email to the list, the proposed fix
  ! wasn't extensively tested.
  !
  
  ! ! Move the first QM atom to be at the origin.
  ! !   iqm_one = iqmatoms(1)
  ! !   frac(1:3) = x(1,iqm_one)*recip(1,1:3)+x(2,iqm_one)*recip(2,1:3)+ &
  ! !               x(3,iqm_one)*recip(3,1:3)
  
  !   ! Moving the first QM atom can be an issue if the box size is small:
  !   ! atoms could end up sticking outside the box.  We can change this to
  !   ! calculate the center of the coordinates of the QM region and move this
  !   ! instead, which is much less likely to put atoms outside the box.  This
  !   ! could cause problems if the QM region diffuses significantly, but if
  !   ! that happens we are in a whole world of problems anyway.  Such a case
  !   ! should trigger the cutoff too large error.  This will calculate the
  !   ! center of coordinates for the QM atoms.
  ! xtmp = zero
  ! ytmp = zero
  ! ztmp = zero
  ! do j = 1, qmmm_struct%nquant
  !   jqmatom = iqmatoms(j)
  !   xtmp = xtmp + x(1,jqmatom)
  !   ytmp = ytmp + x(2,jqmatom)
  !   ztmp = ztmp + x(3,jqmatom)
  ! end do
  ! one_nquant = 1.0d0/real(qmmm_struct%nquant)
  ! xtmp = xtmp * one_nquant
  ! ytmp = ytmp * one_nquant
  ! ztmp = ztmp * one_nquant

  !
  ! The problem with the approach shown above is that it ignores
  ! periodicity in the averaging. If the QM region consisted of
  ! 2 adjacent molecules that were wrapped to either side of the
  ! box, for example, then the arithemtic mean isn't the central
  ! most point of the QM region. In fact, it is one of the furthest
  ! points from the QM region.
  ! The modification below calculates the "circular mean"
  ! of each fractional coordinate
  ! 
  xcos=zero
  ycos=zero
  zcos=zero
  xsin=zero
  ysin=zero
  zsin=zero
  do j = 1, qmmm_struct%nquant
     jqmatom = iqmatoms(j)
     frac(1:3) = x(1,jqmatom)*recip(1,1:3) &
          &    + x(2,jqmatom)*recip(2,1:3) &
          &    + x(3,jqmatom)*recip(3,1:3)
     xcos = xcos + cos(TWOPI*frac(1))
     xsin = xsin + sin(TWOPI*frac(1))
     ycos = ycos + cos(TWOPI*frac(2))
     ysin = ysin + sin(TWOPI*frac(2))
     zcos = zcos + cos(TWOPI*frac(3))
     zsin = zsin + sin(TWOPI*frac(3))
  end do
  one_nquant = 1.0d0/real(qmmm_struct%nquant)
  xcos = xcos * one_nquant
  xsin = xsin * one_nquant
  ycos = ycos * one_nquant
  ysin = ysin * one_nquant
  zcos = zcos * one_nquant
  zsin = zsin * one_nquant
  frac(1) = atan2(xsin,xcos)/TWOPI
  frac(2) = atan2(ysin,ycos)/TWOPI
  frac(3) = atan2(zsin,zcos)/TWOPI
  xtmp = frac(1)*ucell(1,1) + frac(2)*ucell(1,2) + frac(3)*ucell(1,3)
  ytmp = frac(1)*ucell(2,1) + frac(2)*ucell(2,2) + frac(3)*ucell(2,3)
  ztmp = frac(1)*ucell(3,1) + frac(2)*ucell(3,2) + frac(3)*ucell(3,3)

  
  ! And this will now calculate the fractional coordinates
  ! of the QM center in the unit cell (simulation box).
  frac(1:3) = xtmp*recip(1,1:3)+ytmp*recip(2,1:3)+ &
              ztmp*recip(3,1:3)
  frac(1:3) = frac(1:3) - anint(frac(1:3))
  offset(1) = frac(1)*ucell(1,1) + frac(2)*ucell(1,2) + frac(3)*ucell(1,3)
  offset(2) = frac(1)*ucell(2,1) + frac(2)*ucell(2,2) + frac(3)*ucell(2,3)
  offset(3) = frac(1)*ucell(3,1) + frac(2)*ucell(3,2) + frac(3)*ucell(3,3)

  ! This will now translate all QM atoms such that their
  ! center of coordinates is at the origin, then transform
  ! them into fractional coordinates in the simulation box,
  ! image them, and finally store the imaged cartesian atom
  ! coordinates in the QM/MM real_scratch array.
  do j = 1, qmmm_struct%nquant
    jqmatom = iqmatoms(j)
    xx = x(1,jqmatom) - offset(1)
    yy = x(2,jqmatom) - offset(2)
    zz = x(3,jqmatom) - offset(3)
    frac(1:3) = xx*recip(1,1:3) + yy*recip(2,1:3) + zz*recip(3,1:3)
    frac(1:3) = frac(1:3) - anint(frac(1:3))
    real_scratch(1,j) = frac(1)*ucell(1,1) + frac(2)*ucell(1,2) + &
                        frac(3)*ucell(1,3)
    real_scratch(2,j) = frac(1)*ucell(2,1) + frac(2)*ucell(2,2) + &
                        frac(3)*ucell(2,3)
    real_scratch(3,j) = frac(1)*ucell(3,1) + frac(2)*ucell(3,2) + &
                        frac(3)*ucell(3,3)
  end do

  ! do i=1,qmmm_struct%nquant
  !    iqmatom = iqmatoms(i)
  !    write(6,'(a,i3,i6,3f9.3)')"WWCRD ",i,iqmatom,real_scratch(1:3,i)
  ! end do
  
  ! Compute the bounds of the QM region, then stretch those
  ! boundaries with the QM interaction cutoff distance.
  xbnd0 = zero
  xbnd1 = zero
  ybnd0 = zero
  ybnd1 = zero
  zbnd0 = zero
  zbnd1 = zero
  do j = 1,qmmm_struct%nquant
    xbnd0 = min(xbnd0, real_scratch(1,j))
    xbnd1 = max(xbnd1, real_scratch(1,j))
    ybnd0 = min(ybnd0, real_scratch(2,j))
    ybnd1 = max(ybnd1, real_scratch(2,j))
    zbnd0 = min(zbnd0, real_scratch(3,j))
    zbnd1 = max(zbnd1, real_scratch(3,j))
  enddo
 
  ! write(6,'(a,2f9.3)')"xbnd ",xbnd0,xbnd1
  ! write(6,'(a,2f9.3)')"ybnd ",ybnd0,ybnd1
  ! write(6,'(a,2f9.3)')"zbnd ",zbnd0,zbnd1

  pair_cutoff2 = pair_cutoff * pair_cutoff
  xbnd0 = xbnd0 - pair_cutoff - box_buffer
  ybnd0 = ybnd0 - pair_cutoff - box_buffer
  zbnd0 = zbnd0 - pair_cutoff - box_buffer
  xbnd1 = xbnd1 + pair_cutoff + box_buffer
  ybnd1 = ybnd1 + pair_cutoff + box_buffer
  zbnd1 = zbnd1 + pair_cutoff + box_buffer

  
  ! Check if QM region plus cutoff around it is too large for this box.
  ! The sphere method is the simplest check, but we can get more
  ! sophisticated using the distance between parallel faces later if an
  ! octahedral box or some other trclinic system is in play.  The
  ! sphere is calculated in ew_box.f and used here.
  if ((xbnd1-xbnd0 > two*sphere) .or. (ybnd1-ybnd0 > two*sphere) .or. &
      (zbnd1-zbnd0 > two*sphere)) then
    write(6,*) " ****************************************************"
    write(6,*) " ERROR: QM region + cutoff larger than box dimension:"
    write(6,'(2X,"QM-MM Cutoff = ",f8.4)') pair_cutoff
    write(6,*) "  Coord   Lower     Upper    Size    Radius of largest ", &
      "sphere inside unit cell"
    write(6,'(5X,"X",4(2X,f8.3))') xbnd0, xbnd1, xbnd1-xbnd0, sphere
    write(6,'(5X,"Y",4(2X,f8.3))') ybnd0, ybnd1, ybnd1-ybnd0, sphere
    write(6,'(5X,"Z",4(2X,f8.3))') zbnd0, zbnd1, zbnd1-zbnd0, sphere
    write(6,*) " ****************************************************"
    call sander_bomb("QM_CHECK_PERIODIC<qm_mm.f>", &
                     "QM region + cutoff larger than box", &
                     "cannot continue, need larger box.")
  endif

  ! Compute the QM region boundaries, within the simulation box
  ! fractional coordinates, but where the uncentered QM atoms
  ! actually reside.
  bxbnd(1) = xbnd0 + offset(1)
  bxbnd(2) = xbnd1 + offset(1)
  bxbnd(3) = ybnd0 + offset(2)
  bxbnd(4) = ybnd1 + offset(2)
  bxbnd(5) = zbnd0 + offset(3)
  bxbnd(6) = zbnd1 + offset(3)

  ! Fill qm_xcrd in a periodic setting.  First, move the center of
  ! the QM BOUNDING region to the origin, much as was done for the
  ! coordinates of all QM atoms earlier.
  offset(1) = (bxbnd(2) + bxbnd(1))*half
  offset(2) = (bxbnd(4) + bxbnd(3))*half
  offset(3) = (bxbnd(6) + bxbnd(5))*half

  ! Loop over all atom positions to put them into the simulation box
  ! fractional coordinates, translate them in such a way that would
  ! center the QM region bounding box on the origin, then select those
  ! atoms within bounding box + cutoff.  As in the first pass, this
  ! will transform into box space, image, and transform back to real
  ! space.  Atoms exactly on the bounding box faces are not counted in
  ! the QM/MM cutoff region.  Use the QM/MM integer scratch array for
  ! a mask.  This will reimage all QM atoms as well as nearby MM atoms.
  qmmm_scratch%qm_int_scratch(1:natom) = 0
  do m = 1, natom
    xx = x(1,m) - offset(1)
    yy = x(2,m) - offset(2)
    zz = x(3,m) - offset(3)
    frac(1:3) = xx*recip(1,1:3) + yy*recip(2,1:3) + zz*recip(3,1:3)
    frac(1:3) = frac(1:3) - anint(frac(1:3))
    xx = frac(1)*ucell(1,1) + frac(2)*ucell(1,2) + frac(3)*ucell(1,3)
    if ((xx > bxbnd(1)-offset(1)) .and. (xx < bxbnd(2)-offset(1))) then
      yy = frac(1)*ucell(2,1) + frac(2)*ucell(2,2) + frac(3)*ucell(2,3)
      if ((yy > bxbnd(3)-offset(2)) .and. (yy < bxbnd(4)-offset(2))) then
        zz = frac(1)*ucell(3,1) + frac(2)*ucell(3,2) + frac(3)*ucell(3,3)
        if ((zz > bxbnd(5)-offset(3)) .and. (zz < bxbnd(6) -offset(3))) then

          ! This atom is inside the box.  Record its shifted position.
          qmmm_scratch%qm_int_scratch(m)=1
          real_scratch(1,m) = xx
          real_scratch(2,m) = yy
          real_scratch(3,m) = zz
        endif
      endif
    endif
  enddo

  ! Fill the qm coordinate array with the imaged QM atoms
  do n = 1, qmmm_struct%nquant
    m = iqmatoms(n)
    qmmm_struct%qm_coords(1:3,n) = real_scratch(1:3,m)

    ! Mask out the actual QM atoms so that QM:QM pairs
    ! are not included in the pair list building next.
    qmmm_scratch%qm_int_scratch(iqmatoms(n)) = 0
  enddo
  call timer_stop_start(TIME_QMMMCOORDSX, TIME_QMMMLISTBUILD)

  ! Now calculate the pair list and fill qm_xcrd.  As in the
  ! non-periodic version of this subroutine above, n1 keeps
  ! a count of the number of atoms in the pair list.
  include_atom = .false.
  n1 = 0
  do m = 1, natom

    ! Loop over all atoms, but excluding QM atoms
    if (qmmm_scratch%qm_int_scratch(m) > 0) then
      check_cut: do j = 1, qmmm_struct%nquant

        ! Find all MM atoms that are within the cutoff of this QM atom
        xx = qmmm_struct%qm_coords(1,j)
        yy = qmmm_struct%qm_coords(2,j)
        zz = qmmm_struct%qm_coords(3,j)

        ! Calculate the distance and see if it is within the cutoff
        dx2 = ((xx - real_scratch(1,m)) * (xx - real_scratch(1,m)) + &
               (yy - real_scratch(2,m)) * (yy - real_scratch(2,m)) + &
               (zz - real_scratch(3,m)) * (zz - real_scratch(3,m)))
        if (dx2 < pair_cutoff2) then

          ! We include the atom unless this is a QM/MM link pair
          if (qmmm_struct%mm_link_mask(m) .and. &
              (.not. include_link_pair_atoms)) then
            exit check_cut
          end if
          include_atom = .true.
          exit check_cut
        end if
      end do check_cut
      if (include_atom) then
        n1 = n1 + 1
        qmmm_struct%qm_mm_pair_list(n1) = m
        qmmm_struct%qm_xcrd(1,n1) = real_scratch(1,m)
        qmmm_struct%qm_xcrd(2,n1) = real_scratch(2,m)
        qmmm_struct%qm_xcrd(3,n1) = real_scratch(3,m)
        qmmm_struct%qm_xcrd(4,n1) = scaled_mm_chrgs(m)
        include_atom = .false.
      end if
    end if
  end do

  ! Record the number of QM:MM pairs
  qmmm_struct%qm_mm_pairs = n1

end subroutine qm_fill_qm_xcrd_periodic

!------------------------------------------------------------------------------
! get_qm2_forces: calculate energy and forces with built-in semiempirical or
!                 DFTB methods
!
! Arguments:
!   master:            flag to indicate that this thread is the master in an
!                      MPI setting (imported from the data structure
!                      qm2_struct)
!   calc_mchg_scf:     flag to activate calculation of Mulliken charges
!                      (imported from the data structure qm2_struct)
!   natom:             the number of REAL atoms in the system (not just the QM
!                      region)
!   born_radii:        Effective GB radii for all atoms (array of natom _REAL_
!                      numbers), only used when doing qm with gb (and
!                      qm_gb==2).  These are calculated via an initial call to
!                      the egb subroutine.
!   one_born_radii:    Inverse GB radii for atoms (array of natom _REAL_
!                      numbers)
!   coords:            Cartesian coordinates for all atoms in the system
!                      (3*natom _REAL_ numbers, not just the QM region)
!   scaled_mm_charges: the scaled molecular mechanics charges on all atoms (the
!                      scaling factor is the inverse of 18.2223, which takes
!                      the charges into proton units--sodium cation would read
!                      +1.0 here)
!   qmmm_nml:          the QM/MM namelist read in from the input file (it
!                      contains the QM method and other details)
!   qmmm_struct:       the primary structure for storing information that comes
!                      out of the QM calculation, including energy quantities
!                      (coulombic_eng, etc.) and forces (dxyzqm) to be returned
!   qmmm_scratch:      a collection of scratch spaces (_REAL_ arrays and the
!                      like) needed for QM-related computations
!   scf_mchg:          the array of SCF Mulliken charges (these are what gets
!                      mapped to the Ewald long-ranged electrostatic
!                      calculation).  The nquant_nlink attribute of the
!                      qmmm_struct_type is the total number of QM atoms in the
!                      system (number of REAL quantum atoms + number of link
!                      atoms).  scf_mchg returns results.
!   escf:              the SCF energy (returned)
!------------------------------------------------------------------------------
subroutine get_qm2_forces(master, calc_mchg_scf, natom, born_radii, &
                          one_born_radii, coords, scaled_mm_charges, &
                          qmmm_nml, qmmm_struct, qmmm_scratch, scf_mchg, escf)

  use qmmm_nml_module, only : qmmm_nml_type
  use qmmm_struct_module, only : qmmm_struct_type
  use qmmm_module, only : qmmm_scratch_structure, qmmm_opnq
  use constants, only : zero, EV_TO_KCAL
  use abfqmmm_module, only: abfqmmm_param

  implicit none

  logical, intent(in) :: master
  logical, intent(in) :: calc_mchg_scf
  integer, intent(in) :: natom
  _REAL_ , intent(in) :: born_radii(natom), one_born_radii(natom)
  _REAL_ , intent(in) :: coords(natom*3)
  _REAL_ , intent(in) :: scaled_mm_charges(natom)
  type(qmmm_nml_type)         , intent(inout) :: qmmm_nml
  type(qmmm_struct_type)      , intent(inout) :: qmmm_struct
  type(qmmm_scratch_structure), intent(inout) :: qmmm_scratch
  _REAL_ , intent(inout) :: scf_mchg(qmmm_struct%nquant_nlink)
  _REAL_ , intent(out) :: escf

  integer :: i, j, iqmp
  logical, save :: first_call = .true.

  ! Setup is needed if this is the first call to get_qm2_forces().  The
  ! first_call condition is turned off, but back on again if abfqmmm is
  ! active.  This ensures that the setup is always done for every step
  ! of abfqmmm.
  if (first_call) then
    first_call = .false.
    if (abfqmmm_param%abfqmmm == 1) first_call = .true.
    call timer_start(TIME_QMMMSETUP)

    ! Load semiempirical parameters.  This Also does a lot of memory
    ! allocation and pre-calculates all the STO-6G orbital expansions.
#ifdef API
    call qm2_load_params_and_allocate(.true.)
#else
    call qm2_load_params_and_allocate(.false.)
#endif
    call timer_stop(TIME_QMMMSETUP)

#ifndef API
    if (master) then

      ! Print a summary about memory usage.  All memory allocation
      ! must have happened by now if the numbers reported to the
      ! mdout file are to be accurate.
      call qm_print_dyn_mem(natom,qmmm_struct%qm_mm_pairs)

      ! Finally print the result header that was skipped in sander.
      if (abfqmmm_param%abfqmmm /= 1 .or. &
          (abfqmmm_param%qmstep == 1 .and. abfqmmm_param%system == 1)) then
        write(6,'(/80("-")/"   4.  RESULTS",/80("-")/)')
      end if
    end if
#endif
  end if
  ! End of the first_call conditional

  ! Parallel: Calculate RIJ and many related equations here.
  ! Necessary memory allocation is done inside the routine.
  ! Store the results in memory to save time later.
  call timer_start(TIME_QMMMRIJEQNS)
  call qm2_calc_rij_and_eqns(qmmm_struct%qm_coords, &
                             qmmm_struct%nquant_nlink,qmmm_struct%qm_xcrd, &
                             natom, qmmm_struct%qm_mm_pairs)
  call timer_stop_start(TIME_QMMMRIJEQNS,TIME_QMMMENERGY)

  ! Parallel: Calculate SCF Energy
  call qm2_energy(escf, scf_mchg, natom, born_radii, one_born_radii, &
                  coords, scaled_mm_charges)
  call timer_stop(TIME_QMMMENERGY)

  ! Calculate forces due to QM atoms
  call timer_start(TIME_QMMMFQM)
  qmmm_struct%dxyzqm=zero
  if (qmmm_nml%qmtheory%DFTB) then

    ! Partially Parallel
    call qm2_dftb_get_qm_forces(qmmm_struct%dxyzqm)
  else

    ! Parallel
    call qm2_get_qm_forces(qmmm_struct%dxyzqm)
  end if

  call timer_stop(TIME_QMMMFQM)

  call timer_start(TIME_QMMMFQMMM)

  if (qmmm_nml%qmmm_int > 0 .and. (qmmm_nml%qmmm_int /= 5) ) then
    qmmm_struct%dxyzcl=zero
    if (qmmm_nml%qmtheory%DFTB) then

      ! Parallel
      iqmp = qmmm_struct%qm_mm_pairs
      call qm2_dftb_get_qmmm_forces(qmmm_struct%dxyzcl, &
             qmmm_struct%dxyzqm, qmmm_scratch%qm_real_scratch, &
             qmmm_scratch%qm_real_scratch(natom+1:natom+iqmp), &
             qmmm_scratch%qm_real_scratch(2*natom+1:2*natom+iqmp), &
             qmmm_scratch%qm_real_scratch(3*natom+1:3*natom+iqmp))
    else

      ! Parallel
      if (qmmm_nml%qmmm_switch) then

        ! Calculate Mulliken charges that will be
        ! used in the QM/MM switching function
        do i=1,qmmm_struct%nquant_nlink
          call qm2_calc_mulliken(i,scf_mchg(i))
        end do
      end if
      call qm2_get_qmmm_forces(qmmm_struct%dxyzqm, qmmm_struct%qm_xcrd, &
                               qmmm_struct%dxyzcl, scf_mchg)
    end if
  end if

  call timer_stop(TIME_QMMMFQMMM)

  ! Calculate Mulliken charges for later printing, if needed.
  ! Note: at present we calculate the mulliken charges even if
  ! we don't need them for printing since one might want to use
  ! them for dipole calculations etc.  It is not very expensive
  ! to calculate them, even on every MD step.
  if (master) then
    if (qmmm_nml%qmtheory%DFTB .or. calc_mchg_scf .or. &
        qmmm_nml%qm_ewald == 2 .or. qmmm_nml%qmmm_switch ) then

      ! Mulliken charges have already been calculated and stored.
      ! Do nothing.
    else
      do i = 1,qmmm_struct%nquant_nlink
        call qm2_calc_mulliken(i, scf_mchg(i))
      end do
    end if
  end if

  ! Print some extra information if verbosity level is > 0
  if (master) then
    call qm2_print_energy(qmmm_nml%verbosity, qmmm_nml%qmtheory, &
                          escf, qmmm_struct)
    if (qmmm_nml%verbosity > 3) then
      if (qmmm_nml%qm_ewald>0) then
        write (6, '("QMMM: QM - MM Coulombic Interaction = ", f18.8, &
               &" eV (", f18.8," KCal/mol)")') qmmm_struct%coulombic_eng, &
               qmmm_struct%coulombic_eng * EV_TO_KCAL
      end if
      if (qmmm_opnq%useOPNQ) then
        write (6, '("QMMM: QM - MM OPNQ correction = ", f18.8, " eV (", &
               &f18.8, " KCal/mol)")') qmmm_opnq%OPNQCorrection, &
               qmmm_opnq%OPNQCorrection*EV_TO_KCAL
        write (6, '("QMMM: QM - MM vdW correction = ", f18.8, " eV (", f18.8, &
               &" KCal/mol)")') qmmm_opnq%vdWCorrection, &
               qmmm_opnq%vdWCorrection*EV_TO_KCAL
        write (6,'("QMMM: QM - MM Total OPNQ correction = ", f18.8, " eV (", &
               &f18.8," KCal/mol)")') &
               qmmm_opnq%vdWcorrection + qmmm_opnq%OPNQCorrection, &
               (qmmm_opnq%vdWCorrection + qmmm_opnq%OPNQCorrection)*EV_TO_KCAL
      end if
      write (6,'("QMMM:")')
      write (6,'("QMMM: Forces on QM atoms from SCF calculation")')
      write (6,'("QMMM: Atm ",i6,": ",3f20.14)') (j, qmmm_struct%dxyzqm(1,j), &
             qmmm_struct%dxyzqm(2,j), qmmm_struct%dxyzqm(3,j), j=1, &
             qmmm_struct%nquant_nlink)
      if (qmmm_nml%verbosity > 4) then

        !Also print info in KJ/mol
        write (6,'("QMMM:")')
        write (6,'("QMMM: Forces on QM atoms from SCF calculation (KJ/mol)")')
        write (6,'("QMMM: Atm ",i6,": ",3f20.14)') &
               (j, qmmm_struct%dxyzqm(1,j)*4.184d0, &
               qmmm_struct%dxyzqm(2,j)*4.184d0, &
               qmmm_struct%dxyzqm(3,j)*4.184d0, j=1, qmmm_struct%nquant_nlink)
      end if
    end if
  end if

  return

  ! Provide an entry into this subroutine so that "first_call" can be reset
  entry get_qm2_forces_reset
  first_call = .true.

end subroutine get_qm2_forces
