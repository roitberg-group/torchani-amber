#include "../include/dprec.fh"
module qm2_extern_module
! ----------------------------------------------------------------
! Interface for QM and QM/MM MD with SANDER and external programs
!
! Currently supports:
! QM           with ADF and GAMESS
! QM and QM/MM with TeraChem, Gaussian and Orca
!
! Initial implementation for ADF by
! Matthew Clark and Prithvi Undavalli
! (SDSC LSSI summer high school students)
! under supervision of
! Andreas Goetz and Ross Walker (SDSC)
! 
! Date: August 2010
!
! Extensions by Andreas Goetz (agoetz@sdsc.edu)
! Date: November 2010 and later
!
! MRCC interface by Bence Hegely
! Date: February 2017
!
! ----------------------------------------------------------------

  use qmmm_module, only: qmmm_nml
  use qm2_extern_adf_module   , only: get_adf_forces
  use qm2_extern_gms_module   , only: get_gms_forces
  use qm2_extern_tc_module    , only: get_tc_forces
  use qm2_extern_gau_module   , only: get_gau_forces
  use qm2_extern_orc_module   , only: get_orc_forces
  use qm2_extern_nw_module    , only: get_nw_forces
  use qm2_extern_mrcc_module  , only: get_mrcc_forces
  use qm2_extern_quick_module , only: get_quick_forces
  use qm2_extern_reaxff_puremd_module , only: get_reaxff_puremd_forces
#ifdef MPI
  use qm2_extern_genmpi_module, only: get_genmpi_forces
#endif
  use qm2_extern_qc_module    , only: get_qc_forces
#ifdef LIO
  use qm2_extern_lio_module   , only: get_lio_forces 
#endif 
#ifdef FIREBALL
  use qm2_extern_fb_module   , only: get_fb_forces
#endif
#ifdef TORCHANI
  use qm2_extern_torchani_module, only: get_torchani_forces
#endif
  implicit none

  private
  public :: qm2_extern_get_qm_forces, qm2_extern_finalize, qm2_extern_using_ani
  
  contains

  subroutine qm2_extern_get_qm_forces(nstep, nqmatoms, qmcoords, qmtypes, &
       qmcharges, nclatoms, clcoords, mmtypes, escf, dxyzqm, dxyzcl, id)

    use file_io_dat
#if defined(MPI)
#   include "parallel.h"
#endif /* MPI */
#include "../include/md.h" /* We are only using this for values of nstlim and maxcyc */

    integer, intent(in) :: nstep
    integer, intent(in) :: nqmatoms             ! Number QM of atoms
    _REAL_,  intent(in) :: qmcoords(3,nqmatoms) ! QM atom coordinates
    integer, intent(in) :: qmtypes(nqmatoms)    ! QM atom types (nuclear charge in au)
    _REAL_,  intent(inout) :: qmcharges(nqmatoms) ! QM charges
    integer, intent(in) :: nclatoms             ! Number of MM atoms
    _REAL_,  intent(in) :: clcoords(4,nclatoms) ! MM atom coordinates and charges in au
    integer, intent(in) :: mmtypes(nclatoms)    ! MM atom types
    _REAL_, intent(out) :: escf                 ! SCF energy
    _REAL_, intent(out) :: dxyzqm(3,nqmatoms)   ! SCF QM gradient
    _REAL_, intent(out) :: dxyzcl(3,nclatoms)   ! SCF MM gradient

    ! List of supported external programs; will search for namelists in this order
    character(len=20), save :: extern_program
    character(len=3), intent(in) :: id
    logical, save :: first_call = .true.
    logical, save :: do_gradient = .true.

#if defined(MPI)
    ! In parallel runs, only one thread needs to call the EXTERN program
    escf = 0.0d0
    dxyzqm = 0.0d0
    dxyzcl = 0.0d0
    if ( mytaskid /= 0 ) return
   
    if(first_call) then
      ! If we are doing an mpi run, print out a warning on nesting MPI executables
      write(6,'(a)') "| !!!!!!!!!!!!!!!!!!!!!!!!!   WARNING   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
      write(6,'(a)') "| Not all MPI implementations fully support system calls that execute"
      write(6,'(a)') "| MPI parallel programs. In this case it will not be possible to run "
      write(6,'(a)') "| MPI parallel versions of both sander and the external QM program.  "
      write(6,'(a)') "| !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
   end if
#endif /* MPI */

    if(first_call) then
      ! If doing post-processing
      if( nstlim==0 .or. maxcyc==0 ) do_gradient = .false. 
      call select_program(extern_program)
      call check_electrostatic_embedding(nclatoms, qmmm_nml%qmmm_int, extern_program)
      call print_citation_information(extern_program)
      call print_constants()
      first_call = .false.
    end if

    ! Call chosen program
    ! Note that qmmm_nml%spin actually holds spin multiplicity!
    select case (extern_program)
    case('adf')
      call get_adf_forces(do_gradient, nstep, ntpr, id, nqmatoms, qmcoords,&
           qmtypes, escf, dxyzqm, qmmm_nml%qmcharge, qmmm_nml%spin)
    case('gau')
      call get_gau_forces(do_gradient, nstep, ntpr, id, nqmatoms, qmcoords,&
           qmtypes, nclatoms, clcoords, escf, dxyzqm, dxyzcl, &
           qmmm_nml%qmcharge, qmmm_nml%spin)
    case('gms')
      call get_gms_forces(do_gradient, nstep, ntpr, id, nqmatoms, qmcoords,&
           qmtypes, escf, dxyzqm, qmmm_nml%qmcharge, qmmm_nml%spin)
    case('tc')
      call get_tc_forces( do_gradient, nstep, ntpr, id, nqmatoms, qmcoords,&
           qmtypes, nclatoms, clcoords, escf, dxyzqm, dxyzcl, &
           qmmm_nml%qmcharge, qmmm_nml%spin)
    case('orc')
      call get_orc_forces( do_gradient, nstep, ntpr, id, nqmatoms, qmcoords,&
           qmtypes, nclatoms, clcoords, escf, dxyzqm, dxyzcl, &
           qmmm_nml%qmcharge, qmmm_nml%spin)
    case('nw')
      call get_nw_forces( do_gradient,        ntpr, id, nqmatoms, qmcoords,&
           qmtypes, nclatoms, clcoords, escf, dxyzqm, dxyzcl, &
           qmmm_nml%qmcharge, qmmm_nml%spin)
    case('qc')
      call get_qc_forces( do_gradient, nstep, ntpr, id, nqmatoms, qmcoords,&
           qmtypes, nclatoms, clcoords, escf, dxyzqm, dxyzcl, &
           qmmm_nml%qmcharge, qmmm_nml%spin)
    case('mrcc')
      call get_mrcc_forces(do_gradient, nstep, ntpr, id, nqmatoms, qmcoords,&
           qmtypes, nclatoms, clcoords, escf, dxyzqm, dxyzcl, &
           qmmm_nml%qmcharge, qmmm_nml%spin)
    case('quick')
      call get_quick_forces(do_gradient, nstep, ntpr, id, nqmatoms, qmcoords,&
           qmtypes, nclatoms, clcoords, escf, dxyzqm, dxyzcl, &
           qmmm_nml%qmcharge, qmmm_nml%spin)
    case('reaxff')
      call get_reaxff_puremd_forces(nqmatoms, qmcoords, qmtypes, qmcharges,&
           nclatoms, clcoords, mmtypes, escf, dxyzqm, &
           dxyzcl, qmmm_nml%qmcharge)
#ifdef MPI
    case('genmpi')
      call get_genmpi_forces( do_gradient, nstep, ntpr, id, nqmatoms, qmcoords,&
           qmtypes, nclatoms, clcoords, escf, dxyzqm, dxyzcl, &
           qmmm_nml%qmcharge, qmmm_nml%spin)
#endif
    case('lio')
#ifdef LIO
      call get_lio_forces( nqmatoms, qmcoords,&
           nclatoms, clcoords, escf, dxyzqm, dxyzcl)
#else
      call sander_bomb("qm2_extern_get_qm_forces", "Lio is not enabled", &
           "Check your installation or reconfigure with the -lio option.")
#endif
    case('fb')
#ifdef FIREBALL
      call get_fb_forces( do_gradient, nstep, ntpr, id, nqmatoms, qmcoords,&
           qmtypes, nclatoms, clcoords, escf, dxyzqm, dxyzcl, &
           qmmm_nml%qmcharge, qmmm_nml%spin)
#else
      call sander_bomb("qm2_extern_get_qm_forces", "Fireball is not enabled", &
           "Check your installation or reconfigure with the -fireball option.")
#endif
      case('ani')
#ifdef TORCHANI
      call get_torchani_forces( nstep, ntpr, nqmatoms, qmcoords,&
      nclatoms, clcoords, escf, dxyzqm, dxyzcl, id)
#else
      call sander_bomb(&
          "qm2_extern_get_qm_forces",&
          "TorchANI-Amber is not enabled",&
          "Sander's QM/MM subroutine found the 'ani' namelist, "&
          //" but this Sander executable lacks TorchaANI-Amber support."&
          //" Install TorchANI-Amber and recompile Sander to use it"&
      )
#endif
    case default
      call sander_bomb("qm2_extern_get_qm_forces","External namelist not found", &
           "Please check your input.")
    end select

  end subroutine qm2_extern_get_qm_forces

  logical function qm2_extern_using_ani()

    implicit none

    character(len=20) :: extern_program

    call select_program(extern_program)
    qm2_extern_using_ani = (trim(extern_program) == 'ani')

  end function qm2_extern_using_ani


  ! Select external program to use
  subroutine select_program(extern_program)

    implicit none

    integer :: i, ifind
    character(len=20) :: programs(14) = (/'ani   ', 'adf   ', 'gms   ', 'tc    ', 'gau   ', &
                                         'orc   ', 'nw    ', 'qc    ', 'genmpi', &
                                         'lio   ','mrcc  ', 'fb    ', 'quick ', &
                                         'reaxff'/)
    character(len=20), intent(out) :: extern_program

    ! Select which external program to use
    extern_program='none'
    do i=1, size(programs)
      rewind 5
      call nmlsrc(programs(i),5,ifind)
      if(ifind>0) then
        extern_program=programs(i)
        exit
      end if
    end do

#ifndef MPI
    if (trim(extern_program) == 'genmpi') &
      call sander_bomb("qm2_extern_select_program", "cannot use genmpi", &
            "You must compile with MPI to use 'genmpi' external QM")
#else
#  ifdef MPI_1
    if (trim(extern_program) == 'genmpi') &
       call sander_bomb("qm2_extern_select_program", &
            'unsupported MPI version (need MPI version 2 or larger).', &
            'Will quit now.')
#  endif
#endif /* MPI */

  end subroutine select_program


  ! Check whether program supports electronic embedding
  subroutine check_electrostatic_embedding(nclatoms, qmmm_int, extern_program)

    implicit none
    
    integer, intent(in) :: nclatoms
    integer, intent(in) :: qmmm_int
    character(len=*), intent(in) :: extern_program

    if (nclatoms > 0) then
       ! if electrostatic embedding in use
       if ( qmmm_int /= 5 ) then
          ! QM/MM with electrostatic embedding not possible with ADF, or GAMESS at present
          if ( (extern_program == 'adf') .or. (extern_program == 'gms') ) then
             call sander_bomb("qm2_extern_get_qm_forces","nquant /= natom", &
                  trim(extern_program)//" does not support QM/MM with electronic embedding")
          end if
       end if
    end if

  end subroutine check_electrostatic_embedding


  ! Print citation information
  subroutine print_citation_information(extern_program)

    implicit none

    character(len=*), intent(in) :: extern_program

    write (6, '(/a//a/a/a/a/a)') &
         '| Please also cite the following work for the use of the QM/MM interface:', &
         '| A. W. G"otz, M. A. Clark, R. C. Walker', &
         '| "An extensible interface for ab initio QM/MM molecular dynamics simulations', &
         '|  with AMBER"' , &
         '| J. Comput. Chem. 35 (2014) 95-108.', &
         '| DOI: 10.1002/jcc.23444'

    if ( trim(extern_program) == 'tc' ) then
       write (6,'(/a/a/a/a/a)') &
            '| C. M. Isborn , A. W. G"otz , M. A. Clark , R. C. Walker, T. J. Martinez', &
            '| "Electronic Absorption Spectra from MM and ab initio QM/MM Molecular Dynamics:', &
            '|  Environmental Effects on the Absorption Spectrum of Photoactive Yellow Protein"', &
            '| J. Chem. Theory Comput. 8 (2012) 5092-5106', &
            '| DOI: 10.1021/ct3006826'
    end if

    if(trim(extern_program)=='mrcc') then
     write (6,'(/a/a//a/a/a/a/a/a)') &
          '| If results obtained with the MRCC code are published,', &
          '| an appropriate citation would be', &
          '| MRCC, a quantum chemical program suite written by M. Kállay,',&
          '| Z. Rolik, J. Csontos, I. Ladjánszki, L. Szegedy, B. Ladóczki,',&
          '| G. Samu, K. Petrov, M. Farkas, P. Nagy, D. Mester, and B. Hégely.',&
          '| See also',&
          '| Z. Rolik, L. Szegedy, I. Ladjánszki, B. Ladóczki, and M. Kállay,',&
          '| J. Chem. Phys. 139, 094105 (2013), as well as: www.mrcc.hu'
    endif

    if ( trim(extern_program) == 'fb' ) then
       write (6,'(/a/a/a/a/a/a)') &
            '| Jesus I. Mendieta-Moreno, Ross C. Walker, James P. Lewis,', &
            '| Paulino Gomez-Puertas, Jesus Mendieta, and Jose Ortega', &
            '| "FIREBALL / AMBER :An Efficient Local-Orbital DFT', &
            '| QM/MM Method for Biomolecular Systems"', &
            '| J. Chem. Theory Comput., 2014, 10 (5), pp 2185–2193', &
            '| DOI: 10.1021/ct500033w'
    end if

    if ( trim(extern_program) == 'reaxff' ) then
       write (6,'(/a/a/a/a/a/a)') &
            '| Ali Rahnamoun, Mehmet Cagri Kaymak, Madushanka Manathunga,', &
            '| Andreas W. Götz, Adri C. T. van Duin, Kenneth M. Merz,', &
            '| and Hasan Metin Aktulga', &
            '| "ReaxFF/AMBER - A Framework for Hybrid Reactive/Nonreactive', &
            '| Force Field Molecular Dynamics Simulations"', &
            '| QM/MM Method for Biomolecular Systems"', &
            '| J. Chem. Theory Comput., 2020 16 (12), pp 7645-7654 ', &
            '| DOI: 10.1021/acs.jctc.0c00874'
    end if

  end subroutine print_citation_information

  ! Print information about natural constants
  subroutine print_constants()

    use constants, only: CODATA08_AU_TO_KCAL, CODATA08_A_TO_BOHRS, CODATA08_AU_TO_DEBYE
    
    implicit none

    ! print constants that are used for conversion
    write (6,'(3(/,a),/)') ' Constants for unit conversion taken from', &
                           ' Mohr, Taylor, Newell, Rev. Mod. Phys. 80 (2008) 633-730', &
                           ' and using the thermochemical calorie (1 cal = 4.184 J):'
    write (6, '(a, es19.12)') ' A_TO_BOHRS  = ', CODATA08_A_TO_BOHRS
    write (6, '(a, es17.10)') ' AU_TO_KCAL  = ', CODATA08_AU_TO_KCAL
    write (6, '(a, es15.8)')  ' AU_TO_DEBYE = ', CODATA08_AU_TO_DEBYE

  end subroutine print_constants


  ! Used for tasks that must be run after the last step
  subroutine qm2_extern_finalize()
  
    use qm2_extern_tc_module    , only: tc_finalize
#ifdef MPI
    use qm2_extern_genmpi_module, only: genmpi_finalize
#endif
    use qm2_extern_reaxff_puremd_module, only: reaxff_puremd_finalize
 
    character(len=20) :: extern_program
    call select_program(extern_program)
 
    select case (extern_program)
      case('tc')
        call tc_finalize()
#ifdef MPI
      case('genmpi')
        call genmpi_finalize()
#endif
     case('lio')
#ifdef LIO
       call lio_finalize()
#endif
     case('reaxff')
       call reaxff_puremd_finalize()
    end select
 
  end subroutine qm2_extern_finalize


end module qm2_extern_module
