! MODIFIED BY TORCHANI-AMBER-INTERFACE
!*******************************************************************************
!
! Module: external_module
!
! Description: This module houses the capabilities of calling external libraries
!              to compute energies and forces, instead of using the force field ones.
!
!              Adapted here by Vinicius Wilian D. Cruzeiro and Delaram Ghoreishi
!
!
!*******************************************************************************

#include "copyright.h"
#include "../include/dprec.fh"
#include "../include/assert.fh"

module external_module

  private

  ! general
  character(256), public :: extprog

  ! mbx
  character(256), public :: json

  ! torchani
  logical, public :: use_double_precision
  logical, public :: use_cuda_device
  logical, public :: use_cell_list
  integer, public :: cuda_device_index
  character(256), public :: model_type
  integer, public :: model_index

  ! The active namelist:

  private       :: extpot

  namelist /extpot/      extprog, json, use_cuda_device, cuda_device_index, use_double_precision, &
                         model_type, model_index, use_cell_list

  public external_init, gb_external, pme_external

  contains

  subroutine external_init(ix,ih,xx)

    use memory_module,   only : natom, m04, nres, m02, i02, i100, lmass
    use qmmm_module,     only : get_atomic_number

    IMPLICIT NONE

    integer :: i, ifind, cnt
    _REAL_ :: coord(3*natom)
    character(len=5), allocatable :: monomers(:)
    integer, allocatable :: nats(:)
    character(len=5), allocatable :: at_name(:)
    integer :: nmon, val
    integer :: ix(*)
    character(len=4) :: ih(*)
    _REAL_ :: xx(*)
    integer :: atomicnumber(natom)
    logical :: isvalid, errFlag
    integer :: model_type_for_c
    
    extprog = ''

    ! mbx
    json = ''

    ! torchani
    use_cuda_device = .false.
    use_double_precision = .true.
    cuda_device_index = -1
    model_type_for_c = 0
    model_type = 'ani1x'
    model_index = -1
    use_cell_list = .false.

    ! Read input in namelist format:

    rewind(5)                  ! Insurance against maintenance mods.

    call nmlsrc('extpot', 5, ifind)

    if (ifind .ne. 0) then        ! Namelist found. Read it:
      read(5, nml = extpot)
    else                          ! No namelist was present,
      write(6, '(a)') 'ERROR: Could not find the namelist extpot in the mdin file'
      call mexit(6, 1)
    end if
    ! torchani defaults, if cuda_device_index is not set, then it is set to
    ! device 0 for cuda and "default cpu", which is -1 for cpu
    if (cuda_device_index == -1 .and. use_cuda_device) cuda_device_index = 0
    if (cuda_device_index == -1 .and. .not. use_cuda_device) cuda_device_index = -1

    do i=1, natom
      if(ix(i100) .eq. 1) then
        atomicnumber(i) = ix(i100+i)
      else
        call get_atomic_number(ih(m04+i-1), xx(lmass+i-1), atomicnumber(i), errFlag)
      end if
    end do

    ! For MBX
    if (extprog .eq. 'mbx') then
#ifdef MBX
      nmon = nres

      allocate(nats(nmon),monomers(nmon),at_name(natom))

      cnt = 1
      do i=1,nmon
         isvalid = .True.
         if (ih(m02+i-1) == "WAT") then
            val = 3
            monomers(i) = "h2o"
            at_name(cnt+0) = "O"
            if (atomicnumber(cnt+0) .ne. 8) isvalid = .False.
            at_name(cnt+1) = "H"
            if (atomicnumber(cnt+1) .ne. 1) isvalid = .False.
            at_name(cnt+2) = "H"
            if (atomicnumber(cnt+2) .ne. 1) isvalid = .False.
            cnt = cnt + val
         else if (ih(m02+i-1) == "N2O") then
            val = 7
            monomers(i) = "n2o5"
            at_name(cnt+0) = "O"
            if (atomicnumber(cnt+0) .ne. 8) isvalid = .False.
            at_name(cnt+1) = "N"
            if (atomicnumber(cnt+1) .ne. 7) isvalid = .False.
            at_name(cnt+2) = "N"
            if (atomicnumber(cnt+2) .ne. 7) isvalid = .False.
            at_name(cnt+3) = "O"
            if (atomicnumber(cnt+3) .ne. 8) isvalid = .False.
            at_name(cnt+4) = "O"
            if (atomicnumber(cnt+4) .ne. 8) isvalid = .False.
            at_name(cnt+5) = "O"
            if (atomicnumber(cnt+5) .ne. 8) isvalid = .False.
            at_name(cnt+6) = "O"
            if (atomicnumber(cnt+6) .ne. 8) isvalid = .False.
            cnt = cnt + val
         else
            write(6, '(a,a,a)') 'ERROR: The residue ',ih(m02+i-1),' is not recognized by MBX!'
            call mexit(6, 1)
         end if
         if (val == ix(i02+i)-ix(i02+i-1)) then
            nats(i) = val
         else
            write(6, '(a,a,a)') 'ERROR: The number of atoms in residue ',ih(m02+i-1),' does not match the expected by MBX!'
            call mexit(6, 1)
         end if
         if (.not. isvalid) then
            write(6, '(a,a,a)') 'ERROR: The order or type of the atoms in residue ',ih(m02+i-1),&
                                    ' does not match the expected by MBX!'
            call mexit(6, 1)
         end if
      end do

      do i =1, natom
        at_name(i)=trim(at_name(i))//CHAR(0)
      end do
      do i=1,nmon
        monomers(i) = trim(monomers(i))//CHAR(0)
      end do

      if (json /= '') then
        call initialize_system(coord, nats, at_name, monomers, nmon, trim(json)//CHAR(0))
      else
        call initialize_system(coord, nats, at_name, monomers, nmon)
      end if
#endif
    else if (extprog == "torchani") then
#ifdef TORCHANI_
      if (model_type == 'custom') then
        model_type_for_c = -1
      else if (model_type == 'ani1x') then
        model_type_for_c = 0
      else if (model_type == 'ani1ccx') then
        model_type_for_c = 1
      else if (model_type == 'ani2x') then
        model_type_for_c = 2
      else
        write(6, '(a,a,a)') 'ERROR: Model type of', trim(model_type), &
          'not recognized, available models are "custom", "ani1x", "ani1ccx" and "ani2x"'
        call mexit(6, 1)
      end if 
      call torchani_init_atom_types(atomicnumber, natom, use_cuda_device, &
                                    cuda_device_index, use_double_precision, &
                                    model_type_for_c, model_index, use_cell_list)
#endif /* TORCHANI_ */
    else
      write(6, '(a,a,a)') 'ERROR: External program ',trim(extprog),&
             ' is not valid! Please set a valid value in the extprog flag'
      call mexit(6, 1)
    end if

  end subroutine

  subroutine gb_external(crd, frc, pot_ene)

    use memory_module,   only : natom

    IMPLICIT NONE

    _REAL_   ::  crd(*)
    _REAL_   ::  frc(*)
    _REAL_   ::  pot_ene
    integer  ::  i, i3
    _REAL_   ::  coord(3*natom), grads(3*natom)
    ! Torchani uses double precision variables, 
    ! otherwise there are alignment issues with tensors
    ! so compiling sander with single precision will lead to errors!!

    if (extprog == 'torchani') then
#ifdef TORCHANI_
      call torchani_energy_force(crd, natom, frc, pot_ene)
#endif /* TORCHANI_ */
    ! For MBX
    else if (extprog .eq. 'mbx') then
#ifdef MBX
      do i = 1, natom
        i3=3*(i-1)
        coord(3*(i-1)+1) = crd(i3+1)
        coord(3*(i-1)+2) = crd(i3+2)
        coord(3*(i-1)+3) = crd(i3+3)
      end do

      call get_energy_g(coord, natom, pot_ene, grads)

      do i = 1, natom
        i3=3*(i-1)
        frc(i3+1) = frc(i3+1) - grads(3*(i-1)+1)
        frc(i3+2) = frc(i3+2) - grads(3*(i-1)+2)
        frc(i3+3) = frc(i3+3) - grads(3*(i-1)+3)
      end do
#endif
    end if

  end subroutine

  subroutine pme_external(crd, frc, pot_ene)

    use memory_module, only : natom, i70, ix
    use nblist, only: ucell
    IMPLICIT NONE
#ifdef TORCHANI_
  ! necessary for nspm
#include "../include/md.h"
!  ! necessary for i70
!#include "../include/memory.h"
#endif


    double precision           ::  crd(*)
    double precision           ::  frc(*)
    double precision           ::  pot_ene
    integer                    ::  i, i3
    double precision           ::  coord(3*natom), grads(3*natom), box(9)

    if (extprog == 'torchani') then
#ifdef TORCHANI_
      call torchani_energy_force_pbc(crd, natom, frc, ucell, pot_ene)
#endif
    ! For MBX
    else if (extprog .eq. 'mbx') then
#ifdef MBX
      do i = 1, natom
        i3=3*(i-1)
        coord(3*(i-1)+1) = crd(i3+1)
        coord(3*(i-1)+2) = crd(i3+2)
        coord(3*(i-1)+3) = crd(i3+3)
      end do

      box(1) = ucell(1,1)
      box(2) = ucell(2,1)
      box(3) = ucell(3,1)
      box(4) = ucell(1,2)
      box(5) = ucell(2,2)
      box(6) = ucell(3,2)
      box(7) = ucell(1,3)
      box(8) = ucell(2,3)
      box(9) = ucell(3,3)

      call get_energy_pbc_g(coord, natom, box, pot_ene, grads)

      do i = 1, natom
        i3=3*(i-1)
        frc(i3+1) = frc(i3+1) - grads(3*(i-1)+1)
        frc(i3+2) = frc(i3+2) - grads(3*(i-1)+2)
        frc(i3+3) = frc(i3+3) - grads(3*(i-1)+3)
      end do
#endif
    end if

  end subroutine

end module external_module
