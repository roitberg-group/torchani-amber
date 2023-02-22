! MODIFIED BY TORCHANI-AMBER-INTERFACE
!*******************************************************************************
!
! Module: external_mod
!
! Description: This module houses the capabilities of calling external libraries
!              to compute energies and forces, instead of using the force field ones.
!
!              Adapted here by Vinicius Wilian D. Cruzeiro and Delaram Ghoreishi
!               
!*******************************************************************************

! Considerations for development with C/C++ interface
! The unit cell vectors are the columns of `ucell`,
! as determined by the following pmemd code in `pbc.F90`:
! 
!     ucell(1, 1) = a
!     ucell(2, 1) = 0.d0
!     ucell(3, 1) = 0.d0
!     ucell(1, 2) = b * cos(factor * gamma)
!     ucell(2, 2) = b * sin(factor * gamma)
!     ucell(3, 2) = 0.d0
!     ucell(1, 3) = c * cos(factor * beta)
!     ucell(2, 3) =  (b * c * cos(factor * alpha) - ucell(1, 3) * &
!                    ucell(1, 2))/ucell(2, 2)
!     ucell(3, 3) = sqrt(c * c - ucell(1, 3) * ucell(1, 3) - ucell(2, 3) * &
!                   ucell(2, 3))
! 
! This means that inside the C++ interface and due to the row major ordering
! of C unit cell vectors will be rows. As usuall, forces and coordinates
! are in Fortran:
!     forces(3, N)
!     coordinates(3, N)
! 
! In C/C++ they are transposed:
!     forces[N, 3]
!     coordinates[N, 3]


module external_mod

  ! extprog, json, nn_path, nn_cnt, use_cuda_device, cuda_device_index,
  ! use_double_precision, model_type, model_index
  use external_dat_mod 

  IMPLICIT NONE

  ! The active namelist:

  private       :: extpot

  namelist /extpot/      extprog, &
                         json, &
                         nn_path, &
                         nn_cnt, &
                         use_cuda_device, &
                         use_cell_list, &
                         cuda_device_index, &
                         use_double_precision, &
                         model_type, &
                         model_index

  contains

  subroutine external_init()

    use prmtop_dat_mod, only :  natom, & ! number of atoms 
                                nres,  & ! number of residues
                                gbl_labres, & ! global labels of residues
                                gbl_res_atms, & ! global residues of atoms
                                atm_mass, & ! atomic masses
                                atm_igraph, & 
                                atm_atomicnumber, & ! atomic numbers
                                loaded_atm_atomicnumber ! whether the atomic numbers were loaded by prmtop
    use file_io_dat_mod, only : mdout, &  ! MD output file
                                mdin  ! MD input file
    use pmemd_lib_mod, only : get_atomic_number ! function to obtain atomic numbers
    use file_io_mod, only : nmlsrc  ! funcion for namelist search
    use pmemd_lib_mod, only : mexit ! exit the pmemd program "gracefully"

    IMPLICIT NONE
    
    ! general 
    integer :: atomicnumber(natom) ! The atomic number of every atom
    integer :: ifind ! only for namelist

    ! mbx 
    integer :: i ! for loops
    double precision :: coord(3*natom)
    integer :: cnt
    character(len=5), allocatable :: monomers(:)
    integer, allocatable :: nats(:)
    character(len=5), allocatable :: at_name(:)
    integer :: nmon
    integer :: val
    logical :: isvalid ! whether atomic numbers are consistent with atom types

    ! torchani
    integer :: model_type_for_c
  
    extprog = ''

    ! mbx
    json = ''

    ! unused, old ANI-Amber
    nn_path = ''
    nn_cnt = 0

    ! torchani
    use_cuda_device = .false.
    use_double_precision = .true.
    cuda_device_index = -1
    model_type = 'ani1x' 
    model_type_for_c = 0
    model_index = -1
    use_cell_list = .false.

    ! Here the namelist is read from the MD input file and all 
    ! variables written to the MD input file that have to be retrieved by 
    ! extpot are read
    ! this avoids nasty error messages if the namelist is not found by 
    ! the raw read command
    rewind(mdin)                  ! Insurance against maintenance mods.
    call nmlsrc('extpot', mdin, ifind)
    if (ifind .ne. 0) then        ! Namelist found. Read it:
      read(mdin, nml = extpot)
    else                          ! No namelist was present,
      write(mdout, '(a)') 'ERROR: Could not find the namelist extpot in the mdin file'
      call mexit(mdout, 1)
    end if
    ! torchani defaults, if cuda_device_index is not set, then it is set to
    ! device 0 for cuda and "default cpu", which is -1 for cpu
    if (cuda_device_index == -1 .and. use_cuda_device) cuda_device_index = 0
    if (cuda_device_index == -1 .and. .not. use_cuda_device) cuda_device_index = -1
    
    ! Here if prmtop has already loaded the atomic numbers then 
    ! atomicnumber is directly set to the loaded values by prmtop
    ! otherwise the atomic numbers are loaded with the get_atomic_number function
    do i = 1, natom
      if(loaded_atm_atomicnumber) then
          atomicnumber(i) = atm_atomicnumber(i)
      else
        call get_atomic_number(atm_igraph(i), atm_mass(i), atomicnumber(i))
      end if
    end do
    ! afterwards a program specific section follows, depending on the 
    ! name of the external program different setup routines are performed
    ! these routines are performed only once, when master_setup is called

    if (extprog .eq. 'mbx') then
#ifdef MBX
      nmon = nres

      allocate(nats(nmon),monomers(nmon),at_name(natom))

      cnt = 1
      do i=1,nmon
         isvalid = .True.
         if (gbl_labres(i) == "WAT") then
            val = 3
            monomers(i) = "h2o"
            at_name(cnt+0) = "O"
            if (atomicnumber(cnt+0) .ne. 8) isvalid = .False.
            at_name(cnt+1) = "H"
            if (atomicnumber(cnt+1) .ne. 1) isvalid = .False.
            at_name(cnt+2) = "H"
            if (atomicnumber(cnt+2) .ne. 1) isvalid = .False.
            cnt = cnt + val
         else if (gbl_labres(i) == "N2O") then
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
            write(mdout, '(a,a,a)') 'ERROR: The residue ',gbl_labres(i),' is not recognized by MBX!'
            call mexit(mdout, 1)
         end if
         if (val == gbl_res_atms(i+1)-gbl_res_atms(i)) then
            nats(i) = val
         else
            write(mdout, '(a,a,a)') 'ERROR: The number of atoms in residue ',gbl_labres(i),' does not match the expected by MBX!'
            call mexit(mdout, 1)
         end if
         if (.not. isvalid) then
            write(mdout, '(a,a,a)') 'ERROR: The order or type of the atoms in residue ',gbl_labres(i),&
                                    ' does not match the expected by MBX!'
            call mexit(mdout, 1)
         end if
      end do

      do i =1, natom
        at_name(i)=trim(at_name(i))//CHAR(0)
      end do
      do i=1,nmon
        monomers(i) = trim(monomers(i))//CHAR(0)
      end do

      if (json .ne. '') then
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
      write(mdout, '(a,a,a)') 'ERROR: External program ',trim(extprog)//CHAR(0),&
             ' is not valid! Please set a valid value in the extprog flag'
      call mexit(mdout, 1)
    end if

  end subroutine

! subroutine for generalized born forces
subroutine gb_external(crd, frc, pot_ene)
  use prmtop_dat_mod,    only :  natom
  implicit none

  ! general 
  double precision           ::  crd(:,:) ! intent in, the coordinates, 3 x N
  double precision           ::  frc(:,:) ! forces, intent out , 3 x N
  double precision           ::  pot_ene ! potential energy, intent out

  ! for mbx
  integer                    ::  i
  double precision           ::  coord(3*natom), grads(3*natom)
  
if (extprog == "torchani") then
#ifdef TORCHANI_
    call torchani_energy_force(crd, natom, frc, pot_ene)
#endif /* TORCHANI_ */
  else if (extprog == 'mbx') then
#ifdef MBX
    do i = 1, natom
      coord(3*(i-1)+1) = crd(1,i)
      coord(3*(i-1)+2) = crd(2,i)
      coord(3*(i-1)+3) = crd(3,i)
    end do
    call get_energy_g(coord, natom, pot_ene, grads)
    do i = 1, natom
      frc(1,i) = -grads(3*(i-1)+1)
      frc(2,i) = -grads(3*(i-1)+2)
      frc(3,i) = -grads(3*(i-1)+3)
    end do
#endif
end if

end subroutine

! subroutine for particle mesh ewald forces
subroutine pme_external(crd, frc, pot_ene)
  use prmtop_dat_mod,    only :  natom
  use pbc_mod,           only :  ucell, recip
  implicit none

  ! general  
  double precision           ::  crd(:,:) ! intent in, the coordinates, 3 x N
  double precision           ::  frc(:,:) ! forces, intent out , 3 x N
  double precision           ::  pot_ene ! potential energy, intent out

  ! for mbx
  integer                    ::  i, j
  double precision           ::  coord(3*natom), grads(3*natom), box(9)
  
if (extprog == "torchani") then
#ifdef TORCHANI_
    call torchani_energy_force_pbc(crd, natom, frc, ucell, pot_ene)
#endif /* TORCHANI_ */
else if (extprog .eq. 'mbx') then
#ifdef MBX
    do i = 1, natom
      coord(3*(i-1)+1) = crd(1,i)
      coord(3*(i-1)+2) = crd(2,i)
      coord(3*(i-1)+3) = crd(3,i)
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
      frc(1,i) = -grads(3*(i-1)+1)
      frc(2,i) = -grads(3*(i-1)+2)
      frc(3,i) = -grads(3*(i-1)+3)
    end do
#endif
end if

end subroutine

end module external_mod
