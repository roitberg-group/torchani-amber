! MODIFIED BY TORCHANI-AMBER-INTERFACE
! Another module needed to avoid cyclic dependencies

module external_dat_mod

  implicit none

  integer, parameter            :: max_fn_len = 256

  public
  ! general 
  character(max_fn_len), save :: extprog

  ! mbx
  character(max_fn_len), save :: json

  ! unused, old ANI-Amber
  character(max_fn_len), save :: nn_path
  integer, save               :: nn_cnt
  
  ! torchani
  logical, save :: use_double_precision
  logical, save :: use_cuda_device 
  logical, save :: use_cell_list
  integer, save :: cuda_device_index
  character(max_fn_len), save :: model_type
  integer, save :: model_index

end module external_dat_mod
