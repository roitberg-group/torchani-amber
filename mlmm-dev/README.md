# Manual for the usage of TORCHANI as a QM engine in AMBER (ML/MM)

First, the variable `ifqnt` in the &cntrl namelist must be set to 1 to turn on
the qmmm calculation, qm_theory must be set to `extern`, and the &ani namelist
must be included bellow the &qmmm namelist.

qmmm_int = 5 o 2 are available to use with the interface. If set to 2, the
interface will deal with the qmmm coupling. If set to 5, the coupling will be
plain mechanical embedding, as implemented in amber, using the forcefield
charges for the QM region (they are read from the topolofy file). Note that if
`qmmm_int` is set to 2 and the external coupling is turned off, as well as the
polarization of the qm region (see below), the coupling will also be mechanical
embedding using the charges from the forcefield. However, in this case it will
be calculated by the interface, which does not use periodic boundary
conditions, and it is not extensively tested (will do so when charges
prediction by ani are implemented).

Available keywords for the &ani namelist in AMBER mdin (with their defaults)

      model_type ='ani1x'
      model_index = -1
      use_double_precision = .true.
      use_cuda_device = .true.
      cuda_device_index = 0
      use_torchani_charges = .false.
      charges_model_type = 'ani1x-mbis'
      charges_model_index = -1
      polarize_qm_charges = .true.
      distort_qm_energy = .true.
      distortion_k = 0.5d0
      use_extcoupling = .false.
      extcoupling_program = 'none'
      use_numerical_qmmm_forces = .false.
      write_xyz = .false.
      write_forces = .false.
      write_charges = .false.
      use_charges_derivatives = .true.
      switching_program = 'none'
      use_switching_function = .false.
      switching_property = 'forces'
      qlow  = 0.2
      qhigh = 0.3
      pol_H = 3.08
Description:


- `model_type` (string)
  The neural network to choose. Currently possible values are `"ani1x"`,
  `"ani1ccx"`, `"ani2x"`,`"animbis"`  and `"custom"`. For use of "custom" see
  section *Support for custom models*. Default is `"ani1x"`.

- `model_index` (int)
  Used to select a specific model from a model ensemble. Since the ANI-1x,
  ANI-2x and ANI-1ccx models are an ensemble of 8 networks each, this flag must
  be between 0 and 7 if used (models are 0-indexed). The default is to use the
  whole ensemble (set to -1). It is highly recommended that you do not set this
  flag unless you know exactly what you are doing. Using an ensemble of models
  provides a significantly higher accuracy than using one model only.

- `use_double_precision` (bool)
  Flag that determines if the network runs in double or single precision
  Default is `.true.` (Its recommended to use double precision for dynamics).

- `use_cuda_device` (bool)
  Flag that determines if the network runs in a CUDA device or in CPU
  Default is `.true.`. If the flag is set to true and a CUDA device can't be
  found torchani will exit with an error. CUDA provides a very significant
  performance boost over CPU.

- `cuda_device_index` (int)
  The index of the cuda device. If `use_cuda_device` is `.false.` this flag
  should be omitted (or set to -1). If `use_cuda_device` is `.true.` then it
  has to be set to a positive integer, and it is set by default to 0. Only set
  it to a different GPU index if the CPU can access more than one CUDA device.

- `use_torchani_charges` (bool)
   Only available if If `model_type` is set to `animbis`. Partial charges for
   the QM atoms will be predicted by torchani (wB97X/tzvp/MBIS level, in
   vacuo), in each step, and the interface uses the forcefield charges to
   calculate the eelectrostatic coupling between the QM and MM regions, at the
   mechanical embedding level.

- `use_charges_derivatives` (bool)
   Only available if If `model_type` is set to `animbis` and
   `use_torchani_charges` is `.true.`. It consideres the predicted charges
   dependence on atomic coordinates for forces calculation. Makes the code a
   bit slower for large systems, but it is still recommended to set it `true`.

- `polarize_qm_charges`(bool)
   Only used if `qmmm_int` is set to 2. A polarization correction is included
   to account the effect of the MM region.

- `distort_qm_energy`(bool)
   Only used if `qmmm_int` is set to 2. The in vacuo QM energy is corrected
   with a term called distortion energy, that is assumed to be proportional and
   of opposite sign to the polarization energy.

- `distortion_k`(double precision)
   Only used if `distort_qm_energy` is `.true.`. The proporcionality constant
   fot the distortion correction.

- `write_xyz` (bool)
   Writes xyz coordinates of QM region.

- `write_forces` (bool)
   Writes forces acting on QM atoms.

- `write_charges` (bool)
   Only used if `use_torchani_charges` is `.true.`. Writes partial charges
   predicted by torchani.

- `pol_H` (double precision)
  Only used  if `polarize_qm` is `.true.`. It allowes the user to manually
  indicate the atomic polarizability desired for Hydrogen atoms. The same can
  be done for all atoms of the second period (pol_H, pol_C, pol_N, etc.)

NOT EXTENSIVELY TESTED:

- `use_switching_function` (bool)
  If set to `true`, torchani estimates how similar the prediction between the
  different models is. If it is too high, the interface starts mixing the
  energy estimated by torchani with that of an external software (as if it were
  switching to a different potential energy surface).

- `switching_program` (string)
  The program used for switching. Currenlty available options are `orca` and
  `lio`. A new namelist must be included below the &ani namelist. The
  corresponding name must be `switching_orc` or `switching_lio`, and the
  keywords available are the same as if those programs were used as qm engines.

- `switching_property` (string)
  The mixing of the potential energy surfaces can be done in terms of forces or
  energies. More details will be added soon, but for now use `forces`.

- `qlow` and `qhigh` (double precision)
  These values correspond to the smoothing function used to switch (or mix) the
  potential energy surface.

- `use_extcoupling` (bool)
  Uses external software (external to TORCHANI) to calculate the QM/MM
  interaction.

- `extcoupling_program` (string)
  The program used for external coupling. Available options are 'amber-dftb'
  (uses builtin DFTB code from AMBER), 'orca' and 'lio'. For lio and orca one
  needs to include a 'extcoupling_orc' or 'extcoupling_lio'.

## Usage

An example `mdin` input file could be to run an NVT MD with ani as a qmmm
engine in amber, using the forcefield charges do compute the coupling term:


```
 &cntrl
    imin=0,
    ntx=5, nmropt=0,
    ntwr=100,ntpr=10,ntwx=100,ioutfm=1,ntxo=1,
    nstlim=5000,dt=0.001,
    ntt=3,tempi=300.0,temp0=300.0,gamma_ln=5.0,
    ntp=0,
    ntb=1,
    ntf=1,ntc=2,
    cut=10.0,
    ifqnt=1,
  &end
/
 &qmmm
  qm_theory='EXTERN',
  qmmask=':1',
  qmmm_int = 5,
  qmshake=0,
  qm_ewald=0,
  qmcut=15.0,
/
 &ani
  model_type='ani2x',
  use_cuda_device= .true. ,
/
```

Same but using charges predicted with TORCHANI and a polarizable mechanical embedding:


```
 &cntrl
    imin=0,
    ntx=5, nmropt=0,
    ntwr=100,ntpr=10,ntwx=100,ioutfm=1,ntxo=1,
    nstlim=5000,dt=0.001,
    ntt=3,tempi=300.0,temp0=300.0,gamma_ln=5.0,
    ntp=0,
    ntb=1,
    ntf=1,ntc=2,
    cut=10.0,
    ifqnt=1,
  &end
/
 &qmmm
  qm_theory='EXTERN',
  qmmask=':1',
  qmmm_int = 2,
  qmshake=0,
  qm_ewald=0,
  qmcut=15.0,
/
 &ani
  model_type = 'animbis',
  use_cuda_device= .true. ,
  use_torchani_charges = .true. ,
/
```

Similar but using DFTB for coupling (+ correction to the QM term):

```
 &cntrl
  imin=0,
  ntx=5, nmropt=0,
  ntwr=100,ntpr=10,ntwx=100,ioutfm=1,ntxo=1,
  nstlim=5000,dt=0.001,
  ntt=3,tempi=300.0,temp0=300.0,gamma_ln=5.0,
  ntp=0,
  ntb=1,
  ntf=1,ntc=2,
  cut=10.0,
  ifqnt=1,
 &end
/
 &qmmm
  qm_theory='EXTERN',
  qmmask=':1',
  qmmm_int = 2,
  qmshake=0,
  qm_ewald=0,
  qmcut=15.0,
/
 &ani
  use_cuda_device= .true. ,
  extcoupling_program='amber-dftb',
  use_extcoupling = .true.,
/
```
