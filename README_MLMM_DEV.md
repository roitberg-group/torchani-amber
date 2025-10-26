# Dev notes on ML/MM

Some *advanced* options are not extensively tested, or are meant to be used for dev or
debug situations only.

Some of the advanced options correspond to simulation protocols that technically should
work, but are untested. If you want to specify any of theseyou need to also specify
`allow_untested_protocols=.true.`. Using either mlmm_embedding=0 with MBIS geometry
dependent charges, or mlmm_embedding=1 with fixed topology charges is one of these
cases. The other case is setting `qmmm_int` to anything different from `1` (the
default).

- `qmmm_int = 0` completely disregards the coupling between the MM and ML (i.e. QM)
  parts of the system, it can be used for debugging.
- `qmmm_int = 5` Makes Sander manage the MM/ML coupling as mechanical embedding. This
  may be slightly better in some situations, since ANI doesn't take into account PBC
  when calculating the ML/MM interaction. In this case the charges will *always* be the
  FF charges, as read from the topology file. Any extra options specified in the `&ani`
  namelist, that pertain the ML/MM interaction, will not be taken into account.

In older versions of the interface, `polarize_qm_charges` and `distort_qm_energy` were
allowed options. Please use `mlmm_coupling = 1`, which will enable **both options**, or
`mlmm_coupling = 0`, which will disable both. If you really want to disregard the
distortion contribution only, use both `mlmm_coupling = 1` and `distortion_k = 0.0`.

The other extra available *advanced* options are, in format `<option> = <default>  (type)`:

General:
- `use_numerical_qmmm_forces = .false.` (bool)
   Wheter to calculate the ML/MM coupling numerically.
- `use_charges_derivatives = .true.` (bool)
   Only used if `use_torchani_charges=.true.`. It consideres the predicted charges
   dependence on atomic coordinates for forces calculation. Makes the code a bit slower
   for large systems, but it is still recommended to set it `true`.
- `distortion_k = 0.4d0` (double)
   Proportionality constant for the distortion correction
- `pol_<element-symbol>` (double)
   Fixed atomic polarizability associated with a given element. Element symbols
   up to `Ne` are supported (`pol_H = ..., pol_C = ..., ...`).

Experimental *switching* feature:
- `use_switching_function` (bool)
  If set to `.true.`, torchani estimates how similar the prediction between the
  different models is. If it is too high, the interface starts mixing the
  energy estimated by torchani with that of an external software (as if it were
  switching to a different potential energy surface).
- `switching_program` (string)
  The name of the QM switching program. Available options `'orca'`, or `'lio'`.
  The corresponding `orc` or `lio` namelists should also be
  included.
- `qlow` and `qhigh` (double precision)
  Parameters of the function used to mix the potential energy surfaces.

Experimental *Extcoupling* feature:
- `use_extcoupling` (bool)
  Dispatch a QM program as a helper to calculate the QM/MM interaction.
- `extcoupling_program` (string)
  The name of the QM helper program. Available options are `'amber-dftb'`
  (uses builtin DFTB code in Amber), `'orca'` and `'lio'`. If `'lio'` or `'orca'`
  are specified, the `orc` or `lio` namelists should also be included.

An example `&ani` namelist for use with the *Extcoupling* feature:

```raw
&ani
  use_cuda_device= .true. ,
  extcoupling_program='amber-dftb',
  use_extcoupling =.true.,
/
```

A full example of an input for a simulation with `qmmm_int=5`:

```raw
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
/
&qmmm
    qm_theory='EXTERN',
    qmmask=':1',
    qmmm_int=5,  ! Advanced option
    qmshake=0,
    qm_ewald=0,
    qmcut=15.0,
/
&ani
    model_type='ani2x',
    use_cuda_device=.true.,
    allow_untested_protocols=.true.,  ! Required for qmmm_int=5
/
```
