# Dev notes on ML/MM

Some *advanced* options are not extensively tested, or are meant to be used for dev or
debug situations only.

Some of the advanced options correspond to simulation protocols that technically should
work, but are untested. If you want to specify any of these you need to also specify
`allow_untested_protocols=.true.`. Using either `mlmm_coupling='coulombic'` with MBIS
geometry dependent charges, or `mlmm_coupling='simple-polarizable'` with fixed topology
charges is one of these cases. The other case is setting `qmmm_int` to anything
different from `1` (the default), which also requires `mlmm_coupling='none'`.

- `qmmm_int = 0` completely disregards the coupling between the MM and ML (i.e. QM)
  parts of the system, it can be used for debugging.
- `qmmm_int = 5` Makes Sander manage the MM/ML coupling as mechanical embedding. This
  may be slightly better in some situations, since ANI doesn't take into account PBC
  when calculating the ML/MM interaction. In this case the charges will *always* be the
  FF charges, as read from the topology file. Any extra options specified in the
  `&mlip` namelist, that pertain the ML/MM interaction, will not be taken into account.

In older versions of the interface, `polarize_qm_charges` and `distort_qm_energy` were
allowed options. Please use `mlmm_coupling = 'simple-polarizable'`, which will enable
**both options**, or `mlmm_coupling = 'coulombic'`, which will disable both.

The other extra available *advanced* options are, in format `<option> = <default>  (type)`:

General:
- `pol_<element-symbol>` (double)
   Atomic polarizability (in atomic units) associated with a given element, used with
   `mlmm_coupling = 'simple-polarizable'` (`pol_H = ..., pol_C = ..., ...`). Elements up
   to `Ar` have a default, the polarizabilities of other elements in the ML region must
   be specified.
- `inv_pol_dielectric = 0.5` (double)
   Inverse dielectric of the polarization correction, only used with
   `mlmm_coupling = 'simple-polarizable'`.

Experimental *hybrid ML-QM* feature:
- `use_hybrid_mlqm = .false.` (bool)
  If set to `.true.`, torchani estimates how similar the prediction between the
  different models is. If it is too high, the interface starts mixing the
  energy estimated by torchani with that of an external software (as if it were
  switching to a different potential energy surface). Requires an ensemble model.
- `hybrid_mlqm_method` (string)
  The name of the QM program. Available options are `'orca'`, `'gau'` and `'lio'`. The
  corresponding `orc`, `gau` or `lio` namelists should also be included.
- `hybrid_mlqm_switch_low = 0.2` and `hybrid_mlqm_switch_high = 0.3` (double)
  Parameters of the function used to mix the potential energy surfaces.

Experimental *QM coupling* feature:
- `use_qm_coupling = .false.` (bool)
  Dispatch a QM program as a helper to calculate the QM/MM interaction. Requires
  `qmmm_int = 1` and `mlmm_coupling = 'none'`, and can't be used together with
  `use_hybrid_mlqm`.
- `qm_coupling_method` (string)
  The name of the QM helper program. Available options are `'amber-dftb'`
  (uses builtin DFTB code in Amber), `'orca'`, `'gau'` and `'lio'`. If `'orca'`,
  `'gau'` or `'lio'` are specified, the `orc`, `gau` or `lio` namelists should also be
  included.

An example `&mlip` namelist for use with the *QM coupling* feature:

```raw
&mlip
  use_cuda_device= .true. ,
  qm_coupling_method='amber-dftb',
  use_qm_coupling =.true.,
  mlmm_coupling = 'none',
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
&mlip
    model_type='ani2x',
    use_cuda_device=.true.,
    allow_untested_protocols=.true.,  ! Required for qmmm_int=5
    mlmm_coupling='none',  ! Required for qmmm_int=5
/
```
