# TorchANI-Amber interface

This interface works with the AMBER suite of programs (specifically pmemd and
sander). It allows energies and forces to be calculated with a deep neural
network from the ANI family of Behler-Parrinello style NNPs in every step of a
molecular dynamics simulation or minimization. Forces and energies are then
sent to Amber and used to propagate the positions. Currently
([ANI-1x](https://aip.scitation.org/doi/10.1063/1.5023802), ANI-2x (preprint)
and [ANI-1ccx](https://www.nature.com/articles/s41467-019-10827-4) ensembles
(each composed of 8 networks) and their respective individual networks are
available to use as calculators. ANI-1x and ANI-1ccx support HCNO elements.
ANI-2x supports in addition F, Cl and S.

Useful links:

- [Download the AmberTools source distribution](https://ambermd.org/AmberTools.php)
- [Download the NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
- [Download NVIDIA cuDNN](https://developer.nvidia.com/cudnn)
- [Anaconda](https://www.anaconda.com/products/individual)
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- [Official PyTorch site](https://pytorch.org/get-started/locally/)

## Installing

The main supported way to build and install TorchANI-Amber interface is by using
`cmake`, which you should call from within a `conda` (or `mamba`) environment. The
necessary steps are described next. Other procedures may work, but are untested.

A GCC version that supports C++17 is needed to compile TorchANI-Amber (typically > 9 is
enough, it is tested with 11.4). A tested GCC version is included in the
`environment.yaml` file and installed by default.

1. Clone this repo and cd into it
    ```bash
    git clone --recurse-submodules git@github.com:roitberg-group/torchani-amber.git
    cd torchani-amber
    ```
2. Create a new `conda` (or `mamba`) environment and activate it. The `environment.yaml`
    file has a correct environment, tested to work correctly with TorchANI-Amber. It
    contains:
    - TorchANI's required dependencies, including PyTorch
    - CUDA Toolkit and cuDNN libraries necessary to build the extensions and interface
    - GFortran and OpenMPI, which are needed to compile Sander and Pmemd (serial and MPI)
    ```bash
    conda env create --file ./environment.yaml
    conda activate ani-amber
    ```
3. Install TorchANI (python), together with its compiled extensions
    ```bash
    pip install --no-deps --no-build-isolation --config-settings=--global-option=ext -v -e ./submodules/torchani_sandbox
    ```
4. Build and install TorchANI-Amber using the `run-cmake` script
    *ADVANCED:* If you want to perform your custom modifications to the build, this is
    the moment to do it. Check `run-cmake` and the `CMakeLists.txt` for more info.
    By default the installation script runs the tests (which are fast), you can avoid
    this by using the `-T` flag. For more options do `run-cmake -h`.
    ```bash
    ./run-cmake
    ```
    After this is done, you can safely deactivate the environment, it is no longer
    needed. However, *don't remove it*. The compiled binaries will depend on the cuda
    and torch dynamic libraries in the env to run correctly.
    ```bash
    conda deactivate ani-amber
    ```
5. Compile Amber from source. Amber will automatically find TorchANI-Amber and link it
   to both `pmemd` and `sander`. You can refer to [the amber
   website](https://ambermd.org/) for info on how to obtain and install Amber. You can
   use this cmake configuration as a template to generate the buildsystem.
    ```bash
    cmake \
        -S./amber-src-dir/ \
        -B./amber-build-dir/ \
        -DCMAKE_INSTALL_PREFIX=/amber-install-dir/ \
        -DCMAKE_PREFIX_PATH=$HOME/.local/ \
        -DCOMPILER=GNU
    ```
<!-- Is it a problem to use amber's miniconda / python? maybe not? -->

IMPORTANT: If you compile Sander or Pmemd with TorchANI-Amber enabled, the `sander|pmemd`
binaries *will depend on the torchani libraries being present to run correctly*. This
is true even when running CPU-only calculations, or calculations that don't use torchani
at all.

## Details on building Amber

When building Amber make sure that:

- You are *not* using `conda`'s compilers to build `Amber` (the ones provided by default
  in the env). Currently there are some incompatiblities with internal Amber libraries,
  which means some `sander` and `pmemd` features will not work if you do this (e.g.
  netCDF output).
- The install prefix for `TorchANI-Amber` is in the `cmake` search path.
  If `TorchANI-Amber` was installed to `~/.local/lib` (the default, which doesn't need `sudo`).
  you may need to add `CMAKE_PREFIX_PATH=${HOME}/.local/`. If this is correctly done,
  then `Torchani` will show up in the list of enabled software that Amber prints
  when installing.

## About CUDA and LD_LIBRARY_PATH

TODO: Double check this.

TorchANI-Amber is tested with a specific version of the CUDA Toolkit. It is recommended
that the CUDA Toolkit be installed using *conda* (or *mamba*) as in the provided
installation instructions. When installing the library or running the executables,
however, the paths to the correct CUDA Toolkit's linked libraries may be overriden if
the system has a different Toolkit available.

This should not cause problems in principle, since the libraries will only be overriden
*if they are compatible*, if you want to avoid this you can remove the CUDA Toolkit libs
from `LD_LIBRARY_PATH` before building the library or running Sander or Pmemd.

Note that this situation is pretty rare, most probably you will not experience any
issues regarding this.

## LibTorch (C++) and PyTorch (Python) compatibility

NOTE: This is a non-issue if you use the installation script, since the same binaries
are used for both LibTorch and PyTorch. You can safely skip this section
if that is the case.

Its important that the TorchANI models used are JIT-compiled using the same PyTorch
version as the LibTorch version linked to the libraries. For example, if
`torch.__version__ == 2.5` for TorchANI, then the linked LibTorch must also be 2.5,
otherwise LibTorch may fail to load the models, or load it incorrectly.

## CPU-only support

TorchANI-Amber can run CPU-only, but even in this case it depends on the the CUDA
Toolkit, cuDNN and LibTorch.

## Usage

Familiarity with Amber, Pmemd and/or Sander is assumed in what follows.

To use TorchANI-Amber to run full-ML simulations you must include three different
namelists:
- First the usual `&cntrl` namelist, which *must have* the flag `iextpot = 1`, together
  with the usual simulation configuration.
- Second, the `&extpot` namelist, with the only setting `extprog = 'TORCHANI'`.
- Third, the `&ani` namelist, which has the actual `TorchANI` configuration.

The `&ani` namelist has the following basic options:
- `model_type` (string)
   The neural network to choose. Possible values are `"ani1x"`, `"ani1ccx"`, `"ani2x"`,
   `anidr`. For usage of custom models see section *Support for custom
   models*. Default is `"ani1x"` (case sensitive).
- `use_double_precision` (*bool*)
   Determines whether the network runs using float64 parameters. Defaults to `.true.` We
   recommend this setting for accurate dynamics.
- `use_cuda_device` (*bool*)
   Determines whether the network runs in a CUDA enabled GPU. Default is `.true.`. If
   the flag is set to `.true.` and a CUDA enabled GPU can't be found, TorchANI-Amber
   will exit with an error. CUDA acceleration provides a very significant performance
   boost over CPU.

There are also some advanced options:
- `use_cuaev` (bool)
   Whether to use the cuAEV cuda extension to accelerate potentials that support it.
- `use_amber_neighborlist` (bool)
   Whether to let Sander | Pmemd handle the neighborlist calculation.
- `use_torch_cell_list` (bool)
   Whether to use the TorchANI `CellList` to accelerate internal neighborlist
   calculations.
- `model_index` (int)
   Select a specific model (0-indexed) from a model ensemble. The default is to use the
   whole ensemble (set to -1). We recommend you do *not* set this flag unless you know
   exactly what you are doing. Using an ensemble provides a significantly higher
   accuracy than using a single model.
- `cuda_device_index` (int)
   The index of the CUDA enabled GPU. If `use_cuda_device` is `.true.` then it can be
   set to a (0-indexed) device integer. By default it is set to `0`. It only makes sense
   to change this flag if you can access more than one CUDA enabled GPU in your machine.

An example `mdin` input file with the correct format follows:

```
&cntrl
    iextpot = 1 ! Required to run full-ML TorchANI-Amber
    ! ... Add the rest of the Sander options here
/
&extpot
    extprog = "TORCHANI"  ! Required to run full-ML TorchANI-Amber
/
&ani
    model_type = "ani2x"
    use_double_precision = .true.
    use_cuda_device = .true.
    use_cuaev = .true.
    ! ... Add the rest of the TorchANI-Amber config options here
/
```

## Usage of the interface for ML/MM

TorchANI-Amber is also integrated with the QM/MM Sander subsystem, which means you can
perform ML/MM simulations with it. Sander is *required* for this, Pmemd is not
supported.

If you want to run this kind of dynamics, **instead** of setting `iextpot = 1` and
including the `&extpot` namelist, you should set `ifqnt = 1`, and include the `&qmmm`
namelist.

Many options can be used in the `&qmmm` namelist, but `qmmm_int = 1` (default),
`qm_ewald = 0` (default), and `qmmm_theory = 'EXTERN'` (must be specified) are *required*
to run ML/MM simulations with TorchANI-Amber.

The `&ani` namelist remains the same, with the following extra available options:

Output related options:
- `write_xyz` (bool)
   Dump xyz coordinates of QM region as a `.xyz` file
- `write_forces` (bool)
   Dump forces acting on QM atoms as a `.dat` file
- `write_charges` (bool)
   Dump charges of the QM atoms as a `.dat` file
- `write_charges_grad` (bool)
   Dump charge derivatives w.r.t. coords of the QM atoms as a `.dat` file

ML/MM and electrostatic related options:
- `use_torchani_charges` (bool)
   **This option can only be specified if a charge-predicting model is selected.**
   Currently the only available `model_type` that supports it is `animbis`. Partial
   charges for the QM atoms will be predicted by TorchANI (MBIS atomic charges at the
   `wB97X/def2-TZVPP` level of theory, *in vacuo*, for ANI-mbis) in each step. This
   charges are geometry-dependent, and the derivativse w.r.t. coordinates are used to
   calculate their contribution to the forces.
- `mlmm_coupling` (int = `0` or `1`)
   Currently available are: `0` (*coulombic coupling*) and `1` (*simple polarizable*
   coupling).

We recommend using one of the following two settings:
- *mlmm_coupling = 1* and *use_torchani_charges=.true.* (variable nn-predicted charges)
- *mlmm_coupling = 0* and *use_torchani_charges=.false.* (fixed topology charges)

TODO: Check what the defaults are for qm_ewald and qm_mask

A template for the first setting (simple polarizable with variable charges) is:

```raw
&cntrl
    ifqnt = 1  ! Required for all ML/MM TorchANI-Amber dynamics
    ! ... Add extra simulation settings here
/
&qmmm
    qm_theory = 'EXTERN'  ! Required for all ML/MM TorchANI-Amber dynamics
    qm_ewald = 0  ! Required for Sander EXTERN QM/MM
    qmmm_int = 1  ! Required, let TorchANI-Amber handle the ML/MM coupling
    qmmask = ':1',  ! Select the first molecule as the QM-region
    qmcut = 15.0  ! Recommended
/
&ani
    model_type = 'animbis'  ! Charge-predicting model. Currently available: 'animbis'
    use_torchani_charges = .true.  ! Use geometry dependent, nn-predicted charges
    mlmm_coupling = 1  ! Simple polarizable coupling
    ! ... Add the rest of the TorchANI-Amber config options here
/
```

An example of the second (coulombic with fixed charges, i.e. mechanical embedding):

```raw
&cntrl
    ifqnt = 1  ! Required for all ML/MM TorchANI-Amber dynamics
    ! ... Add extra simulation settings here
/
&qmmm
    qm_theory = 'EXTERN'  ! Required for all ML/MM TorchANI-Amber dynamics
    qm_ewald = 0  ! Required for Sander EXTERN QM/MM
    qmmm_int = 1  ! Required, let TorchANI-Amber handle the ML/MM coupling
    qmmask = ':1'  ! Select the first molecule as the QM-region
    qmcut = 15.0  ! Recommended
/
&ani
    model_type = 'ani2x'  ! Select any model
    use_torchani_charges = .false.  ! Use fixed topology charges
    mlmm_coupling = 0  ! Coulombic coupling
    ! ... Add the rest of the TorchANI-Amber config options here
/
```

Many experimental options are also available for ML/MM. Please don't use experimental
options unless you are a developer or you know exactly what you are doing, they are not
extensively tested and we make no guarantees or claims regarding the results obtained
with them. For more information consult the developer *dev notes on ML/MM*.

## Limitations

The following are not yet supported:

- Generalized Born dynamics
- Constant pH and constant redox potential dynamics
- Thermodynamic integration (TI)
- External electric fields
- Berendsen barostat for NPT dynamics.

It is possible that some of these limitations will be lifted in the future. The only
compatible flags for `igb` in the Amber `&ctrl` namelist are `igb = 0` (PBC, vacuum) and
`igb = 6` (no PBC, vacuum).

## Support for custom models

Custom models are supported by passing a full path to the jit-compiled file to
`model_type`. Custom models have the following limitations:

The easiest way to fullfil requirements needed for usage of custom models is for your
model to be an instance of `ANI` or `ANIq`. This already has quite a high flexibility,
since they are highly customizable.

Alternatively, subclassing `ANI` or `ANIq` and overriding `compute_from_neighbors(...)`
is possible if you need more freedom. This is more complex however. Consult the
`TorchANI 2.0` source code if you want to see a reference implementation of
`compute_from_neighbors`. Be warned, `use_cuaev` and `network_index` may not be
supported in this case, depending on your model.

ADVANCED: The exact requirements are as follows. If the model outputs atomic energies,
`forward` must have the following signature:

```python
def forward(
    self,
    species_coords: tuple[Tensor, Tensor]
    cell: Tensor | None,
    pbc: Tensor | None,
    charge: int,
    atomic: bool,  # Controls atomic energy decomposition
    ensemble_values: bool,  # Controls whether the model outputs ensemble values
) -> tuple[Tensor, Tensor, Tensor]:
    # Where the output is a tuple:
    #     - species (shape: [1, atoms]), energies, atomic_numbers
    #     - energies (shape: [1,])
    #     - atomic_charges (shape: [1, atoms])
    # For more information about the signature consult the `TorchANI 2.0` docs
    # and source code.
    ...
    return species, energies, atomic_charges
```

If the model doesn't support atomic charges, the signature is the same, but with
`tuple[Tensor, Tensor]` instead, omitting `atomic_charges`.

If you want to use the internal Amber neighborlists, your should additionally
support the following method:

```python
@torch.jit.export
def compute_from_external_neighbors(
    self,
    species: Tensor,
    coords: Tensor,
    neighbor_idxs: Tensor,  # External neighbors
    shifts: Tensor,  # External shifts that have to be applied to wrap PBC
    charge: int = 0,
    atomic: bool = False,
    ensemble_values: bool = False,
) -> tuple[Tensor, Tensor | None]:
    # Where the output is a tuple:
    #     - energies (shape: [1,])
    #     - atomic_charges (shape: [1, atoms]) (or None)
    ...
    return energies, atomic_charges
```

EXPERIMENTAL: If you want to use the 'switching' feature, the model should correctly
respect the `ensemble_values` contract. Energies and atomic charges must have
an extra dim prepended in this case, which indexes the models in the network.

## Amber integration tests

To run the `Amber` integration tests do `pytest -v ./tests/test_sander.py` (a working Sander
binary is assumed to be on `PATH`). This will run CPU and CUDA tests for the ML/MM
and Full-ML Amber integrations.

## Dev notes on ML/MM

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
- `use_numerical_qmmm_ofrces = .false.` (bool)
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
