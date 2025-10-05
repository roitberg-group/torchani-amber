<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/roitberg-group/torchani-amber/main/aniamber-logo-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/roitberg-group/torchani-amber/main/aniamber-logo-light.png">
  <img alt="TorchANI + Amber logo" src="https://raw.githubusercontent.com/roitberg-group/torchani-amber/main/aniamber-logo-light.png">
</picture>
</div>

Interface enabling molecular dynamics or minimizations with ANI-style NN-IPs (neural
network interatomic potentials) *and other general NN-IPs*, in the Amber software suite.
Energies and forces are calculated with the deep NNs at each step of the simulation,
and propagated by the MD engine.
Different modes are available, allowing for both "full ML" simulations, and simulations
where intermolecular non-bonded interactions are calculated by the force field. ML/MM
simulations are also possible. Both `sander` and `pmemd` are supported.

Built-in models are:
- [ANI-1x](https://aip.scitation.org/doi/10.1063/1.5023802)
    (wB97X) Supports H, C, N, O elements. No charged systems
- [ANI-2x](https://pubs.acs.org/doi/10.1021/acs.jctc.0c00121)
    (wB97X) Supports H, C, N, O, S, F, Cl elements. No charged systems
- [ANI-1ccx](https://www.nature.com/articles/s41467-019-10827-4)
    (wB97X) Supports H, C, N, O, S, F, Cl elements. No charged systems

The modified TorchANI 2.0 models:
- [ANI-2xr](https://chemrxiv.org/engage/chemrxiv/article-details/6890d92523be8e43d6b9bbba)
  (wB97X) Supports H, C, N, O, S, F, Cl elements. No charged systems. Includes
  repulsive interactions and smooth PES
- [ANI-2dr](https://chemrxiv.org/engage/chemrxiv/article-details/6890d92523be8e43d6b9bbba)
    (B973c) Supports H, C, N, O, S, F, Cl elements. No charged systems. Includes
    repulsive interactions, D3, and smooth PES

Other NN-IPs supported out of the box:
- [AimNet 2](https://pubs.rsc.org/en/content/articlelanding/2025/sc/d4sc08572h)
    Both wB97M-D3BJ and B973c are supported, with and without DSF.
- [Nutmeg](https://pubs.rsc.org/en/content/articlelanding/2025/sc/d4sc08572h)
    Small, medium and large are supported (wB97M-D3BJ)

Including your custom NN-IP is simple if you follow the TorchANI 2.0 API

## Installing from source

Useful links:

- [Download the AmberTools source distribution](https://ambermd.org/AmberTools.php)
- [Download the NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
- [Miniconda and Mamba releases](https://conda-forge.org/miniforge/)
- [Official PyTorch installation instructions](https://pytorch.org/get-started/locally/)

The main supported way to build and install the TorchANI-Amber interface is with
`cmake`, from within a `conda` (or `mamba`) environment. The necessary steps are
described next. Other procedures may work, but are untested. GCC >= 12.2 is required.

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
    pip install --no-deps --no-build-isolation --config-settings=--global-option=ext -v -e ./submodules/torchani_sandbox_pub
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
        ...
        -DCMAKE_PREFIX_PATH=$HOME/.local/
        -DCOMPILER=GNU
        ...
    ```
<!-- Is it a problem to use amber's miniconda / python? maybe not? -->

IMPORTANT: If you compile `sander` or `pmemd` with TorchANI-Amber enabled, the
`sander|pmemd` binaries *will depend on the torchani libraries being present to run
correctly*. This is true even when running CPU-only calculations, or calculations that
don't use torchani at all.

When building Amber make sure the install prefix for `TorchANI-Amber` is in the `cmake`
search path. If `TorchANI-Amber` was installed to `~/.local/lib` (the default, which
doesn't need `sudo`). you may need to add `CMAKE_PREFIX_PATH=${HOME}/.local/`. If this
is correctly done, then `Torchani` will show up in the list of enabled software that
Amber prints when installing.

## CPU-only support

TorchANI-Amber can run CPU-only, but even in this case it depends on the the CUDA
Toolkit, cuDNN and LibTorch.

## Usage

Familiarity with Amber, Pmemd and/or Sander is assumed in what follows.

To use TorchANI-Amber to run full-ML simulations, you need three different
namelists in your input file:
- The usual `&cntrl` namelist, which *must have* the flag `iextpot = 1`, together
  with the usual simulation configuration.
- Second, the `&extpot` namelist, with the only setting `extprog = 'TORCHANI'`.
- Third, the `&ani` namelist, with the actual `TorchANI` configuration.

The `&ani` namelist supports the following basic options:
- `model_type` (*string*)
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
- `use_cuaev` (*bool*)
   Whether to use the cuAEV cuda extension to accelerate potentials that support it.
- `use_amber_neighborlist` (*bool*)
   Whether to let Sander | Pmemd handle the neighborlist calculation.
- `model_index` (*int*)
   Select a specific model (0-indexed) from a model ensemble. The default is to use the
   whole ensemble (set to -1). We recommend you do *not* set this flag unless you know
   exactly what you are doing. Using an ensemble provides a significantly higher
   accuracy than using a single model.
- `cuda_device_index` (*int*)
   The index of the CUDA enabled GPU. If `use_cuda_device` is `.true.` then it can be
   set to a (0-indexed) device integer. By default it is set to `0`. It only makes sense
   to change this flag if you can access more than one CUDA enabled GPU in your machine.

An example `mdin` input file with the correct format:

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
with them. For more information consult the [ML/MM developer
notes](/README_MLMM_DEV.md).

## Limitations

The following are not yet supported, although it is possible that some of these
limitations will be lifted in the future:

- Implicit solvent Generalized Born dynamics. Only `igb = 0` (PBC) and `igb = 6` (no
  PBC, vacuum) are supported.
- Constant pH and constant redox potential dynamics
- Thermodynamic integration (TI)
- External electric fields
- Berendsen barostat for NPT dynamics.

## Support for custom models

Custom models are supported by passing a full path to the jit-compiled file to
`model_type`. Custom models have the following limitations:

The easiest way to fullfil requirements needed for usage of custom models is for your
model to be an instance of `ANI` or `ANIq`. This is flexible, since they are highly
customizable.

Alternatively, subclassing `ANI` or `ANIq` and overriding `compute_from_neighbors(...)`
is also supported. This is more complex. Consult the `TorchANI 2.0` source code for a
reference implementation of `compute_from_neighbors`. `use_cuaev` and `network_index`
may not be supported in this case, depending on your model.

*ADVANCED*: The exact requirements are as follows. If the model outputs atomic energies,
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

*EXPERIMENTAL*: If you want to use the 'switching' feature, the model should correctly
respect the `ensemble_values` contract. Energies and atomic charges must have
an extra dim prepended in this case, which indexes the models in the network.

## Tests

To run the `Amber` integration tests do `pytest -v ./tests/test_sander.py` (a working Sander
binary is assumed to be on `PATH`). This will run CPU and CUDA tests for the ML/MM
and Full-ML Amber integrations.

## Notes on LibTorch (C++) and PyTorch (Python) compatibility

**NOTE**: This is a non-issue if you use the normal installation instructions,
but it may be an issue if you use your own custom models.

Its important that the models used are JIT-compiled using the same PyTorch
version as the LibTorch version linked to the libraries. For example, if
`torch.__version__ == 2.8` when compiling JIT'ed models, then the linked LibTorch must
also be 2.8, otherwise LibTorch may fail to load the models, or load them incorrectly.
