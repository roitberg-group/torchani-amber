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
    By default the installation script runs the tests, you can avoid this by
    using the `-T` flag. For more options do `run-cmake -h`
    ```bash
    ./run-cmake
    ```
5. Compile Amber from source. Amber will automatically find TorchANI-Amber and link it
    to both `pmemd` and `sander`. You can refer to [the amber
    website](https://ambermd.org/) for info on how to obtain and install Amber. You can
    use this configuration as a template, which requires that you run `cmake` while the
    `ani-amber` env is activated:
    ```bash
    cmake \
        -S./path/to/your/amber/source-dir/ \
        -B./path/to/your/amber/build-dir/ \
        -DCMAKE_INSTALL_PREFIX=/path/to/your/amber/install-dir/ \
        -DCMAKE_PREFIX_PATH=$HOME/.local/lib \
        -DCOMPILER=MANUAL \
        -DCMAKE_C_COMPILER="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc" \
        -DCMAKE_CXX_COMPILER="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++" \
        -DCMAKE_Fortran_COMPILER="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gfortran" \
        -DCMAKE_CUDA_HOST_COMPILER="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++" \
        -DDOWNLOAD_MINICONDA=FALSE \
        -DBUILD_PYTHON=FALSE
    ```

## Details on building Amber

When building Amber make sure that:

- You are building using the same compilers as those used for TorchANI-Amber. This
  is done with `COMPILER=MANUAL` and `CMAKE_<lang>_COMPILER=...` in the template config,
  which instruct Amber to use the compilers in the current conda env.
- You are *not* using Amber's Miniconda or Amber's python. This is
  done with the `DOWNLOAD_MINICONDA=FALSE` and `BUILD_PYTHON=FALSE`.
- `~/.local/lib/` is in the search file for `cmake`. This is only needed if installing
  to `~/.local/lib/` (the default, which doesn't need `sudo`). If installing for example
  to `/usr/local` it is not needed. Done with `CMAKE_PREFIX_PATH=${HOME}/.local/lib`.
- Amber can find `TorchANI-Amber` (it should appear in the list of enabled
  software that Amber prints when installing).

## About the provided conda environment

There is no need to activate the `ani-amber` conda env after Amber has been built, since
the path to the needed libraries baked in, but *don't remove it*, since this will remove
the needed libraries from your system.

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
   `anidr`, and `"custom"`. For usage of "custom" see section *Support for custom
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
- `use_external_neighborlist` (bool)
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

Many options can be used in the `&qmmm` namelist, but `qmmm_int = 2` and `qmmm_theory =
'EXTERN'` are *required* to run ML/MM simulations with TorchANI-Amber.

The `&ani` namelist remains the same, with the following extra available options:

Output related options:
- `write_xyz` (bool)
   Writes xyz coordinates of QM region.
- `write_forces` (bool)
   Writes forces acting on QM atoms.
- `write_charges` (bool)
   Only used if `use_torchani_charges` is `.true.`. Writes partial charges
   predicted by the chosen model.

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
- *mlmm_coupling = 0* and *use_torchani_charges=.true.* (variable nn-predicted charges)
- *mlmm_coupling = 1* and *use_torchani_charges=.false.* (fixed topology charges)

TODO: Check what the defaults are for qm_ewald and qm_mask

A template for the first setting (simple polarizable with variable charges) is:

```raw
&cntrl
    ifqnt = 1  ! Required for all ML/MM TorchANI-Amber dynamics
    ! ... Add extra simulation settings here
/
&qmmm
    qm_theory = 'EXTERN'  ! Required for all ML/MM TorchANI-Amber dynamics
    qmmm_int = 2  ! Required, let TorchANI-Amber handle the ML/MM coupling
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
    qmmm_int = 2  ! Required, let TorchANI-Amber handle the ML/MM coupling
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
with them. For more information consult the developer [**README**](./mlmm-dev/README.md)
file.

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

TODO: Fix this section
