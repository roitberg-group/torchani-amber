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

The main supported way to build and install the torchani interface is using `CMake`, and
calling it from inside a `conda` (or `mamba`) environment. The necessary steps are
described next. Different build+install procedures are not tested as of now.

Note that a GCC version that supports C++17 is needed
to compile torchani-amber (typically > 9 is enough, it is tested with 11.4).

```bash
# (0) Clone this repo and cd into it
git clone --recurse-submodules git@github.com:roitberg-group/torchani-amber.git
cd torchani-amber

# (1) Create a new conda (or mamba) environment
# This environment will contain:
# - torchani's required dependencies, including pytorch
# - CUDA Toolkit and cuDNN libraries necessary to build the extensions and interface
conda env create --file ./environment.yaml

# (2) Activate the environment
conda activate ani-amber

# (4) Install the python torchani submodule, together with the cuaev extension
cd ./submodules/torchani_sandbox
pip install --no-deps --no-build-isolation --config-settings=--global-option=ext -v -e .
cd ../..

# (5) Build and install libtorchani using the cmake.sh script
# ADVANCED: If you want to perform your custom modifications to the build, check the
# script and/or CMakeLists.txt file before running
bash ./run-cmake.sh

# (6) You may have to add ~/.local/bin to your PATH, if it isn't already
# there, since by default torchani is installed into ~/.local
mkdir -P ~/.local/lib
# If using bash, for example, run:
cat LD_LIBRARY_PATH="$HOME/.local/lib:$LD_LIBRARY_PATH" >> ~/.bashrc

# (7) compile Amber using cmake (Amber will automatically find torchani and link it
# to its MD engines, Sander and Pmemd)
conda install -c conda-forge gfortran_linux-64=11.4 openmpi=4.1.5  # Sander, Pmemd deps
# Follow instructions in https://ambermd.org/
```

TODO: Expand this section

By default the installation script runs the tests

## About the conda environment

You don't need to run things inside the `ani` environment since the built artifacts have
the path to the needed libraries baked in, but *don't remove it*, since this will remove
the installed libraries from your system

## About CUDA and LD_LIBRARY_PATH

TODO: Double check this.

TorchANI-Amber is tested with a specific version of the CUDA Toolkit. It is recommended
that the CUDA Toolkit be installed using *conda* (or *mamba*). When installing the library,
however, the path's to the correct CUDA Toolkit's linked libraries may get overriden if
the system has a different Toolkit available and LD_LIBRARY_PATH is set to point there
(as the CUDA installation instructions unfortunately recommend).

This should not cause problems in principle, since the libraries will supposedly only be
overriden if compatible, but if this is problematic to you, it is recommended to wrap
torchani in a script that removes the system's cuda libraries from LD_LIBRARY_PATH.

Note that this situation is pretty rare, most probably you will not experience any
issues regarding this.

## LibTorch / PyTorch version compatibility

Its important that the TorchANI models used are compiled in python with JIT using the
same PyTorch version as the LibTorch version used to run the model in C++. For example,
if `torch.__version__ == 2.5` for TorchANI, then LibTorch must also be 2.5, otherwise
LibTorch may fail to load the models, or load it incorrectly.

This is a non-issue if you use the installation script, since the same binaries are used
for both LibTorch and PyTorch in that case.

## CPU-only support

TorchANI-Amber can run CPU-only, but even in this case it depends on the the CUDA
Toolkit, cuDNN and LibTorch. There are no plans to address this limitation.

## Usage

TODO: Write something more comprehensive.

Familiarity with Pmemd and/or Sander is assumed in what follows.

To use the potential with Amber you must include three different namelists:
- First the usual `&cntrl` namelist, which *must have* the flag `iextpot = 1`, together
  with the usual simulation configuration.
- Second, the `&extpot` namelist, which consists only on the value `extprog =
  'torchani'`.
- Third, the `&ani` namelist, which has the actual `TorchANI` configuration.

The `&ani` namelist has the following options:
- `model_type` (string)
   The neural network to choose. Possible values are `"ani1x"`, `"ani1ccx"`, `"ani2x"`,
   `anidr`, and `"custom"`. For usage of "custom" see section *Support for custom
   models*. Default is `"ani1x"`.
- `use_double_precision` (*bool*)
   Determines whether the network runs using float64 parameters. Defaults to `.true.` We
   recommend to use double precision for accurate dynamics.
- `use_cuda_device` (*bool*)
   Flag that determines if the network runs in a CUDA device or in CPU. Default is
   `.true.`. If the flag is set to true and a CUDA device can't be found TorchANI-Amber
   will exit with an error. CUDA provides a very significant performance boost over CPU.

There are also some advanced options:
- `use_cuaev` (bool)
   Whether to use the cuAEV cuda extension to accellerate TorchANI potentials.
- `use_external_neighborlist` (bool)
   Whether to let Sander / Pmemd handle the neighborlist calculation.
- `use_torch_cell_list` (bool)
   Whether to use the Torch `CellList` to accellerate TorchANI's internal neighborlist
   calculation.
- `model_index` (int)
   Used to select a specific model (0-indexed) from a model ensemble.
   The default is to use the whole ensemble (set to -1). It is highly recommended that
   you do not set this flag unless you know exactly what you are doing. Using an
   ensemble of models provides a significantly higher accuracy than using one model
   only.
- `cuda_device_index` (int)
   The index of the CUDA enabled GPU. If `use_cuda_device` is `.false.` this flag should
   be set to `-1`. If `use_cuda_device` is `.true.` then it can be set to a positive
   integer. By default it is set to `0`. It only makes sense to change this flag if
   you can access more than one CUDA enabled GPU in your machine.

An example `mdin` input file with the correct format follows:

```
&cntrl
    iextpot = 1 ! Required to run full-ML TorchANI-Amber
    imin = 0
    nstlim = 10000
    dt = 0.001
    ntf = 2
    ntc = 2
    temp0 = 300.0
    cut = 8.0
    ntt = 3
    gamma_ln = 2.0
    ...  ! Add the rest of the Sander options here
/

&extpot
    extprog = "torchani"  ! Required to run full-ML TorchANI-Amber
/

&ani
    model_type = "ani2x"
    use_double_precision = .true.
    use_cuda_device = .true.
    use_cuaev = .true.
    ...  ! Add the rest of the TorchANI-Amber config options here
/
```

## Usage of the interface for ML/MM

TorchANI can run in an ML/MM context. Sander is required for this, Pmemd is not
supported. In order to run ani with ML/MM, the `&qmmm` namelist must be included
*instead of* the `&extpot` namelist. Additionally, `extpot = 1` should not be specified.
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
   Note that this option can only be specified if a charge-predicting model is selected.
   Currently the only available `model_type` that supports it is `animbis`. Partial
   charges for the QM atoms will be predicted by TorchANI (MBIS atomic charges at the
   `wB97X/def2-TZVPP` level of theory, *in vacuo*, for ANI-mbis) in each step. This
   charges are geometry-dependent, and the derivativse w.r.t. coordinates are used to
   calculate their contribution to the forces.
- `mlmm_coupling` (int)
   Note that this option can only be specified if `qmmm_int = 2`.
   Currently available options are: `0` (*coulombic coupling*) and `1` (*simple
   polarizable* coupling). We recommend using the *simple polarizable* coupling if
   predicting **with use_torchani_charges=.true.**, and the *coulombic coupling* if predicting
   **with use_torchani_charges=.false.** (which implies fixed topology charges).

This means there are two main setups we recommend. The first one is the *simple
polarizable* coupling using ANI-predicted MBIS charges:

```raw
&cntrl
    ifqnt=1  ! Required for ML/MM TorchANI-Amber dynamics
    ...
/

&qmmm
    qmmm_int = 2,  ! Let TorchANI-Amber handle the ML/MM coupling
    qmmask = ':1',  ! Select the first molecule as the QM-region
    qm_theory = 'EXTERN'  ! Required
    qm_ewald = 0  ! Required
    qmshake = 0  ! Recommended
    qmcut = 15.0  ! Recommended
/

&ani
    model_type = 'animbis'  ! Charge-predicting model. Currently available: 'animbis'
    use_torchani_charges = .true.
    mlmm_coupling = 1  ! Simple polarizable coupling
    ...  ! Add the rest of the TorchANI-Amber config options here
/
```

The second is using *coulombic* coupling, with fixed topology charges (i.e. mechanical
embedding, ME). Here we can choose to either let TorchANI handle the interaction
(Doesn't take PBC into account TODO:???), or allow Sander to handle it

An example of TorchANI-Amber handling the ME:

```raw
&cntrl
    ifqnt=1  ! Required for ML/MM TorchANI-Amber dynamics
    ...
/

&qmmm
    qmmm_int = 2  ! Let TorchANI-Amber handle the ML/MM coupling
    qmmask = ':1'  ! Select the first molecule as the QM-region
    qm_theory = 'EXTERN'  ! Required
    qm_ewald = 0  ! Required
    qmshake = 0  ! Recommended
    qmcut = 15.0  ! Recommended
/

&ani
    model_type = 'ani2x'  ! Select any model
    use_torchani_charges = .false.  ! Required
    mlmm_coupling = 0  ! Coulombic coupling
    ...  ! Add the rest of the TorchANI-Amber config options here
/
```

And of Sander handling the ME:

```raw
&cntrl
    ifqnt=1  ! Required for ML/MM TorchANI-Amber dynamics
    ...
/

&qmmm
    qmmm_int = 5  ! Let Sander handle the ML/MM coupling (Coulombic coupling)
    qmmask = ':1'  ! Select the first molecule as the QM-region
    qm_theory = 'EXTERN'  ! Required
    qm_ewald = 0  ! Required
    qmshake = 0  ! Recommended
    qmcut = 15.0  ! Recommended
/

&ani
    model_type = 'ani2x'  ! Select any model
    use_torchani_charges = .false.  ! Required
    ...  ! Add the rest of the TorchANI-Amber config options here
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
