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

## Instructions

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

## Note about the conda environment

You don't need to run things inside the `ani-amber` environment since the built
artifacts have the path to the needed libraries baked in, but *don't remove
it*, since this will remove the installed libraries from your system

## Note about CUDA and LD_LIBRARY_PATH

Torchani is tested with a specific version of the CUDA Toolkit. It is
recommended that the CUDA Toolkit be installed using conda (or mamba). When
installing the library, however, the path's to the correct CUDA Toolkit's liked
libraries may get overriden if the system has a different Toolkit available and
LD_LIBRARY_PATH is set to point there (as the CUDA installation instructions
unfortunately recommend).

This should not cause problems in principle, since the libraries will
supposedly only be overriden if compatible, but if this is problematic to you,
it is recommended to wrap torchani in a script that removes the system's cuda
libraries from LD_LIBRARY_PATH.

Note that this situation is pretty rare, most probably you will not experience
any issues regarding this.

## Requirements

- Linux operating system
- cmake 3.16 or higher
- gcc 9.3.0
- git
- catch2 (library for unit tests only, included in the distribution)
- python 3.8 (for generating the models and some test data)
- torchani (latest version, for generating the models and some test data)
- Amber24
- LibTorch 2.3

Through conda:

- PyTorch 1.13.1
- CUDA Toolkit 11.8
- cuDNN 8.3.2

## LibTorch / PyTorch compatibility

It is important that the TorchANI models is compiled with JIT using the same
PyTorch version as the LibTorch version used to run the model in C++.  For
example, if `torch.__version__ == 1.13.1` for TorchANI, then LibTorch must also
be 1.13.1, otherwise LibTorch may fail to load the model, or load it
incorrectly.

NOTE: The library can be run CPU-only, but the CUDA Toolkit, cuDNN and LibTorch
dependencies exist **even if you want to run CPU-only** this is a current
limitation that will probably remain like this, there are no plans to address
it.

## Installation

The following steps are necessary to install the library and link it correctly:

1. Extract Amber20 (don't compile it) into `AMBERHOME`. The AmberTools
    suite (sans pmemd) can be downloaded from [the AmberTools download page](https://ambermd.org/AmberTools.php)
1. Install CUDA Toolkit 11.6 and cuDNN 8.3.2 for CUDA 11.6 from
   [the NVIDIA CUDA downloads page](https://developer.nvidia.com/cuda-downloads) and
   [the NVIDIA cuDNN download page](https://developer.nvidia.com/cudnn) respectively.
1. Run `./install.sh --amber --amberhome <AMBERHOME>`

It may also be helpful to run `./install.sh --help`. Note: It is *highly
recommended* that you run this script inside an encapsulated python environment
(such as [Anaconda](https://www.anaconda.com/products/individual) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)) otherwise you risk polluting
your system's python installation, since this script installs python packages.

After installation python is not actually needed anymore by the interface.

### Installation Detail

This section is only useful if standard installation fails for you or you
want to perform some extra customization steps, otherwise feel free to skip
it.

If you want to skip installation of PyTorch (pip, CUDA 10.2, nightly version), numpy and the latest torchani version
you can install with `./install.sh --no-python-packages`.

A quick way to manually install these packages is to do this:

```bash
git clone https://github.com/aiqm/torchani /path/to/torchani
pip install numpy
pip install --pre torch -f https://download.pytorch.org/whl/nightly/cu102/torch_nightly.html
cd /path/to/torchani
pip install -e .
```

You can also get the instructions to download the latest pytorch from [the official
PyTorch site](https://pytorch.org/get-started/locally/)

If you want to skip testing (no real reason to skip testing,
tests should be very fast): `./install.sh --no-tests`.

If you want to build only the C++ wrapper:

```bash
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

If for some reason the LibTorch download fails you can do it manually by
following the download instructions from [the official PyTorch site](https://pytorch.org/get-started/locally/)
and extracting into `./lib/`

After extracting your directory structure should look like this:

```
README.md
CMakeLists.txt
...
lib/
    catch2/
    libtorch/
        bin/
        include/
        lib/
        share/
        ...
```

**TODO**: remove this and integrate with amber

```
If you want to only modify, configure and build amber you can call
`./install.sh --only-amber --amberhome AMBERHOME`
Alternatively, to only modify the files you can export AMBERHOME and run the
script ``modify_amber_and_link.sh``:

export AMBERHOME=/path/to/amberhome
./amber_files/modify_amber_and_link.sh

Also, you can copy and modify the files manually. Check the README
file inside `amber_files/` for info on this, but this is time consuming and
error prone. By default the flags passed to `./configure` are `-torchani
-noX11 --skip-python gnu` and only pmemd and sander are built, but you can
run configure again and/or build all of Amber the usual way if needed as long
as you pass `-torchani` as a flag.
```

## Usage

Familiarity with pmemd and/or sander is assumed in what follows.

To use the potential with Amber you must include the flag `iextpot = 1` in the
`ctrl` namelist. An `extpot` namelist must also be included in the input file
(`mdin`). This namelist has to be included *in addition to* the cntrl namelist,
where all the usual simulation parameters are specified. The `extpot` namelist
has some options, described next, all of which except for `extprog` have
default values.

- `extprog` (string)
  Required to be `"torchani"`

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

- `model_type` (string)
  The neural network to choose. Currently possible values are `"ani1x"`,
  `"ani1ccx"`, `"ani2x"` and `"custom"`. For use of "custom" see section *Support for
  custom models*. Default is `"ani1x"`.

- `model_index` (int)
  Used to select a specific model from a model ensemble. Since the ANI-1x, ANI-2x and
  ANI-1ccx models are an ensemble of 8 networks each, this flag must be between
  0 and 7 if used (models are 0-indexed). The default is to use the whole
  ensemble (set to -1). It is highly recommended that you do not set this flag
  unless you know exactly what you are doing. Using an ensemble of models
  provides a significantly higher accuracy than using one model only.

An example `mdin` input file could be:

```
&cntrl
    imin = 0
    ntx = 1
    ntpr = 10
    ntwx = 10
    nstlim = 1000 ! total steps
    dt = 0.001
    ntf = 2  ! turn SHAKE on
    ntc = 2  ! turn SHAKE on
    temp0 = 300.0
    cut = 8.0
    ntt = 3  ! langevin thermostat
    gamma_ln = 2.0
    ig = -1
    iextpot = 1 ! needed for external potential
/

&extpot
    extprog = "torchani"
    model_type = "ani1x"
    use_double_precision = .true.
    use_cuda_device = .true.
/
```

## Usage in an ML/MM framework

See the [**README**](./mlmm/README.md) file in the mlmm folder.

## Limitations

The interface is provided as is, with no guarantees that it will work with any
specific type of dynamics.

The potential hasn't been tested with dynamics other than serial (non MPI, non
OPENMP) constant temperature (with and without NMR restraints and SHAKE
constraints) and constant pressure (with the MC barostat). Any other kind of
dynamics may fail to work correctly.

- Generalized Born dynamics
- Constant pH and constant redox potential are not supported
- Thermodynamic integration (TI) is not supported
- External electric fields are not supported.
- Only the Monte Carlo barostat (`barostat = 3`) is allowed for NPT dynamics.

It is possible that some of these limitations will be lifted in the future.
The only accepted flags for `igb` in the Amber `ctrl` namelist are `igb = 0`
(the default, periodic boundary conditions, vacuum) and `igb = 6` (non periodic
boundary conditions, vacuum).

## Support for custom models

Custom models (subclassing BuiltinModel from the torchani python library are
supported as long as they expose the same functions to jit (same API). Note
however that the support of custom models is not guaranteed since the specifics
of this classes are not stable and subject to changing at any time so don't
rely on this.

To add a custom model you should JIT-compile it inside python and save it into
a file under ``TORCHANI_ROOT/jit/custom.pt``, where ``TORCHANI_ROOT`` is the
source root:

```python
import torch
from torchani.models import BuiltinModel

class CustomModel(BuiltinModel):
    ...

custom_model = CustomModel()
torch.jit.save(torch.jit.script(custom_model), './jit/custom.pt')
```

The custom model can then be loaded by setting `model_type = "custom"` in the
`extpot` namelist.

