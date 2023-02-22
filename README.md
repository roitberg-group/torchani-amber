# Torchani-Amber interface

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

## Note about hardcoded LibTorch paths

Some versions of libtorch have hardcoded paths inside
/path/to/libtorch/share/cmake/Caffe2/Caffe2Targets.cmake, which you need to
change to ${CUDA_TOOLKIT_ROOT_DIR} instead of /usr/local/cuda to make things
work correctly in case installation fails due to issues with the linker)

## Note about Ampere GPUs (e.g. A100 nodes)

The latest Ampere GPU introduced new TF32 tensor cores,
which can speed-up matrix multiplications and convolutions of float32 by ~7x
while sacrificing precision. This feature is ON by default, which means, if you
are using Ampere and you don't turn it OFF, then all your networks will be
evaluated at lower precision. For most popular AI applications it is a
good idea to turn this on, but this might not be the case for neural network
potentials. To disable TF32 try:

```python
import torch
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
```

## Requirements

- Linux operating system
- gcc (tested on 9.3.0 and 8.4.0)
- cmake 3.13 or higher
- git (necessary to obtain the latest torchani version) (?)
- catch2 (library for unit tests only, included in the distribution)
- python 3.8 or higher (for generating the models and some test data)
- torchani (latest version, for generating the models and some test data)
- Amber20 (23?)
- PyTorch 1.5.0 or higher  (1.13?)
- LibTorch 1.5.0 or higher  (1.13?)
- CUDA Toolkit 10.2  (11.6?)
- cuDNN 7.6  (8?)

*LibTorch/PyTorch versions warning*: It is crucial that Torchani is compiled
with JIT using the same torch version as the LibTorch version used to run the
model in C++ (or a lower one).  For example, if `torch.__version__ == 1.5.0`
for Torchani, then LibTorch must also be 1.5.0 or *higher*, otherwise LibTorch
will fail to load the model, or load it incorrectly.

Note that libtorch has shared libraries in it so it can't really be included
with the binary. Also note that the dependance on the CUDA Toolkit, cuDNN and
LibTorch is there also if you want to only run the model on GPU. This is due to
the intrinsic way LibTorch works.

## Installation

The following steps are necessary to install the library and link it correctly:

1. Extract Amber20 (No need to install it) into `AMBERHOME`. The AmberTools
    suite (sans pmemd) can be downloaded from [the AmberTools download page](https://ambermd.org/AmberTools.php)
1. Install CUDA Toolkit 10.2 and cuDNN from
   [the NVIDIA CUDA downloads page](https://developer.nvidia.com/cuda-downloads) and
   [the NVIDIA cuDNN download page](https://developer.nvidia.com/cudnn) respectively.
1. Run `./install.sh --amber --amberhome <AMBERHOME>`

This installs the necessary python packages, downloads and extracts LibTorch
and installs the `libtorchani` shared library and tests it. Afterwards it
modifies the necessary Amber files, configures Amber for torchani with some
default flags and builds pmemd and sander, then it runs some short simulations
using ANI-1x and times them. For more information about manually doing some of
the installation tasks read the section *Installation detail*.

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
a file called "custom.pt" which you should copy inside the `jit/` directory:

```python
import torch
import torchani

class CustomModel(torchani.BuiltinModel):
    ...

custom_model = CustomModel()
torch.jit.save(torch.jit.script(custom_model), './jit/custom.pt')
```

The custom model can then be loaded by setting `model_type = "custom"` in the
`extpot` namelist.
