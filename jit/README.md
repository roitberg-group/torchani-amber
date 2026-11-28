# JIT-compiled models

This directory contains the scripts that compile models into TorchScript with the JIT
Torch compiler, so that LibTorch can run them in a C++ environment. The compiled
`*.pt` files are written to this directory.

`python jit.py` compiles the following MLIP models, if they are not found here
(`--force` recompiles them):

- `ani1x.pt`, `ani1ccx.pt`, `ani2x.pt`, `ani2xr.pt`, `ani2dr.pt`, `animbis.pt`,
  `animbisv.pt`
- The inference-optimized `*-infer.pt` versions of the above, with `--infer`
- `animbis-custom-mlmm.pt`, an ANI-MBIS wrapper that implements `compute_mlmm`
  (see *Support for custom ML/MM models* in the top-level `README.md`)

Optionally, it also generates:

- With `--aimnet2`, the AimNet2 models `aimnet2/aimnet2-{b973c,wb97m}-[mbis-]{nocut,dsf}.pt`,
  from the bundled weights (`aimnet2_*_0.raw.pt`). It needs `ase`.
- With `--nutmeg`, the Nutmeg models `nutmeg/nutmeg-{small,medium,large}.pt`, from the
  bundled weights (`nutmeg-*.raw.pt`).

The CMake build runs `jit.py` when `TORCHANI_JIT_COMPILE_MODELS` is `ON` (the default
is `OFF`, and it requires `TORCHANI_LINK_TO_PYTORCH_LIBS`). It passes `--aimnet2` when
`TORCHANI_JIT_COMPILE_AIMNET2` is `ON` (the default).

libtorchani doesn't load the built-in models from this directory. It loads them from
the MLIPs dir (`TORCHANI_MLIPS_INSTALL_DIR`), where the CMake build downloads the
compiled models from Hugging Face when `TORCHANI_DOWNLOAD_JIT_MODELS` is `ON` (the
default). To use any other model, pass its path as `model_type` (see *Support for
custom models* in the top-level `README.md`).
