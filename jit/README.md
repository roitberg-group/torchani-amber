# JIT compiled models from torchani

This directory contains models compiled into Torchscript with the JIT Torch
compiler. The models can be interpreted by LibTorch on a C++ environment.

Currently available models are:

- ANI-1x
- ANI-2x
- ANI-1ccx
- ANI-2x-MBIS

Currently "custom" corresponds to ANI-DR

The AIMNet2 models (`aimnet2/aimnet2-*.pt`) are generated from the bundled weights
(`aimnet2_*_0.raw.pt`) by `python jit.py --aimnet2`, which the CMake build runs
when `JIT_COMPILE_AIMNET2` is `ON` (the default); it needs `ase`.

If you want to add a custom model you can add it here with the name `custom.pt`
