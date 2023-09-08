# JIT compiled models from torchani

This directory contains models compiled into Torchscript with the JIT Torch
compiler. The models can be interpreted by LibTorch on a C++ environment.

Currently available models are:

- ANI-1x
- ANI-2x
- ANI-1ccx
- ANI-2x-MBIS

Currently "custom" corresponds to ANI-DR

If you want to add a custom model you can add it here with the name `custom.pt`
