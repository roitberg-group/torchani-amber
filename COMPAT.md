# Model compatibility

What libtorchani passes to, and accepts from, each model format: TorchScript (`.pt`),
Python models (a directory with a `model.py`) and AOTInductor packages (`.pt2`).
TorchScript and Python models accept several older or optional conventions, listed
below. AOTI packages have a single fixed contract.

## TorchScript and Python models

Both formats call the same methods: `forward`, and optionally
`compute_from_external_neighbors`, `compute_mlmm`, `compute_multipole_mlmm`,
`set_active_members`, `set_extra_features` and `set_strategy`.

### Inputs

| Convention | TorchScript (`src/torchscript.cpp`) | Python (`src/python_model.cpp`) |
|---|---|---|
| How arguments are passed | Positionally, in the order of the schema, `spin_mult` / `mm_cutoff` inserted at their schema position | Leading arguments positionally, the rest by name (`cell`, `pbc`, `charge`, `atomic`, `ensemble_values`, `molecule_idxs`, ...) |
| `charge` as a plain `int` (old models) instead of an `int64` tensor of shape `(molecules,)` | Accepted, if `forward`'s schema types `charge` as `int` | Accepted, if `forward` annotates `charge` as `int` (also the string `"int"`, from `from __future__ import annotations`) |
| `forward` without a `charge` argument | Error | Error, unless `forward` takes `**kwargs` |
| No `spin_mult` argument | Accepted if `spin_mult = 1`, error otherwise | Same |
| ML/MM methods without `spin_mult`, `mm_cutoff` | Accepted if the value is the default (spin 1, cutoff 0), error otherwise | Same. Whether an argument is accepted is checked with `inspect.signature` on every call |

The `int` charge, `spin_mult` and `mm_cutoff` are detected from the
schema / signature of the method, so a model never receives an argument it doesn't
declare.

### Outputs

The README only documents a dict with `"energies"` and the extra outputs (e.g.
`"atomic_charges"`). Both formats accept these forms for every compute method
(`forward`, `compute_from_external_neighbors`, `compute_mlmm`,
`compute_multipole_mlmm`):

| Output | Energies | Extra outputs |
|---|---|---|
| `{"energies": ..., <name>: <tensor>, ...}` | `"energies"` | The other tensor items |
| `energies` | The tensor | None |
| `(species, energies)` | Index 1 | None |
| `(species, energies, atomic_charges)` | Index 1 | `atomic_charges` |
| `(species, energies, {<name>: <tensor>, ...})` | Index 1 | The dict items |
| `(energies, {<name>: <tensor>, ...})` | Index 0 | The dict items |
| `(energies, atomic_charges)` | Index 0 | `atomic_charges` |
| `(energies, None)` | Index 0 | None |

A tuple whose first element is floating point holds the energies first, otherwise the
first element is the (integer) species. This is how both formats tell
`(energies, atomic_charges)` from `(species, energies)`.

Differences between the formats:

| Convention | TorchScript (`src/torchscript.cpp`) | Python (`src/python_model.cpp`) |
|---|---|---|
| Non-tensor values in a dict | Skipped in a top-level dict, an error inside a tuple | Skipped silently |
| Output that matches none of the forms | Error from TorchScript, or libtorchani exits (`forward` returning a non-tuple, non-dict, non-tensor) | Error |
| Atomic charges of shape `(1, molecules, atoms)` (old TorchANI) | Squeezed to `(molecules, atoms)` | Same (shared `computeOutput` in `src/model_output.cpp`) |

### Other methods

| Convention | TorchScript | Python |
|---|---|---|
| No `set_strategy` | No-op | Same |
| `set_strategy` rejects the strategy | Error | Same |
| No `set_active_members` / `set_extra_features` | Error when `model_index` / extra features are requested | Same |

Forces, MM forces, QBC and the QBC gradients are computed by libtorchani with autograd
from the returned energies, the same way for both formats.

### TorchANI models

The model classes of the `torchani` submodule (`submodules/torchani/torchani/arch.py`)
all take `charge` and `spin_mult` as tensors, and return from
`compute_from_external_neighbors` a dict with `"energies"` and their extra outputs.
Their `forward` returns:

| Class | Models | `forward` returns | Extra outputs |
|---|---|---|---|
| `ANI` | e.g. ANI1x, ANI2x | `SpeciesEnergies(species, energies)` | None |
| `ANIq` | ANImbis | `SpeciesEnergiesAtomicCharges(species, energies, atomic_charges)` | `atomic_charges` |
| `MultiHeadANI` | ANImbisv | `SpeciesEnergiesResults(species, energies, results)`, `results` a dict | `atomic_charges`, `atomic_volumes` |

All of them are supported by both the TorchScript and the Python path. Run through the
Python path (`make_model` returning e.g. `torchani.models.ANImbisv()`), ANI2x, ANImbis
and ANImbisv give the same outputs as their TorchScript files for `forward`,
`compute_from_external_neighbors` and QBC (max abs difference 3e-17, float64, CPU).
None of them has `compute_mlmm` or `compute_multipole_mlmm`; the ML/MM wrappers in
`jit/` add those.

## AOTInductor packages (`.pt2`)

An AOTI package has no flexible signature: it holds one compiled model per computation,
device (`cpu`, `cuda`) and precision (`f32`, `f64`), named
`<computation>/<device>/<dtype>`, and each has fixed inputs and outputs
(`src/compile.h`):

| Model | Inputs | Outputs |
|---|---|---|
| `single_point` | `species, coords, net_charge, spin_mult` | `energies, forces, atomic_charges` |
| `single_point_pbc` | `species, coords, cell, pbc, net_charge, spin_mult` | `energies, forces, atomic_charges` |
| `external_neighbors` | `species, coords, neighbor_idxs, shifts, net_charge, spin_mult` | `energies, forces, atomic_charges` |
| `custom_mlmm` | `species, ml_coords, mm_coords, mm_charges, net_charge, spin_mult, mm_cutoff` | `energies, forces, mm_forces, atomic_charges` |
| `multipole_mlmm` | as `custom_mlmm`, with `mm_dipoles, mm_quads` after `mm_charges` | `energies, forces, mm_forces, atomic_charges` |

- `net_charge` and `spin_mult` are always `int64` tensors of shape `(molecules,)`, and
  `mm_cutoff` is always a float tensor of shape `(molecules,)`. There is no legacy `int`
  charge and no optional argument.
- The package computes the forces (and MM forces) itself, libtorchani doesn't run
  autograd on them.
- A model that doesn't predict atomic charges must return an empty tensor for them.
- Not supported: `atomic`, `ensemble_values`, `molecule_idxs`, QBC, PBC in ML/MM,
  selecting ensemble members (`model_index`) and extra features. `set_strategy` is a
  no-op.
- Only CPU and CUDA, and only float32 and float64.
- The models are loaded on first use, so a missing model, or a file that isn't an AOTI
  package (e.g. with `model_format = 'pt2-aoti'` forced on a `.pt`), only fails on the
  first calculation, not when the model is loaded.
- `do_torch_compile`, `torch_compile_mode` and `skip_py_preamble` are ignored, with a
  warning.
