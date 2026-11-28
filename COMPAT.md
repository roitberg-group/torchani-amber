# Model compatibility

What libtorchani passes to, and accepts from, each model format: TorchScript (`.pt`),
Python models (a directory with a `model.py`) and AOTInductor packages (`.pt2`).
TorchScript and Python models accept several older or optional conventions, listed
below. AOTI packages have a single fixed contract.

## TorchScript and Python models

Both formats follow the same contract, implemented in `src/torchscript.cpp` and
`src/python_model.cpp`. They call the same methods: `forward`, and optionally
`compute_from_external_neighbors`, `compute_mlmm`, `compute_multipole_mlmm`,
`set_active_members`, `set_extra_scalar_atomic_features` and `set_strategy`.

### Inputs

The leading arguments of each method are passed positionally, and the rest by name:

| Method | Positional | By name |
|---|---|---|
| `forward` | `(species, coords)` | `cell`, `pbc`, `charge`, `atomic`, `ensemble_values`, and if applicable `molecule_idxs`, `spin_mult` |
| `compute_from_external_neighbors` | `species, coords, neighbor_idxs, shifts` | `charge`, `atomic`, `ensemble_values`, and if applicable `molecule_idxs`, `spin_mult` |
| `compute_mlmm` | `(species, ml_coords), mm_coords, mm_charges` | as `forward`, and if applicable `mm_cutoff` |
| `compute_multipole_mlmm` | as `compute_mlmm`, then `mm_dipoles, mm_quads` | as `compute_mlmm` |

Since arguments are bound by name, the order of the parameters after the positional ones
doesn't matter, but their names do. Parameters that libtorchani doesn't pass need a
default. Passing an argument the method doesn't declare is an error (TorchScript:
"Unknown keyword argument", Python: `TypeError`).

- `atomic` is always `False`. `ensemble_values` is `True` only when QBC is requested.
- `cell` and `pbc` are `None` without PBC.
- `molecule_idxs` is only passed if it has a value. Models that name it
  `_molecule_idxs` (old TorchANI) work, except when it is needed.
- `charge` is an `int64` tensor of shape `(molecules,)`, or a plain `int` if the method
  declares it as `int` (old models). This is detected from the TorchScript schema, or
  from the Python annotation (also the string `"int"`, from
  `from __future__ import annotations`). All compute methods must agree on it.
- `spin_mult` is an `int64` tensor of shape `(molecules,)`, and `mm_cutoff` a float
  tensor of shape `(molecules,)`. Each is only passed if the method declares it. If it
  doesn't, only the default value (spin 1, cutoff 0) is supported.

Differences between the formats:

| Convention | TorchScript | Python |
|---|---|---|
| Method without a `charge` argument | Error | Error, unless it takes `**kwargs` |
| Method that takes `**kwargs` | Not possible in TorchScript | Receives `spin_mult`, and in the ML/MM methods `mm_cutoff`, too |
| How `spin_mult` and `mm_cutoff` are detected | From the schema | With `inspect.signature`, on every call |

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

`atomic_charges` and `atomic_volumes` of shape `(n, molecules, atoms)` are averaged over
the first dimension, for both formats (`src/model_output.cpp`). This covers old TorchANI
charges, `(1, molecules, atoms)`, and per-member values of ensembles. With QBC the
energies must have shape `(members, molecules)`, with at least 2 members.

Differences between the formats:

| Convention | TorchScript | Python |
|---|---|---|
| Non-tensor values in a dict | Skipped in a top-level dict, an error inside a tuple | Skipped silently |
| Output that matches none of the forms | Error from TorchScript, or libtorchani exits (`forward` returning a non-tuple, non-dict, non-tensor) | Error |
| Tuple whose extra element is not a tensor, dict or `None` | Error | Extra element ignored |
| `forward` returning `(energies,)` | libtorchani exits | Accepted |

### Other methods

| Convention | TorchScript | Python |
|---|---|---|
| No `set_strategy` | No-op | Same |
| `set_strategy` rejects the strategy | Error | Same |
| No `set_active_members` / `set_extra_scalar_atomic_features` | Error when `model_index` / extra features are requested | Same |

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
- With the built-in ML/MM couplings (`coulombic`, `simple-polarizable` and their GEEP
  variants), the package's forces are added to the coupling forces, which libtorchani
  calculates with autograd. This requires fixed charges and volumes
  (`use_mlip_atomic_charges = F`, `use_mlip_atomic_volumes = F`), since the package's
  charges and volumes have no autograd graph. `mlip_calc_energy_atomic_charges_retain`
  is not supported, for the same reason.
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
