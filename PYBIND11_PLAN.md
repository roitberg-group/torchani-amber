# Loading non-scriptable Python models through an embedded interpreter

Goal: run an MLIP that can't be TorchScripted or AOTI-compiled by calling its Python
`forward` from libtorchani, using pybind11 to embed CPython in the md engine process.

A working prototype is in `pybind11_prototype/`:

| File | Purpose |
|---|---|
| `toy-model/model.py` | Example model package: an `nn.Module` that is neither scriptable (dict of functions) nor AOTI-friendly (`.tolist()` control flow), plus `make_model` |
| `python_model.h/.cpp` | `PythonModel`: starts the interpreter, imports `model.py`, calls `make_model`, calls the model, computes forces in C++ |
| `main.cpp` | Calls the model the way the md engine would |
| `CMakeLists.txt` | Build, linking libtorch, libtorch_python and libpython |

To run it:

```bash
cmake -S pybind11_prototype -B build-proto && cmake --build build-proto
./build-proto/run_python_model pybind11_prototype/toy-model
```

## Model packaging convention

A Python model is a directory:

```
<model-name>/
    model.py      # required, defines make_model
    ...           # anything else the model needs: weights, sibling modules, config
```

`model.py` must define

```python
def make_model(device: str, dtype: torch.dtype) -> torch.nn.Module:
    ...
```

which returns the model already on `device`, with its floating point tensors in `dtype`.
libtorchani then calls `.to(device=device, dtype=dtype)` and `.eval()` on it, which
are no-ops if `make_model` already did that (unless `skip_py_preamble` is set, see
below). Weights are loaded by `make_model` itself, relative
to the file, e.g. `torch.load(Path(__file__).parent / "weights.pt")`. The returned
module must have the same methods (`forward`, and optionally `compute_mlmm`, etc.) and
return conventions as the TorchScript models.

The model is loaded from its directory, not from a `.pt` file, so nothing is unpickled
by libtorchani and how the model was saved doesn't matter.

The energy and forces match a pure-Python run (checked with torch 2.13, Python 3.11, CPU).

## How it works

1. **Interpreter startup.** The first time a Python model is loaded,
   `py::initialize_interpreter(false)` starts CPython in the process (`false` stops
   Python from installing its own SIGINT handler over the md engine's handler). The
   interpreter is never finalized, because `import torch` can't run a second time after
   `Py_Finalize`. `add_program_dir_to_path` is `false`: otherwise pybind11 puts the
   current directory (the md run dir) first on `sys.path`, and any `.py` file there
   shadows installed modules (found when torchani's own `ase.py` shadowed the `ase`
   package).
2. **Model loading.** First the sibling names are checked (see below). Then
   `<model-dir>` is prepended to `sys.path` so that `model.py` can import sibling
   modules, and `<model-dir>/model.py` is imported by path with
   `importlib.util.spec_from_file_location` + `exec_module`, under the name
   `torchani_user_model_<hash of the path>`. That name is unique per directory and is
   registered in `sys.modules`, since e.g. `dataclasses` (with string annotations) and
   `pickle` look modules up there and fail if they are missing. Then
   `make_model(device.str(), dtype)` builds
   the model. Unless `skip_py_preamble` is true, `.to(device=..., dtype=...)` and
   `.eval()` are then called on the result, to fix models that `make_model` didn't move,
   cast or put in eval mode. Both are no-ops for a model that is already correct
   (`nn.Module.to` modifies the module in place). `skip_py_preamble` is for models where
   those calls are wrong, e.g. a model that keeps some float buffers in another dtype on
   purpose. Converting the dtype in Python is convenient because the Python `.to` only
   converts floating point tensors, unlike the C++ `Module::to`.
   If `do_torch_compile` is true, each of `forward`, `compute_from_external_neighbors`,
   `compute_mlmm` and `compute_multipole_mlmm` that the model has is replaced by
   `torch.compile(model.<method>)` (an instance attribute that shadows the class
   method). `mode=torch_compile_mode` is passed to `torch.compile` (one of `"default"`,
   `"reduce-overhead"`, `"max-autotune"`, `"max-autotune-no-cudagraphs"`, checked before
   anything is loaded). Its other kwargs (`fullgraph`, `dynamic`, `backend`, `options`,
   ...) are not passed.
3. **Tensor passing without copies.** `<torch/csrc/utils/pybind.h>` provides pybind11
   type casters for `at::Tensor` <-> `torch.Tensor`. Both sides wrap the same
   `TensorImpl`, so nothing is copied and autograd graphs carry across.
4. **Forces.** The model returns an energy that is still attached to the graph. C++
   calls `torch::autograd::grad` on it exactly as `TorchScriptMlipModel` does, so the
   existing `computeOutput` logic in `src/torchscript.cpp` can be reused unchanged. The
   GIL is released around the backward call, because backward may run Python code
   (for example a custom `autograd.Function`) on autograd worker threads, and holding the
   GIL there would deadlock.
5. **GIL policy.** Right after starting the interpreter, the GIL is released
   (`PyEval_SaveThread`), and every call into Python takes it with
   `py::gil_scoped_acquire`, so the model can be called from any thread, and from a md
   engine that is itself a Python program. The C++ autograd backward runs without the
   GIL.

## Linking: what's needed, and where it comes from

| Dependency | Needed at link time? | Source |
|---|---|---|
| pybind11 | No. It is **header-only** and has no library to link. | Bundled with the PyTorch wheel in `torch/include/pybind11` (`embed.h` included). Also available from `pip install pybind11`, `conda install pybind11`, or a git submodule, but the bundled copy is guaranteed to match the version torch was built with. |
| libpython3.X.so | Yes | The Python installation (conda envs ship `lib/libpython3.X.so`). CMake: `find_package(Python COMPONENTS Interpreter Development.Embed)` -> `Python::Python`. |
| libtorch_python.so | Yes (it holds `THPVariable_Wrap`/`Unpack`, used by the tensor casters) | **Only in the PyTorch wheel** (`torch/lib`). The standalone LibTorch zip does not include it. |
| libtorch / libc10 | Yes (already linked) | They must be the **same files** that `import torch` loads, so they have to come from the PyTorch wheel. |

Consequences for the build:

- The feature requires `TORCHANI_LINK_TO_PYTORCH_LIBS=ON`. With
  `TORCHANI_DOWNLOAD_AND_INSTALL_LIBTORCH=ON`, `import torch` would load a second copy of
  libc10/libtorch next to the one libtorchani already linked, which leads to crashes or
  duplicate operator registrations. That combination should be rejected at configure
  time.
- libtorchani becomes tied to one Python minor version (the libpython it linked to) and
  one torch build. At runtime the md engine must see the same conda env (`PYTHONHOME`,
  or simply running from the activated env so that `sys.prefix` resolves).
- In the prototype, `find_package(Torch)` refused to configure without a CUDA toolkit,
  so the prototype CMake uses include/lib paths taken from `torch.__file__`. In the real
  build, `Torchani_PytorchHelper` already finds the wheel, and only
  `torch_python` + `Python::Python` need to be added.

## Integration into libtorchani

`ModelFormat::PYTHON` and `MlipModel::from_python_model_path` already exist in
`src/mlip.cpp` (the latter currently throws). `detectModelFormat` currently classifies a
`torch.save(model)` archive as `PYTHON`; with the directory convention it should instead
return `PYTHON` when the path is a directory containing `model.py` (and stop treating
pickled archives as loadable). Steps:

1. Add `src/python_model.{h,cpp}` with `PythonMlipModel : public mlip::MlipModel`,
   holding a `py::object`. Map each virtual to a Python method, the same way
   `TorchScriptMlipModel` maps to TorchScript methods:
   - `computeSinglePoint` -> `forward`
   - `computeFromExternalNeighbors` -> `compute_from_external_neighbors`
   - `computeMlmm` / `computeMultipoleMlmm` -> `compute_mlmm` / `compute_multipole_mlmm`
   - `supports*` -> `hasattr(model, name)`
   - `useOnlyEnsembleMember`, `setExtraFeatures`, `setStrategy` -> the same exported
     methods

   `python_model.cpp` uses no TorchScript (`torch::jit`) code, since TorchScript will be
   removed from PyTorch. The Python result is parsed directly into tensors (the
   energies, plus extra outputs from None, a tensor of atomic charges, or a dict of
   tensors). Only `computeOutput` (forces, MM forces, QBC and their autograd), which
   uses only tensors, moved from `src/torchscript.cpp` to `src/model_output.{h,cpp}`
   and is shared by both formats. Python models always get `charge` as a tensor, and
   every argument after the leading positional ones is passed by name, with
   `spin_mult` and `mm_cutoff` only passed if the method accepts them (as found with
   `inspect.signature`).
2. Make `from_python_model_path` construct it from the model directory. Guard the whole
   thing with a CMake option (`TORCHANI_SUPPORT_PYTHON_MODELS`, default OFF), so that
   builds against the LibTorch zip don't need Python. With the option OFF, the current
   error message stays.
3. Add the `&mlip` namelist settings below.
4. `model_type` already accepts a path (the "not a builtin model" branch in
   `src/torchani.cpp`), so `model_type = '/path/to/<model-name>'` selects a Python
   model. `detectModelFormat` must check for a directory before opening the path with
   `PyTorchStreamReader`, which fails on directories.

### Namelist settings

All go in the `&mlip` namelist. All except `model_format` are only meaningful for
Python models:

| Setting | Type | Default | Effect |
|---|---|---|---|
| `model_format` | string | `'auto'` | How `model_type` is loaded: `'auto'` (detect from the path), `'torchscript'`, `'pt2-aoti'` or `'python'` (force that loader) |
| `skip_py_preamble` | bool | `.false.` | Don't call `.to(device=..., dtype=...)` and `.eval()` on the result of `make_model` |
| `do_torch_compile` | bool | `.false.` | Replace each of `forward`, `compute_from_external_neighbors`, `compute_mlmm`, `compute_multipole_mlmm` that the model has with `torch.compile(<method>, mode=torch_compile_mode)` |
| `torch_compile_mode` | string | `'default'` | One of `'default'`, `'reduce-overhead'`, `'max-autotune'`, `'max-autotune-no-cudagraphs'` |

Example:

```
&mlip
    model_type = '/path/to/my-model',
    model_format = 'python',
    use_double_precision = .true.,
    do_torch_compile = .true.,
    torch_compile_mode = 'max-autotune-no-cudagraphs',
/
```

Changes needed:

- `src/config.cpp`, `defaultMlipConfig()`: add
  `{"model_format", {std::string{"auto"}, NamelistType::STR}}`,
  `{"skip_py_preamble", {false, NamelistType::BOOL}}`,
  `{"do_torch_compile", {false, NamelistType::BOOL}}` and
  `{"torch_compile_mode", {std::string{"default"}, NamelistType::STR}}`. The parser,
  type checking and unknown-key errors then work as for the other options. The values
  of `model_format` and `torch_compile_mode` are checked with `isOneOf` in the config
  validation, like
  `mlmm_coupling`, so a typo fails when the namelist is read rather than when the model
  is loaded. A non-default `torch_compile_mode` with `do_torch_compile = .false.` is
  also an error, checked there too.
- `GlobalConfig::printInfo()`: add the four keys to the printed list.
- Passing them to the model: a `ModelOptions` struct in `src/mlip.h` (the prototype
  has the same struct in `python_model.h`):

  ```cpp
  struct ModelOptions {
      bool skip_py_preamble = false;
      bool do_torch_compile = false;
      std::string torch_compile_mode = "default";
  };
  ```

  `from_model_path(path, device, dtype, options, format)` passes it on to
  `from_python_model_path`. `src/torchani.cpp` fills it from the `&mlip` config.
- `model_format`: `mlip::ModelFormat` (`AUTO`, `TORCHSCRIPT`, `PT2_AOTI`, `PYTHON`) and
  the `format` parameter of `from_model_path` already exist, but `src/torchani.cpp`
  never passes `format`, so it is always `AUTO`. `src/torchani.cpp` maps the string to
  the enum (`'auto'` -> `AUTO`, `'torchscript'` -> `TORCHSCRIPT`, `'pt2-aoti'` ->
  `PT2_AOTI`, `'python'` -> `PYTHON`) and passes it. It stays a separate parameter, not
  a `ModelOptions` member, since it applies to every format and is never ignored.
  - A forced format skips `detectModelFormat` and goes straight to that loader, so a
    mismatch fails inside the loader (e.g. `torch::jit::load` on a `.pt2`, or "Python
    model file not found" for `'python'` with a file path). `'pt2-aoti'` packages are
    loaded lazily, so for them the mismatch only fails on the first calculation. The existing catch block in
    `src/torchani.cpp` adds the path to the error.
  - With a forced `'torchscript'` or `'pt2-aoti'`, the `ModelOptions` are ignored as
    described below, the same as for auto-detected formats.
  - A built-in model (`model_type = 'ani2x'`, etc.) with any `model_format` other than
    `'auto'` (including `'torchscript'`) is rejected during config validation. The
    format of a built-in model is decided by libtorchani, not by the user.
- Every loaded model knows its resolved format (never `AUTO`):
  - `MlipModel` gets a private `const ModelFormat m_format` member, set by a protected
    `MlipModel(ModelFormat format)` constructor, and a non-virtual
    `format() const -> ModelFormat` accessor. Each subclass passes its own format
    (`TorchScriptMlipModel` -> `TORCHSCRIPT`, `CompiledMlipModel` -> `PT2_AOTI`,
    `PythonMlipModel` -> `PYTHON`), so it is read-only and can't disagree with the
    loader that was used.
  - `src/mlip.cpp` gets the string <-> enum conversions (`'torchscript'`, `'pt2-aoti'`,
    `'python'`, `'auto'`), used both for parsing `model_format` and for printing.
  - After step (5) "Load model" in `src/torchani.cpp`, the format is printed with
    `LOG(INFO) << "MLIP Model format: " << ...`, next to the existing
    "MLIP Model full path" message. Like the other `LOG(INFO)` messages it goes to
    stderr, and only shows with `TORCH_CPP_LOG_LEVEL=INFO` (see `README.md`), not in the
    md engine's output file.
- For a TorchScript or AOTI model, `from_model_path` ignores the options instead of
  failing (the format is only known at load time, not when the namelist is parsed):
  - `do_torch_compile = .true.` (and with it `torch_compile_mode`) and
    `skip_py_preamble = .true.` are each ignored with a `utils::printWarning`, saying
    that the option only applies to Python models.
- `README.md`: document the four under the advanced `&mlip` options, including the
  CUDA graph caveats of `'reduce-overhead'` and `'max-autotune'` (see Caveats).
- `tests/test_config.cpp`: check the defaults, that they parse from a namelist (like
  the existing `use_cuaev` checks), that an invalid `model_format` or
  `torch_compile_mode` throws, and that a built-in model with a non-`'auto'`
  `model_format` throws.
- Model loading tests: each forced format loads a model of that format, `format()`
  returns the expected value for auto-detected and forced formats, and a forced format
  that doesn't match the file throws.

## Caveats

- **Performance.** Every step pays the Python call overhead (around tens of
  microseconds) and runs eager mode. That is negligible for large ML regions, but
  noticeable for very small systems. `torch.compile` can still be used inside the
  Python model.
- **Security.** Loading a model runs its `model.py`. Only trusted model directories
  should be loaded (the same trust level as running any Python script).
- **Sibling module names.** Because `<model-dir>` goes on `sys.path`, sibling modules
  are imported by plain name and cached in `sys.modules`, so two models in one process
  that both had e.g. `layers.py` would silently share the first one, and a sibling named
  e.g. `typing.py` would shadow the stdlib module. To turn both into errors, before
  `<model-dir>` is added to `sys.path`, every importable name in it (`*.py`, `*.so`, and
  dirs with an `__init__.py`; `model.py` excepted) is looked up with
  `importlib.util.find_spec`. If it is found outside `<model-dir>` (a previously loaded
  model, an installed package, the stdlib), loading fails with an error saying that
  file names must be unique among all loaded Python models and installed packages.
  Loading the same dir twice is allowed. There is no extra state in libtorchani: the
  check uses the import system's own state. Files that are never imported (e.g. a
  `train.py` script) are also checked, which can give false positives; dirs without
  `__init__.py` (e.g. `weights/`) are not.
- **Python stdout.** Python buffers `print` output, and since the interpreter is never
  finalized, the buffer is never flushed at exit, so output could be lost.
  `PythonModel` calls `sys.stdout.flush()` after `make_model` and after each model call.
  Output printed right before an exception is still lost unless the user sets
  `PYTHONUNBUFFERED=1`.
- **CUDA graph compile modes.** `'reduce-overhead'` and `'max-autotune'` use CUDA
  graphs on GPU. They need fixed input shapes: `forward` has a constant number of ML
  atoms, but `compute_mlmm` and `compute_multipole_mlmm` see a varying number of MM
  atoms within the cutoff, so every new count recompiles and records a new graph.
  Also, each call overwrites the output memory of the previous one, so any tensor
  libtorchani keeps across steps must be `.clone()`d. `'max-autotune'` can take minutes
  to compile, paid in every MPI rank unless the inductor cache is shared.
- **MPI.** Each rank gets its own interpreter, which is fine. Python's own threads
  (`threading`) inside the model would need the GIL policy revisited.
- **Exceptions.** A Python exception surfaces as `py::error_already_set`, which derives
  from `std::exception`. Existing C++ error paths catch it, and `what()` holds the
  Python traceback.
- **Static/exit-time destruction.** Because the interpreter is never finalized,
  `py::object`s that are destroyed after `main` returns are harmless. If a finalize step
  is ever added, every `py::object` must be released before it.
