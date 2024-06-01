from dataclasses import dataclass
import typing as tp
from copy import deepcopy
from pathlib import Path
import argparse
import warnings

from tqdm import tqdm
from rich.console import Console

# Force terminal is needed to display color output in CMake
console = Console(highlight=False, force_terminal=True)
JIT_DIR = Path(__file__).resolve().parent


@dataclass
class ModelSpec:
    cls: str
    neighborlist: str
    use_cuda_ops: bool
    model_indices: tp.List[tp.Optional[int]]

    @property
    def kwargs(self) -> tp.Dict[str, tp.Any]:
        _kwargs: tp.Dict[str, tp.Any] = {"neighborlist": self.neighborlist}
        if self.cls == "ANIdr":
            _kwargs["use_cuda_ops"] = self.use_cuda_ops
            return _kwargs
        _kwargs["use_cuda_extension"] = self.use_cuda_ops
        _kwargs["use_cuaev_interface"] = self.use_cuda_ops
        return _kwargs

    def file_path(self, index: tp.Optional[int]) -> Path:
        if index not in self.model_indices:
            raise ValueError(
                f"Index must be one of the allowed model indices {self.model_indices}"
            )
        parts = []
        if self.use_cuda_ops:
            parts.append("cuaev")

        if self.neighborlist == "cell_list":
            parts.append("celllist")
        elif self.neighborlist == "full_pairwise":
            parts.append("stdlist")
        elif self.neighborlist == "external":
            parts.append("externlist")

        if index is not None:
            parts.append(str(index))
        suffix = "-".join(parts)
        return Path(JIT_DIR / f"{self.cls.lower()}-{suffix}.pt")


def _check_which_models_need_compilation(
    force_recompilation: bool,
    external_neighborlist: bool,
    cuaev: bool,
) -> tp.List[ModelSpec]:
    model_names_and_sizes = {
        "ANI1x": 8,
        "ANI1ccx": 8,
        "ANI2x": 8,
        "ANIdr": 7,
        # "ANIala": 1,
        # "ANImbis": 8,
    }
    model_kwargs: tp.List[tp.Dict[str, tp.Any]] = [
        {
            "neighborlist": "full_pairwise",
            "use_cuda_ops": False,
        },
        {
            "neighborlist": "cell_list",
            "use_cuda_ops": False,
        },
    ]
    if external_neighborlist:
        model_kwargs.append(
            {
                "neighborlist": "external",
                "use_cuda_ops": False,
            }
        )

    if cuaev:
        _model_kwargs = deepcopy(model_kwargs)
        for d in _model_kwargs:
            d.update({"use_cuda_ops": True})
        model_kwargs.extend(_model_kwargs)

    specs = []
    for name, size in model_names_and_sizes.items():
        idxs: tp.List[tp.Optional[int]] = [None]
        if size > 1:
            idxs.extend(range(size))
        for kwargs in model_kwargs:
            spec = ModelSpec(cls=name, model_indices=idxs, **kwargs)
            needed_indices = []
            for idx in spec.model_indices:
                file_path = spec.file_path(idx)
                if not file_path.exists() or force_recompilation:
                    needed_indices.append(idx)
            if needed_indices:
                spec.model_indices = needed_indices
                specs.append(spec)
    if specs:
        console.print("-- JIT - Will attempt to compile the following models:")
        for s in specs:
            _name = [
                f"-- {s.cls}({'cuAEV' if s.use_cuda_ops else 'pyAEV'}, {s.neighborlist})"  # noqa
            ]
            if None in s.model_indices:
                _name.append("full-ensemble")
            single_networks = []
            for idx in s.model_indices:
                if idx is None:
                    continue
                single_networks.append(str(idx))
            if single_networks:
                _name.append(f"single-networks:{','.join(single_networks)}")
            console.print(", ".join(_name))
    return specs


def _reset_jit_registry():
    torch._C._jit_clear_class_registry()
    torch.jit._recursive.concrete_type_store = torch.jit._recursive.ConcreteTypeStore()
    torch.jit._state._clear_class_state()


def _jit_compile_and_save_models_in_spec(spec: ModelSpec) -> bool:
    # First construction of models will trigger download of the model data if needed
    try:
        model = getattr(torchani.models, spec.cls)(**spec.kwargs)
    except Exception as e:
        console.print(f"-- JIT - {e}", style="yellow")
        console.print(f"-- JIT - Failed to instantiate {spec}", style="yellow")
        return False
    # TODO: Is this needed?
    model.requires_grad_(False)
    indices = deepcopy(spec.model_indices)
    if None in indices:
        indices.remove(None)
        _reset_jit_registry()
        torch.jit.save(torch.jit.script(model), spec.file_path(None))
    for index in indices:
        _reset_jit_registry()
        torch.jit.save(torch.jit.script(model[index]), spec.file_path(index))
    return True


def _disable_jit_optimizations() -> None:
    # Avoid potential issues with JIT compilation by disabling these
    torch._C._jit_set_profiling_executor(False)
    torch._C._jit_set_profiling_mode(False)
    torch._C._jit_override_can_fuse_on_cpu(False)
    torch._C._jit_set_texpr_fuser_enabled(False)
    if tuple(map(int, torch.__version__.split("."))) < (2, 3):
        # Avoid nvfuser bugs for old pytorch versions
        # https://github.com/pytorch/pytorch/issues/84510)
        torch._C._jit_set_nvfuser_enabled(False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--disable-optimizations",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--external-neighborlist",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--cuaev",
        action="store_true",
        default=True,
    )
    args = parser.parse_args()
    model_specs = _check_which_models_need_compilation(
        force_recompilation=args.force,
        external_neighborlist=args.external_neighborlist,
        cuaev=args.cuaev,
    )
    if model_specs:
        # If we actually need to compile something we import torch and torchani
        # here, since the imports are slow, otherwise we skip the immports
        console.print(
            "JIT - Importing torch in order to JIT-compile models...",
        )
        import torch

        # Disable some annoying torchani warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import torchani
        if args.disable_optimizations:
            console.print(
                "-- JIT - Disabling optimizations before scripting",
            )
            _disable_jit_optimizations()

    success = True
    console.print("-- JIT - Starting to compile models")
    for spec in tqdm(model_specs, leave=False):
        success = _jit_compile_and_save_models_in_spec(spec)

    if success:
        if model_specs:
            console.print("-- JIT - Done", style="green")
        else:
            console.print("-- JIT - Done (nothing to compile)", style="green")
    else:
        console.print(
            "-- JIT - Done, but failed to instantiate some models", style="yellow"
        )
