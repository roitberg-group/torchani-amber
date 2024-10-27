import sys
from dataclasses import dataclass
import typing as tp
from pathlib import Path
import argparse

from tqdm import tqdm
from rich.console import Console

# Force terminal is needed to display color output in CMake
console = Console(highlight=False, force_terminal=True)
JIT_DIR = Path(__file__).resolve().parent


@dataclass
class ModelSpec:
    cls: str
    neighborlist: str

    @property
    def kwargs(self) -> tp.Dict[str, tp.Any]:
        return {"neighborlist": self.neighborlist, "strategy": "cuaev"}

    def file_path(self) -> Path:
        if self.neighborlist == "cell_list":
            suffix = "celllist"
        elif self.neighborlist == "full_pairwise":
            suffix = "stdlist"
        return Path(JIT_DIR / f"{self.cls.lower()}-{suffix}.pt")


def _check_which_models_need_compilation(
    force_recompilation: bool,
) -> tp.List[ModelSpec]:
    model_names = ("ANI1x", "ANI1ccx", "ANI2x", "ANIdr", "ANIala", "ANImbis")
    neighborlists = ["full_pairwise", "cell_list"]

    specs = []
    for name in model_names:
        for nl in neighborlists:
            spec = ModelSpec(cls=name, neighborlist=nl)
            if spec.file_path().exists() and not force_recompilation:
                continue
            specs.append(spec)
    if specs:
        console.print("-- JIT - Will attempt to compile the following models:")
        for s in specs:
            console.print(f"-- {s.cls}({s.neighborlist})")  # noqa
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
    # TODO: is this still needed?
    _reset_jit_registry()
    torch.jit.save(torch.jit.script(model), spec.file_path())
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
        "--force",
        action="store_true",
        default=False,
    )
    try:
        args = parser.parse_args()
        model_specs = _check_which_models_need_compilation(
            force_recompilation=args.force,
        )
        if model_specs:
            # If we actually need to compile something we import torch and torchani
            # here, since the imports are slow, otherwise we skip the imports
            console.print(
                "JIT - Importing torch in order to JIT-compile models...",
            )
            import torch
            import torchani

            # TODO Does this do anything?
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
            console.print("-- JIT - Done, but failed for some models", style="yellow")
    except Exception as e:
        console.print(f"-- JIT - Failed with exception {type(e)}: {e}", style="red")
        sys.exit(1)
