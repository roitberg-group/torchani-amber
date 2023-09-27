from pathlib import Path
import argparse
import warnings

import torch
from rich.console import Console

# Disable annoying torchani warnings about MNP and cuAEV
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import torchani

console = Console()


def _reset_jit_registry():
    torch._C._jit_clear_class_registry()
    torch.jit._recursive.concrete_type_store = torch.jit._recursive.ConcreteTypeStore()
    torch.jit._state._clear_class_state()


def _jit_compile_and_save_whole_model_and_submodels(
    model: torchani.models.BuiltinModel,
    name: str,
    title: str,
    path: Path,
    force_recompile: bool = False,
) -> None:
    console.print(f"-- JIT - Compiling {title} to TorchScript")
    model.requires_grad_(False)
    full_model_path = path / f"{name}.pt"
    _reset_jit_registry()
    skip_all = True
    if force_recompile or not full_model_path.is_file():
        torch.jit.save(torch.jit.script(model), str(full_model_path))
        skip_all = False
    for j, _ in enumerate(model):
        jth_model_path = path / f"{name}-{j}.pt"
        _reset_jit_registry()
        if force_recompile or not jth_model_path.is_file():
            script_model = torch.jit.script(model[j])
            torch.jit.save(script_model, str(jth_model_path))
            skip_all = False
    if skip_all:
        console.print("-- JIT - Skipped, already present")
    else:
        console.print("-- JIT - Done")


def _disable_jit_optimizations() -> None:
    console.print("-- JIT - Disabling optimizations")
    # Avoid potential issues with JIT compilation by disabling these
    torch._C._jit_set_profiling_executor(False)
    torch._C._jit_set_profiling_mode(False)
    torch._C._jit_override_can_fuse_on_cpu(False)
    torch._C._jit_set_texpr_fuser_enabled(False)
    torch._C._jit_set_nvfuser_enabled(False)


# First construction of models will trigger download of the model data
_MODELS = {
    "ANI1x": torchani.models.ANI1x,
    "ANI1ccx": torchani.models.ANI1ccx,
    "ANI2x": torchani.models.ANI2x,
    "ANIdr": torchani.models.ANIdr,
    "ANImbis": torchani.models.ANI2xCharges,
}

# This maps has kwargs -> suffix
_SUFFIX_MAP = {
    (): "standard",
    ("cell_list",): "torch-cell-list",
    ("external_cell_list",): "external-cell-list",
    ("use_cuaev_interface", "use_cuda_extension"): "cuaev",
    (
        "use_cuaev_interface",
        "use_cuda_extension",
        "cell_list",
    ): "cuaev-torch-cell-list",
    (
        "use_cuaev_interface",
        "use_cuda_extension",
        "external_cell_list",
    ): "cuaev-external-cell-list",
}


# Save the jit-compiled version of all available builtin models
def _main(
    disable_optimizations: bool,
    force_recompile: bool,
    # normal network options
    standard: bool,
    torch_cell_list: bool,
    external_cell_list: bool,
    # cuaev options
    cuaev: bool,
    cuaev_torch_cell_list: bool,
    cuaev_external_cell_list: bool,
) -> None:
    console.print("-- JIT - Compiling builtin models to TorchScript")
    current_path = Path(__file__).resolve().parent

    if disable_optimizations:
        _disable_jit_optimizations()

    options = {
        (): standard,
        ("cell_list",): torch_cell_list,
        ("external_cell_list",): external_cell_list,
        ("use_cuaev_interface", "use_cuda_extension"): cuaev,
        (
            "use_cuaev_interface",
            "use_cuda_extension",
            "cell_list",
        ): cuaev_torch_cell_list,
        (
            "use_cuaev_interface",
            "use_cuda_extension",
            "external_cell_list",
        ): cuaev_external_cell_list,
    }

    for name, Model in _MODELS.items():
        for labels, choice in options.items():
            if not choice:
                continue
            console.print(
                f"-- JIT - Compiling {name} "
                f"{'with ' + str(labels) if labels else 'standard'}"
            )
            kwargs = {label: True for label in labels}
            try:
                model = Model(**kwargs)
            except Exception as e:
                console.print(f"-- JIT - {e}", style="yellow")
                console.print(
                    f"-- JIT - Could not compile {name} "
                    f"{'with ' + str(labels) if labels else 'standard'}",
                    style="yellow",
                )
            suffix = _SUFFIX_MAP[labels]
            model.requires_grad_(False)
            _jit_compile_and_save_whole_model_and_submodels(
                model,
                f"{name.lower()}-{suffix}",
                name,
                current_path,
                force_recompile,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--disable-optimizations",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--external-cell-list",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--cuaev-external-cell-list",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--standard",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--force-recompile",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--torch-cell-list",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--cuaev",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--cuaev-torch-cell-list",
        action="store_true",
        default=True,
    )
    args = parser.parse_args()
    _main(
        disable_optimizations=args.disable_optimizations,
        force_recompile=args.force_recompile,
        standard=args.standard,
        external_cell_list=args.external_cell_list,
        torch_cell_list=args.torch_cell_list,
        cuaev=args.cuaev,
        cuaev_torch_cell_list=args.cuaev_torch_cell_list,
        cuaev_external_cell_list=args.cuaev_external_cell_list,
    )
