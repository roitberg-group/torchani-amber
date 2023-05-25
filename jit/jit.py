import typing as tp
from pathlib import Path
import argparse
import warnings

import torch

# Disable annoying torchani warnings about MNP and cuAEV
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import torchani


def _reset_jit_bullshit():
    torch._C._jit_clear_class_registry()
    torch.jit._recursive.concrete_type_store = torch.jit._recursive.ConcreteTypeStore()
    torch.jit._state._clear_class_state()


def _jit_compile_and_save_whole_model_and_submodels(
    model: torchani.models.BuiltinModel,
    name: str,
    title: str,
    path: Path
) -> None:
    print(f"JIT compiling {title} to TorchScript")
    model.requires_grad_(False)
    full_model_path = path / f"{name}.pt"
    _reset_jit_bullshit()
    torch.jit.save(torch.jit.script(model), str(full_model_path))
    for j, _ in enumerate(model):
        jth_model_path = path / f"{name}_{j}.pt"
        _reset_jit_bullshit()
        script_model = torch.jit.script(model[j])
        torch.jit.save(script_model, str(jth_model_path))
    print("Done")


def _disable_jit_optimizations() -> None:
    print("Disabling JIT optimizations")
    # Avoid potential issues with JIT compilation by disabling these
    torch._C._jit_set_profiling_executor(False)
    torch._C._jit_set_profiling_mode(False)
    torch._C._jit_override_can_fuse_on_cpu(False)
    torch._C._jit_set_texpr_fuser_enabled(False)
    torch._C._jit_set_nvfuser_enabled(False)


# First construction of models will trigger download of the model data
_MODELS = {
    "ANI-1x": torchani.models.ANI1x,
    "ANI-1ccx": torchani.models.ANI1ccx,
    "ANI-2x": torchani.models.ANI2x,
    "ANI-mbis": torchani.models.ANI2xCharges,
}


# Save the jit-compiled version of all available builtin models
def _main(
    disable_optimizations: bool,
    external_cell_list: bool,
    torch_cell_list: bool,
) -> None:
    print("JIT compiling builtin models to TorchScript...")
    current_path = Path(__file__).resolve().parent

    if disable_optimizations:
        _disable_jit_optimizations()
    options = {
        "": True,
        "cell_list": torch_cell_list,
        "external_cell_list": external_cell_list,
    }
    for name, Model in _MODELS.items():
        str1, str2 = name.split("-")
        for label, choice in options.items():
            if not choice:
                continue
            print(f"JIT compiling {name} with choice {label or 'standard'} = {choice}")
            suffix = "".join(["_", "_".join(label.split("_")[:-1])]) if label else ""
            kwargs: tp.Dict[str, bool] = {}
            if label:
                kwargs.update({label: choice})
            try:
                model = Model(**kwargs)
            except Exception as e:
                print(e)
                print(
                    f"Could not generate model {name} with choice {label or 'standard'}"
                )
                print(" It may not be available in your torchani version?")
            model.requires_grad_(False)
            _jit_compile_and_save_whole_model_and_submodels(
                model, f"{str1.lower()}{str2}{suffix}", name, current_path
            )
    print("Done with all models")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--disable-optimizations", action="store_true", default=False)
    parser.add_argument("--torch-cell-list", action="store_true", default=False)
    parser.add_argument("--external-cell-list", action="store_true", default=False)
    args = parser.parse_args()
    _main(args.disable_optimizations, args.external_cell_list, args.torch_cell_list)
