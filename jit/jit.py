import typing as tp
from pathlib import Path
import argparse
import warnings

import torch

# Disable annoying torchani warnings about MNP and cuAEV
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import torchani


def _jit_compile_and_save_whole_model_and_submodels(
    model: torchani.models.BuiltinModel, name: str, title: str, path: Path
) -> None:
    print(f"JIT compiling {title} to TorchScript")
    model.requires_grad_(False)
    full_model_path = path / f"{name}.pt"
    torch.jit.save(torch.jit.script(model), str(full_model_path))
    for j in range(len(model)):
        jth_model_path = path / f"{name}_{j}.pt"
        script_model = torch.jit.script(model[j])
        torch.jit.save(script_model, str(jth_model_path))
    print("Done")


def _avoid_jit_optimizations():
    print("Avoiding JIT optimizations")
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
}


# Save the jit-compiled version of all available builtin models
def _main(
    avoid_optimizations: bool,
    external_cell_list: bool,
    torch_cell_list: bool,
) -> None:
    print("JIT compiling builtin models to TorchScript...")
    current_path = Path(__file__).resolve().parent

    if avoid_optimizations:
        _avoid_jit_optimizations()
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
    parser.add_argument("--avoid-optimizations", action="store_true", default=False)
    parser.add_argument("--torch-cell-list", action="store_true", default=False)
    parser.add_argument("--external-cell-list", action="store_true", default=False)
    args = parser.parse_args()
    _main(args.avoid_optimizations, args.external_cell_list, args.torch_cell_list)
