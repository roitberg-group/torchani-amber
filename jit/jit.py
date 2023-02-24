from pathlib import Path
import argparse

import torch

import torchani


def _store_model(model: torch.nn.Module, name: str, title: str, path: Path) -> None:
    print(f"JIT compiling {title} to TorchScript")
    model.requires_grad_(False)
    torch.jit.save(torch.jit.script(model), path.joinpath(f'{name}.pt').as_posix())
    for j in range(len(model)):
        script_model = torch.jit.script(model[j])
        torch.jit.save(script_model, path.joinpath(f'{name}_{j}.pt').as_posix())
    print("Done")


# Save all available builtin models to jit script files
def _main(
    avoid_optimizations: bool,
    external_cell_list: bool,
    torch_cell_list: bool,
) -> None:
    print("JIT compiling builtin models to TorchScript...")
    current_path = Path(__file__).resolve().parent
    if avoid_optimizations:
        print("Avoiding JIT optimizations")
        # avoid potential issues with JIT compilation by disabling
        # internal torch optimizations
        torch._C._jit_set_profiling_executor(False)
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_override_can_fuse_on_cpu(False)
        torch._C._jit_set_texpr_fuser_enabled(False)
        torch._C._jit_set_nvfuser_enabled(False)

    # first construction of models will trigger download of the model data
    model_map = {
        "ANI-1x": torchani.models.ANI1x,
        "ANI-1ccx": torchani.models.ANI1ccx,
        "ANI-2x": torchani.models.ANI2x,
    }

    for name, model_class in model_map.items():
        str1 = name.split("-")[0]
        str2 = name.split("-")[1]
        model = model_class()
        model.requires_grad_(False)
        _store_model(model, f"{str1.lower()}{str2}", name, current_path)
        if torch_cell_list:
            print(f"JIT compiling {name} with Torch CellList to TorchScript...")
            try:
                model_cell_list = model_class(cell_list=True)
            except Exception as e:
                print(e)
                print(f"Could not generate model {name} with torch cell list")
                print(" Are you sure this is available in your torchani version?")
            model.requires_grad_(False)
            _store_model(
                model_cell_list,
                f"{str1.lower()}{str2}_cell",
                name,
                current_path
            )
        if external_cell_list:
            print(f"JIT compiling {name} with External CellList to TorchScript...")
            try:
                model_cell_list = model_class(cell_list=True)
            except Exception as e:
                print(e)
                print(f"Could not generate model {name} with external cell list")
                print(" Are you sure this is available in your torchani version?")
            model.requires_grad_(False)
            _store_model(
                model_cell_list,
                f"{str1.lower()}{str2}_external_cell",
                name,
                current_path
            )
    print("Done with all models")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--avoid-optimizations", action="store_true", default=False)
    parser.add_argument("--torch-cell-list", action="store_true", default=False)
    parser.add_argument("--external-cell-list", action="store_true", default=False)
    args = parser.parse_args()
    _main(
        args.avoid_optimizations,
        args.external_cell_list,
        args.torch_cell_list
    )
