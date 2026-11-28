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
    infer: bool = False

    @property
    def kwargs(self) -> dict[str, tp.Any]:
        return {"neighborlist": "adaptive", "strategy": "cuaev"}

    def file_path(self) -> Path:
        if self.infer:
            return Path(JIT_DIR / f"{self.cls.lower()}-infer.pt")
        return Path(JIT_DIR / f"{self.cls.lower()}.pt")


def _check_which_models_need_compilation(
    force_recompilation: bool,
    infer: bool = False,
) -> list[ModelSpec]:
    model_names: tuple[str, ...] = (
        "ANI1x",
        "ANI1ccx",
        "ANI2x",
        "ANImbis",
        "ANImbisv",
        "ANI2dr",
        "ANI2xr",
        # "ANIr2s",
        # "ANIr2s_water",
        # "ANIr2s_ch3cn",
        # "ANIr2s_chcl3",
    )
    all_specs = [ModelSpec(cls=name) for name in model_names]
    if infer:
        all_specs.extend(ModelSpec(cls=name, infer=True) for name in model_names)

    specs = []
    for spec in all_specs:
        if spec.file_path().exists() and not force_recompilation:
            continue
        specs.append(spec)
    if specs:
        console.print("-- JIT - Will attempt to compile the following models:")
        for s in specs:
            console.print(f"-- {s.file_path().stem}")  # noqa
    return specs


def _jit_compile_and_save_models_in_spec(spec: ModelSpec) -> bool:
    # First construction of models will trigger download of the model data if needed
    try:
        model = getattr(torchani.models, spec.cls)(**spec.kwargs)
    except Exception as e:
        console.print(f"-- JIT - {e}", style="yellow")
        console.print(f"-- JIT - Failed to instantiate {spec}", style="yellow")
        return False
    if spec.infer:
        model.potentials["nnp"].neural_networks = model.potentials[
            "nnp"
        ].neural_networks.to_infer_model(use_mnp=False)
    torch.jit.save(torch.jit.script(model), spec.file_path())
    return True


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--infer",
        action="store_true",
        default=False,
        help="Also compile the inference-optimized (*-infer.pt) version of each model",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--aimnet2",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--nutmeg",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()
    if args.aimnet2:
        # The wrappers are written once (they are only rewritten with --force); if
        # their dependencies (e.g. ase) are missing, warn and go on with the rest
        aimnet2_files = [
            JIT_DIR / "aimnet2" / f"aimnet2-{lot}-{mbis}{suffix}.pt"
            for lot in ("b973c", "wb97m")
            for mbis in ("", "mbis-")
            for suffix in ("nocut", "dsf")
        ]
        if all(f.exists() for f in aimnet2_files) and not args.force:
            console.print("-- JIT - AimNet2 wrappers found, not generating them again")
            args.aimnet2 = False
        else:
            try:
                from aimnet2_wrapper import generate_aimnet2_wrapper
            except ImportError as e:
                console.print(
                    f"-- JIT - Can't generate the AimNet2 wrappers ({e});"
                    " the AimNet2 models will not be available",
                    style="yellow",
                )
                args.aimnet2 = False
    if args.nutmeg:
        from nutmeg_wrapper import generate_nutmeg_wrapper

    try:
        model_specs = _check_which_models_need_compilation(
            force_recompilation=args.force,
            infer=args.infer,
        )
        if model_specs:
            # If we actually need to compile something we import torch and torchani
            # here, since the imports are slow, otherwise we skip the imports
            console.print(
                "JIT - Importing torch in order to JIT-compile models...",
            )
            import torch
            import torchani
        success = True
        console.print("-- JIT - Starting to compile models")
        for spec in tqdm(model_specs, leave=False):
            if not _jit_compile_and_save_models_in_spec(spec):
                success = False

        if success:
            if model_specs:
                console.print("-- JIT - Done", style="green")
            else:
                console.print("-- JIT - Done (nothing to compile)", style="green")
        else:
            console.print("-- JIT - Done, but failed for some models", style="yellow")
        custom_mlmm_file = JIT_DIR / "animbis-custom-mlmm.pt"
        if args.force or not custom_mlmm_file.exists():
            console.print("-- JIT - Generating ANImbis custom ML/MM wrapper")
            from custom_mlmm_wrapper import generate_custom_mlmm_wrapper

            generate_custom_mlmm_wrapper(
                custom_mlmm_file, neighborlist="adaptive", strategy="cuaev"
            )
        if args.aimnet2:
            console.print("-- JIT - Generating AimNet2 Wrappers")
            # jit one model to cache the NNPotential class
            import torchani  # noqa
            import torch  # noqa

            generate_aimnet2_wrapper()
        if args.nutmeg:
            console.print("-- JIT - Generating Nutmeg Wrappers")
            generate_nutmeg_wrapper()
    except Exception as e:
        console.print(f"-- JIT - Failed with exception {type(e)}: {e}", style="red")
        sys.exit(1)
