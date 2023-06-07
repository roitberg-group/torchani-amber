import typing as tp
import warnings
from pathlib import Path

import torch

# Disable annoying torchani warnings about mnp and cuaev
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from torchani import units


def _save_jit_compiled_model_results_to_file(
    file_path: Path, device: str, model_jit_file: Path
) -> None:
    model = torch.jit.load(str(model_jit_file), device).to(torch.double)

    coordinates = torch.tensor(
        [[[3.0, 3.0, 4.0], [1.0, 2.0, 1.0]]],
        dtype=torch.double,
        requires_grad=True,
        device=device,
    )
    atomic_numbers = torch.tensor([[1, 6]], dtype=torch.long, device=device)
    input_ = (atomic_numbers, coordinates)
    output = model(input_)
    energy = output.energies * units.HARTREE_TO_KCALMOL
    force = -torch.autograd.grad(energy.sum(), coordinates, retain_graph=True)[0][0]
    with open(file_path, "w+") as f:
        lines_ = [f"{energy.item()}\n"]
        lines_.extend([f"{v}\n" for v in force.flatten()])
        if hasattr(output, "atomic_charges"):
            charges = output.atomic_charges.flatten()
            lines_.extend([f"{v}\n" for v in charges])
            for c in charges:
                deriv_charges = torch.autograd.grad(
                    c,
                    coordinates,
                    retain_graph=True
                )[0]
                lines_.extend([f"{v}\n" for v in deriv_charges.flatten()])
        f.writelines(lines_)


def _generate_cpu_or_cuda_values(
    model_jit_files: tp.Iterable[Path], device: str, tests_dir: Path
) -> None:
    assert device in {"cuda", "cpu"}
    for f in model_jit_files:
        if "2x" in f.name:
            suffix = "_2x"
        elif "mbis" in f.name:
            suffix = "_mbis"
        elif "1x" in f.name:
            suffix = "_1x"
        elif "1ccx" in f.name:
            suffix = "_1ccx"
        if "_0" not in f.name:
            suffix = f"{suffix}_ensemble"

        results_file = (
            tests_dir / f'test_values_{device}{suffix}.txt'
        )
        _save_jit_compiled_model_results_to_file(
            results_file,
            device,
            model_jit_file=f,
        )
    print(f"Generated {device.upper()} test values")


def _main(model_jit_files: tp.Iterable[Path], tests_dir: Path) -> None:
    if torch.cuda.is_available():
        # Disable tf32 for accuracy
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        # Disable fp16 for accuracy
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        _generate_cpu_or_cuda_values(model_jit_files, "cuda", tests_dir)
    else:
        warnings.warn(
            "WARNING: Couldn't generate CUDA test values, no CUDA device detected.\n"
            " Test values are needed to run CUDA unit tests.\n"
            " In the future run 'python ./test/generate_test_values.py'\n"
            " This will generate the values."
        )

    _generate_cpu_or_cuda_values(model_jit_files, "cpu", tests_dir)


if __name__ == "__main__":
    tests_dir = Path(__file__).resolve().parent.parent / "test"
    jit_dir = Path(__file__).resolve().parent.parent / "jit"
    model_jit_files = [
        jit_dir / "ani1x.pt",
        jit_dir / "ani1ccx.pt",
        jit_dir / "ani2x.pt",
        jit_dir / "animbis.pt",
        jit_dir / "ani1x_0.pt",
        jit_dir / "ani1ccx_0.pt",
        jit_dir / "ani2x_0.pt",
        jit_dir / "animbis_0.pt",
    ]
    _main(model_jit_files, tests_dir)
