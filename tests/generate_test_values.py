import typing as tp
import warnings
from pathlib import Path

import torch

# Disable annoying torchani warnings about mnp and cuaev
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from torchani import units


def _save_jit_compiled_model_results_to_file(
    file_path: Path,
    device: str,
    model_jit_file: Path,
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
    # Avoid qbc for non-ensembles
    if file_path.stem.endswith("ensemble"):
        qbcs = model.energies_qbcs(input_).qbcs * units.HARTREE_TO_KCALMOL
        qbc_deriv = torch.autograd.grad(qbcs.sum(), coordinates, retain_graph=True)[0][0]
    with open(file_path, mode="w") as f:
        lines_ = [f"{energy.item()}\n"]
        lines_.extend([f"{v}\n" for v in force.flatten()])
        if file_path.stem.endswith("ensemble"):
            lines_.extend([f"{qbcs.item()}\n"])
            lines_.extend([f"{v}\n" for v in qbc_deriv.flatten()])
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
    model_jit_files: tp.Iterable[Path],
    device: str,
    tests_dir: Path,
) -> None:
    assert device in {"cuda", "cpu"}
    for f in model_jit_files:
        if "1x" in f.name:
            suffix = "_1x"
        elif "2x" in f.name:
            suffix = "_2x"
        elif "1ccx" in f.name:
            suffix = "_1ccx"
        elif "mbis" in f.name:
            suffix = "_mbis"
        elif "dr" in f.name:
            suffix = "_dr"
        if "-0" not in f.name:
            suffix = f"{suffix}_ensemble"

        _save_jit_compiled_model_results_to_file(
            tests_dir / f'test_values_{device}{suffix}.txt',
            device,
            model_jit_file=f,
        )
    print(f"Generated {device.upper()} test values")


def _main(
    model_jit_files: tp.Iterable[Path],
    tests_dir: Path,
) -> None:
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
    tests_dir = Path(__file__).resolve().parent.parent / "tests"
    jit_dir = Path(__file__).resolve().parent.parent / "jit"
    model_jit_files = [
        jit_dir / "ani1x-standard.pt",
        jit_dir / "ani1ccx-standard.pt",
        jit_dir / "ani2x-standard.pt",
        jit_dir / "animbis-standard.pt",
        jit_dir / "anidr-standard.pt",
        jit_dir / "ani1x-standard-0.pt",
        jit_dir / "ani1ccx-standard-0.pt",
        jit_dir / "ani2x-standard-0.pt",
        jit_dir / "animbis-standard-0.pt",
        jit_dir / "anidr-standard-0.pt",
    ]
    _main(model_jit_files, tests_dir)
