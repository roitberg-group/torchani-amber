import warnings
from pathlib import Path

import torch

from torchani import units

_TESTS_DIR = Path(__file__).resolve().parent.parent / 'test'
_JIT_DIR = Path(__file__).resolve().parent.parent / "jit"


def _save_tests_to_file(file_path: Path, device: str) -> None:
    jit_model_file = _JIT_DIR / 'ani1x_0.pt'
    model = torch.jit.load(str(jit_model_file), device).to(torch.double)

    coordinates = torch.tensor(
                      [[[3., 3., 4.], [1.0, 2.0, 1.0]]],
                      dtype=torch.double,
                      requires_grad=True,
                      device=device
                  )
    atomic_numbers = torch.tensor([[1, 6]], dtype=torch.long, device=device)
    input_ = (atomic_numbers, coordinates)

    energy = model(input_).energies * units.HARTREE_TO_KCALMOL
    force = -torch.autograd.grad(energy.sum(), coordinates)[0][0]
    with open(file_path, 'w+') as f:
        lines_ = [f'{energy.item()}\n']
        lines_.extend([f'{v}\n' for v in force.flatten()])
        f.writelines(lines_)


def _main() -> None:
    if torch.cuda.is_available():

        # disable tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        # disable fp16
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        _save_tests_to_file(
            _TESTS_DIR / 'test_values_cuda.txt',
            "cuda"
        )
        print("Generated CUDA test values")
    else:
        warnings.warn(
            "WARNING: Couldn't generate CUDA test values, no CUDA device detected.\n"
            " Test values are needed to run CUDA unit tests.\n"
            " In the future run 'python ./test/generate_test_values.py'\n"
            " This will generate the values."
        )
    _save_tests_to_file(
        _TESTS_DIR / 'test_values_cpu.txt',
        "cpu",
    )
    print("Generated CPU test values.")


if __name__ == "__main__":
    _main()
