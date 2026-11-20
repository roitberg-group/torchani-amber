import torch
from pathlib import Path

torch_version = torch.__version__.split("+")[0]
cuda_version = torch.version.cuda
if cuda_version is None:
    cuda_maj_min = "none"
else:
    cuda_maj, cuda_min = cuda_version.split(".")
    cuda_maj_min = f"{cuda_maj}.{cuda_min}"
_path = Path(torch.__file__).resolve().parent
print(
    f"{torch_version};{cuda_maj_min};{_path}",
    end="",
)
