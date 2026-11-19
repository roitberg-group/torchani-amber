import torch
from pathlib import Path

torch_maj, torch_min = torch.__version__.split(".")[:2]
cuda_version = torch.version.cuda
if cuda_version is None:
    cuda_maj_min = "none"
    cudnn_maj = "none"
else:
    cuda_maj, cuda_min = cuda_version.split(".")
    cuda_maj_min = f"{cuda_maj}.{cuda_min}"
    cudnn_raw = str(torch.backends.cudnn.version())
    cudnn_maj = int(cudnn_raw[:-4])
_path = Path(torch.__file__).resolve().parent
print(
    f"{torch_maj}.{torch_min};{cuda_maj_min};{cudnn_maj};{_path}",
    end="",
)
