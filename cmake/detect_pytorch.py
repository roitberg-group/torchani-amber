import torch
from pathlib import Path

torch_maj, torch_min, torch_patch = torch.__version__.split(".")
cuda_version = torch.version.cuda
assert cuda_version is not None
cuda_maj, cuda_min = cuda_version.split(".")
cudnn_raw = str(torch.backends.cudnn.version())
cudnn_maj = int(cudnn_raw[:-4])
cudnn_min = int(cudnn_raw[-4:-2])
_path = Path(torch.__file__).resolve().parent
print(
    f"{torch_maj}.{torch_min};{cuda_maj}.{cuda_min};{cudnn_maj}.{cudnn_min};{_path}",
    end="",
)
