import itertools
import typing as tp
from dataclasses import dataclass, asdict
from pathlib import Path
import jinja2

env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(Path(__file__).parent.parent / "templates/"),
    undefined=jinja2.StrictUndefined,
    autoescape=jinja2.select_autoescape(),
    trim_blocks=True,
    lstrip_blocks=True,
)


@dataclass
class MlmmConfig:
    protocol: tp.Literal["me", "mbispol"] = "me"


@dataclass
class CudaConfig:
    cuaev: bool = True


template = env.get_template("input.mdin.jinja")

Pairlists = tp.Literal["cell_list", "all_pairs", "amber"]


@dataclass
class Config:
    cuda: tp.Optional[CudaConfig]
    mlmm: tp.Optional[MlmmConfig]
    shake: bool
    vacuum: bool
    float64: bool
    # TODO test external neighborlist, "amber"
    neighborlist: Pairlists


bools = (True, False)
mlmm_me = MlmmConfig()
mlmm_mbispol = MlmmConfig("mbispol")
cuda_configs = (None, CudaConfig(True), CudaConfig(False))
mlmm_configs = (None, MlmmConfig("me"), MlmmConfig("mbispol"))
neighbor_configs: tp.Tuple[Pairlists, ...] = ("cell_list", "all_pairs")


for tup in itertools.product(
    cuda_configs, mlmm_configs, bools, bools, bools, neighbor_configs
):
    config = Config(*tup)

    if config.mlmm and config.neighborlist != "all_pairs":
        continue

    if not config.vacuum and config.neighborlist == "all_pairs":
        continue

    string = template.render(**asdict(config))
    fname_parts = [f"mlmm_{config.mlmm.protocol}"] if config.mlmm else ["fullml"]
    fname_parts.append("shake" if config.shake else "noshake")
    fname_parts.append("vac" if config.vacuum else "wat")
    fname_parts.append("f64" if config.float64 else "f32")
    fname_parts.append(config.neighborlist)
    d = Path(__file__).parent / f"{'-'.join(fname_parts)}"
    d.mkdir(exist_ok=True)
    (d / "input.mdin").write_text(string)
