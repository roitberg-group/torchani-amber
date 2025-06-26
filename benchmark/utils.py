import typing as tp
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typer import Abort
from rich.console import Console
import jinja2

console = Console()


# For now only supports slurm scheduler
def send_to_scheduler(
    cluster: str,
    gpu: str,
    name: str,
    hours: int,
    unique_id: str,
    core_num: int,
    install_kind: str,
    env: tp.Optional[jinja2.Environment] = None,
) -> None:
    if cluster == "hpg":
        assert gpu in ["2080ti", "b200", "l4", ""]
        if gpu == "b200":
            partition = "hpg-b200"
        elif gpu == "l4":
            partition = "hpg-turin"
        else:
            partition = "gpu"
    else:
        partition = ""
        console.print(f"Unknown cluster {cluster}", style="red")
        raise Abort()
    gpu = f"{gpu}:1" if gpu else "1"
    if env is None:
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(Path(__file__).parent / "templates/"),
            undefined=jinja2.StrictUndefined,
            autoescape=jinja2.select_autoescape(),
            trim_blocks=True,
            lstrip_blocks=True,
        )
    arg_list = sys.argv[1:]
    for j, arg in enumerate(deepcopy(arg_list)):
        if arg in ["--hpg"]:
            arg_list[j] = ""
        elif arg == "--gpu":
            arg_list[j] = ""
            arg_list[j + 1] = ""
        elif arg == "--install-kind":
            arg_list[j] = ""
            arg_list[j + 1] = ""
    args = " ".join(arg_list)

    j = 0
    sched_root = Path(Path.home(), "Sched", "ani-amber")
    sched_root.mkdir(exist_ok=True, parents=True)
    sched_fpath = sched_root / f"{unique_id}-v{str(j).zfill(3)}-{name}.slurm.sh"
    while sched_fpath.is_file():
        j += 1
        sched_fpath = sched_root / f"{unique_id}-v{str(j).zfill(3)}-{name}.slurm.sh"

    tmpl = env.get_template(f"{cluster}.slurm.sh.jinja").render(
        name=name,
        gpu=gpu,
        args=args,
        unique_id=unique_id,
        partition=partition,
        install_kind=install_kind,
        hours=hours,
        core_num=core_num,
        version=str(j).zfill(3),
        cli_app_dir=str(Path(__file__).parent.resolve()),
    )
    sched_fpath.write_text(tmpl)
    console.print("Launching batch script ...")
    subprocess.run(["sbatch", str(sched_fpath)], cwd=sched_root, check=True)


SUPPORTED_MODELS = [
    "ani1x",
    "ani1ccx",
    "ani2x",
    "ani2xr",
    "ani2dr",
    "animbis",
    "aniala",
    "anir2s",
    "anir2s_water",
    "anir2s_chcl3",
    "anir2s_ch3cn",
    "aimnet2-b973c-dsf",
    "aimnet2-b973c-ewald",
    "aimnet2-b973c-nocut",
    "aimnet2-b973c-mbis-dsf",
    "aimnet2-b973c-mbis-ewald",
    "aimnet2-b973c-mbis-nocut",
    "aimnet2-wb97m-dsf",
    "aimnet2-wb97m-ewald",
    "aimnet2-wb97m-nocut",
    "aimnet2-wb97m-mbis-dsf",
    "aimnet2-wb97m-mbis-ewald",
    "aimnet2-wb97m-mbis-nocut",
    "nutmeg-small",
    "nutmeg-medium",
    "nutmeg-large",
]
