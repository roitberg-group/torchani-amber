from enum import Enum
from typer import Option, Typer
import typing_extensions as tpx
import typing as tp
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


from rich.console import Console
from tqdm import tqdm


console = Console()
ROOT = Path(__file__).resolve().parent.parent


app = Typer()


class Opt(Enum):
    PY = "py"
    CU = "cu"
    JIT = "jit"
    COMPILE = "compile"


@app.command()
def run(
    cuda: tpx.Annotated[
        bool,
        Option("--cuda/--no-cuda", help="Use a CUDA enabled gpu for benchmark"),
    ] = True,
    optims: tpx.Annotated[
        tp.Optional[tp.List[Opt]],
        Option(
            "-o",
            "--optim",
            help="Torch optimizations to perform. Default is all",
            show_default=False,
        ),
    ] = None,
    sync: tpx.Annotated[
        bool,
        Option(
            "-s/-S",
            "--sync/--no-sync",
            help="Whether to sync cuda threads in between function calls",
        ),
    ] = True,
    use_tqdm: tpx.Annotated[
        bool,
        Option("-t/-T", "--tqdm/--no-tqdm", help="Whether to use a progress bar"),
    ] = True,
    num_warm_up: tpx.Annotated[
        int,
        Option("-n", "--num-warm-up", help="Num of warm up steps"),
    ] = 200,
    num_profile: tpx.Annotated[
        int,
        Option("-n", "--num-profile", help="Num of profiling steps"),
    ] = 30,
) -> None:
    import torch
    from torchani.utils import SYMBOLS_2X
    from torchani.nn import SpeciesConverter
    from torchani.neighbors import AdaptiveList, neighbors_to_triples
    from torchani._testing import make_molecs

    from tool_utils import Timer

    if optims is None:
        optims = [Opt.PY, Opt.JIT, Opt.COMPILE, Opt.CU]

    target_atomic_density = 0.1
    # num_atoms / cell_size has to be a constant, equal to the water atomic density
    # which is around ~0.1 Ang^-1

    no_tqdm = not use_tqdm
    device = torch.device("cuda" if cuda else "cpu")

    if cuda:
        console.print(f"CUDA sync {'[green]ON[/green]' if sync else '[red]OFF[/red]'}")

    # Set up required models for benchmark
    symbols = SYMBOLS_2X
    models: tp.Dict[str, tp.Any] = {}
    if Opt.PY in optims:
        models["triples"] = neighbors_to_triples
    if Opt.COMPILE in optims:
        models["triples-compile"] = torch.compile(neighbors_to_triples)
    if Opt.JIT in optims:
        models["triples-jit"] = torch.jit.script(neighbors_to_triples)
    converter = SpeciesConverter(symbols=symbols).to(device)
    nl = AdaptiveList().to(device)

    # Loop over molecules and calculate timings
    cutoff = 5.2
    for k, fn in models.items():
        console.print(f"Model kind={k}")
        timer = Timer(modules_and_fns=[], nvtx=False, sync=sync)
        atoms_num: tp.Iterable[int]
        atoms_num = range(30, 700, 10)

        for n in map(int, atoms_num):
            cell_side = (n / target_atomic_density) ** (1 / 3)
            molecs = make_molecs(
                num_warm_up + num_profile,
                n,
                cell_side,
                pbc=False,
                device=device,
                seed=1234,
            )
            slice_ = slice(None, num_warm_up)
            for j, (_species, _coords) in tqdm(
                enumerate(zip(molecs.atomic_nums[slice_], molecs.coords[slice_])),
                desc=f"Warm up {n} atoms, cell side length {cell_side}",
                disable=no_tqdm,
                total=num_warm_up,
                leave=False,
            ):
                coords = _coords.unsqueeze(0)
                elem_idxs = converter(_species.unsqueeze(0))
                neighbors = nl(cutoff, elem_idxs, coords, cell=None, pbc=None)
                _ = fn(neighbors)
                if cuda:
                    torch.cuda.empty_cache()
            if k != "dummy":
                timer.start_profiling()
            slice_ = slice(num_warm_up, num_warm_up + num_profile)
            for j, (_species, _coords) in tqdm(
                enumerate(zip(molecs.atomic_nums[slice_], molecs.coords[slice_])),
                desc=f"Profiling {n} atoms, cell side length {cell_side}",
                disable=no_tqdm,
                total=num_profile,
                leave=False,
            ):
                coords = _coords.unsqueeze(0)
                elem_idxs = converter(_species.unsqueeze(0))
                neighbors = nl(cutoff, elem_idxs, coords=coords, cell=None, pbc=None)
                timer.start_range(str(n))
                _ = fn(neighbors)
                timer.end_range(str(n))
                if cuda:
                    torch.cuda.empty_cache()
            if k != "dummy":
                timer.stop_profiling()
                timer.dump_csv(Path(__file__).parent / f"{k}.csv")
        timer.display()


@app.command()
def plot() -> None:
    def prettify(label) -> str:
        return f"{label.replace('aev', 'AEV')} fwd + bwd"

    fig, ax = plt.subplots()
    size = 0.5
    alpha = 0.7
    colors = (
        "tab:purple",
        "tab:blue",
        "tab:green",
        "tab:orange",
        "darkred",
        "teal",
        "magenta",
    )
    for c, k in zip(
        colors,
        (
            "triples",
            "triples-old",
            # "triples-jit",
            # "triples-compile",
        ),
    ):
        csv = Path(__file__).parent / f"{k}.csv"
        if not csv.is_file():
            continue
        df = pd.read_csv(csv, sep=",")
        ax.scatter(
            df["timing"], df["median"], label=prettify(k), s=size, color=c, alpha=alpha
        )
        ax.set_xlabel("Num. atoms")
        ax.set_ylabel("Median walltime per call, CUDA (ms)")
        # ax.set_title("PBC benchmark")
        ax.legend()
    # plt.savefig("/home/ipickering/Figures/no-pbc-neighborlist")
    plt.show()


if __name__ == "__main__":
    app()
