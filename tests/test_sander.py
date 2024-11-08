r"""TorchANI-Amber integration tests

If the envvar TORCHANI_AMBER_KEEP_TEST_DIRS=1, then the inputs and outputs
are not removed after the tests are run, this may be useful for debugging.

If the envvar TORCHANI_AMBER_EXPECTTEST=1 then the ``.dat``, ``.xyz``, and ``.traj``
outputs of the test are saved into the `expect/` directory

If you are debugging old branches use TORCHANI_AMBER_LEGACY_TEST=1
"""

from numpy.typing import NDArray
from numpy.testing import assert_allclose
import numpy as np
import shutil
import os
import tempfile
import typing as tp
import itertools
import unittest
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path

from parameterized import parameterized
import jinja2

this_dir = Path(__file__).parent
Pairlists = tp.Literal["cell_list", "all_pairs", "amber"]


@dataclass
class MlmmConfig:
    protocol: tp.Literal["me", "mbispol"] = "me"


@dataclass
class CudaConfig:
    cuaev: bool = True


@dataclass
class RunConfig:
    cuda: tp.Optional[CudaConfig]
    mlmm: tp.Optional[MlmmConfig]
    shake: bool
    vacuum: bool
    float64: bool
    neighborlist: Pairlists

    @property
    def name(self) -> str:
        parts = [f"MLMM_{self.mlmm.protocol}"] if self.mlmm else ["FullML"]
        parts.append("shake" if self.shake else "noshake")
        parts.append("vacuum" if self.vacuum else "water")
        parts.append("f64" if self.float64 else "f32")
        parts.append("cuda" if self.cuda else "cpu")
        if self.cuda and self.cuda.cuaev:
            parts.append("cuaev")
        parts.append(self.neighborlist.replace("_", ""))
        return "_".join(parts)

    @property
    def explicit_solvent(self) -> bool:
        return not self.vacuum

    @property
    def fullml(self) -> bool:
        return self.mlmm is None

    @property
    def cpu(self) -> bool:
        return self.cuda is None


# Generate all configs to be tested
bools = (True, False)
cuda_configs = (None, CudaConfig(True), CudaConfig(False))
mlmm_configs = (None, MlmmConfig("me"), MlmmConfig("mbispol"))
# TODO test external neighborlist, "amber"
neighbor_configs: tp.Tuple[Pairlists, ...] = ("cell_list", "all_pairs")
configs: tp.List[RunConfig] = []
for tup in itertools.product(
    cuda_configs, mlmm_configs, bools, bools, bools, neighbor_configs
):
    config = RunConfig(*tup)
    if os.environ.get("TORCHANI_AMBER_LEGACY_TEST") == "1":
        if not config.mlmm:
            continue
        if config.neighborlist != "all_pairs":
            continue
        if config.cuda and config.cuda.cuaev:
            continue

    # Only test Float 64 in a few Mlmm cases, never on FullML. Its too slow.
    if config.float64 and not (
        config.mlmm and config.cuda and config.cuda.cuaev and config.vacuum
    ):
        continue

    # Mlmm is only tested with all-pairs
    if config.mlmm:
        if config.neighborlist != "all_pairs":
            continue

    # All-Pairs PBC with full-ml is too slow
    if not config.mlmm:
        if config.explicit_solvent and config.neighborlist == "all_pairs":
            continue

    configs.append(config)


env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(Path(__file__).parent / "templates/"),
    undefined=jinja2.StrictUndefined,
    autoescape=jinja2.select_autoescape(),
    trim_blocks=True,
    lstrip_blocks=True,
)


def name_func(fn, idx, param) -> str:
    (config,) = param.args
    return f"{fn.__name__}_{param.args[0].name}"


class AmberIntegration(unittest.TestCase):
    def setUp(self) -> None:
        self.d: tp.Optional[tempfile.TemporaryDirectory] = None

    def cleanUp(self) -> None:
        if self.d is not None:
            self.d.cleanup()

    @parameterized.expand(configs, name_func=name_func)
    def testSander(self, config: RunConfig) -> None:
        if os.environ.get("TORCHANI_AMBER_KEEP_TEST_DIRS") == "1":
            test_dir = Path(__file__).parent / config.name
            test_dir.mkdir(exist_ok=True)
        else:
            self.d = tempfile.TemporaryDirectory()
            test_dir = Path(self.d.name)
        if os.environ.get("TORCHANI_AMBER_LEGACY_TEST") == "1":
            string = env.get_template("input.mdin.jinja").render(
                legacy=True, **asdict(config)
            )
        else:
            string = env.get_template("input.mdin.jinja").render(
                legacy=False, **asdict(config)
            )
        (test_dir / "input.mdin").write_text(string)
        self._run_sander(test_dir)

        # Generate expected values
        expect = this_dir / "expect"
        expect.mkdir(exist_ok=True)
        for f in sorted(test_dir.iterdir()):
            if f.suffix in [".dat", ".traj", ".xyz"]:
                expect_file = (expect / config.name).with_suffix(f".{f.name}")
                if expect_file.exists() and not config.cuda:
                    expect_text = expect_file.read_text()
                    expect_arr = self.parse_amber_output(expect_text, f.suffix)
                    this_text = f.read_text()
                    this_arr = self.parse_amber_output(this_text, f.suffix)
                    assert_allclose(
                        this_arr,
                        expect_arr,
                        rtol=1e-7 if not config.cuda else 1e-5,
                        atol=1e-5 if not config.cuda else 1e-4,
                        err_msg=f"\nDiscrepancy was found on {expect_file.name}\n",
                    )
                if os.environ.get("TORCHANI_AMBER_EXPECTTEST") == "1":
                    shutil.copy(f, (expect / config.name).with_suffix(f".{f.name}"))

    @staticmethod
    def parse_amber_output(text: str, suffix: str) -> NDArray[np.float32]:
        if suffix == ".xyz":
            lines = text.split("\n")
            non_comment = [line for line in lines if "TORCHANI" not in line]
            text = " ".join(non_comment)
        if suffix == ".traj":
            text = " ".join(text.split("\n")[1:])  # Get rid of file name
        return np.array([float(s) for s in text.split() if s.strip()], dtype=np.float32)

    def _run_sander(self, dir: Path) -> None:
        # prmtop and inpcrd correspond to a solvated ALA dipeptide
        this_dir = Path(__file__).parent
        out = subprocess.run(
            [
                "sander",
                "-i",
                "input.mdin",
                "-p",
                this_dir / "system.prmtop",
                "-c",
                this_dir / "system.inpcrd",
                "-o",
                "system.mdout",
                "-r",
                "system.restart",
                "-x",
                "system.traj",
                "-inf",
                "system.mdinfo",
                "-O",  # Overwrite
            ],
            cwd=dir,
            shell=False,
            capture_output=True,
            text=True,
        )
        if out.returncode != 0:
            print("Process stdout", out.stdout)
            print("Process stderr", out.stdout)
            mdout = dir / "system.mdout"
            if mdout.exists():
                print("Amber mdout", mdout.read_text())
            raise RuntimeError("Error when calling sander")


if __name__ == "__main__":
    unittest.main(verbosity=2)
