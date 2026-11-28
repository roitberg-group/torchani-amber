r"""TorchANI-Amber integration tests

If the envvar TORCHANI_AMBER_KEEP_TEST_DIRS=1, then the inputs and outputs
are not removed after the tests are run, this may be useful for debugging.

If the envvar TORCHANI_AMBER_OVERWRITE_EXPECTED=1 then the ``.dat``, ``.xyz``, ``mdcrd``
outputs of the test are saved into the `expect/` directory

If the envvar TORCHANI_SKIP_CUDA_TESTS is set, the tests that use CUDA are skipped
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

import netCDF4 as netcdf
from parameterized import parameterized
import jinja2

this_dir = Path(__file__).parent


@dataclass
class MlmmConfig:
    protocol: tp.Literal["me", "mbispol"] = "me"


@dataclass
class CudaConfig:
    cuaev: bool = True


@dataclass
class RunConfig:
    cuda: CudaConfig | None
    mlmm: MlmmConfig | None
    shake: bool
    vacuum: bool
    float64: bool
    use_amber_neighborlist: bool
    netcdf: bool = False

    @property
    def name(self) -> str:
        parts = [f"MLMM_{self.mlmm.protocol}"] if self.mlmm else ["FullML"]
        parts.append("shake" if self.shake else "noshake")
        parts.append("vacuum" if self.vacuum else "water")
        parts.append("f64" if self.float64 else "f32")
        parts.append("cuda" if self.cuda else "cpu")
        if self.cuda and self.cuda.cuaev:
            parts.append("cuaev")
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
mlmm_configs = (
    None,
    MlmmConfig("me"),
    MlmmConfig("mbispol"),
)
configs: list[RunConfig] = []
for tup in itertools.product(cuda_configs, mlmm_configs, bools, bools, bools, bools):
    config = RunConfig(*tup)
    if config.cuda and "TORCHANI_SKIP_CUDA_TESTS" in os.environ:
        continue

    # Only test Float 64 in a few Mlmm cases, never on FullML. Its too slow.
    if config.float64 and not (
        config.mlmm and config.cuda and config.cuda.cuaev and config.vacuum
    ):
        continue

    # Not implemented
    if config.mlmm and config.use_amber_neighborlist:
        continue

    # Not applicable
    if config.fullml and config.vacuum and config.use_amber_neighborlist:
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
        self.d: tempfile.TemporaryDirectory | None = None

    def tearDown(self) -> None:
        if self.d is not None:
            self.d.cleanup()

    @parameterized.expand(
        [c for c in configs if (c.fullml and not c.use_amber_neighborlist)],
        name_func=name_func,
    )
    def testPmemd(self, config: RunConfig) -> None:
        self._testPmemd(config, f"pmemd_{config.name}")

    # TODO: CPU + Water test fails, not sure why (!) difference is very small,
    # may be some sort of rounding error issue?
    @parameterized.expand(
        [c for c in configs if (c.fullml and c.use_amber_neighborlist)],
        name_func=name_func,
    )
    def testPmemdFromNeighbors(self, config: RunConfig) -> None:
        self._testPmemd(config, f"pmemd_from_neighbors_{config.name}")

    def _testPmemd(self, config: RunConfig, dirname: str) -> None:
        if os.environ.get("TORCHANI_AMBER_KEEP_TEST_DIRS") == "1":
            test_dir = Path(__file__).parent / f"dump_{dirname}"
            test_dir.mkdir(exist_ok=True)
        else:
            self.d = tempfile.TemporaryDirectory()
            test_dir = Path(self.d.name)
        string = env.get_template("mdin.jinja").render(**asdict(config))
        (test_dir / "mdin").write_text(string)
        self._run_engine("pmemd", test_dir)

        self._check_outputs(config, test_dir)

    @parameterized.expand(
        [c for c in configs if (c.fullml and c.use_amber_neighborlist)],
        name_func=name_func,
    )
    def testSanderFromNeighbors(self, config: RunConfig) -> None:
        self._testSander(config, f"sander_from_neighbors_{config.name}")

    @parameterized.expand(
        [c for c in configs if not c.use_amber_neighborlist],
        name_func=name_func,
    )
    def testSander(self, config: RunConfig) -> None:
        self._testSander(config, f"sander_{config.name}")

    @parameterized.expand(["pmemd", "sander"])
    def testNetCDFWorks(self, engine: str) -> None:
        config = RunConfig(
            cuda=None,
            mlmm=None,
            shake=True,
            vacuum=True,
            float64=False,
            use_amber_neighborlist=True,
            netcdf=True,
        )
        string = env.get_template("mdin.jinja").render(**asdict(config))
        self.d = tempfile.TemporaryDirectory()
        test_dir = Path(self.d.name)
        (test_dir / "mdin").write_text(string)
        self._run_engine(engine, test_dir)
        path = test_dir / "mdcrd"
        ds = netcdf.Dataset(str(path), "r", format="NETCDF3_64BIT_OFFSET")
        coordinates = ds["coordinates"][:].data
        self.assertTrue((coordinates > 1e-3).any())

    def _testSander(self, config: RunConfig, dirname: str) -> None:
        if os.environ.get("TORCHANI_AMBER_KEEP_TEST_DIRS") == "1":
            test_dir = Path(__file__).parent / f"dump_{dirname}"
            test_dir.mkdir(exist_ok=True)
        else:
            self.d = tempfile.TemporaryDirectory()
            test_dir = Path(self.d.name)

        # Fix amber neighborlist
        string = env.get_template("mdin.jinja").render(**asdict(config))
        (test_dir / "mdin").write_text(string)
        self._run_engine("sander", test_dir)

        self._check_outputs(config, test_dir)

    def _check_outputs(self, config: RunConfig, test_dir: Path) -> None:
        # Generate expected values
        expect = this_dir / "expect"
        expect.mkdir(exist_ok=True)
        overwrite_expected = os.environ.get("TORCHANI_AMBER_OVERWRITE_EXPECTED") == "1"
        # Never overwrite with amber's neighborlist
        if overwrite_expected and not config.use_amber_neighborlist:
            for f in sorted(test_dir.iterdir()):
                if f.suffix in [".dat", ".xyz"] or f.name == "mdcrd":
                    shutil.copy(f, (expect / config.name).with_suffix(f".{f.name}"))

        expect_files = sorted(expect.glob(f"{config.name}.*"))
        if not expect_files and not overwrite_expected:
            self.fail(f"No expected outputs for {config.name} in {expect}")
        if not config.shake and config.use_amber_neighborlist:
            # Lots of slack for amber neighborlist, since it uses slightly ops
            # for mapping to the central cell.
            rtol = 0.0005 if not config.cuda else 0.0
            atol = 1.5e-3
        elif config.cuda:
            # Extra slack for cuda
            rtol = 0.0
            atol = 1.0002e-3
        else:
            rtol = 1e-7
            atol = 1e-5
        for expect_file in expect_files:
            f = test_dir / expect_file.name.removeprefix(f"{config.name}.")
            self.assertTrue(
                f.is_file(), f"{f.name} was not written, but {expect_file.name} exists"
            )
            expect_arr = self.parse_amber_output(expect_file.read_text(), f.name)
            this_arr = self.parse_amber_output(f.read_text(), f.name)
            assert_allclose(
                this_arr,
                expect_arr,
                rtol=rtol,
                atol=atol,
                err_msg=f"\nDiscrepancy was found on {expect_file.name}\n",
            )

    @staticmethod
    def parse_amber_output(text: str, fname: str) -> NDArray[np.float32]:
        if fname.endswith(".xyz"):
            lines = text.split("\n")
            non_comment = [
                line for line in lines if "ML/MM" not in line and "TORCHANI" not in line
            ]
            text = " ".join(non_comment)
        if fname == "mdcrd":
            text = " ".join(text.split("\n")[1:])  # Get rid of file name
        return np.array([float(s) for s in text.split() if s.strip()], dtype=np.float32)

    def _run_engine(self, name: str, dir: Path) -> None:
        # prmtop and inpcrd correspond to a solvated ALA dipeptide
        this_dir = Path(__file__).parent
        shutil.copy(this_dir / "prmtop", dir / "prmtop")
        shutil.copy(this_dir / "inpcrd", dir / "inpcrd")
        out = subprocess.run(
            [
                name,
                "-i",
                "mdin",
                "-p",
                "prmtop",
                "-c",
                "inpcrd",
                "-o",
                "mdout",
                "-r",
                "restart",
                "-x",
                "mdcrd",
                "-inf",
                "mdinfo",
                "-O",  # Overwrite
            ],
            cwd=dir,
            shell=False,
            capture_output=True,
            text=True,
        )
        if out.returncode != 0:
            print("Process stdout: ", out.stdout)
            print("Process stderr: ", out.stderr)
            mdout = dir / "mdout"
            if mdout.exists():
                print("Amber mdout", mdout.read_text())
            raise RuntimeError("Error when calling sander")


if __name__ == "__main__":
    unittest.main(verbosity=2)
