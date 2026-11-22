r"""Compare the C++ namelist parser in src/fortran_utils.cpp with f90nml

The parser is driven through the torchani_namelist_dump executable, which is built
together with the unit tests. The files in namelists/ are sander and pmemd inputs taken
from the Amber repository
"""

import json
import re
import subprocess
import tempfile
import typing as tp
import unittest
from pathlib import Path

import f90nml
from parameterized import parameterized

this_dir = Path(__file__).parent
DUMP_EXE = this_dir.parent / "build" / "torchani_namelist_dump"
NAMELIST_FILES = sorted((this_dir / "namelists").iterdir())

# f90nml takes a lone "&end" line outside a group as the start of a group named "end".
# These lines appear after non-namelist sections of some inputs (e.g. DISANG=...).
# Replacing them with "/" doesn't change the result of parsing with f90nml otherwise
LONE_END_REGEX = re.compile(r"^[ \t]*[&$]end[ \t]*$", re.IGNORECASE | re.MULTILINE)

EDGE_CASES = [
    ("null", "&g a= /"),
    ("null_comma", "&g a=, /"),
    ("null_middle", "&g a=1,,3 /"),
    ("null_first", "&g a=,3 /"),
    ("trailing_comma", "&g a=1, /"),
    ("repeat", "&g a=3*2 /"),
    ("repeat_null", "&g a=2* , 5 /"),
    ("repeat_string", "&g a=2*'x' /"),
    ("blank_separated", "&g a = 1 2 3 /"),
    ("override", "&g a=1,2,3 a=5 /"),
    ("override_null", "&g a=1 a=, /"),
    ("logicals", "&g a=t c=.true. d=true e=F /"),
    ("numbers", "&g b=1.0d-3 c=.5 d=5. e=1e3 f=-0 g=+3 h=1.0E+02 /"),
    ("strings", "&g a='it''s' b=\"x/y!z\" /"),
    ("end_terminator", "&g a=1 &end"),
    ("dollar_terminator", "$g a=1 $end"),
    ("uppercase", "&G A=1 /"),
    ("comment", "&g\n a=1 ! comment / here\n b=2\n/"),
    ("no_blank_terminator", "&g a=1/"),
    ("junk_after_terminator", "&g a=1 / junk\n&h b=2 /"),
    ("end_then_slash", "&g a=1 &end\n /\n &h b=2 /"),
    ("repeated_group", "title\n&g a=1 /\n&g a=2 /\nDISANG=x\n"),
]


def as_list(value: tp.Any) -> list[tp.Any]:
    return value if isinstance(value, list) else [value]


# Typed, so that e.g. 1, 1.0 and True don't compare equal
def as_typed(groups: dict[str, list[dict[str, list[tp.Any]]]]) -> dict[str, tp.Any]:
    typed_groups: dict[str, tp.Any] = {}
    for name, same_name_groups in groups.items():
        typed_groups[name] = []
        for group in same_name_groups:
            typed_group = {}
            for key, values in group.items():
                typed_group[key] = [(type(v).__name__, v) for v in values]
            typed_groups[name].append(typed_group)
    return typed_groups


def parse_with_f90nml(text: str) -> dict[str, list[dict[str, list[tp.Any]]]]:
    namelist = f90nml.reads(LONE_END_REGEX.sub("/", text))
    groups: dict[str, list[dict[str, list[tp.Any]]]] = {}
    for name, group_or_groups in namelist.items():
        for group in as_list(group_or_groups):
            values = {key: as_list(value) for key, value in group.items()}
            groups.setdefault(name, []).append(values)
    return groups


def parse_with_cpp(path: Path) -> dict[str, list[dict[str, list[tp.Any]]]]:
    proc = subprocess.run(
        [str(DUMP_EXE), str(path)], capture_output=True, text=True, check=True
    )
    groups: dict[str, list[dict[str, list[tp.Any]]]] = {}
    for group in json.loads(proc.stdout):
        groups.setdefault(group["name"], []).append(group["values"])
    return groups


@unittest.skipIf(not DUMP_EXE.is_file(), f"{DUMP_EXE} not found, build the tests first")
class TestNamelistParser(unittest.TestCase):
    def testFilesFound(self) -> None:
        self.assertGreater(len(NAMELIST_FILES), 0)

    @parameterized.expand([(p.name, p) for p in NAMELIST_FILES])
    def testAmberInput(self, _name: str, path: Path) -> None:
        expect = parse_with_f90nml(path.read_text())
        self.assertIn("cntrl", expect)
        self.assertEqual(as_typed(parse_with_cpp(path)), as_typed(expect))

    @parameterized.expand(EDGE_CASES)
    def testEdgeCase(self, _name: str, text: str) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "input"
            path.write_text(text)
            self.assertEqual(
                as_typed(parse_with_cpp(path)), as_typed(parse_with_f90nml(text))
            )


if __name__ == "__main__":
    unittest.main()
