from pathlib import Path

import pytest

DEFAULT_DUMP_EXE = Path(__file__).parent.parent / "build" / "torchani_namelist_dump"


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--dump-exe",
        type=Path,
        default=DEFAULT_DUMP_EXE,
        help="Path to torchani_namelist_dump, used by test_namelists.py"
        " (default: build/torchani_namelist_dump)",
    )


# Runs after the -k / -m deselection, so deselected tests don't need the executable
@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    dump_exe = config.getoption("dump_exe")
    namelist_tests_selected = any(i.path.name == "test_namelists.py" for i in items)
    if namelist_tests_selected and not dump_exe.is_file():
        raise pytest.UsageError(
            f"{dump_exe} not found\n"
            "If the tests were built in a different build dir, run:\n"
            "    pytest tests/test_namelists.py"
            " --dump-exe <build-dir>/torchani_namelist_dump\n"
            "If they were not built, build them first with:\n"
            "    cmake --build <build-dir> --target torchani_namelist_dump"
        )
