r"""Clean dumped output from tests"""
import shutil
from pathlib import Path

for f in Path(__file__).parent.glob("dump_*"):
    if f.is_dir():
        shutil.rmtree(f)
