import sys
import subprocess
from pathlib import Path

if __name__ == "__main__":
    # prmtop and inpcrd correspond to a solvated ALA dipeptide
    this_dir = Path(__file__).parent
    for d in sorted(this_dir.iterdir()):
        if not d.is_dir():
            continue
        if "mlmm" in sys.argv and "mlmm" not in d.name:
            continue
        print(f"Running input {d.name}")
        subprocess.run(
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
                "system.restart.nc",
                "-x",
                "system.traj.nc",
                "-inf",
                "system.mdinfo",
                "-O",  # Overwrite
            ],
            cwd=d,
            shell=False,
            check=True,
        )
