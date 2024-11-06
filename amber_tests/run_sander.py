import subprocess
from pathlib import Path

# prmtop and inpcrd correspond to a solvated ALA dipeptide
this_dir = Path(__file__).parent
for d in sorted(this_dir.iterdir()):
    if not d.is_dir():
        continue
    print(f"Running input {d.name}")
    subprocess.run(
        [
            "sander",
            "-i",
            "input.md.in",
            "-p",
            this_dir / "system.prmtop",
            "-c",
            this_dir / "system.inpcrd",
            "-o",
            "system.md.out",
            "-r",
            "system.restart.nc",
            "-x",
            "system.traj.nc",
            "-O",  # Overwrite
        ],
        cwd=d,
        shell=False,
        check=True,
    )
