import argparse
from pathlib import Path

# Make a bulk system for nutmeg
if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="make_bulk")
    parser.add_argument("--num-waters", type=int, default=630)
    parser.add_argument("--solute-path", type=str, default="")
    args = parser.parse_args()
    if not args.solute_path:
        solute_path = Path("water")
        solute = []
    else:
        solute_path = Path(args.solute_path)
        solute = solute_path.read_text().split()
    wat = (Path(__file__).parent / "water.charges").read_text().split()
    for _ in range(args.num_waters):
        solute.extend(wat)
    out_name = f"{solute_path.stem}-bulk-{args.num_waters}.charges"
    out_path = Path(__file__).parent / out_name
    out_path.write_text("\n".join(solute))
