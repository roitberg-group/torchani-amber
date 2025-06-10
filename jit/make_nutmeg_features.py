import sys
from pathlib import Path

try:
    from rdkit import Chem
    from rdkit.Chem import rdPartialCharges
except ImportError:
    print(
        "rdkit is required to generate nutmeg features but it could not be found."
        " Please 'conda install -c conda-forge rdkit' or 'pip install rdkit'"
    )


# Order of the Amber PDB files is equal to the prmtop order
def main() -> None:
    # Usage: first argument is a path to the pdb file
    if len(sys.argv) > 1:
        path = Path(sys.argv[1]).resolve()
    else:
        path = Path(__file__).parent / "ace-ala-nme.pdb"
    mol = Chem.MolFromPDBFile(str(path), removeHs=False)
    rdPartialCharges.ComputeGasteigerCharges(mol)
    charges = [atom.GetDoubleProp("_GasteigerCharge") for atom in mol.GetAtoms()]
    # Write charges as "filename.charges"
    path.with_suffix(".charges").write_text("\n".join(map(str, charges)))


if __name__ == "__main__":
    main()
