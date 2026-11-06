import re
from pathlib import Path
import pandas as pd
from rdkit import Chem

# Exocyclic bond SMARTS
_EXOCYCLIC_SMARTS = Chem.MolFromSmarts("[R;!$([+1,-1]~[-1,+1])]-&!@[!R]")

# Dummy atom regexes
_DUMMY_LABELED = re.compile(r"\[(\d+)\*\]")
_DUMMY_BARE    = re.compile(r"\*")

# Return indices of all exocyclic single bonds in the molecule
def _exocyclic_bond_indices(mol: Chem.Mol) -> list[int]:

    matches   = mol.GetSubstructMatches(_EXOCYCLIC_SMARTS)
    seen      = set()
    bond_idxs = []

    for a1, a2 in matches:
        bond = mol.GetBondBetweenAtoms(a1, a2)
        if bond is not None and bond.GetIdx() not in seen:
            bond_idxs.append(bond.GetIdx())
            seen.add(bond.GetIdx())
    return bond_idxs

# Convert a SMILES string to fragSMILES notation
def smiles_to_fragsmiles(smiles: str) -> str | None:

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    bond_idxs = _exocyclic_bond_indices(mol)

    if not bond_idxs:
        return Chem.MolToSmiles(mol)

    dummy_labels = [(i, i) for i in range(len(bond_idxs))]
    frag_mol     = Chem.FragmentOnBonds(mol, bond_idxs, dummyLabels=dummy_labels)
    frag_smi     = Chem.MolToSmiles(frag_mol)
    frag_smi     = _DUMMY_LABELED.sub(r"<\1>", frag_smi)
    frag_smi     = _DUMMY_BARE.sub(r"<0>", frag_smi)

    return frag_smi

# Convert a list of SMILES to fragSMILES
def fragsmiles_batch(smiles_list: list[str], fallback: bool = True) -> list[str]:

    result = []

    for smi in smiles_list:

        fs = smiles_to_fragsmiles(smi)

        if fs is None:
            result.append(smi if fallback else "")
        else:
            result.append(fs)

    return result

# Compute and save fragSMILES for a processed dataset
def save_fragsmiles(df: pd.DataFrame, out_path: Path, smiles_col: str = "SMILES") -> None:

    out_path = Path(out_path)

    if out_path.exists():
        print(f"  fragSMILES already exists, skipping: {out_path}")
        return

    smiles = df[smiles_col].tolist()
    print(f"  Computing fragSMILES for {len(smiles)} molecules...", end="", flush=True)

    frags = fragsmiles_batch(smiles, fallback=True)
    print(" done.")

    out_path.write_text("\n".join(frags))
    print(f"  Saved fragSMILES → {out_path}")
