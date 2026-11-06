# SMILES curation based on CurateMol from MolForge (https://github.com/molML/molforge).
# Tautomer canonicalization is intentionally disabled: it can alter peptide stereochemistry,
# and large peptides often exceed the tautomer limit, making canonicalization inconsistent.

import re
from typing import Optional, Tuple, List
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit.Chem.rdmolops import RemoveHs, GetFormalCharge

# Neutralization patterns for common charged groups
NEUTRALIZATION_PATTERNS = (
    ("[n+;H]", "n"),
    ("[N+;!H0]", "N"),
    ("[$([O-]);!$([O-][#7])]", "O"),
    ("[S-;X1]", "S"),
    ("[$([N-;X2]S(=O)=O)]", "N"),
    ("[$([N-;X2][C,N]=C)]", "N"),
    ("[n-]", "[nH]"),
    ("[$([S-]=O)]", "S"),
    ("[$([N-]C=O)]", "N"),
)

# Neutralization reactions precompiled for efficiency
_NEUTRALIZATION_REACTIONS: List[Tuple[Chem.Mol, Chem.Mol]] = []
for _smarts, _smiles in NEUTRALIZATION_PATTERNS:

    _reactant = Chem.MolFromSmarts(_smarts)
    _product  = Chem.MolFromSmiles(_smiles, False)

    if _reactant and _product:
        _NEUTRALIZATION_REACTIONS.append((_reactant, _product))


# Initialize salt remover
_SALT_REMOVER = SaltRemover()


# CURATION STEPS

# Desalt molecule
def _desalt_molecule(mol: Chem.Mol) -> Tuple[Optional[Chem.Mol], bool]:

    original_smiles = Chem.MolToSmiles(mol)

    if "." not in original_smiles:
        return mol, True

    stripped_mol = _SALT_REMOVER.StripMol(mol, dontRemoveEverything=True)

    if stripped_mol and stripped_mol.GetNumAtoms() > 0:

        if "." in Chem.MolToSmiles(stripped_mol):
            frags        = Chem.GetMolFrags(stripped_mol, asMols=True)
            stripped_mol = max(frags, key=lambda m: m.GetNumAtoms())

        return stripped_mol, True

    return None, False

# Remove isotope information
def _remove_isotope(mol: Chem.Mol) -> Tuple[Optional[Chem.Mol], bool]:

    for atom in mol.GetAtoms():
        atom.SetIsotope(0)

    smiles  = Chem.MolToSmiles(mol)
    cleaned = re.sub(r"\[([CNOPS])\]", r"\1", smiles)

    mol = Chem.MolFromSmiles(cleaned)

    if mol is None:
        return None, False

    return mol, True

# Remove explicit hydrogens
def _remove_hydrogens(mol: Chem.Mol) -> Tuple[Optional[Chem.Mol], bool]:

    params = Chem.rdmolops.RemoveHsParameters()

    params.removeWithWedgedBond     = True
    params.removeDefiningBondStereo = True
    params.updateExplicitCount      = True

    mol = RemoveHs(mol, params)

    if mol:
        has_no_h = all(atom.GetSymbol() != "H" for atom in mol.GetAtoms())
        return mol, has_no_h

    return mol, False

# Neutralize charges
def _neutralize_molecule(mol: Chem.Mol) -> Tuple[Optional[Chem.Mol], bool]:

    if GetFormalCharge(mol) == 0:
        return mol, True

    for reactant, product in _NEUTRALIZATION_REACTIONS:

        while mol.HasSubstructMatch(reactant):
            rms = AllChem.ReplaceSubstructs(mol, reactant, product)
            mol = rms[0]

    return mol, GetFormalCharge(mol) == 0

# Sanitize molecule
def _sanitize_molecule(mol: Chem.Mol) -> Tuple[Optional[Chem.Mol], bool]:

    sanitize_flags = Chem.SanitizeFlags.SANITIZE_ALL
    sanitize_fail  = Chem.SanitizeMol(
        mol,
        catchErrors = True,
        sanitizeOps = sanitize_flags,
    )

    if sanitize_fail:
        return mol, False

    return mol, True

# Standardize SMILES
def standardize_smiles(smiles: str) -> str | None:

    if pd.isna(smiles) or not smiles:
        return None

    try:
        mol = Chem.MolFromSmiles(str(smiles))

        if mol is None:
            return None

        steps = [
            _desalt_molecule,
            _remove_isotope,
            _remove_hydrogens,
            _neutralize_molecule,
            _sanitize_molecule,
        ]

        for step in steps:
            mol, success = step(mol)
            if not success or mol is None:
                return None

        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)

    except Exception:
        return None
