from typing import Optional
import numpy as np
import pandas as pd
from rdkit import Chem

# Count amide bonds in a molecule (from Mol object)
def count_amide_bonds(mol: Chem.Mol) -> int:

    if mol is None:
        return 0

    amide_pattern = Chem.MolFromSmarts("[C;X3](=[O;X1])[N;X3]")
    matches       = mol.GetSubstructMatches(amide_pattern)

    return len(matches)

# Get number of atoms in the largest ring
def get_ring_atoms_count(mol: Chem.Mol) -> int:

    if mol is None:
        return 0

    ring_info  = mol.GetRingInfo()
    atom_rings = ring_info.AtomRings()

    if not atom_rings:
        return 0

    largest_ring_size = max(len(ring) for ring in atom_rings)

    return largest_ring_size


# Get ring size from SMILES string
def get_ring_atoms_count_from_smiles(smiles: str) -> Optional[int]:

    if pd.isna(smiles) or not smiles:
        return None

    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None
        return get_ring_atoms_count(mol)
    except Exception:
        return None

# Calculate the Amide Ratio (AR) = 3 × amide bonds / macrocycle ring atoms (Feng et al., NPMMPD)
def calculate_amide_ratio(smiles: str) -> float:

    if pd.isna(smiles) or not smiles:
        return np.nan

    try:
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            return np.nan

        n_amides  = count_amide_bonds(mol)
        ring_size = get_ring_atoms_count(mol)

        if ring_size == 0:
            return np.nan

        ar = (3 * n_amides) / ring_size

        return round(ar, 3)

    except Exception:
        return np.nan

# P2SMI SMILES GENERATION

try:
    from p2smi.utilities.smilesgen import constrained_peptide_smiles, what_constraints
    P2SMI_AVAILABLE = True

except ImportError:
    P2SMI_AVAILABLE = False

# Three-letter to one-letter standard amino acid codes
THREE_TO_ONE = {
    'Ala': 'A', 'Cys': 'C', 'Asp': 'D', 'Glu': 'E', 'Phe': 'F',
    'Gly': 'G', 'His': 'H', 'Ile': 'I', 'Lys': 'K', 'Leu': 'L',
    'Met': 'M', 'Asn': 'N', 'Pro': 'P', 'Gln': 'Q', 'Arg': 'R',
    'Ser': 'S', 'Thr': 'T', 'Val': 'V', 'Trp': 'W', 'Tyr': 'Y',
}


# Normalize a peptide sequence string to the one-letter form P2SMI expects
def _normalize_sequence(seq: str) -> Optional[str]:

    if pd.isna(seq):
        return None

    seq_str = str(seq).strip()
    if not seq_str:
        return None

    if '-' in seq_str:
        tokens = seq_str.split('-')
    elif ' ' in seq_str:
        tokens = seq_str.split()
    else:
        tokens = list(seq_str)

    result = []

    for token in tokens:
        token = token.strip()
        if not token:
            continue

        token_cap = token.capitalize()

        if token_cap in THREE_TO_ONE:
            result.append(THREE_TO_ONE[token_cap])

        elif len(token) == 1 and token.upper() in 'ACDEFGHIKLMNPQRSTVWY':
            result.append(token.upper())

        else:
            result.append(token.upper())

    return ''.join(result) if result else None

# Check if cyclic_info indicates head-to-tail cyclization
def _is_head_to_tail(cyclic_info: str) -> bool:

    if not cyclic_info:
        return False

    info_lower  = cyclic_info.lower().strip()
    ht_patterns = [
        'head-to-tail', 'head to tail', 'h-t', 'ht',
        'backbone', 'n-c', 'n-to-c', 'homodetic'
    ]

    return any(pattern in info_lower for pattern in ht_patterns)

# Convert peptide sequence to SMILES
def sequence_to_smiles(seq: str, cyclic_info: Optional[str] = None, is_cyclic: bool = False) -> Optional[str]:

    if not P2SMI_AVAILABLE:
        return None

    normalized_seq = _normalize_sequence(seq)
    if normalized_seq is None:
        return None

    try:
        is_ht = is_cyclic or (cyclic_info and _is_head_to_tail(cyclic_info))

        # Head-to-tail cyclization
        if is_ht:

            result = constrained_peptide_smiles(normalized_seq, 'HT')

            if isinstance(result, tuple):
                return result[2]

            return result

        # Try to detect disulfide from sequence (if contains C)
        if 'C' in normalized_seq:
            try:
                constraints = what_constraints(normalized_seq)
                for _, pattern in constraints:
                    if pattern.startswith('SS'):
                        result = constrained_peptide_smiles(normalized_seq, pattern)
                        if isinstance(result, tuple):
                            return result[2]
                        return result
            except Exception:
                pass

        # Try auto-detection of any constraint
        try:
            constraints = what_constraints(normalized_seq)
            if constraints:
                for _, pattern in constraints:
                    if pattern == 'HT' or pattern.startswith('SS'):
                        result = constrained_peptide_smiles(normalized_seq, pattern)
                        if isinstance(result, tuple):
                            return result[2]
                        return result
        except Exception:
            pass

        # No cyclic SMILES could be generated
        return None

    except Exception:
        return None
