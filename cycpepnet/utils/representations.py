import logging
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Crippen, rdFingerprintGenerator

logger = logging.getLogger(__name__)

try:
    from mapchiral.mapchiral import encode as _mapchiral_fp
    MAPCHIRAL_AVAILABLE = True
except ImportError:
    MAPCHIRAL_AVAILABLE = False
    _mapchiral_fp = None

# MAP6C (radius=3) and MAP4C (radius=2) share the same underlying package
MAP6C_AVAILABLE = MAPCHIRAL_AVAILABLE
MAP4C_AVAILABLE = MAPCHIRAL_AVAILABLE

try:
    from mxfp.mxfp import MXFPCalculator as _MXFPCalculator
    MXFP_AVAILABLE = True
except ImportError:
    MXFP_AVAILABLE = False
    _MXFPCalculator = None

MXFP_DIM = 217

try:
    from hyper_fingerprints import Encoder as _HDF_Encoder
    HDF_AVAILABLE = True
except ImportError:
    HDF_AVAILABLE = False
    _HDF_Encoder = None

HDF_DIM        = 2048
HDF_DEPTH      = 2
HDF_ATOM_TYPES = ["B", "Br", "C", "Cl", "F", "I", "N", "O", "P", "S", "Se", "Si", "Te"]

# Calculate Morgan fingerprint from SMILES
def calculate_morgan_fingerprint(

    smiles       : str,
    radius       : int  = 3,
    n_bits       : int  = 4096,
    use_features : bool = False,
    use_chirality: bool = True,
    counted      : bool = False

) -> np.ndarray | None:

    if pd.isna(smiles) or not smiles:
        return None

    try:

        mol = Chem.MolFromSmiles(str(smiles))

        if mol is None:
            return None

        if use_features:
            generator = rdFingerprintGenerator.GetMorganGenerator(
                radius                  = radius,
                fpSize                  = n_bits,
                includeChirality        = use_chirality,
                atomInvariantsGenerator = rdFingerprintGenerator.GetMorganFeatureAtomInvGen()
            )
        else:
            generator = rdFingerprintGenerator.GetMorganGenerator(
                radius           = radius,
                fpSize           = n_bits,
                includeChirality = use_chirality
            )

        if counted:
            fp = generator.GetCountFingerprintAsNumPy(mol)
        else:
            fp = generator.GetFingerprintAsNumPy(mol)

        return fp

    except Exception as e:
        logger.debug(f"Failed to compute Morgan fingerprint for '{smiles}': {e}")
        return None

# Calculate MAPC fingerprint from SMILES (radius=3 → MAP6C, radius=2 → MAP4C)
def calculate_mapchiral_fingerprint(

    smiles    : str,
    dimensions: int = 2048,
    radius    : int = 3

) -> np.ndarray | None:

    if not MAPCHIRAL_AVAILABLE:
        raise ImportError("MAPC fingerprints require 'mapchiral' package")

    if pd.isna(smiles) or not smiles:
        return None

    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None

        fp = _mapchiral_fp(mol, max_radius=radius, n_permutations=dimensions)

        return fp

    except Exception as e:
        logger.debug(f"Failed to compute MAPC fingerprint (radius={radius}) for '{smiles}': {e}")
        return None

_mxfp_calc = None

# Calculate the MXFP fingerprint of one SMILES, reusing a lazily created calculator
def calculate_mxfp_fingerprint(smiles: str) -> np.ndarray | None:

    global _mxfp_calc

    if not MXFP_AVAILABLE:
        raise ImportError("MXFP fingerprints require the 'mxfp' package")

    if pd.isna(smiles) or not smiles:
        return None

    if _mxfp_calc is None:
        _mxfp_calc = _MXFPCalculator(dimensionality='2D')

    try:
        fp = _mxfp_calc.mxfp_from_smiles(str(smiles))
        if fp is None:
            return None
        return np.array(fp, dtype=np.float32)
    except Exception as e:
        logger.debug(f"Failed to compute MXFP fingerprint for '{smiles}': {e}")
        return None

# Save MXFP fingerprints to .npy file
def save_mxfp_fingerprints(

    df         : pd.DataFrame,
    output_path: str,
    smiles_col : str = 'SMILES',
    desc       : str = "Calculating MXFP fingerprints"

) -> np.ndarray:

    output_path = Path(output_path)
    if output_path.exists():
        print(f"  MXFP fingerprints already exist, skipping: {output_path}")
        return np.load(output_path)

    fps = []

    for smiles in tqdm(df[smiles_col], desc=f"  {desc}", ncols=100):
        fp = calculate_mxfp_fingerprint(smiles)
        if fp is not None:
            fps.append(fp)
        else:
            fps.append(np.zeros(MXFP_DIM, dtype=np.float32))

    fp_array = np.array(fps, dtype=np.float32)
    np.save(str(output_path), fp_array)
    print(f"Saved: {output_path} (shape: {fp_array.shape})")

    return fp_array

# Load MXFP fingerprints from .npy file
def load_mxfp_fingerprints(input_path: str) -> np.ndarray:
    return np.load(input_path)


_hdf_encoder = None

# Save HDF fingerprints to .npy file
def save_hdf_fingerprints(

    df         : pd.DataFrame,
    output_path: str,
    smiles_col : str       = 'SMILES',
    dimension  : int       = HDF_DIM,
    depth      : int       = HDF_DEPTH,
    seed       : int       = 0,
    atom_types : list[str] | None = None,
    desc       : str       = "Calculating HDF fingerprints"

) -> np.ndarray:

    global _hdf_encoder

    if atom_types is None:
        atom_types = HDF_ATOM_TYPES

    output_path = Path(output_path)

    if output_path.exists():
        print(f"  HDF fingerprints already exist, skipping: {output_path}")
        return np.load(output_path)

    if not HDF_AVAILABLE:
        raise ImportError("HDF fingerprints require the 'hyper-fingerprints' package")

    if (
        _hdf_encoder is None
        or _hdf_encoder.dimension  != dimension
        or _hdf_encoder.depth      != depth
        or _hdf_encoder.seed       != seed
        or _hdf_encoder.atom_types != atom_types
        or not _hdf_encoder.normalize
    ):

        _hdf_encoder = _HDF_Encoder(dimension=dimension, depth=depth, seed=seed, atom_types=atom_types, backend="numpy", normalize=True)

    smiles_list  = df[smiles_col].tolist()
    valid_smiles = [s if (isinstance(s, str) and s) else "C" for s in smiles_list]
    batch_size   = 64
    n            = len(valid_smiles)
    fp_array     = np.empty((n, dimension), dtype=np.float32)
    pbar         = tqdm(total=n, desc=f"  {desc}", ncols=100, unit="mol")

    for start in range(0, n, batch_size):
        chunk = valid_smiles[start : start + batch_size]
        fp_array[start : start + len(chunk)] = _hdf_encoder.encode(chunk).astype(np.float32)
        pbar.update(len(chunk))

    pbar.close()

    np.save(str(output_path), fp_array)
    print(f"Saved: {output_path} (shape: {fp_array.shape})")

    return fp_array

# Load HDF fingerprints from .npy file
def load_hdf_fingerprints(input_path: str) -> np.ndarray:
    return np.load(input_path)

# Calculate PyG graph from SMILES

# Atom features configuration
ATOM_FEATURES = {

    'atomic_num': list(range(1, 119)),
    'chirality' : [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER,
        ],
    'degree'               : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'formal_charge'        : [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    'num_hs'               : [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'num_radical_electrons': [0, 1, 2, 3, 4],
    'hybridization': [
        Chem.rdchem.HybridizationType.UNSPECIFIED,
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
        Chem.rdchem.HybridizationType.OTHER,
    ],
    'is_aromatic': [False, True],
    'is_in_ring' : [False, True],
}

# Bond features configuration
BOND_FEATURES = {

    'bond_type': [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ],

    'stereo': [
        Chem.rdchem.BondStereo.STEREONONE,
        Chem.rdchem.BondStereo.STEREOANY,
        Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOE,
        Chem.rdchem.BondStereo.STEREOCIS,
        Chem.rdchem.BondStereo.STEREOTRANS,
    ],

    'is_conjugated': [False, True],
    'is_in_ring'   : [False, True],
}

# One-hot encode a value given an allowable set
def _one_hot_encode(value: Any, allowable_set: list[Any]) -> list[int]:

    encoding = [0] * (len(allowable_set) + 1)

    if value in allowable_set:
        encoding[allowable_set.index(value)] = 1

    else:
        encoding[-1] = 1

    return encoding

# Extract atom features
def get_atom_features(atom: Chem.Atom) -> list[int]:

    features = []

    features.extend(_one_hot_encode(atom.GetAtomicNum(), ATOM_FEATURES['atomic_num']))
    features.extend(_one_hot_encode(atom.GetChiralTag(), ATOM_FEATURES['chirality']))
    features.extend(_one_hot_encode(atom.GetTotalDegree(), ATOM_FEATURES['degree']))
    features.extend(_one_hot_encode(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge']))
    features.extend(_one_hot_encode(atom.GetTotalNumHs(), ATOM_FEATURES['num_hs']))
    features.extend(_one_hot_encode(atom.GetNumRadicalElectrons(), ATOM_FEATURES['num_radical_electrons']))
    features.extend(_one_hot_encode(atom.GetHybridization(), ATOM_FEATURES['hybridization']))
    features.extend(_one_hot_encode(atom.GetIsAromatic(), ATOM_FEATURES['is_aromatic']))
    features.extend(_one_hot_encode(atom.IsInRing(), ATOM_FEATURES['is_in_ring']))

    return features

# Extract bond features
def get_bond_features(bond: Chem.Bond) -> list[int]:

    features = []

    features.extend(_one_hot_encode(bond.GetBondType(), BOND_FEATURES['bond_type']))
    features.extend(_one_hot_encode(bond.GetStereo(), BOND_FEATURES['stereo']))
    features.extend(_one_hot_encode(bond.GetIsConjugated(), BOND_FEATURES['is_conjugated']))
    features.extend(_one_hot_encode(bond.IsInRing(), BOND_FEATURES['is_in_ring']))

    return features

# Convert SMILES to PyG Data object
def smiles_to_pyg_data(smiles: str):

    if pd.isna(smiles) or not smiles:
        return None

    try:
        import torch
        from torch_geometric.data import Data

        # Convert SMILES to RDKit Mol
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None

        # Get atom features
        atom_features = []
        for atom in mol.GetAtoms():
            atom_features.append(get_atom_features(atom))

        # Get bond connectivity and features
        edge_indices  = []
        edge_features = []

        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            bond_feats = get_bond_features(bond)

            # Add both directions (undirected graph)
            edge_indices.append([i, j])
            edge_indices.append([j, i])
            edge_features.append(bond_feats)
            edge_features.append(bond_feats)

        # Handle molecules with no bonds (single atoms)
        if len(edge_indices) == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr  = torch.zeros((0, get_bond_feature_dim()), dtype=torch.float32)
        else:
            edge_index = torch.tensor(np.array(edge_indices, dtype=np.int64).T,  dtype=torch.long)
            edge_attr  = torch.tensor(np.array(edge_features, dtype=np.float32), dtype=torch.float32)

        return Data(
            x          = torch.tensor(np.array(atom_features, dtype=np.float32), dtype=torch.float32),
            edge_index = edge_index,
            edge_attr  = edge_attr,
        )

    except Exception as e:
        logger.debug(f"Failed to convert SMILES to PyG graph '{smiles}': {e}")
        return None


PHYSICOCHEMICAL_COLUMNS = ['MW', 'LogP', 'TPSA', 'HBD', 'HBA', 'RotBonds', 'FractionCSP3']

# Calculate physicochemical properties from SMILES
def calculate_physicochemical_properties(smiles: str) -> dict[str, float] | None:

    if pd.isna(smiles) or not smiles:
        return None

    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None

        return {
            'MW'          : round(Descriptors.ExactMolWt(mol), 2),
            'LogP'        : round(Crippen.MolLogP(mol), 3),
            'TPSA'        : round(Descriptors.TPSA(mol), 2),
            'HBD'         : Lipinski.NumHDonors(mol),
            'HBA'         : Lipinski.NumHAcceptors(mol),
            'RotBonds'    : Lipinski.NumRotatableBonds(mol),
            'FractionCSP3': round(Lipinski.FractionCSP3(mol), 3),
        }

    except Exception as e:
        logger.debug(f"Failed to compute physicochemical properties for '{smiles}': {e}")
        return None


# Apply physicochemical property calculation to a DataFrame
def apply_physicochemical_properties(

    df        : pd.DataFrame,
    smiles_col: str = 'SMILES',
    desc      : str = "Calculating physicochemical properties"

) -> pd.DataFrame:

    df = df.copy()
    tqdm.pandas(desc=f"  {desc}", ncols=100)

    props = df[smiles_col].progress_apply(calculate_physicochemical_properties)

    props_df = pd.DataFrame(props.tolist(), index=df.index)

    for col in PHYSICOCHEMICAL_COLUMNS:
        df[col] = props_df[col]

    return df

# Save Morgan fingerprints (binary) to .npy file
def save_morgan_fingerprints(

    df           : pd.DataFrame,
    output_path  : str,
    smiles_col   : str  = 'SMILES',
    radius       : int  = 3,
    n_bits       : int  = 4096,
    use_chirality: bool = True,
    desc         : str  = "Calculating ECFP6-Binary fingerprints"

) -> np.ndarray:

    output_path = Path(output_path)
    if output_path.exists():
        print(f"  ECFP6b fingerprints already exist, skipping: {output_path}")
        return np.load(output_path)

    fps = []

    for smiles in tqdm(df[smiles_col], desc=f"  {desc}", ncols=100):

        fp = calculate_morgan_fingerprint(smiles, radius=radius, n_bits=n_bits, use_chirality=use_chirality, counted=False)

        if fp is not None:
            fps.append(fp)

        else:
            fps.append(np.zeros(n_bits, dtype=np.uint8))

    fp_array = np.array(fps, dtype=np.uint8)

    np.save(str(output_path), fp_array)

    print(f"Saved: {output_path} (shape: {fp_array.shape})")

    return fp_array

# Save Morgan fingerprints (count) to .npy file
def save_morgan_count_fingerprints(

    df           : pd.DataFrame,
    output_path  : str,
    smiles_col   : str  = 'SMILES',
    radius       : int  = 3,
    n_bits       : int  = 4096,
    use_chirality: bool = True,
    desc         : str  = "Calculating ECFP6-Count fingerprints"

) -> np.ndarray:

    output_path = Path(output_path)
    if output_path.exists():
        print(f"  ECFP6c fingerprints already exist, skipping: {output_path}")
        return np.load(output_path)

    fps = []

    for smiles in tqdm(df[smiles_col], desc=f"  {desc}", ncols=100):

        fp = calculate_morgan_fingerprint(smiles, radius=radius, n_bits=n_bits, use_chirality=use_chirality, counted=True)

        if fp is not None:
            fps.append(fp)
        else:
            fps.append(np.zeros(n_bits, dtype=np.uint32))

    fp_array = np.array(fps, dtype=np.uint32)

    np.save(str(output_path), fp_array)

    print(f"Saved: {output_path} (shape: {fp_array.shape})")

    return fp_array

# Save PyG graphs to .pt file
def save_pyg_graphs(

    df         : pd.DataFrame,
    output_path: str,
    smiles_col : str = 'SMILES',
    desc       : str = "Generating molecular graphs"

) -> list:

    import torch

    output_path = Path(output_path)
    if output_path.exists():
        print(f"  PyG graphs already exist, skipping: {output_path}")
        return torch.load(output_path, weights_only=False)

    graphs = [
        smiles_to_pyg_data(smiles)
        for smiles in tqdm(df[smiles_col], desc=f"  {desc}", ncols=100)
    ]

    torch.save(graphs, str(output_path))

    print(f"Saved: {output_path} ({len(graphs)} graphs)")

    return graphs

# Save Morgan binary fingerprints at radius=2 (ECFP4) to .npy file
def save_morgan_r2_binary_fingerprints(

    df           : pd.DataFrame,
    output_path  : str,
    smiles_col   : str  = 'SMILES',
    n_bits       : int  = 4096,
    use_chirality: bool = True,
    desc         : str  = "Calculating ECFP4-Binary fingerprints"

) -> np.ndarray:

    output_path = Path(output_path)
    if output_path.exists():
        print(f"  ECFP4b fingerprints already exist, skipping: {output_path}")
        return np.load(output_path)

    fps = []

    for smiles in tqdm(df[smiles_col], desc=f"  {desc}", ncols=100):

        fp = calculate_morgan_fingerprint(smiles, radius=2, n_bits=n_bits, use_chirality=use_chirality, counted=False)

        if fp is not None:
            fps.append(fp)
        else:
            fps.append(np.zeros(n_bits, dtype=np.uint8))

    fp_array = np.array(fps, dtype=np.uint8)
    np.save(str(output_path), fp_array)
    print(f"Saved: {output_path} (shape: {fp_array.shape})")

    return fp_array


# Save Morgan count fingerprints at radius=2 (ECFP4) to .npy file
def save_morgan_r2_count_fingerprints(

    df           : pd.DataFrame,
    output_path  : str,
    smiles_col   : str  = 'SMILES',
    n_bits       : int  = 4096,
    use_chirality: bool = True,
    desc         : str  = "Calculating ECFP4-Count fingerprints"

) -> np.ndarray:

    output_path = Path(output_path)
    if output_path.exists():
        print(f"  ECFP4c fingerprints already exist, skipping: {output_path}")
        return np.load(output_path)

    fps = []

    for smiles in tqdm(df[smiles_col], desc=f"  {desc}", ncols=100):

        fp = calculate_morgan_fingerprint(smiles, radius=2, n_bits=n_bits, use_chirality=use_chirality, counted=True)

        if fp is not None:
            fps.append(fp)
        else:
            fps.append(np.zeros(n_bits, dtype=np.uint32))

    fp_array = np.array(fps, dtype=np.uint32)
    np.save(str(output_path), fp_array)
    print(f"Saved: {output_path} (shape: {fp_array.shape})")

    return fp_array


# Save MAP6C fingerprints to .npy file
def save_map6c_fingerprints(

    df         : pd.DataFrame,
    output_path: str,
    smiles_col : str = 'SMILES',
    dimensions : int = 2048,
    radius     : int = 3,
    desc       : str = "Calculating MAP6C fingerprints"

) -> np.ndarray:

    output_path = Path(output_path)
    if output_path.exists():
        print(f"  MAP6C fingerprints already exist, skipping: {output_path}")
        return np.load(output_path)

    if not MAP6C_AVAILABLE:
        raise ImportError("MAP6C fingerprints require 'mapchiral' package")

    fps = []

    for smiles in tqdm(df[smiles_col], desc=f"  {desc}", ncols=100):

        fp = calculate_mapchiral_fingerprint(smiles, dimensions=dimensions, radius=radius)

        if fp is not None:
            fps.append(fp)

        else:
            fps.append(np.zeros(dimensions, dtype=np.uint64))

    fp_array = np.array(fps, dtype=np.uint64)

    np.save(str(output_path), fp_array)

    print(f"Saved: {output_path} (shape: {fp_array.shape})")

    return fp_array

# Save MAP4C fingerprints (radius=2) to .npy file
def save_map4c_fingerprints(

    df         : pd.DataFrame,
    output_path: str,
    smiles_col : str = 'SMILES',
    dimensions : int = 2048,
    radius     : int = 2,
    desc       : str = "Calculating MAP4C fingerprints"

) -> np.ndarray:

    output_path = Path(output_path)
    if output_path.exists():
        print(f"  MAP4C fingerprints already exist, skipping: {output_path}")
        return np.load(output_path)

    if not MAP4C_AVAILABLE:
        raise ImportError("MAP4C fingerprints require 'mapchiral' package")

    fps = []

    for smiles in tqdm(df[smiles_col], desc=f"  {desc}", ncols=100):

        fp = calculate_mapchiral_fingerprint(smiles, dimensions=dimensions, radius=radius)

        if fp is not None:
            fps.append(fp)

        else:
            fps.append(np.zeros(dimensions, dtype=np.uint64))

    fp_array = np.array(fps, dtype=np.uint64)

    np.save(str(output_path), fp_array)

    print(f"Saved: {output_path} (shape: {fp_array.shape})")

    return fp_array

# Load ECFP6b fingerprints from .npy file
def load_morgan_binary_fingerprints(input_path: str) -> np.ndarray:
    return np.load(input_path)

# Load ECFP6c fingerprints from .npy file
def load_morgan_count_fingerprints(input_path: str) -> np.ndarray:
    return np.load(input_path)

# Load ECFP4b fingerprints from .npy file
def load_morgan_r2_binary_fingerprints(input_path: str) -> np.ndarray:
    return np.load(input_path)

# Load ECFP4c fingerprints from .npy file
def load_morgan_r2_count_fingerprints(input_path: str) -> np.ndarray:
    return np.load(input_path)

# Load MAP6C fingerprints from .npy file
def load_map6c_fingerprints(input_path: str) -> np.ndarray:
    return np.load(input_path)

# Load MAP4C fingerprints from .npy file
def load_map4c_fingerprints(input_path: str) -> np.ndarray:
    return np.load(input_path)

# Load PyG graphs from .pt file
def load_pyg_graphs(input_path: str) -> list:
    import torch
    return torch.load(input_path, weights_only=False)

# Feature dimension information

# Return the dimension of atom feature vectors
def get_atom_feature_dim() -> int:

    dim = 0

    for _, values in ATOM_FEATURES.items():
        dim += len(values) + 1

    return dim

# Return the dimension of bond feature vectors
def get_bond_feature_dim() -> int:

    dim = 0

    for _, values in BOND_FEATURES.items():
        dim += len(values) + 1

    return dim
