from pathlib import Path
import pandas as pd
from rdkit import Chem
from cycpepnet.process._shared import export_representations, processor_cli
from cycpepnet.utils import PHYSICOCHEMICAL_COLUMNS, apply_physicochemical_properties, apply_with_progress, get_processed_dataset_dir, print_section_header, save_processed_data, standardize_smiles

# Curate the NPMMPD permeability dataset: one row per compound-assay measurement
def main(stage="all", original_root="data/original"):
    data_dir   = Path(original_root) / Path("npmmpd")
    output_dir = get_processed_dataset_dir("npmmpd", create=True)
    csv_path   = output_dir / "npmmpd.csv"

    if csv_path.exists():
        print(f"  \n  Processed csv already exists, loading: {csv_path}")
        final_df = pd.read_csv(csv_path)

    else:
        print_section_header("PROCESSING DATA")

        # Read the CSV file
        df = pd.read_csv(data_dir / "Overall.csv")

        # Select relevant columns
        columns_to_extract = ['ID', 'Standardise_SMILES', 'Standardized_Value','Standardized_Endpoint',
        'Assay', "Citation", 'Amide_Ratio', 'Macrocycle_Ring_Size', 'Source Type']

        initial_df = df[columns_to_extract].copy()

        # Rename for clarity
        initial_df = initial_df.rename(columns={
            'Standardise_SMILES'            : 'SMILES',
            'Standardized_Value'            : 'Value',
            'Standardized_Endpoint'         : 'Endpoint',
        })

        # Filter out entries with Amide_Ratio < 0.3
        initial_count     = len(initial_df)
        final_df          = initial_df[initial_df['Amide_Ratio'] >= 0.3].copy()
        filtered_by_amide = initial_count - len(final_df)

        # Classify by Amide Ratio
        final_df['Molecule_Type'] = final_df['Amide_Ratio'].apply(lambda x: 'Peptidic' if x >= 0.7 else 'Semi-peptidic')

        # Filter out entries without value
        entries_before_value_filter = len(final_df)
        final_df                    = final_df.dropna(subset=['Value']).copy()
        filtered_by_value           = entries_before_value_filter - len(final_df)

        # Remove sentinel/placeholder values (e.g. -9999)
        entries_before_outlier_filter = len(final_df)
        final_df                      = final_df[final_df['Value'] != -9999].copy()
        filtered_by_outlier           = entries_before_outlier_filter - len(final_df)

        # Standardize SMILES
        print("")
        final_df['SMILES'] = apply_with_progress(final_df['SMILES'], standardize_smiles, "Standardizing SMILES")

        # Remove entries where SMILES standardization failed
        before_std_filter = len(final_df)
        final_df          = final_df.dropna(subset=['SMILES']).copy()
        filtered_by_std   = before_std_filter - len(final_df)

        # Calculate physicochemical properties
        final_df = apply_physicochemical_properties(final_df, smiles_col='SMILES')

        # Count backbone assembly bonds within the macrocyclic ring as sequence length
        _amide_ring_s = Chem.MolFromSmarts('[C;R](=[O])[N;R]')
        _ester_ring_s = Chem.MolFromSmarts('[C;R](=[O])[O;R;!H]')
        # Backbone amide/ester bonds inside the macrocycle, used as the sequence length
        def _count_ring_assembly_bonds(smi: str) -> int | None:
            mol = Chem.MolFromSmiles(str(smi))
            if mol is None:
                return None
            return (len(mol.GetSubstructMatches(_amide_ring_s))
                    + len(mol.GetSubstructMatches(_ester_ring_s)))
        final_df['Seq_Length'] = final_df['SMILES'].apply(_count_ring_assembly_bonds)

        # Select final columns in desired order (no ring size filter - include all cyclic peptides)
        final_columns = ['ID', 'SMILES', 'Value', 'Endpoint', 'Assay', 'Citation', 'Amide_Ratio', 'Molecule_Type', 'Macrocycle_Ring_Size', 'Seq_Length', 'Source Type'] + PHYSICOCHEMICAL_COLUMNS

        final_df = final_df[final_columns]

        print("")
        print(f"  Filtered out {filtered_by_amide:,} entries with Amide_Ratio < 0.3 (non-peptidic)")
        print(f"  Filtered out {filtered_by_value:,} entries without activity value")
        print(f"  Removed {filtered_by_outlier:,} entries with Value == -9999 (sentinel values)")
        print(f"  Removed {filtered_by_std:,} entries with invalid SMILES after standardization")
        print("")

        save_processed_data(final_df, output_dir, "npmmpd.csv")

    # Generate and save molecular representations in separate files
    if stage != "curate":
        export_representations(final_df, output_dir)


if __name__ == "__main__":
    processor_cli(main)
