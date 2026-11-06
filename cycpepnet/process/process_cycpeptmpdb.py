from pathlib import Path
import pandas as pd
from cycpepnet.process._shared import export_representations, processor_cli
from cycpepnet.utils import PHYSICOCHEMICAL_COLUMNS, apply_physicochemical_properties, apply_with_progress, calculate_amide_ratio, get_processed_dataset_dir, get_ring_atoms_count_from_smiles, print_section_header, save_processed_data, standardize_smiles

# Curate the CycPeptMPDB permeability dataset: one row per compound-assay measurement
def main(stage="all", original_root="data/original"):
    data_dir   = Path(original_root) / Path("cycpeptmpdb")
    output_dir = get_processed_dataset_dir("cycpeptmpdb", create=True)
    csv_path   = output_dir / "cycpeptmpdb.csv"

    if csv_path.exists():
        print(f"  \n  Processed csv already exists, loading: {csv_path}")
        final_df = pd.read_csv(csv_path)

    else:
        print_section_header("PROCESSING DATA")

        # Read the CSV file
        df = pd.read_csv(data_dir / "CycPeptMPDB_Peptide_All.csv", low_memory=False)

        # Select relevant columns including assay type columns, Source, Monomer_Length, Sequence
        columns_to_extract = ['ID', 'SMILES', 'Sequence', 'Source', 'Monomer_Length', 'Molecule_Shape','PAMPA', 'Caco2', 'MDCK', 'RRCK']
        df_subset = df[columns_to_extract].copy()

        # Convert wide format to long format: one row per compound-assay combination
        assay_columns = ['PAMPA', 'Caco2', 'MDCK', 'RRCK']

        melted_df = df_subset.melt(
            id_vars    = ['ID', 'SMILES', 'Sequence', 'Source', 'Monomer_Length', 'Molecule_Shape'],
            value_vars = assay_columns,
            var_name   = 'Assay',
            value_name = 'Permeability'
        )

        # Remove rows where permeability value is NaN
        before_numeric_filter    = len(melted_df)
        final_df                 = melted_df.dropna(subset=['Permeability']).copy()
        filtered_by_permeability = before_numeric_filter - len(final_df)

        # Remove entries with Permeability == -10 (detection limit artifact)
        entries_before_outlier_filter = len(final_df)
        final_df                      = final_df[final_df['Permeability'] != -10].copy()
        filtered_by_outlier           = entries_before_outlier_filter - len(final_df)

        # Filter peptide length (keep 3-50 residues)
        entries_before_length_filter = len(final_df)
        final_df                     = final_df[(final_df['Monomer_Length'] >= 3) & (final_df['Monomer_Length'] <= 50)].copy()
        filtered_by_length           = entries_before_length_filter - len(final_df)

        # Standardize SMILES
        print("")
        final_df['SMILES'] = apply_with_progress(final_df['SMILES'], standardize_smiles, "Standardizing SMILES")

        # Remove entries where SMILES standardization failed
        before_std_filter = len(final_df)
        final_df          = final_df.dropna(subset=['SMILES']).copy()
        filtered_by_std   = before_std_filter - len(final_df)

        # Calculate Amide Ratio and filter non-peptidic entries (AR >= 0.3)
        final_df['Amide_Ratio']  = apply_with_progress(final_df['SMILES'], calculate_amide_ratio, "Calculating Amide Ratio")
        entries_before_ar_filter = len(final_df)
        final_df                 = final_df[final_df['Amide_Ratio'] >= 0.3].copy()
        filtered_by_amide_ratio  = entries_before_ar_filter - len(final_df)

        # Classify by Amide Ratio
        final_df['Molecule_Type'] = final_df['Amide_Ratio'].apply(lambda x: 'Peptidic' if x >= 0.7 else 'Semi-peptidic')

        # Calculate Ring Size
        final_df['Ring_Size'] = apply_with_progress(final_df['SMILES'], get_ring_atoms_count_from_smiles, "Calculating Ring Size")

        # Calculate physicochemical properties
        final_df = apply_physicochemical_properties(final_df, smiles_col='SMILES')

        # Select final columns in desired order
        final_columns = ["ID", "SMILES", "Sequence", "Source", "Monomer_Length", "Amide_Ratio", "Molecule_Type", "Ring_Size", "Molecule_Shape", "Assay", "Permeability"] + PHYSICOCHEMICAL_COLUMNS
        final_df      = final_df[final_columns]

        print("")
        print("  Reshaped data to long format with one row per compound-assay combination")
        print(f"  Filtered out {filtered_by_permeability:,} entries with NaN Permeability values")
        print(f"  Removed {filtered_by_outlier:,} entries with Permeability == -10 (detection limit)")
        print(f"  Removed {filtered_by_length:,} entries outside peptide length range (3-50 residues)")
        print(f"  Removed {filtered_by_std:,} entries with invalid SMILES after standardization")
        print(f"  Removed {filtered_by_amide_ratio:,} entries with Amide Ratio < 0.3")
        print("")

        save_processed_data(final_df, output_dir, "cycpeptmpdb.csv")

    # Generate and save molecular representations in separate files
    if stage != "curate":
        export_representations(final_df, output_dir)


if __name__ == "__main__":
    processor_cli(main)
