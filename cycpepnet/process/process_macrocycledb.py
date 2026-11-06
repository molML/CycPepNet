from pathlib import Path
import pandas as pd
from cycpepnet.process._shared import export_representations, processor_cli
from cycpepnet.utils import PHYSICOCHEMICAL_COLUMNS, apply_physicochemical_properties, apply_with_progress, calculate_amide_ratio, get_processed_dataset_dir, get_ring_atoms_count_from_smiles, print_section_header, save_processed_data, standardize_smiles

# Curate the MacrocycleDB binding dataset: one row per compound-target activity
def main(stage="all", original_root="data/original"):
    data_dir   = Path(original_root) / Path("macrocycledb")
    output_dir = get_processed_dataset_dir("macrocycledb", create=True)
    csv_path   = output_dir / "macrocycledb.csv"

    if csv_path.exists():
        print(f"  \n  Processed csv already exists, loading: {csv_path}")
        final_df = pd.read_csv(csv_path)

    else:
        print_section_header("PROCESSING DATA")
        print("")

        # Read the CSV files
        all_df            = pd.read_csv(data_dir / "All.csv")
        bio_activities_df = pd.read_csv(data_dir / "Biological Activities.csv")
        target_all_df     = pd.read_csv(data_dir / "Target-All.csv")
        linker_df         = pd.read_csv(data_dir / "Linker.csv")

        # Select relevant columns from each dataframe (including Total Peptide Count for length)
        all_subset    = all_df[['MCD_ID', 'M_type', 'SMILES', 'InChI Key', 'Molecular Formula', 'Total Peptide Count']].copy()
        bio_subset    = bio_activities_df[['MCD_ID', 'Target', 'A_type', 'A_value', 'doi']].copy()
        target_subset = target_all_df[['Target', 'Target_Name', 'Species', 'Target_type']].copy()
        linker_subset = linker_df[['MCD_ID', 'Linker', 'Linker_type', 'Linker_length', 'Ligand_sml']].copy()

        # Rename columns to standardized format

        all_subset = all_subset.rename(columns={
            'MCD_ID': 'ID',
            'M_type': 'Molecule_Category',
            'SMILES': 'SMILES',
            'InChI Key': 'InChIKey',
            'Molecular Formula': 'Molecular_Formula',
            'Total Peptide Count': 'Peptide_Length'
        })

        bio_subset = bio_subset.rename(columns={
            'MCD_ID': 'ID',
            'Target': 'Target',
            'A_type': 'Activity_Type',
            'A_value': 'Activity_Value',
            'doi'   : 'DOI',
        })

        target_subset = target_subset.rename(columns={
            'Target'     : 'Target',
            'Target_Name': 'Target_Name',
            'Target_type': 'Target_Type',
        })

        linker_subset = linker_subset.rename(columns={
            'MCD_ID'       : 'ID',
            'Linker'       : 'Linker_ID',
            'Linker_type'  : 'Linker_Type',
            'Linker_length': 'Linker_Length',
            'Ligand_sml'   : 'Ligand_SMILES'
        })

        # FIRST MERGE: All.csv + Biological Activities.csv on ID
        merged_step1 = pd.merge(all_subset, bio_subset, on='ID', how='inner')

        # SECOND MERGE: Result + Target-All.csv on Target
        merged_step2 = pd.merge(merged_step1, target_subset, on='Target', how='left')

        # THIRD MERGE: Result + Linker.csv on ID
        merged_step3 = pd.merge(merged_step2, linker_subset, on='ID', how='left')

        # Fill missing Target information with 'Unknown'
        merged_step3['Target_Type'] = merged_step3['Target_Type'].fillna('Unknown')

        # Reorder columns (initial selection - Amide_Ratio added later)
        final_columns = ['ID', 'Molecule_Category', 'SMILES', 'InChIKey', 'Molecular_Formula', 'Peptide_Length', 'Activity_Type', 'Activity_Value', 'DOI',
                         'Target', 'Target_Name', 'Target_Type','Species', 'Linker_ID', 'Linker_Type', 'Linker_Length', 'Ligand_SMILES']

        final_df = merged_step3[final_columns]

        # Filter to keep only 'Synthetic Macrocycle' and 'Cyclic Peptide' types
        final_df = final_df[final_df['Molecule_Category'].isin(['Synthetic Macrocycle', 'Cyclic Peptide', 'Cyclic peptide'])]

        # Filter to remove entries without Activity_Value
        entries_before_activity_filter = len(final_df)
        final_df                       = final_df.dropna(subset=['Activity_Value'])
        entries_removed_no_activity    = entries_before_activity_filter - len(final_df)

        # Normalize Activity_Type to remove duplicates (e.g., Kd vs KD, Ki vs KI)
        final_df['Activity_Type'] = final_df['Activity_Type'].str.strip().str.upper()

        # Calculate Amide Ratio on original SMILES
        final_df['Amide_Ratio'] = apply_with_progress(final_df['SMILES'], calculate_amide_ratio, "Calculating Amide Ratio")

        # Filter non-peptidic entries (AR < 0.3)
        entries_before_ar_filter = len(final_df)
        final_df                 = final_df[final_df['Amide_Ratio'] >= 0.3]
        entries_removed_ar       = entries_before_ar_filter - len(final_df)

        # Classify by Amide Ratio
        final_df['Molecule_Type'] = final_df['Amide_Ratio'].apply(lambda x: 'Peptidic' if x >= 0.7 else 'Semi-peptidic')

        # Filter peptide length (3 to 50 residues)
        entries_before_length_filter = len(final_df)
        final_df                     = final_df[(final_df['Peptide_Length'] >= 3) & (final_df['Peptide_Length'] <= 50)]
        entries_removed_length       = entries_before_length_filter - len(final_df)

        # Standardize SMILES
        final_df['SMILES'] = apply_with_progress(final_df['SMILES'], standardize_smiles, "Standardizing SMILES")

        # Remove entries where SMILES standardization failed
        before_std_filter    = len(final_df)
        final_df             = final_df.dropna(subset=['SMILES']).copy()
        entries_removed_std  = before_std_filter - len(final_df)

        # Calculate Ring Size
        final_df['Ring_Size'] = apply_with_progress(final_df['SMILES'], get_ring_atoms_count_from_smiles, "Calculating Ring Size")

        # Normalize Target names to fix nomenclature inconsistencies
        target_normalization_map = {
            'GTPase KRas': 'KRAS',
            'GTPase NRas': 'NRAS',
            'GTPase HRas': 'HRAS',
            'Furin': 'FURIN',
        }

        final_df['Target'] = final_df['Target'].replace(target_normalization_map)

        # Calculate physicochemical properties
        final_df = apply_physicochemical_properties(final_df, smiles_col='SMILES')

        # Reorder final columns (include Amide_Ratio, Molecule_Type and Ring_Size)
        final_columns = ['ID', 'Molecule_Category', 'SMILES', 'InChIKey', 'Molecular_Formula', 'Peptide_Length', 'Amide_Ratio', 'Molecule_Type', 'Ring_Size', 'Activity_Type',
                         'Activity_Value', 'DOI', 'Target', 'Target_Name', 'Target_Type','Species', 'Linker_ID', 'Linker_Type', 'Linker_Length', 'Ligand_SMILES'] + PHYSICOCHEMICAL_COLUMNS

        final_df = final_df[final_columns]

        print("")
        print("  Filtered Synthetic Macrocycle and Cyclic Peptide molecule type entries")
        print(f"  Removed {entries_removed_no_activity:,} entries without Activity_Value")
        print(f"  Removed {entries_removed_ar:,} non-peptidic entries (Amide Ratio < 0.3)")
        print(f"  Removed {entries_removed_length:,} entries outside peptide length range (3-50, using Total Peptide Count)")
        print(f"  Removed {entries_removed_std:,} entries with invalid SMILES after standardization")
        print("")

        save_processed_data(final_df, output_dir, "macrocycledb.csv")

    # Generate and save molecular representations in separate files
    if stage != "curate":
        export_representations(final_df, output_dir)


if __name__ == "__main__":
    processor_cli(main)
