from pathlib import Path
import pandas as pd
from cycpepnet.process._shared import export_representations, processor_cli
from cycpepnet.utils import PHYSICOCHEMICAL_COLUMNS, apply_physicochemical_properties, apply_with_progress, calculate_amide_ratio, get_processed_dataset_dir, get_ring_atoms_count_from_smiles, print_section_header, save_processed_data, standardize_smiles

# Curate the PepMSND half-life dataset
def main(stage="all", original_root="data/original"):
    data_dir   = Path(original_root) / Path("pepmsnd")
    output_dir = get_processed_dataset_dir("pepmsnd", create=True)
    csv_path   = output_dir / "pepmsnd.csv"

    if csv_path.exists():
        print(f"  \n  Processed csv already exists, loading: {csv_path}")
        final_df = pd.read_csv(csv_path)

    else:
        print_section_header("PROCESSING DATA")

        # Read the Excel file
        df = pd.read_excel(data_dir / "Dataset.xlsx")

        # Select relevant columns
        columns_to_extract = ['ID', 'SMILES', 'Sequence', 'Length', 'N-ter MOD', 'C-ter MOD', 'CHEMICAL MODIFICATION',
                              'Cyclic/Linear', 'T1/2(hour)', 'Test Sample', 'In-Vivo/Vitro']

        final_df = df[columns_to_extract].copy()

        # Rename columns for easier handling
        final_df = final_df.rename(columns={
            'N-ter MOD'            : 'N_Terminal_Modification',
            'C-ter MOD'            : 'C_Terminal_Modification',
            'CHEMICAL MODIFICATION': 'Chemical_Modification',
            'Cyclic/Linear'        : 'Cyclic_Linear',
            'T1/2(hour)'           : 'Half_Life_Hours',
            'Test Sample'          : 'Test_Sample',
            'In-Vivo/Vitro'        : 'In_Vivo_Vitro'
        })

        # Rename test sample values for consistency
        final_df['Test_Sample'] = final_df['Test_Sample'].str.strip()
        final_df['Test_Sample'] = final_df['Test_Sample'].str.title()
        final_df['Test_Sample'] = final_df['Test_Sample'].str.replace('Mice', 'Mouse', regex=False)

        # Filter out linear peptides
        initial_count_before_linear = len(final_df)
        final_df                    = final_df[final_df['Cyclic_Linear'] != 'Linear'].copy()
        filtered_linear             = initial_count_before_linear - len(final_df)

        # Filter peptide length (keep 3-50 residues)
        entries_before_length_filter = len(final_df)
        final_df                     = final_df[(final_df['Length'] >= 3) & (final_df['Length'] <= 50)].copy()
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

        # Apply readable names to modifications in the dataframe before saving
        n_term_mapping = {
            'AC-'                : 'Acetylation (AC-)',
            'Pr-(propionylation)': 'Propionylation (Pr-)',
            'Pyr'                : 'Pyroglutamic acid (Pyr)',
            'pGlu-'              : 'Pyroglutamate (pGlu-)'
        }

        final_df['N_Terminal_Modification'] = final_df['N_Terminal_Modification'].fillna('No modification').replace(n_term_mapping)

        # C-Terminal modifications
        c_term_mapping = {
            'amide': 'Amidation (-NH2)'
        }
        final_df['C_Terminal_Modification'] = final_df['C_Terminal_Modification'].fillna('No modification').replace(c_term_mapping)

        # Chemical modifications
        chem_mod_mapping = {
            'Ahx = aminohexanoic acid'   : 'Aminohexanoic acid (Ahx)',
            'Orn,Cha: cyclohexylalanine' : 'Ornithine & Cyclohexylalanine',
            'Thz：Thiazolidine'           : 'Thiazolidine (Thz)'
        }

        final_df['Chemical_Modification'] = final_df['Chemical_Modification'].fillna('No modification').replace(chem_mod_mapping)

        # Calculate physicochemical properties
        final_df = apply_physicochemical_properties(final_df, smiles_col='SMILES')

        # Select final columns in desired order
        final_columns = ['ID', 'SMILES', 'Sequence', 'Length', 'Amide_Ratio', 'Molecule_Type', 'Ring_Size', 'N_Terminal_Modification', 'C_Terminal_Modification',
                         'Chemical_Modification', 'Cyclic_Linear', 'Half_Life_Hours', 'Test_Sample', 'In_Vivo_Vitro'] + PHYSICOCHEMICAL_COLUMNS

        final_df = final_df[final_columns]

        print("")
        print(f"  Filtered {filtered_linear:,} linear peptides")
        print(f"  Removed {filtered_by_length:,} entries outside peptide length range (3-50 residues)")
        print(f"  Removed {filtered_by_std:,} entries with invalid SMILES after standardization")
        print(f"  Removed {filtered_by_amide_ratio:,} entries with Amide Ratio < 0.3")
        print("")

        save_processed_data(final_df, output_dir, "pepmsnd.csv")

    # Generate and save molecular representations in separate files
    if stage != "curate":
        export_representations(final_df, output_dir)


if __name__ == "__main__":
    processor_cli(main)
