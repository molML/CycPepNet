import re
from pathlib import Path
import pandas as pd
from cycpepnet.process._shared import export_representations, processor_cli
from cycpepnet.utils import PHYSICOCHEMICAL_COLUMNS, apply_physicochemical_properties, apply_with_progress, calculate_amide_ratio, get_processed_dataset_dir, get_ring_atoms_count_from_smiles, print_section_header, print_subsection_header, save_processed_data, sequence_to_smiles, standardize_smiles

# Curate the PEPlife2 half-life dataset, building SMILES from sequences where needed
def main(stage="all", original_root="data/original"):
    data_dir   = Path(original_root) / Path("peplife2")
    output_dir = get_processed_dataset_dir("peplife2", create=True)
    csv_path   = output_dir / "peplife2.csv"

    if csv_path.exists():
        print(f"  \n  Processed csv already exists, loading: {csv_path}")
        final_df = pd.read_csv(csv_path)

    else:
        print_section_header("PROCESSING DATA")

        # Read CSV file
        df = pd.read_csv(data_dir / "Peplife 2.0.csv")

        # Select relevant columns
        initial_df = df[['ID', 'PMID', 'Sequence', 'Length', 'Linear/Cyclic', 'N-ter MOD', 'C-ter MOD', 'Chem-MOD', 'Chirality', 'Origin',
                         'Nature', 'Incubation Time', 'Concentration', 'Half Life', 'Assay', 'Protease', 'Test Sample', 'Vivo/Vitro']].copy()

        # Extract Cyclic_Info from Linear/Cyclic
        initial_df['Cyclic_Info'] = initial_df['Linear/Cyclic'].apply(
            lambda value: value[value.index('(')+1:value.index(')')].strip()
            if pd.notna(value) and '(' in str(value) and ')' in str(value)
            else 'Unknown'
        )

        initial_df = initial_df.drop(columns=['Linear/Cyclic'])

        # Filter out entries where Sequence is N.A.
        entries_before_na_filter = len(initial_df)
        final_df                 = initial_df[~initial_df['Sequence'].isin(['N.A.'])].copy()
        filtered_na_sequences    = entries_before_na_filter - len(final_df)

        # Filter out rows where Half Life is missing
        initial_count         = len(final_df)
        final_df              = final_df.dropna(subset=['Half Life']).copy()
        filtered_no_half_life = initial_count - len(final_df)

        # Rename columns first
        final_df = final_df.rename(columns={
            'N-ter MOD'      : 'N_Terminal_Modification',
            'C-ter MOD'      : 'C_Terminal_Modification',
            'Chem-MOD'       : 'Chemical_Modification',
            'Incubation Time': 'Incubation_Time',
            'Half Life'      : 'Half_Life',
            'Test Sample'    : 'Test_Sample'
        })

        # Apply readable names to modifications
        n_term_mapping                      = {'Ac'  : 'Acetylation (Ac)',  'Free': 'No modification'}
        final_df['N_Terminal_Modification'] = final_df['N_Terminal_Modification'].fillna('No modification').replace(n_term_mapping)

        c_term_mapping                      = {'NH2' : 'Amidation (-NH2)', 'Free': 'No modification'}
        final_df['C_Terminal_Modification'] = final_df['C_Terminal_Modification'].fillna('No modification').replace(c_term_mapping)

        final_df['Chemical_Modification']   = final_df['Chemical_Modification'].fillna('No modification')

        # Parse Half Life column into numeric Value and Error
        half_life_values          = []
        half_life_errors          = []
        filtered_qualified_values = 0
        filtered_ranges           = 0

        # Iterate through each value in the Half_Life column and apply parsing logic

        for value in final_df['Half_Life']:

            # Handle NaN
            if pd.isna(value):
                half_life_values.append(None)
                half_life_errors.append(None)
                continue

            value_str = str(value).strip()

            # Handle N.A. and descriptive terms
            if value_str.upper() in ['N.A.'] or \
               any(word in value_str.lower() for word in ['stable', 'intact', 'degraded']):
                half_life_values.append(None)
                half_life_errors.append(None)
                continue

            # Remove parenthetical notes
            if '(' in value_str and ')' in value_str:
                value_str = value_str[:value_str.index('(')].strip()

            # Handle qualified values
            if value_str and value_str[0] in '<>~∼':
                half_life_values.append(None)
                half_life_errors.append(None)
                filtered_qualified_values += 1
                continue

            range_found = False

            # Handle range values
            for pattern in [r'(\d+\.?\d*)\s*to\s*(\d+\.?\d*)', r'(\d+\.?\d*)\s*-\s*(\d+\.?\d*)']:
                match = re.search(pattern, value_str)
                if match:
                    half_life_values.append(None)
                    half_life_errors.append(None)
                    filtered_ranges += 1
                    range_found = True
                    break

            if range_found:
                continue

            half_life_error = None

            # Handle values with ± or +
            if '±' in value_str:
                parts = value_str.split('±', 1)
                value_str, half_life_error = parts[0].strip(), f"± {parts[1].strip()}"

            elif ' + ' in value_str:
                parts = re.split(r'\s*\+\s*', value_str, 1)
                value_str = parts[0].strip()
                half_life_error = f"± {parts[1].strip()}" if len(parts) > 1 else None

            value_clean = value_str.replace('h', '').strip()

            try:
                half_life_values.append(float(value_clean))
                half_life_errors.append(half_life_error)

            except (ValueError, TypeError):
                half_life_values.append(None)
                half_life_errors.append(None)

        final_df['Half_Life_Value'] = half_life_values
        final_df['Half_Life_Error'] = half_life_errors

        final_df = final_df.drop(columns=['Half_Life'])

        entries_before_hl_filter = len(final_df)
        final_df                 = final_df[final_df['Half_Life_Value'].notna()].copy()
        filtered_no_hl_value     = entries_before_hl_filter - len(final_df)

        # Filter peptide length (keep 3-50 residues)
        entries_before_length_filter = len(final_df)
        final_df['Length']           = pd.to_numeric(final_df['Length'], errors='coerce')
        final_df                     = final_df[(final_df['Length'] >= 3) & (final_df['Length'] <= 50)].copy()
        filtered_by_length           = entries_before_length_filter - len(final_df)

        # Generate SMILES from sequences using p2smi
        print_subsection_header("GENERATING SMILES FROM SEQUENCES")

        final_df['SMILES'] = final_df.apply(lambda row: sequence_to_smiles(row['Sequence'], row['Cyclic_Info']), axis=1)

        # Track conversion stats
        total_entries       = len(final_df)
        smiles_success      = final_df['SMILES'].notna().sum()
        smiles_failed       = total_entries - smiles_success

        # Standardize SMILES
        smiles_before_std  = final_df['SMILES'].notna().sum()
        final_df['SMILES'] = apply_with_progress(final_df['SMILES'], standardize_smiles, "Standardizing SMILES")
        smiles_after_std   = final_df['SMILES'].notna().sum()
        smiles_failed_std  = smiles_before_std - smiles_after_std

        # Filter out sequences without valid SMILES
        entries_before_smiles_filter = len(final_df)
        final_df                     = final_df[final_df['SMILES'].notna()].copy()
        filtered_no_smiles           = entries_before_smiles_filter - len(final_df)

        # Calculate Amide Ratio and filter non-peptidic entries (AR >= 0.3)
        final_df['Amide_Ratio']  = apply_with_progress(final_df['SMILES'], calculate_amide_ratio, "Calculating Amide Ratio")
        entries_before_ar_filter = len(final_df)
        final_df                 = final_df[final_df['Amide_Ratio'] >= 0.3].copy()
        filtered_by_amide_ratio  = entries_before_ar_filter - len(final_df)

        # Classify by Amide Ratio
        final_df['Molecule_Type'] = final_df['Amide_Ratio'].apply(lambda x: 'Peptidic' if x >= 0.7 else 'Semi-peptidic')

        # Calculate Ring Size
        final_df['Ring_Size'] = apply_with_progress(final_df['SMILES'], get_ring_atoms_count_from_smiles, "Calculating Ring Size")

        # Categorize Assay
        assay_categories = []

        for assay in final_df['Assay']:

            if pd.isna(assay):
                assay_categories.append('Unknown')
                continue

            assay_lower = str(assay).lower()

            if assay_lower in ['n.a.', 'not mentioned', 'not available', 'not reported']:
                assay_categories.append('Not Available')
            elif any(keyword in assay_lower for keyword in ['lc-ms/ms', 'lc/ms/ms', 'lc-ms-ms', 'uplc-ms/ms', 'hplc-ms/ms', 'hplc–ms/ms', 'hplc, ms-ms']):
                assay_categories.append('LC-MS/MS')
            elif any(keyword in assay_lower for keyword in ['lc-ms', 'lc/ms', 'uplc-ms', 'uplc/ms', 'hplc-ms']):
                assay_categories.append('LC-MS')
            elif 'maldi' in assay_lower:
                assay_categories.append('MALDI-TOF')
            elif 'esi-ms' in assay_lower or 'esi/ms' in assay_lower:
                assay_categories.append('ESI-MS')
            elif 'mass spectrometry' in assay_lower or assay_lower == 'ms':
                assay_categories.append('Mass Spectrometry')
            elif 'elisa' in assay_lower:
                assay_categories.append('ELISA')
            elif 'radioimmunoassay' in assay_lower or assay_lower == 'ria':
                assay_categories.append('RIA')
            elif assay_lower == 'eia' or 'enzyme immunoassay' in assay_lower:
                assay_categories.append('EIA')
            elif 'hplc' in assay_lower and 'ms' not in assay_lower:
                assay_categories.append('HPLC')
            elif 'uplc' in assay_lower and 'ms' not in assay_lower:
                assay_categories.append('UPLC')
            elif 'cell' in assay_lower and 'assay' in assay_lower:
                assay_categories.append('Cell-based Assay')
            elif 'stability' in assay_lower:
                assay_categories.append('Stability Assay')
            elif 'sds' in assay_lower and 'page' in assay_lower:
                assay_categories.append('SDS-PAGE')
            elif 'western' in assay_lower or 'blot' in assay_lower:
                assay_categories.append('Western Blot')
            elif 'radioactivity' in assay_lower or 'scintillation' in assay_lower or 'gamma counter' in assay_lower or 'gamma-counter' in assay_lower:
                assay_categories.append('Radioactivity Assay')
            elif 'dosimetry' in assay_lower:
                assay_categories.append('Dosimetry')
            elif 'imaging' in assay_lower or 'fluorescence' in assay_lower:
                assay_categories.append('Imaging')
            elif any(keyword in assay_lower for keyword in ['uv', 'spectrophotometry', 'absorbance']):
                assay_categories.append('UV-Vis Spectroscopy')
            elif 'nmr' in assay_lower:
                assay_categories.append('NMR')
            else:
                assay_categories.append('Other')

        final_df['Assay_Category'] = assay_categories

        # Categorize Test Sample
        test_sample_categories = []

        for test_sample in final_df['Test_Sample']:

            if pd.isna(test_sample):
                test_sample_categories.append('Unknown')
                continue

            sample_lower = str(test_sample).lower()

            # Specific to general: human first, then specific animals
            if 'human' in sample_lower:
                test_sample_categories.append('Human')
            elif any(word in sample_lower for word in ['rat', 'rats']):
                test_sample_categories.append('Rat')
            elif any(word in sample_lower for word in ['mice', 'mouse']):
                test_sample_categories.append('Mice')
            elif any(word in sample_lower for word in ['dog', 'dogs', 'canine']):
                test_sample_categories.append('Dog')
            elif any(word in sample_lower for word in ['rabbit', 'rabbits']):
                test_sample_categories.append('Rabbit')
            elif any(word in sample_lower for word in ['pig', 'pigs', 'porcine', 'swine', 'minipig']):
                test_sample_categories.append('Pig')
            elif any(word in sample_lower for word in ['bovine', 'cattle', 'calf', 'cow']):
                test_sample_categories.append('Bovine')
            elif 'fetal' in sample_lower or 'foetal' in sample_lower:
                test_sample_categories.append('Fetal')
            else:
                test_sample_categories.append('Other')

        final_df['Test_Sample_Category'] = test_sample_categories

        # Unify 'Other' into 'Others'
        final_df['Assay_Category'] = final_df['Assay_Category'].replace('Other', 'Others')

        # Calculate physicochemical properties
        final_df = apply_physicochemical_properties(final_df, smiles_col='SMILES')

        # Reorder columns (no molecular representations in CSV)
        final_df = final_df[[
            'ID', 'PMID', 'Sequence', 'SMILES', 'Length', 'Amide_Ratio', 'Molecule_Type', 'Ring_Size', 'Cyclic_Info', 'N_Terminal_Modification',
            'C_Terminal_Modification', 'Chemical_Modification', 'Chirality', 'Origin', 'Nature', 'Incubation_Time', 'Concentration', 'Half_Life_Value',
            'Half_Life_Error', 'Assay', 'Assay_Category', 'Protease', 'Test_Sample', 'Test_Sample_Category', 'Vivo/Vitro'] + PHYSICOCHEMICAL_COLUMNS]

        print("")
        print(f"  Filtered out {filtered_na_sequences:,} entries with 'N.A.' or invalid sequence labels")
        print(f"  Filtered out {filtered_no_half_life:,} entries without Half Life data")
        print(f"  Filtered out {filtered_qualified_values:,} entries with qualified values (<, >, ~)")
        print(f"  Filtered out {filtered_ranges:,} entries with range values (e.g., '10-20')")
        print(f"  Filtered out {filtered_no_hl_value:,} entries with no valid Half_Life_Value (non-numeric or empty)")
        print(f"  Filtered out {filtered_by_length:,} entries outside peptide length range (3-50 residues)")
        print(f"  Generated SMILES via p2smi ({smiles_success:,} success, {smiles_failed:,} failed)")
        print(f"  Standardized SMILES ({smiles_after_std:,} success, {smiles_failed_std:,} failed)")
        print(f"  Filtered out {filtered_no_smiles:,} entries without valid SMILES after standardization")
        print(f"  Filtered out {filtered_by_amide_ratio:,} entries with Amide_Ratio < 0.3 (non-peptidic)")
        print("")

        save_processed_data(final_df, output_dir, "peplife2.csv")

    # Generate and save molecular representations in separate files
    if stage != "curate":
        export_representations(final_df, output_dir)


if __name__ == "__main__":
    processor_cli(main)
