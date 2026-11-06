import re
from pathlib import Path
import pandas as pd
from cycpepnet.process._shared import export_representations, processor_cli
from cycpepnet.utils import PHYSICOCHEMICAL_COLUMNS, apply_physicochemical_properties, apply_with_progress, calculate_amide_ratio, get_processed_dataset_dir, get_ring_atoms_count_from_smiles, print_section_header, save_processed_data, standardize_smiles

# Curate the PepTherDia half-life dataset of approved cyclic peptides
def main(stage="all", original_root="data/original"):
    data_dir   = Path(original_root) / Path("peptherdia")
    output_dir = get_processed_dataset_dir("peptherdia", create=True)
    csv_path   = output_dir / "peptherdia.csv"

    if csv_path.exists():
        print(f"  \n  Processed csv already exists, loading: {csv_path}")
        final_df = pd.read_csv(csv_path)

    else:
        print_section_header("PROCESSING DATA")

        # Read CSV file
        df = pd.read_csv(data_dir / "peptherdia_cyclic.csv")

        # Parse Half Life column to numeric hours
        def parse_half_life_to_hours(hl_str: str) -> float | None:

            # Handle missing values
            if pd.isna(hl_str) or not hl_str:
                return None

            hl_str = str(hl_str).strip().lower()

            # Handle N.A., empty, or descriptive (non-numeric) values
            if hl_str in ['n.a.', 'na', '-', '']:
                return None
            if any(word in hl_str for word in ['cannot', "can't", 'negligible', 'not calculated']):
                return None

            # Handle ranges: "1-6 min" or "1 to 6 hours"
            range_match = re.search(r'(\d+\.?\d*)\s*(?:[-–]|to)\s*(\d+\.?\d*)\s*(min|minutes?|h|hrs?|hours?|days?|d|seconds?|s)', hl_str)

            if range_match:
                val1 = float(range_match.group(1))
                val2 = float(range_match.group(2))
                val  = (val1 + val2) / 2
                unit = range_match.group(3)

            else:
                error_match = re.search(r'(\d+\.?\d*)\s*(?:\(?\s*[±+\-]\s*\d+\.?\d*\s*\)?)?\s*(min|minutes?|h|hrs?|hours?|days?|d|seconds?|s)', hl_str)

                if error_match:
                    val  = float(error_match.group(1))
                    unit = error_match.group(2)

                else:
                    simple_match = re.search(r'(\d+\.?\d*)\s*(min|minutes?|h|hrs?|hours?|days?|d|seconds?|s)', hl_str)

                    if simple_match:
                        val  = float(simple_match.group(1))
                        unit = simple_match.group(2)

                    else:
                        return None

            if unit.startswith('s'):
                return val / 3600
            elif unit.startswith('min'):
                return val / 60
            elif unit.startswith('h'):
                return val
            elif unit.startswith('d'):
                return val * 24

            return None

        # Parse half-life values
        half_life_values     = []
        filtered_no_hl_value = 0

        for hl in df['Half_Life_Raw']:

            hl_hours = parse_half_life_to_hours(hl)
            half_life_values.append(hl_hours)

            if hl_hours is None and pd.notna(hl):
                filtered_no_hl_value += 1

        df['Half_Life_Hours'] = half_life_values

        # Remove rows where half-life values are NaN
        before_numeric_filter    = len(df)
        df                       = df.dropna(subset=['Half_Life_Hours']).copy()
        filtered_no_hl_value     = before_numeric_filter - len(df)

        # Standardize SMILES
        df['SMILES'] = apply_with_progress(df['SMILES'], standardize_smiles, "Standardizing SMILES")

        # Filter entries without valid SMILES
        entries_before_smiles_filter = len(df)
        df                           = df.dropna(subset=['SMILES']).copy()
        filtered_no_smiles           = entries_before_smiles_filter - len(df)

        # Calculate Amide Ratio
        df['Amide_Ratio'] = apply_with_progress(df['SMILES'], calculate_amide_ratio, "Calculating Amide Ratio")

        # Filter non-peptidic entries (AR >= 0.3)
        entries_before_ar_filter = len(df)
        df                       = df[df['Amide_Ratio'] >= 0.3].copy()
        filtered_by_amide_ratio  = entries_before_ar_filter - len(df)

        # Classify by Amide Ratio
        df['Molecule_Type'] = df['Amide_Ratio'].apply(lambda x: 'Peptidic' if x >= 0.7 else 'Semi-peptidic')

        # Calculate Ring Size
        df['Ring_Size'] = apply_with_progress(df['SMILES'], get_ring_atoms_count_from_smiles, "Calculating Ring Size")

        # Convert numeric columns and rename for consistency
        df['Length'] = pd.to_numeric(df['Amino_Acid_Count'], errors='coerce')

        # Filter peptide length (keep 3-50 residues)
        entries_before_length_filter = len(df)
        df                           = df[(df['Length'] >= 3) & (df['Length'] <= 50)].copy()
        filtered_by_length           = entries_before_length_filter - len(df)

        # Combine Lipidation and Glycosylation into Chemical_Modification
        def combine_modifications(row):
            mods = []
            lip = row.get('Lipidation')
            gly = row.get('Glycosylation')

            if pd.notna(lip) and str(lip).lower() not in ['no', 'none', '']:
                mods.append(f"Lipidation: {lip}")
            if pd.notna(gly) and str(gly).lower() not in ['no', 'none', '']:
                mods.append(f"Glycosylation: {gly}")

            return '; '.join(mods) if mods else None

        df['Chemical_Modification'] = df.apply(combine_modifications, axis=1)

        # Normalize Bond Type to unified categories
        def normalize_bond_type(bond):

            if pd.isna(bond):
                return None

            bond_lower = str(bond).lower()

            if 'disulfide' in bond_lower:
                if 'head-to-tail' in bond_lower:
                    return 'Disulfide + Head-to-Tail'
                if '2 ' in bond_lower or '3 ' in bond_lower:
                    return 'Multi-Disulfide'
                return 'Disulfide'
            if 'head-to-tail' in bond_lower:
                return 'Head-to-Tail'
            if 'head-to-side' in bond_lower:
                return 'Head-to-Side-Chain'
            if 'side chain-to-tail' in bond_lower:
                return 'Side-Chain-to-Tail'
            if 'thioether' in bond_lower:
                return 'Thioether'
            if 'amide' in bond_lower:
                return 'Amide'
            return 'Other'

        df['Bond_Type'] = df['Bond_Type'].apply(normalize_bond_type)

        # Normalize Route of Administration to primary route
        def normalize_route(route):
            if pd.isna(route):
                return None

            route_upper = str(route).upper().replace(',', ' ').replace(';', ' ')
            route_upper = route_upper.replace('/', ' ').replace('-', ' ')
            tokens = set(route_upper.split())

            if tokens & {'IV', 'INTRAVENOUS'}:
                return 'IV'
            if tokens & {'SC', 'SUBCUTANEOUS'}:
                return 'SC'
            if tokens & {'IM', 'INTRAMUSCULAR'}:
                return 'IM'
            if 'ORAL' in tokens:
                return 'Oral'
            if 'NASAL' in tokens or 'INTRANASAL' in tokens:
                return 'Nasal'
            if 'TOPICAL' in tokens or 'INFILTRATION' in tokens:
                return 'Topical'
            if 'IT' in tokens or 'INTRATHECAL' in tokens:
                return 'Intrathecal'
            return 'Other'

        df['Route'] = df['Route'].apply(normalize_route)

        # Calculate physicochemical properties
        df = apply_physicochemical_properties(df, smiles_col='SMILES')

        # Reorder columns (standardized schema)
        final_columns = [
            'Name', 'SMILES', 'Sequence', 'Length', 'Amide_Ratio', 'Molecule_Type', 'Ring_Size', 'Bond_Type', 'N_Terminal_Modification', 'C_Terminal_Modification',
            'Chemical_Modification', 'Half_Life_Hours', 'Route', 'Bioavailability', 'Indication', 'Origin',] + PHYSICOCHEMICAL_COLUMNS

        final_df = df[[c for c in final_columns if c in df.columns]].copy()

        print("")
        print(f"  Parsed half-life values ({final_df['Half_Life_Hours'].notna().sum():,} valid, {filtered_no_hl_value:,} could not be parsed)")
        print(f"  Filtered out {filtered_no_hl_value:,} entries with invalid half-life values")
        print(f"  Filtered out {filtered_no_smiles:,} entries with invalid SMILES after standardization")
        print(f"  Filtered out {filtered_by_amide_ratio:,} entries with Amide_Ratio < 0.3 (non-peptidic)")
        print(f"  Removed {filtered_by_length:,} entries outside peptide length range (3-50 residues)")
        print("")

        save_processed_data(final_df, output_dir, "peptherdia.csv")

    # Generate and save molecular representations
    if stage != "curate":
        export_representations(final_df, output_dir)


if __name__ == "__main__":
    processor_cli(main)
