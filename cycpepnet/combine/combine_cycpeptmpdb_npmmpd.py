import pandas as pd
from cycpepnet.process._shared import export_representations, processor_cli
from cycpepnet.utils import PHYSICOCHEMICAL_COLUMNS, get_processed_csv_path, get_processed_dataset_dir, print_section_header, save_processed_data

# Combine the CycPeptMPDB and NPMMPD PAMPA datasets into one curated permeability table
def main(stage="all", original_root="data/original"):

    CV_THRESHOLD = 1.0

    output_dir     = get_processed_dataset_dir("cycpeptmpdb_npmmpd", create=True)
    csv_path       = output_dir / "cycpeptmpdb_npmmpd.csv"

    if csv_path.exists():
        print(f"  Processed CSV already exists, loading: {csv_path}")
        df_combined = pd.read_csv(csv_path)

    else:
        print_section_header("COMBINING PAMPA DATASETS")

        # CycPeptMPDB (PAMPA)
        df_cyc = pd.read_csv(get_processed_csv_path("cycpeptmpdb"))
        df_cyc = df_cyc[df_cyc["Assay"] == "PAMPA"].dropna(subset=["Permeability"]).copy()
        df_cyc = df_cyc.rename(columns={"Ring_Size": "Macrocycle_Ring_Size"})
        df_cyc["Source_Dataset"] = "cycpeptmpdb"

        # NPMMPD (PAMPA, Log Peff)
        df_npm = pd.read_csv(get_processed_csv_path("npmmpd"))
        df_npm = df_npm[(df_npm["Assay"] == "PAMPA") & (df_npm["Endpoint"] == "Log Peff")].dropna(subset=["Value"]).copy()
        df_npm = df_npm.rename(columns={"Value": "Permeability"})
        df_npm["Source_Dataset"] = "npmmpd"

        # Align columns
        common_cols = ["SMILES", "Permeability", "Amide_Ratio", "Molecule_Type", "Macrocycle_Ring_Size",
                       "Monomer_Length", "Source_Dataset"] + PHYSICOCHEMICAL_COLUMNS

        df_cyc = df_cyc[[c for c in common_cols if c in df_cyc.columns]]
        df_npm = df_npm[[c for c in common_cols if c in df_npm.columns]]

        print(f"\n  CycPeptMPDB PAMPA:       {len(df_cyc):,} rows  ({df_cyc['SMILES'].nunique():,} unique SMILES)")
        print(f"  NPMMPD PAMPA (Log Peff): {len(df_npm):,} rows  ({df_npm['SMILES'].nunique():,} unique SMILES)")

        # Combine datasets and handle duplicates
        df_all      = pd.concat([df_cyc, df_npm], ignore_index=True)
        agg_rows    = []
        n_singleton = 0
        n_merged    = 0
        n_excluded  = 0

        for smiles, group in df_all.groupby("SMILES", sort=False):

            perm_values = group["Permeability"].dropna()

            if len(perm_values) == 0:
                continue

            if len(group) == 1:
                agg_rows.append(group.iloc[0].to_dict())
                n_singleton += 1
                continue

            mean_p = perm_values.mean()
            std_p  = perm_values.std()
            cv     = std_p / abs(mean_p) if mean_p != 0 else float("inf")

            if cv > CV_THRESHOLD:
                n_excluded += 1
                continue

            row                   = group.iloc[0].to_dict()
            row["Permeability"]   = mean_p
            row["Source_Dataset"] = "+".join(sorted(group["Source_Dataset"].unique()))
            agg_rows.append(row)
            n_merged += 1

        df_combined = pd.DataFrame(agg_rows).reset_index(drop=True)

        n_cyc = df_combined["Source_Dataset"].str.contains("cycpeptmpdb").sum()
        n_npm = df_combined["Source_Dataset"].str.contains("npmmpd").sum()

        print(f"\n  Cross-dataset deduplication (CV threshold = {CV_THRESHOLD:.1f}):")
        print(f"    Singletons (one source only):      {n_singleton:,}")
        print(f"    Merged cross-dataset (CV ≤ {CV_THRESHOLD:.1f}):  {n_merged:,}")
        print(f"    Excluded cross-dataset (CV > {CV_THRESHOLD:.1f}): {n_excluded:,}")
        print(f"\n  Final dataset: {len(df_combined):,} rows ({df_combined['SMILES'].nunique():,} unique SMILES)")
        print(f"      Contains CycPeptMPDB: {n_cyc:,}")
        print(f"      Contains NPMMPD:      {n_npm:,}")

        save_processed_data(df_combined, output_dir, "cycpeptmpdb_npmmpd.csv")

    if stage != "curate":
        export_representations(df_combined, output_dir)


if __name__ == "__main__":
    processor_cli(main)
