import pandas as pd
from cycpepnet.process._shared import export_representations, processor_cli
from cycpepnet.utils import PHYSICOCHEMICAL_COLUMNS, get_processed_csv_path, get_processed_dataset_dir, print_section_header, save_processed_data

# Combine the PEPlife2, PepMSND and PepTherDia half-life datasets into one curated table
def main(stage="all", original_root="data/original"):
    output_dir     = get_processed_dataset_dir("peplife2_pepmsnd_peptherdia", create=True)
    csv_path       = output_dir / "peplife2_pepmsnd_peptherdia.csv"

    def _aggregate_intra(df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for _, group in df.groupby("SMILES", sort=False):
            hl_values = group["Half_Life_Hours"].dropna()
            if len(hl_values) == 0:
                continue
            row                    = group.iloc[0].to_dict()
            row["Half_Life_Hours"] = hl_values.mean()
            rows.append(row)
        return pd.DataFrame(rows).reset_index(drop=True)


    if csv_path.exists():
        print(f"  Processed CSV already exists, loading: {csv_path}")
        df_combined = pd.read_csv(csv_path)

    else:
        print_section_header("COMBINING HALF-LIFE DATASETS (PEPlife2 + PEPmsnd + PEPtherdia)")

        # PEPlife2
        df_pep2 = pd.read_csv(get_processed_csv_path("peplife2"))
        df_pep2 = df_pep2.rename(columns={"Half_Life_Value": "Half_Life_Hours"})
        df_pep2["Source_Dataset"] = "peplife2"

        # PEPmsnd
        df_msnd = pd.read_csv(get_processed_csv_path("pepmsnd"))
        df_msnd["Source_Dataset"] = "pepmsnd"

        # PEPtherdia
        df_ther = pd.read_csv(get_processed_csv_path("peptherdia"))
        df_ther["Source_Dataset"] = "peptherdia"

        # Align to common columns
        common_cols = ["SMILES", "Sequence", "Length", "Half_Life_Hours", "Amide_Ratio", "Molecule_Type",
                       "Ring_Size", "Source_Dataset"] + PHYSICOCHEMICAL_COLUMNS

        df_pep2 = df_pep2[[c for c in common_cols if c in df_pep2.columns]]
        df_msnd = df_msnd[[c for c in common_cols if c in df_msnd.columns]]
        df_ther = df_ther[[c for c in common_cols if c in df_ther.columns]]

        print(f"\n  PEPlife2:   {len(df_pep2):,} rows  ({df_pep2['SMILES'].nunique():,} unique SMILES)")
        print(f"  PEPmsnd:    {len(df_msnd):,} rows  ({df_msnd['SMILES'].nunique():,} unique SMILES)")
        print(f"  PEPtherdia: {len(df_ther):,} rows  ({df_ther['SMILES'].nunique():,} unique SMILES)")

        df_pep2 = _aggregate_intra(df_pep2)
        df_msnd = _aggregate_intra(df_msnd)
        df_ther = _aggregate_intra(df_ther)

        print("\n  After intra-dataset aggregation (replicate averaging):")
        print(f"    PEPlife2:   {df_pep2['SMILES'].nunique():,} unique SMILES")
        print(f"    PEPmsnd:    {df_msnd['SMILES'].nunique():,} unique SMILES")
        print(f"    PEPtherdia: {df_ther['SMILES'].nunique():,} unique SMILES")

        df_all      = pd.concat([df_pep2, df_msnd, df_ther], ignore_index=True)
        agg_rows    = []
        n_singleton = 0
        n_merged    = 0

        for smiles, group in df_all.groupby("SMILES", sort=False):

            hl_values = group["Half_Life_Hours"].dropna()

            if len(hl_values) == 0:
                continue

            if len(group) == 1:
                agg_rows.append(group.iloc[0].to_dict())
                n_singleton += 1
                continue

            row                    = group.iloc[0].to_dict()
            row["Half_Life_Hours"] = hl_values.mean()
            row["Source_Dataset"]  = "+".join(sorted(group["Source_Dataset"].unique()))
            agg_rows.append(row)
            n_merged += 1

        df_combined = pd.DataFrame(agg_rows).reset_index(drop=True)

        print("\n  Cross-dataset deduplication (no CV filter - inter-lab noise is inherent):")
        print(f"    Singletons (one source only): {n_singleton:,}")
        print(f"    Merged cross-dataset:         {n_merged:,}")
        print(f"\n  Final dataset: {len(df_combined):,} rows ({df_combined['SMILES'].nunique():,} unique SMILES)")

        n_pep2 = df_combined["Source_Dataset"].str.contains("peplife2").sum()
        n_msnd = df_combined["Source_Dataset"].str.contains("pepmsnd").sum()
        n_ther = df_combined["Source_Dataset"].str.contains("peptherdia").sum()

        print(f"      Contains peplife2:   {n_pep2:,}")
        print(f"      Contains pepmsnd:    {n_msnd:,}")
        print(f"      Contains peptherdia: {n_ther:,}")

        save_processed_data(df_combined, output_dir, "peplife2_pepmsnd_peptherdia.csv")

    if stage != "curate":
        export_representations(df_combined, output_dir)


if __name__ == "__main__":
    processor_cli(main)
