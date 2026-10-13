#!/usr/bin/env python3
"""
===============================================================================
Nakdong River — Predict RNA Expression for 2019 Samples (7 total)
===============================================================================
Samples to predict:
  19BD_A, 19BD_B               — Before Dam Opening 2019
  19AD1_A, 19AD1_B, 19AD1_C       — After Dam Opening 2019
  19AD2_A, 19AD2_C           — During Week 1 seawater intrusion 2019

Strategy
--------
1. Train GradientBoosting per N-cycling gene on 6 paired 2020 samples
   that have measured DNA metagenome + RNA metatranscriptome:
     20BD_B, 20BD_C, 20AD1_B, 20AD1_C, 20AD2_B, 20AD2_C

   Features:  env conditions + log1p(DNA root count) + CLR top-N taxa
   Target:    log1p(RNA root raw count)

2. Apply trained models to 2019 DNA + 2019 env to predict 2019 RNA
   root raw counts per gene per sample.

3. Distribute predicted root-level counts across the full taxonomy tree
   in proportion to DNA raw counts (same strategy used for missing
   2020 samples in nakdong_predict_missing_rna.py).

4. Calibrate Score and Identity columns from 2020 training RNA:
     Score    = predicted_count × mean_score_per_read
     Identity = taxon-level mean from training RNA (or global mean fallback)

5. Write one xlsx file per 2019 sample in exact metatranscriptome format.

Outputs → mnt/outputs/
  19BD_A_RNA_predicted_function_taxonomy_profile_long.xlsx
  19BD_B_RNA_predicted_function_taxonomy_profile_long.xlsx
  19AD1_A_RNA_predicted_function_taxonomy_profile_long.xlsx
  19AD1_B_RNA_predicted_function_taxonomy_profile_long.xlsx
  19AD1_C_RNA_predicted_function_taxonomy_profile_long.xlsx
  19AD2_A_RNA_predicted_function_taxonomy_profile_long.xlsx
  19AD2_C_RNA_predicted_function_taxonomy_profile_long.xlsx
===============================================================================
"""

import os, glob, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)

# ── Constants ─────────────────────────────────────────────────────────────────
# 2020 training pairs (have both measured DNA and RNA)
TRAIN_SAMPLES = ["20BD_B",   "20BD_C",
                 "20AD1_B", "20AD1_C",
                 "20AD2_B", "20AD2_C"]

# 2019 samples to predict
PREDICT_SAMPLES = ["19BD_A", "19BD_B",
                   "19AD1_A", "19AD1_B", "19AD1_C",
                   "19AD2_A", "19AD2_C"]

# Environmental feature columns (same in both 2019 and 2020 CSVs)

ENV_COLS = ["Salinity", "DO", "NOx",
            "NH4", "PO4", "Temp"]



TOP_N_TAXA = 15
OUT_DIR    = "mnt/outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# Map each training sample to its site group (for identity calibration)
TRAIN_SITE_MAP = {
    "20BD_B":   "20BD",   "20BD_C":   "20BD",
    "20AD1_B": "20AD1", "20AD1_C": "20AD1",
    "20AD2_B": "20AD2", "20AD2_C": "20AD2",
}

# Map each 2019 predict sample to the nearest 2020 site for identity lookup
PRED_SITE_MAP = {
    "19BD_A":   "20BD",    # freshwater pre-intrusion analog
    "19BD_B":   "20BD",
    "19AD1_A":   "20AD1",  # post-intrusion salinity analog 
    "19AD1_B":   "20AD1",
    "19AD1_C":   "20AD1",
    "19AD2_A": "20AD2",  # during seawater intrusion analog 
    "19AD2_C": "20AD2",
}

DNA_STRIP_20 = "_trim.fastq_paired_function_taxonomy_profile_long.xlsx"
DNA_STRIP_19 = "_trim.fastq_paired_function_taxonomy_profile_long.xlsx"
RNA_STRIP    = "_RNA_trim_paired_function_taxonomy_profile_long.xlsx"


# ══════════════════════════════════════════════════════════════════════════════
# 1.  DATA LOADING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def read_root_counts(xlsx_path):
    """Return (df_full, gene_names, raw_cols, score_cols, ident_cols)."""
    df = pd.read_excel(xlsx_path, index_col=0)
    raw_cols   = [c for c in df.columns if ".3.Raw count" in c]
    score_cols = [c for c in df.columns if ".1.Score"     in c]
    ident_cols = [c for c in df.columns if ".2.Identity"  in c]
    gene_names = [c.replace(".3.Raw count", "") for c in raw_cols]
    return df, gene_names, raw_cols, score_cols, ident_cols


def root_row(df):
    return df[(df[".Rank"] == "norank") & (df[".Taxon name"] == "root")]


def clr_transform(mat, eps=1e-6):
    m     = mat.values.astype(float) + eps
    log_m = np.log(m)
    return pd.DataFrame(log_m - log_m.mean(axis=1, keepdims=True),
                        index=mat.index, columns=mat.columns)


def parse_root_counts(file_dict):
    """
    file_dict: {sample_name: xlsx_path}
    Returns DataFrame: samples × genes (root raw counts)
    """
    totals = {}
    for s, path in file_dict.items():
        df, gnames, rcols, _, _ = read_root_counts(path)
        r = root_row(df)
        if len(r):
            totals[s] = dict(zip(gnames, r[rcols].values[0]))
        else:
            totals[s] = {}
    return pd.DataFrame(totals).T.fillna(0)


def parse_genus_taxa(file_dict):
    """
    Returns DataFrame: samples × genus_taxon_names (summed raw counts across genes)
    """
    taxa_dict = {}
    for s, path in file_dict.items():
        df  = pd.read_excel(path, index_col=0)
        rc  = [c for c in df.columns if ".3.Raw count" in c]
        g   = df[df[".Rank"] == "genus"].copy()
        g["_tot"] = g[rc].sum(axis=1)
        taxa_dict[s] = g.set_index(".Taxon name")["_tot"]
    return pd.DataFrame(taxa_dict).T.fillna(0)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  CALIBRATE score_per_read AND mean_identity FROM 2020 TRAINING RNA
# ══════════════════════════════════════════════════════════════════════════════

def calibrate_from_rna(rna_file_dict):
    """
    Compute per gene:
      spr_df       : mean score-per-read across training RNA samples
      mid_glob     : global mean identity (root level)
      mid_site     : DataFrame gene × site → mean identity
      taxon_id_mean: {gene: {taxon: mean_identity}}
    """
    spr_rows, mid_rows = [], []
    taxon_id = {}

    for s, path in rna_file_dict.items():
        site = TRAIN_SITE_MAP.get(s, s)
        df, gnames, rcols, scols, icols = read_root_counts(path)
        r = root_row(df)
        if not len(r):
            continue

        for g, rc, sc, ic in zip(gnames, rcols, scols, icols):
            rv = r[rc].values[0] if rc in r.columns else 0
            sv = r[sc].values[0] if sc in r.columns else np.nan
            iv = r[ic].values[0] if ic in r.columns else np.nan
            if rv > 0 and not np.isnan(sv):
                spr_rows.append({"Gene": g, "spr": sv / rv})
            if rv > 0 and not np.isnan(iv):
                mid_rows.append({"Gene": g, "Site": site, "Identity": iv})

        # Taxon-level identity
        genus_df = df[df[".Rank"] == "genus"].copy()
        for g, rc, ic in zip(gnames, rcols, icols):
            if ic not in df.columns:
                continue
            sub = genus_df[[".Taxon name", ic, rc]].copy()
            sub = sub[sub[rc] > 0]
            for _, row in sub.iterrows():
                taxon = row[".Taxon name"]
                taxon_id.setdefault(g, {}).setdefault(taxon, []).append(row[ic])

    spr_df   = pd.DataFrame(spr_rows).groupby("Gene")["spr"].mean()
    mid_df   = pd.DataFrame(mid_rows)
    mid_glob = mid_df.groupby("Gene")["Identity"].mean()
    mid_site = (mid_df.groupby(["Gene", "Site"])["Identity"]
                .mean().unstack(fill_value=np.nan))
    taxon_id_mean = {g: {t: np.mean(vs) for t, vs in tv.items()}
                     for g, tv in taxon_id.items()}

    return spr_df, mid_glob, mid_site, taxon_id_mean


# ══════════════════════════════════════════════════════════════════════════════
# 3.  FEATURE MATRIX BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_X(samples, dna_gene_df, dna_taxa_df, env_df, gene, top_taxa):
    """
    Feature matrix for one gene across a list of samples.
    Columns: [env vars | log1p(DNA_gene_count) | CLR top taxa]
    """
    env_vals  = env_df.loc[samples, ENV_COLS].values.astype(float)
    dna_gene  = np.log1p(dna_gene_df.reindex(samples)[gene]
                         .fillna(0).values.astype(float)).reshape(-1, 1)
    taxa_avail = [t for t in top_taxa if t in dna_taxa_df.columns]
    taxa_sub   = dna_taxa_df.reindex(samples)[taxa_avail].fillna(0)
    taxa_clr   = clr_transform(taxa_sub).values

    imp = SimpleImputer(strategy="median")
    env_imp  = imp.fit_transform(env_vals)
    dna_imp  = imp.fit_transform(dna_gene)
    taxa_imp = imp.fit_transform(taxa_clr)

    return np.hstack([env_imp, dna_imp, taxa_imp])


# ══════════════════════════════════════════════════════════════════════════════
# 4.  TRAIN ON 2020 DATA, PREDICT 2019 RNA
# ══════════════════════════════════════════════════════════════════════════════

def train_and_predict(dna_gene_all, dna_taxa_all, env_all, rna_gene_tr,
                      genes, predict_samples):
    """
    For each gene:
      - Train GradBoost on 2020 paired samples (TRAIN_SAMPLES)
      - Predict RNA root count for all 2019 predict_samples

    Returns DataFrame: predict_samples × genes (predicted RNA raw counts)
    """
    # Select top taxa by combined mean across training + prediction samples
    all_used = TRAIN_SAMPLES + predict_samples
    avail_taxa = [t for t in dna_taxa_all.columns
                  if t in dna_taxa_all.reindex(all_used).columns]
    top_taxa = (dna_taxa_all.reindex(all_used)[avail_taxa]
                .fillna(0).mean(0)
                .nlargest(TOP_N_TAXA).index.tolist())

    pred_dict = {s: {} for s in predict_samples}

    for gene in genes:
        # Skip if gene absent in training DNA or RNA
        if gene not in dna_gene_all.columns or gene not in rna_gene_tr.columns:
            for s in predict_samples:
                pred_dict[s][gene] = 0.0
            continue

        # Training features and target
        X_tr = build_X(TRAIN_SAMPLES, dna_gene_all, dna_taxa_all,
                       env_all, gene, top_taxa)
        y_tr = np.log1p(rna_gene_tr.loc[TRAIN_SAMPLES, gene]
                        .fillna(0).values.astype(float))

        if np.nanstd(y_tr) < 1e-8:
            # Constant target → predict the mean
            for s in predict_samples:
                pred_dict[s][gene] = float(np.expm1(np.mean(y_tr)))
            continue

        pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scl", StandardScaler()),
            ("mdl", GradientBoostingRegressor(
                n_estimators=200, max_depth=2, learning_rate=0.08,
                subsample=0.8, random_state=SEED)),
        ])
        pipe.fit(X_tr, y_tr)

        # Predict 2019 samples
        X_te    = build_X(predict_samples, dna_gene_all, dna_taxa_all,
                          env_all, gene, top_taxa)
        y_pred  = np.expm1(pipe.predict(X_te))

        for s, val in zip(predict_samples, y_pred):
            pred_dict[s][gene] = max(0.0, float(val))

    return pd.DataFrame(pred_dict).T[genes]


# ══════════════════════════════════════════════════════════════════════════════
# 5.  DISTRIBUTE COUNTS ACROSS TAXONOMY
# ══════════════════════════════════════════════════════════════════════════════

def distribute_counts(dna_template_df, pred_total, gene,
                      spr_df, mid_glob, mid_site, taxon_id_mean, target_site):
    """
    Distribute predicted root-level RNA count across taxonomy rows
    proportionally to DNA counts.

    Returns (rna_raw, rna_score, rna_ident) as Series indexed like dna_template_df.
    """
    dna_col = gene + ".3.Raw count"
    if dna_col not in dna_template_df.columns:
        z = pd.Series(0.0, index=dna_template_df.index)
        return z, z, z

    dna_vals  = dna_template_df[dna_col].fillna(0).astype(float)
    root_mask = dna_template_df[".Taxon name"] == "root"
    dna_total = dna_vals[root_mask].sum()

    if dna_total <= 0 or pred_total <= 0:
        z = pd.Series(0.0, index=dna_template_df.index)
        return z, z, z

    # Proportional scaling
    rna_raw   = (dna_vals * (pred_total / dna_total)).round().astype(float)

    # Score = raw × mean_score_per_read
    spr       = spr_df.get(gene, 0.01)
    rna_score = rna_raw * spr

    # Identity: taxon lookup → site fallback → global fallback → DNA value
    dna_ic = gene + ".2.Identity"
    try:
        fs = float(mid_site.loc[gene, target_site]) if (
            gene in mid_site.index and target_site in mid_site.columns
        ) else np.nan
    except Exception:
        fs = np.nan
    fg = float(mid_glob.get(gene, 80.0))

    ident_vals = []
    for idx, row in dna_template_df.iterrows():
        if rna_raw.loc[idx] == 0:
            ident_vals.append(0.0)
            continue
        taxon   = row[".Taxon name"]
        tv      = taxon_id_mean.get(gene, {}).get(taxon, np.nan)
        if not np.isnan(tv):
            ident_vals.append(tv)
        elif not np.isnan(fs):
            ident_vals.append(fs)
        elif dna_ic in row and not np.isnan(row[dna_ic]):
            ident_vals.append(float(row[dna_ic]))
        else:
            ident_vals.append(fg)

    rna_ident = pd.Series(ident_vals, index=dna_template_df.index)
    rna_ident = rna_ident.where(rna_raw > 0, 0.0)

    return rna_raw, rna_score, rna_ident


# ══════════════════════════════════════════════════════════════════════════════
# 6.  BUILD AND SAVE OUTPUT XLSX
# ══════════════════════════════════════════════════════════════════════════════

def build_predicted_xlsx(sample_name, dna_xlsx_path, predicted_totals,
                         genes, spr_df, mid_glob, mid_site, taxon_id_mean,
                         ref_rna_xlsx):
    """
    Build and save a predicted RNA xlsx that matches metatranscriptome format.

    Column order is taken from `ref_rna_xlsx` (a real 2020 RNA file).
    Taxonomy rows come from the sample's own DNA file.
    """
    target_site = PRED_SITE_MAP.get(sample_name, "20BD")
    print(f"  Building xlsx for {sample_name} (identity analog: {target_site}) …")

    # Load DNA template (taxonomy skeleton for this sample)
    dna_df, dna_genes, _, _, _ = read_root_counts(dna_xlsx_path)

    # Load reference RNA file for column ordering
    ref_df, ref_genes, ref_rcols, ref_scols, ref_icols = read_root_counts(ref_rna_xlsx)

    # Output: start with taxonomy columns
    out_df = dna_df[[".Rank", ".Taxon name"]].copy()

    # Initialise all RNA columns to 0 in reference column order
    rna_col_order = []
    for g, sc, ic, rc in zip(ref_genes, ref_scols, ref_icols, ref_rcols):
        rna_col_order.extend([sc, ic, rc])
    for col in rna_col_order:
        out_df[col] = 0.0

    # Fill predicted values gene by gene
    for g, sc, ic, rc in zip(ref_genes, ref_scols, ref_icols, ref_rcols):
        pred_total = predicted_totals.get(g, 0.0)
        rna_raw, rna_score, rna_ident = distribute_counts(
            dna_df, pred_total, g,
            spr_df, mid_glob, mid_site, taxon_id_mean, target_site)
        out_df[rc] = rna_raw.values
        out_df[sc] = rna_score.round(6).values
        out_df[ic] = rna_ident.values

    # Write xlsx
    sheet_name = (sample_name + "_RNA_predicted")[:31]
    out_path   = os.path.join(OUT_DIR,
        f"{sample_name}_RNA_predicted_function_taxonomy_profile_long.xlsx")
    out_df.to_excel(out_path, sheet_name=sheet_name, index=True)
    print(f"    Saved → {out_path}  "
          f"({out_df.shape[0]} rows × {out_df.shape[1]} cols)")
    return out_path


# ══════════════════════════════════════════════════════════════════════════════
# 7.  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  Nakdong River — Predict RNA Expression for 2019 Samples")
    print("=" * 70)

    # ─── File paths ──────────────────────────────────────────────────────────
    dna_files_2020 = {s: f"mnt/uploads/{s}{DNA_STRIP_20}"
                      for s in TRAIN_SAMPLES}
    dna_files_2019 = {
        s: f"mnt/uploads/{s}_trim.fastq_paired_function_taxonomy_profile_long.xlsx"
        for s in PREDICT_SAMPLES}
    rna_files_2020 = {s: f"mnt/uploads/{s}_RNA_trim_paired_function_taxonomy_profile_long.xlsx"
                      for s in TRAIN_SAMPLES}

    # Reference RNA file (for column ordering in output)
    ref_rna_file = list(rna_files_2020.values())[0]

    # ─── Load all DNA root counts ─────────────────────────────────────────────
    print("\n[1] Loading 2020 training DNA (metagenome) root counts …")
    dna_gene_20 = parse_root_counts(dna_files_2020)
    print(f"    2020 DNA: {dna_gene_20.shape}  genes × samples")

    print("[2] Loading 2019 prediction DNA (metagenome) root counts …")
    dna_gene_19 = parse_root_counts(dna_files_2019)
    print(f"    2019 DNA: {dna_gene_19.shape}")

    # ─── Load genus-level taxa for CLR features ───────────────────────────────
    print("[3] Loading genus-level taxa abundance for feature engineering …")
    dna_taxa_20 = parse_genus_taxa(dna_files_2020)
    dna_taxa_19 = parse_genus_taxa(dna_files_2019)

    # Align taxa columns (union)
    all_taxa = sorted(set(dna_taxa_20.columns) | set(dna_taxa_19.columns))
    dna_taxa_20 = dna_taxa_20.reindex(columns=all_taxa, fill_value=0)
    dna_taxa_19 = dna_taxa_19.reindex(columns=all_taxa, fill_value=0)
    dna_taxa_all = pd.concat([dna_taxa_20, dna_taxa_19]).fillna(0)
    print(f"    Combined taxa matrix: {dna_taxa_all.shape}")

    # Align gene columns (union, fill 0)
    all_genes_dna = sorted(set(dna_gene_20.columns) | set(dna_gene_19.columns))
    dna_gene_20 = dna_gene_20.reindex(columns=all_genes_dna, fill_value=0)
    dna_gene_19 = dna_gene_19.reindex(columns=all_genes_dna, fill_value=0)
    dna_gene_all = pd.concat([dna_gene_20, dna_gene_19]).fillna(0)

    # ─── Load 2020 RNA (training targets) ────────────────────────────────────
    print("[4] Loading 2020 measured RNA (metatranscriptome) root counts …")
    rna_gene_tr = parse_root_counts(rna_files_2020)
    print(f"    RNA training: {rna_gene_tr.shape}")

    # ─── Load environmental data ──────────────────────────────────────────────
    print("[5] Loading environmental data …")
    env_20 = pd.read_csv("mnt/uploads/2020_Nakdong_Envion.csv")
    env_20["Sample_ID"] = env_20["Sample_ID"].str.strip()
    env_20 = env_20.set_index("Sample_ID")

    env_19 = pd.read_csv("mnt/uploads/2019_Environ.csv", encoding="utf-8-sig")
    env_19["Sample_ID"] = env_19["Sample_ID"].str.strip()
    env_19 = env_19.set_index("Sample_ID")

    # Combine env data (same column names)
    env_all = pd.concat([env_20, env_19])[ENV_COLS].astype(float)
    print(f"    Env features: {env_all.shape}  "
          f"cols={', '.join(ENV_COLS)}")

    # ─── Common genes (DNA ∩ RNA) ─────────────────────────────────────────────
    genes = sorted(set(dna_gene_19.columns) & set(rna_gene_tr.columns))
    print(f"\n[6] Genes shared (2019 DNA ∩ 2020 RNA): {len(genes)}")
    print(f"    {genes}")

    # ─── Calibrate Score/Identity from 2020 training RNA ─────────────────────
    print("\n[7] Calibrating Score-per-read and Identity from 2020 RNA …")
    spr_df, mid_glob, mid_site, taxon_id_mean = calibrate_from_rna(rna_files_2020)
    print(f"    Calibrated {len(spr_df)} genes for score-per-read")
    print(f"    Global mean identities: {mid_glob.round(1).to_dict()}")

    # ─── Train on 2020, predict 2019 ─────────────────────────────────────────
    print("\n[8] Training GradBoost per gene on 2020 data, predicting 2019 …")
    pred_df = train_and_predict(
        dna_gene_all, dna_taxa_all, env_all,
        rna_gene_tr, genes, PREDICT_SAMPLES)

    print("\n  Predicted RNA root counts (2019 samples):")
    print(pred_df.round(0).astype(int).to_string())

    # ─── Write predicted xlsx files ───────────────────────────────────────────
    print("\n[9] Building and writing predicted RNA xlsx files …")
    output_paths = []
    for sample in PREDICT_SAMPLES:
        dna_path     = dna_files_2019[sample]
        pred_totals  = pred_df.loc[sample].to_dict()
        path = build_predicted_xlsx(
            sample, dna_path, pred_totals,
            genes, spr_df, mid_glob, mid_site, taxon_id_mean,
            ref_rna_file)
        output_paths.append(path)

    # ─── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  Done. {len(output_paths)} xlsx files written to {OUT_DIR}/")
    print(f"  Files:")
    for p in output_paths:
        print(f"    {os.path.basename(p)}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

