#!/usr/bin/env python3

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from taco_final_pool import PROTOCOL_POOL_SIZE


CELL_LINE_MOTIFS = {
    "hepg2": ["CEBPA", "FOXA2", "HNF4A"],
    "k562": ["GATA1", "GATA1::TAL1", "KLF1"],
}


DATASET_LABELS = {
    "real_hepg2": "Real",
    "uniform_hepg2": "Uniform baseline",
    "gc_matched_hepg2": "GC-matched baseline",
    "markov_hepg2": "Markov baseline",
    "tfbs_guided_hepg2": "TFBS-guided",

    "real_k562": "Real",
    "uniform_k562": "Uniform baseline",
    "gc_matched_k562": "GC-matched baseline",
    "markov_k562": "Markov baseline",
    "tfbs_guided_k562": "TFBS-guided",
}


BASE_APPROACH_ORDER = [
    "Real",
    "Uniform baseline",
    "GC-matched baseline",
    "Markov baseline",
    "TFBS-guided",
]

FINAL_APPROACH_ORDER = [
    "Real",
    "Uniform baseline",
    "GC-matched baseline",
    "Markov baseline",
    "TACO",
    "TFBS-guided",
]

# Muted blue/green tones matching the visual language of Figure 4.7.
FULL_PLOT_COLORS = ["#4D6787", "#7DCCAD", "#FFEA88"]


def configure_plot_style():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def dataset_label(dataset: str):
    if dataset in DATASET_LABELS:
        return DATASET_LABELS[dataset]
    match = re.fullmatch(r"taco_(?:hepg2|k562)_(-?\d+)", str(dataset))
    if match:
        return f"TACO seed {int(match.group(1))}"
    return None


def ordered_approaches(values) -> list[str]:
    present = {str(value) for value in values}
    ordered = [value for value in BASE_APPROACH_ORDER[:-1] if value in present]
    taco = sorted(
        (value for value in present if value.startswith("TACO seed ")),
        key=lambda value: int(value.rsplit(" ", 1)[1]),
    )
    ordered.extend(taco)
    if "TFBS-guided" in present:
        ordered.append("TFBS-guided")
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def taco_seed_from_approach(approach: str):
    match = re.fullmatch(r"TACO seed (-?\d+)", str(approach))
    return int(match.group(1)) if match else None


def make_final_figure_data(table: pd.DataFrame, cell_line: str):
    """Aggregate selected TACO seeds for the final thesis occurrence figure."""
    motifs = CELL_LINE_MOTIFS[cell_line]
    seed_rows = table[
        table["approach"].astype(str).str.startswith("TACO seed ")
    ].copy()
    seed_rows["seed"] = seed_rows["approach"].map(taco_seed_from_approach)
    seed_rows = seed_rows.sort_values("seed")
    if not seed_rows.empty and seed_rows[motifs].isna().any().any():
        raise ValueError(f"Missing TACO motif percentage for {cell_line}.")

    present = set(table["approach"].astype(str))
    approach_order = [
        approach
        for approach in FINAL_APPROACH_ORDER
        if approach in present or (approach == "TACO" and not seed_rows.empty)
    ]

    figure_rows = []
    seed_points = []
    for approach in approach_order:
        for motif in motifs:
            if approach == "TACO":
                values = seed_rows[motif].to_numpy(dtype=float)
                mean = float(values.mean())
                sd = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
                for seed, value in zip(seed_rows["seed"], values):
                    seed_points.append(
                        {
                            "approach": "TACO",
                            "motif": motif,
                            "seed": int(seed),
                            "hit_fraction_percent": float(value),
                        }
                    )
            else:
                values = table.loc[
                    table["approach"].astype(str) == approach, motif
                ].to_numpy(dtype=float)
                if len(values) != 1:
                    raise ValueError(
                        f"Expected one {approach!r} row for {cell_line}, found {len(values)}."
                    )
                mean = float(values[0])
                sd = 0.0

            figure_rows.append(
                {
                    "approach": approach,
                    "motif": motif,
                    "mean_percent": mean,
                    "sd_percent": sd,
                }
            )

    return pd.DataFrame(figure_rows), pd.DataFrame(seed_points), approach_order


def make_figure_output_data(
    figure_data: pd.DataFrame,
    seed_points: pd.DataFrame,
    cell_line: str,
    taco_round: int,
) -> pd.DataFrame:
    """Add explicit seed values and population metadata to final figure data."""
    output = figure_data.copy()
    individual_values = []
    for _, row in output.iterrows():
        if row["approach"] == "TACO":
            values = seed_points[seed_points["motif"] == row["motif"]].sort_values("seed")
            individual_values.append(
                ";".join(f"{value:.12g}" for value in values["hit_fraction_percent"])
            )
        else:
            individual_values.append(f"{row['mean_percent']:.12g}")
    output.insert(0, "cell_line", cell_line)
    output.insert(1, "population", "final_same_policy_pool")
    output.insert(2, "effective_round", int(taco_round))
    output["individual_seed_values_percent"] = individual_values
    return output


def read_all_summaries(input_dir: Path) -> pd.DataFrame:
    files = sorted(input_dir.glob("*/tfbs_summary.csv"))

    if not files:
        raise FileNotFoundError(f"No tfbs_summary.csv files found under {input_dir}")

    dfs = []
    required_source_columns = {
        "dataset",
        "motif",
        "num_sequences",
        "sequences_with_at_least_one_hit",
        "fraction_sequences_with_at_least_one_hit",
        "total_hits",
        "mean_hits_per_sequence",
    }
    for file in files:
        df = pd.read_csv(file)
        missing = required_source_columns - set(df.columns)
        if missing:
            raise ValueError(
                f"Invalid current scanner summary {file}; missing columns: "
                f"{sorted(missing)}"
            )
        df["source_file"] = str(file)
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)

    combined = combined.rename(
        columns={
            "num_sequences": "total_sequences",
            "sequences_with_at_least_one_hit": "sequences_with_hit",
            "fraction_sequences_with_at_least_one_hit": "hit_fraction",
        }
    )

    combined["hit_fraction_percent"] = combined["hit_fraction"] * 100.0

    combined["approach"] = combined["dataset"].map(dataset_label)
    combined = combined.dropna(subset=["approach"])

    return combined


def deduplicate_real_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    real_hepg2 and real_k562 occur in multiple scans. Keep one row per dataset/motif.
    For all non-real datasets there should already be one row per dataset/motif.
    """
    group_cols = ["dataset", "motif"]

    numeric_cols = [
        "total_sequences",
        "sequences_with_hit",
        "hit_fraction",
        "hit_fraction_percent",
        "total_hits",
        "mean_hits_per_sequence",
    ]

    meta_cols = ["approach"]
    if "matrix_id" in df.columns:
        meta_cols.append("matrix_id")

    dedup = (
        df[group_cols + numeric_cols + meta_cols]
        .drop_duplicates(subset=group_cols)
        .copy()
    )

    return dedup


def select_taco_seeds(df: pd.DataFrame, seeds: list[int]) -> pd.DataFrame:
    """Keep only requested TACO seed rows while retaining all other datasets."""
    selected = {int(seed) for seed in seeds}
    taco_seed = df["approach"].map(taco_seed_from_approach)
    return df[taco_seed.isna() | taco_seed.isin(selected)].copy()


def make_percentage_table(df: pd.DataFrame, cell_line: str) -> pd.DataFrame:
    motifs = CELL_LINE_MOTIFS[cell_line]

    # Keep all datasets that contain the cell-line name, including arbitrary TACO seeds.
    cell_df = df[df["dataset"].astype(str).str.contains(cell_line, case=False, regex=False)].copy()
    cell_df = cell_df[cell_df["motif"].isin(motifs)].copy()

    pivot = (
        cell_df
        .pivot_table(
            index="approach",
            columns="motif",
            values="hit_fraction_percent",
            aggfunc="first",
        )
        .reset_index()
    )

    for motif in motifs:
        if motif not in pivot.columns:
            pivot[motif] = 0.0

    pivot["approach"] = pd.Categorical(
        pivot["approach"],
        categories=ordered_approaches(pivot["approach"]),
        ordered=True
    )
    pivot = pivot.sort_values("approach")

    # Add n values
    n_map = (
        cell_df[["approach", "total_sequences"]]
        .drop_duplicates("approach")
        .set_index("approach")["total_sequences"]
        .to_dict()
    )
    pivot.insert(1, "n", pivot["approach"].map(n_map).astype(int))

    return pivot[["approach", "n"] + motifs]


def compute_any_all_from_counts(
    input_dir: Path,
    cell_line: str,
    selected_seeds=None,
) -> pd.DataFrame:
    """
    Compute Any TFBS and All selected TFBS from tfbs_per_sequence_counts.csv files.

    The scan_selected_tfbs.py output is expected in long format:
    dataset, sequence_id, motif, matrix_id, hit_count

    For each dataset, this function pivots to one row per sequence and one
    column per selected motif, then computes:
      - any_tfbs_percent: fraction of sequences with at least one selected motif hit
      - all_selected_tfbs_percent: fraction of sequences with at least one hit for all selected motifs
    """
    motifs = CELL_LINE_MOTIFS[cell_line]
    rows = []

    count_files = sorted(input_dir.glob(f"*{cell_line}*/tfbs_per_sequence_counts.csv"))

    for file in count_files:
        counts = pd.read_csv(file)

        required = {"dataset", "sequence_id", "motif", "hit_count"}
        missing = required - set(counts.columns)
        if missing:
            print(f"Skipping {file}: missing columns {missing}")
            continue

        counts = counts[counts["motif"].isin(motifs)].copy()
        counts = counts[counts["dataset"].astype(str).str.contains(cell_line, case=False, regex=False)].copy()

        if counts.empty:
            continue

        for dataset, sub in counts.groupby("dataset"):
            approach = dataset_label(dataset)
            if approach is None:
                continue
            seed = taco_seed_from_approach(approach)
            if seed is not None and selected_seeds is not None:
                if seed not in {int(value) for value in selected_seeds}:
                    continue

            wide = (
                sub.pivot_table(
                    index="sequence_id",
                    columns="motif",
                    values="hit_count",
                    aggfunc="sum",
                    fill_value=0,
                )
                .reset_index()
            )

            for motif in motifs:
                if motif not in wide.columns:
                    wide[motif] = 0

            total = len(wide)
            if total == 0:
                continue

            any_tfbs = (wide[motifs].sum(axis=1) > 0).mean() * 100.0
            all_tfbs = (wide[motifs].gt(0).all(axis=1)).mean() * 100.0
            total_selected_hits = int(wide[motifs].to_numpy().sum())
            any_sequences_with_hit = int((wide[motifs].sum(axis=1) > 0).sum())
            all_sequences_with_hit = int(wide[motifs].gt(0).all(axis=1).sum())

            rows.append({
                "dataset": dataset,
                "approach": approach,
                "total_sequences": total,
                "any_sequences_with_hit": any_sequences_with_hit,
                "any_hit_fraction": any_tfbs / 100.0,
                "any_tfbs_percent": any_tfbs,
                "any_total_hits": total_selected_hits,
                "any_mean_hits_per_sequence": total_selected_hits / total,
                "all_selected_sequences_with_hit": all_sequences_with_hit,
                "all_selected_hit_fraction": all_tfbs / 100.0,
                "all_selected_tfbs_percent": all_tfbs,
            })

    if not rows:
        print(f"Warning: No any/all TFBS values computed for {cell_line}.")
        return pd.DataFrame(
            columns=[
                "dataset",
                "approach",
                "total_sequences",
                "any_sequences_with_hit",
                "any_hit_fraction",
                "any_tfbs_percent",
                "any_total_hits",
                "any_mean_hits_per_sequence",
                "all_selected_sequences_with_hit",
                "all_selected_hit_fraction",
                "all_selected_tfbs_percent",
            ]
        )

    out = pd.DataFrame(rows).drop_duplicates("dataset")
    out["approach"] = pd.Categorical(
        out["approach"],
        categories=ordered_approaches(out["approach"]),
        ordered=True,
    )

    return out.sort_values("approach")


def make_metric_rows(
    df: pd.DataFrame,
    any_all: pd.DataFrame,
    cell_line: str,
) -> pd.DataFrame:
    """Create one tidy row per dataset and motif/combined occurrence metric."""
    motifs = CELL_LINE_MOTIFS[cell_line]
    cell_df = df[
        df["dataset"].astype(str).str.contains(cell_line, case=False, regex=False)
        & df["motif"].isin(motifs)
    ].copy()
    if "matrix_id" not in cell_df.columns:
        cell_df["matrix_id"] = ""

    metric_rows = cell_df[
        [
            "dataset",
            "approach",
            "motif",
            "matrix_id",
            "total_sequences",
            "sequences_with_hit",
            "hit_fraction",
            "hit_fraction_percent",
            "total_hits",
            "mean_hits_per_sequence",
        ]
    ].copy()

    any_rows = any_all.rename(
        columns={
            "any_sequences_with_hit": "sequences_with_hit",
            "any_hit_fraction": "hit_fraction",
            "any_tfbs_percent": "hit_fraction_percent",
            "any_total_hits": "total_hits",
            "any_mean_hits_per_sequence": "mean_hits_per_sequence",
        }
    )[
        [
            "dataset",
            "approach",
            "total_sequences",
            "sequences_with_hit",
            "hit_fraction",
            "hit_fraction_percent",
            "total_hits",
            "mean_hits_per_sequence",
        ]
    ].copy()
    any_rows["motif"] = "ANY_SELECTED"
    any_rows["matrix_id"] = "ANY_SELECTED"

    all_rows = any_all.rename(
        columns={
            "all_selected_sequences_with_hit": "sequences_with_hit",
            "all_selected_hit_fraction": "hit_fraction",
            "all_selected_tfbs_percent": "hit_fraction_percent",
        }
    )[
        [
            "dataset",
            "approach",
            "total_sequences",
            "sequences_with_hit",
            "hit_fraction",
            "hit_fraction_percent",
        ]
    ].copy()
    all_rows["motif"] = "ALL_3_SELECTED"
    all_rows["matrix_id"] = "ALL_3_SELECTED"
    # ALL_3_SELECTED is an occurrence event, not a motif-hit burden.
    all_rows["total_hits"] = np.nan
    all_rows["mean_hits_per_sequence"] = np.nan

    result = pd.concat([metric_rows, any_rows, all_rows], ignore_index=True)
    metric_order = [*motifs, "ANY_SELECTED", "ALL_3_SELECTED"]
    result["motif"] = pd.Categorical(result["motif"], metric_order, ordered=True)
    result["approach"] = pd.Categorical(
        result["approach"],
        ordered_approaches(result["approach"]),
        ordered=True,
    )
    return result.sort_values(["approach", "motif"]).reset_index(drop=True)


def make_taco_seed_summary(
    metric_rows: pd.DataFrame,
    cell_line: str,
    taco_round=None,
) -> pd.DataFrame:
    taco = metric_rows[
        metric_rows["approach"].astype(str).str.startswith("TACO seed ")
    ].copy()
    taco.insert(0, "cell_line", cell_line)
    taco.insert(1, "seed", taco["approach"].map(taco_seed_from_approach).astype(int))
    if taco_round is not None:
        taco.insert(2, "effective_round", int(taco_round))
    taco = taco.sort_values(["seed", "motif"])
    taco["approach"] = taco["approach"].astype(str)
    taco["motif"] = taco["motif"].astype(str)
    return taco.reset_index(drop=True)


def validate_final_taco_metrics(
    metric_rows: pd.DataFrame,
    cell_line: str,
    selected_seeds,
) -> None:
    """Reject incomplete or mislabeled TACO scan populations before aggregation."""
    selected_seeds = sorted(int(seed) for seed in selected_seeds)
    taco = metric_rows[
        metric_rows["approach"].astype(str).str.startswith("TACO seed ")
    ].copy()
    taco["seed"] = taco["approach"].map(taco_seed_from_approach)
    actual_seeds = sorted(taco["seed"].dropna().astype(int).unique().tolist())
    if actual_seeds != selected_seeds:
        raise ValueError(
            f"Expected TACO seeds {selected_seeds} for {cell_line}, "
            f"found {actual_seeds}."
        )

    expected_metrics = {
        *CELL_LINE_MOTIFS[cell_line],
        "ANY_SELECTED",
        "ALL_3_SELECTED",
    }
    for seed, group in taco.groupby("seed"):
        found_metrics = set(group["motif"].astype(str))
        if found_metrics != expected_metrics or len(group) != len(expected_metrics):
            raise ValueError(
                f"Expected one row for each TFBS metric for {cell_line}/seed "
                f"{int(seed)}; found {sorted(found_metrics)}."
            )
        sequence_counts = set(
            pd.to_numeric(group["total_sequences"], errors="raise").tolist()
        )
        if sequence_counts != {PROTOCOL_POOL_SIZE}:
            raise ValueError(
                f"Final TACO TFBS aggregation requires exactly "
                f"{PROTOCOL_POOL_SIZE} sequences for {cell_line}/seed {int(seed)}; "
                f"found {sorted(sequence_counts)}."
            )


def make_dataset_summary(
    metric_rows: pd.DataFrame,
    cell_line: str,
    selected_seeds,
    taco_round=None,
) -> pd.DataFrame:
    """Summarize TACO across seeds while retaining all other datasets unchanged."""
    selected_seeds = [int(seed) for seed in selected_seeds]
    validate_final_taco_metrics(metric_rows, cell_line, selected_seeds)
    taco = make_taco_seed_summary(metric_rows, cell_line)

    measures = [
        "sequences_with_hit",
        "hit_fraction",
        "hit_fraction_percent",
        "total_hits",
        "mean_hits_per_sequence",
    ]
    rows = []
    non_taco = metric_rows[
        ~metric_rows["approach"].astype(str).str.startswith("TACO seed ")
    ]

    def summarize(group: pd.DataFrame, approach: str, is_taco: bool):
        sequence_counts = group["total_sequences"].dropna().unique()
        if len(sequence_counts) != 1:
            raise ValueError(
                f"Expected one sequence count for {cell_line}/{approach}/"
                f"{group['motif'].iloc[0]}, found {sequence_counts.tolist()}."
            )
        row = {
            "cell_line": cell_line,
            "approach": approach,
            "motif": str(group["motif"].iloc[0]),
            "matrix_id": str(group["matrix_id"].iloc[0]),
            "n_seeds": len(group) if is_taco else 1,
            "n_sequences_per_dataset": int(sequence_counts[0]),
            "source_datasets": ";".join(group["dataset"].astype(str)),
        }
        for measure in measures:
            values = pd.to_numeric(group[measure], errors="coerce").dropna()
            if values.empty:
                row[measure] = np.nan
                row[f"{measure}_sd_across_seeds"] = np.nan
                row[f"{measure}_min_seed"] = np.nan
                row[f"{measure}_max_seed"] = np.nan
            else:
                row[measure] = float(values.mean()) if is_taco else float(values.iloc[0])
                row[f"{measure}_sd_across_seeds"] = (
                    float(values.std(ddof=1)) if is_taco and len(values) > 1 else np.nan
                )
                row[f"{measure}_min_seed"] = float(values.min())
                row[f"{measure}_max_seed"] = float(values.max())
        rows.append(row)

    for _, row in non_taco.iterrows():
        summarize(row.to_frame().T, str(row["approach"]), False)
    for _, group in taco.groupby("motif", sort=False, observed=True):
        summarize(group, "TACO", True)

    result = pd.DataFrame(rows)
    metric_order = [*CELL_LINE_MOTIFS[cell_line], "ANY_SELECTED", "ALL_3_SELECTED"]
    result["approach"] = pd.Categorical(
        result["approach"], FINAL_APPROACH_ORDER, ordered=True
    )
    result["motif"] = pd.Categorical(result["motif"], metric_order, ordered=True)
    result = result.sort_values(["approach", "motif"]).reset_index(drop=True)
    if taco_round is not None:
        result.insert(2, "taco_effective_round", int(taco_round))
    return result


def plot_motif_percentages(
    table: pd.DataFrame,
    cell_line: str,
    output_dir: Path,
):
    configure_plot_style()
    motifs = CELL_LINE_MOTIFS[cell_line]
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_df = table.copy()
    if plot_df.empty:
        raise ValueError(f"No rows available for the final {cell_line} figure.")

    figure_data, seed_points, approach_order = make_final_figure_data(
        plot_df, cell_line
    )
    fig, ax = plt.subplots(figsize=(12, 5.5))
    x = np.arange(len(approach_order))
    width = 0.82 / len(motifs)
    offsets = (np.arange(len(motifs)) - (len(motifs) - 1) / 2) * width

    for motif, color, offset in zip(motifs, FULL_PLOT_COLORS, offsets):
        subset = (
            figure_data[figure_data["motif"] == motif]
            .set_index("approach")
            .loc[approach_order]
        )
        values = subset["mean_percent"].to_numpy(dtype=float)
        errors = subset["sd_percent"].fillna(0.0).to_numpy(dtype=float)
        ax.bar(
            x + offset,
            values,
            width=width,
            yerr=errors,
            color=color,
            edgecolor="black",
            linewidth=0.5,
            capsize=2.5,
            label=motif,
            zorder=3,
        )

        if "TACO" in approach_order and not seed_points.empty:
            motif_points = seed_points[seed_points["motif"] == motif].sort_values("seed")
            taco_x = x[approach_order.index("TACO")] + offset
            for index, value in enumerate(motif_points["hit_fraction_percent"]):
                # A true zero has no valid position on a logarithmic axis. It remains
                # unchanged in all tables and bar data and is therefore not plotted.
                if value <= 0:
                    continue
                jitter = (index - (len(motif_points) - 1) / 2) * 0.012
                ax.scatter(
                    taco_x + jitter,
                    value,
                    s=17,
                    facecolor="white",
                    edgecolor="black",
                    linewidth=0.45,
                    zorder=5,
                )

    ax.set_xlabel("")
    ax.set_ylabel("Sequences with at least one selected FIMO match (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(approach_order, rotation=35, ha="right")
    ax.legend(title="Motif", frameon=False)
    ax.grid(axis="y", linestyle=":", linewidth=0.7)
    ax.set_axisbelow(True)
    # Keep true zeros at zero (and therefore invisible on a log axis) instead
    # of introducing an artificial positive offset.
    ax.set_yscale("log")
    ax.set_ylim(0.01, 105)
    ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=2))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    pdf_path = output_dir / f"{cell_line}_tfbs_percentages_motifs_only_rescan.pdf"
    png_path = output_dir / f"{cell_line}_tfbs_percentages_motifs_only_rescan.png"

    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, dpi=600, bbox_inches="tight")
    plt.close()

    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")
    return figure_data, seed_points


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        default="results/tfbs_analysis/rescan_maxstored10m",
        help="Directory containing rescan subdirectories with tfbs_summary.csv files.",
    )
    parser.add_argument(
        "--output_dir",
        default="results/tfbs_analysis/rescan_maxstored10m",
        help="Directory for combined CSV outputs.",
    )
    parser.add_argument(
        "--figure_dir",
        default="figures/tfbs_analysis/rescan_maxstored10m",
        help="Directory for PDF/PNG plots.",
    )
    parser.add_argument(
        "--taco-seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4],
        help="TACO seeds to aggregate (default: 0 1 2 3 4).",
    )
    parser.add_argument(
        "--taco-round",
        type=int,
        default=100,
        help="Effective TACO round represented by the scan inputs (default: 100).",
    )

    return parser.parse_args(argv)


def main():
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    figure_dir = Path(args.figure_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    combined = read_all_summaries(input_dir)
    combined = deduplicate_real_rows(combined)
    combined = select_taco_seeds(combined, args.taco_seeds)

    combined_path = output_dir / "combined_tfbs_summary_rescan.csv"
    combined.to_csv(combined_path, index=False)
    print(f"Saved {combined_path}")

    for cell_line in ["hepg2", "k562"]:
        table = make_percentage_table(combined, cell_line)
        if table.empty:
            print(f"No {cell_line} datasets found. Skipping cell line.")
            continue

        any_all = compute_any_all_from_counts(
            input_dir, cell_line, selected_seeds=args.taco_seeds
        )
        if any_all.empty:
            raise ValueError(
                f"No per-sequence TFBS counts found for {cell_line}; "
                "Any/All 3 selected TFBS cannot be computed."
            )
        table = table.merge(
            any_all[["approach", "any_tfbs_percent", "all_selected_tfbs_percent"]],
            on="approach",
            how="left",
        )

        metric_rows = make_metric_rows(combined, any_all, cell_line)
        validate_final_taco_metrics(metric_rows, cell_line, args.taco_seeds)
        table_path = output_dir / f"{cell_line}_tfbs_percentages_rescan.csv"
        table.to_csv(table_path, index=False)
        print(f"Saved {table_path}")

        taco_seed_summary = make_taco_seed_summary(
            metric_rows, cell_line, taco_round=args.taco_round
        )
        taco_seed_summary_path = output_dir / f"{cell_line}_taco_seed_summary.csv"
        taco_seed_summary.to_csv(taco_seed_summary_path, index=False)
        print(f"Saved {taco_seed_summary_path}")

        dataset_summary = make_dataset_summary(
            metric_rows,
            cell_line,
            args.taco_seeds,
            taco_round=args.taco_round,
        )
        present_approaches = set(dataset_summary["approach"].astype(str))
        missing_approaches = set(FINAL_APPROACH_ORDER) - present_approaches
        if missing_approaches:
            raise ValueError(
                f"Missing final TFBS comparison groups for {cell_line}: "
                f"{sorted(missing_approaches)}."
            )
        dataset_summary_path = output_dir / f"{cell_line}_tfbs_dataset_summary.csv"
        dataset_summary.to_csv(dataset_summary_path, index=False)
        print(f"Saved {dataset_summary_path}")

        figure_data, seed_points = plot_motif_percentages(
            table,
            cell_line,
            figure_dir,
        )
        figure_output = make_figure_output_data(
            figure_data, seed_points, cell_line, args.taco_round
        )
        figure_data_path = output_dir / f"{cell_line}_tfbs_figure_data.csv"
        figure_output.to_csv(figure_data_path, index=False)
        print(f"Saved {figure_data_path}")

        print(f"\n{cell_line.upper()} table:")
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()
