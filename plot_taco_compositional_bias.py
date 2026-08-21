#!/usr/bin/env python3
"""Plot the dinucleotide-composition bias of TACO-generated sequences.

For each sequence, the frequency of dinucleotide XY is calculated as the
number of overlapping XY occurrences divided by (sequence length - 1).
Dataset-level frequencies are pooled counts divided by the total number of
valid adjacent pairs. The script writes the underlying values as CSV files and
creates the HepG2 and K562 publication figures used in the Results chapter.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from matplotlib.ticker import MultipleLocator, PercentFormatter

from taco_final_pool import load_validated_final_pool


SELECTED_DINUCS = ("AA", "CC", "CG", "GC", "GG", "TT")
DEFAULT_TACO_SEEDS = (0, 1, 2, 3, 4)
DEFAULT_TACO_ROUND = 100
DEFAULT_TACO_PATTERN = (
    "full_runs/{cell}_seed{seed}_b64_ga4/analysis/"
    "{cell}_seed{seed}_all_candidates_annotated.csv"
)
ORIGINAL_COLOR = "#1f77b4"
SEED_GREEN_COLORS = {
    0: "#d9f0d3",
    1: "#a6dba0",
    2: "#5aae61",
    3: "#1b7837",
    4: "#00441b",
}
HATCHES = ("", "///", "\\\\", "xx", "..", "++", "oo", "--")
CELL_LABELS = {"hepg2": "HepG2", "k562": "K562"}

# label, CSV path, selected round, seed (None for Original)
DatasetSpec = tuple[str, Path, Optional[int], Optional[int]]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir if (script_dir / "TACO_MA").is_dir() else script_dir.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=project_root,
        help="Project root containing TACO_MA/data and TACO_MA/results.",
    )
    parser.add_argument(
        "--cell-type",
        choices=tuple(CELL_LABELS),
        required=True,
        help="Cell line to plot.",
    )
    parser.add_argument(
        "--taco-data-dir",
        type=Path,
        default=None,
        help="TACO data root; defaults to PROJECT_ROOT/TACO_MA/data.",
    )
    parser.add_argument(
        "--taco-results-dir",
        type=Path,
        default=None,
        help="TACO result-package root; defaults to PROJECT_ROOT/TACO_MA/results.",
    )
    parser.add_argument(
        "--taco-pattern",
        default=DEFAULT_TACO_PATTERN,
        help="TACO result pattern relative to --taco-results-dir.",
    )
    parser.add_argument(
        "--taco-seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_TACO_SEEDS),
        help="TACO seeds to include.",
    )
    parser.add_argument(
        "--taco-round",
        type=int,
        default=DEFAULT_TACO_ROUND,
        help="Effective TACO round to include.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Directory for reproducibility CSV files; defaults by cell type.",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=None,
        help="Directory for PDF and PNG figures; defaults by cell type.",
    )
    args = parser.parse_args(argv)
    default_results_dir = script_dir / f"results/taco_{args.cell_type}_compositional_bias"
    args.results_dir = args.results_dir or default_results_dir
    args.figure_dir = args.figure_dir or default_results_dir / "figures"
    return args


def build_datasets(args: argparse.Namespace) -> list[DatasetSpec]:
    taco_data_dir = args.taco_data_dir or args.project_root / "TACO_MA/data"
    taco_results_dir = args.taco_results_dir or args.project_root / "TACO_MA/results"
    cell = args.cell_type
    datasets: list[DatasetSpec] = [
        (f"Real {CELL_LABELS[cell]}", taco_data_dir / cell / "mbo.csv", None, None)
    ]
    datasets.extend(
        (
            f"TACO seed{seed}",
            taco_results_dir / args.taco_pattern.format(cell=cell, seed=seed),
            args.taco_round,
            seed,
        )
        for seed in args.taco_seeds
    )
    return datasets


def colors_for_datasets(datasets: Sequence[DatasetSpec]) -> list[str]:
    taco_specs = [spec for spec in datasets if spec[3] is not None]
    fallback = plt.get_cmap("Greens")
    colors: list[str] = []
    for spec in datasets:
        seed = spec[3]
        if seed is None:
            colors.append(ORIGINAL_COLOR)
        elif seed in SEED_GREEN_COLORS:
            colors.append(SEED_GREEN_COLORS[seed])
        else:
            index = taco_specs.index(spec)
            fraction = 0.35 if len(taco_specs) == 1 else 0.3 + 0.6 * index / (len(taco_specs) - 1)
            colors.append(to_hex(fallback(fraction)))
    return colors


def hatches_for_datasets(datasets: Sequence[DatasetSpec]) -> list[str]:
    return [HATCHES[index % len(HATCHES)] for index in range(len(datasets))]


def read_csv_sequences(
    path: Path,
    selected_round: Optional[int],
    expected_cell_line: Optional[str] = None,
    expected_seed: Optional[int] = None,
) -> list[tuple[str, str]]:
    if selected_round is not None:
        if expected_cell_line is None or expected_seed is None:
            raise ValueError(
                "TACO final-pool loading requires expected cell line and seed."
            )
        pool = load_validated_final_pool(
            path,
            expected_cell_line=expected_cell_line,
            expected_seed=expected_seed,
            effective_round=selected_round,
        )
        return [
            (f"sequence_{index}", sequence)
            for index, sequence in enumerate(pool.sequences, start=1)
        ]

    records: list[tuple[str, str]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "sequence" not in (reader.fieldnames or []):
            raise ValueError(f"No 'sequence' column in {path}")
        for row_number, row in enumerate(reader, start=1):
            records.append((f"sequence_{row_number}", row["sequence"].upper()))
    if not records:
        raise ValueError(f"No sequences found in {path}")
    return records


def calculate_frequencies(
    datasets: Sequence[DatasetSpec],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    per_sequence: list[dict[str, object]] = []
    aggregate: list[dict[str, object]] = []

    original_label = datasets[0][0] if datasets else ""
    cell_line = "hepg2" if "hepg2" in original_label.lower() else "k562"
    for dataset, path, selected_round, seed in datasets:
        pooled_counts: Counter[str] = Counter()
        pooled_pairs = 0
        records = read_csv_sequences(path, selected_round, cell_line, seed)
        for sequence_id, sequence in records:
            invalid = set(sequence) - set("ACGT")
            if invalid:
                raise ValueError(
                    f"Non-ACGT symbols {sorted(invalid)} in {path}:{sequence_id}"
                )
            if len(sequence) < 2:
                raise ValueError(f"Sequence shorter than 2 nt in {path}:{sequence_id}")
            counts = Counter(sequence[i : i + 2] for i in range(len(sequence) - 1))
            denominator = len(sequence) - 1
            pooled_counts.update(counts)
            pooled_pairs += denominator
            row: dict[str, object] = {
                "dataset": dataset,
                "source_file": str(path),
                "selected_round": "all" if selected_round is None else selected_round,
                "sequence_id": sequence_id,
                "sequence_length": len(sequence),
                "n_adjacent_pairs": denominator,
                "gc_content": (sequence.count("G") + sequence.count("C")) / len(sequence),
                "mononucleotide_shannon_entropy": -sum(
                    (sequence.count(base) / len(sequence))
                    * math.log2(sequence.count(base) / len(sequence))
                    for base in "ACGT"
                    if sequence.count(base)
                ),
            }
            row.update({dinuc: counts[dinuc] / denominator for dinuc in SELECTED_DINUCS})
            per_sequence.append(row)

        for dinuc in SELECTED_DINUCS:
            aggregate.append(
                {
                    "dataset": dataset,
                    "source_file": str(path),
                    "selected_round": "all" if selected_round is None else selected_round,
                    "n_sequences": len(records),
                    "n_adjacent_pairs": pooled_pairs,
                    "dinucleotide": dinuc,
                    "frequency": pooled_counts[dinuc] / pooled_pairs,
                }
            )
    return per_sequence, aggregate


def make_taco_seed_composition_summaries(
    per_sequence: list[dict[str, object]],
    aggregate: list[dict[str, object]],
    datasets: Sequence[DatasetSpec],
    cell_line: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Return seed values and mean/sample-SD/range across independent seeds."""
    pooled_lookup = {
        (str(row["dataset"]), str(row["dinucleotide"])): float(row["frequency"])
        for row in aggregate
    }
    seed_rows: list[dict[str, object]] = []
    signature = "CG" if cell_line == "hepg2" else "GG"
    for dataset, _, selected_round, seed in datasets:
        if seed is None:
            continue
        rows = [row for row in per_sequence if row["dataset"] == dataset]
        seed_row: dict[str, object] = {
            "cell_line": cell_line,
            "seed": seed,
            "effective_round": selected_round,
            "n_sequences": len(rows),
            "mean_gc_content": statistics.fmean(float(row["gc_content"]) for row in rows),
            "mean_mononucleotide_shannon_entropy": statistics.fmean(
                float(row["mononucleotide_shannon_entropy"]) for row in rows
            ),
        }
        for dinucleotide in SELECTED_DINUCS:
            seed_row[f"pooled_{dinucleotide}_frequency"] = pooled_lookup[
                (dataset, dinucleotide)
            ]
        seed_row[f"mean_per_sequence_{signature}_frequency"] = statistics.fmean(
            float(row[signature]) for row in rows
        )
        seed_rows.append(seed_row)

    metric_names = [
        "mean_gc_content",
        "mean_mononucleotide_shannon_entropy",
        *(f"pooled_{dinucleotide}_frequency" for dinucleotide in SELECTED_DINUCS),
        f"mean_per_sequence_{signature}_frequency",
    ]
    summary_rows: list[dict[str, object]] = []
    for metric in metric_names:
        values = [float(row[metric]) for row in seed_rows]
        summary_rows.append(
            {
                "cell_line": cell_line,
                "metric": metric,
                "n_seeds": len(values),
                "mean_across_seeds": statistics.fmean(values),
                "sample_sd_across_seeds": statistics.stdev(values) if len(values) > 1 else math.nan,
                "min_across_seeds": min(values),
                "max_across_seeds": max(values),
            }
        )
    return seed_rows, summary_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def configure_plot_style() -> None:
    """Apply the compact plotting style shared by both output panels."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, output_stem: Path) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def make_barplot(
    aggregate: list[dict[str, object]],
    output_stem: Path,
    datasets: Sequence[DatasetSpec],
) -> None:
    configure_plot_style()
    fig, ax = plt.subplots(figsize=(6.2, 4.25))
    lookup = {
        (str(row["dataset"]), str(row["dinucleotide"])): float(row["frequency"])
        for row in aggregate
    }
    x_positions = list(range(len(SELECTED_DINUCS)))
    width = min(0.19, 0.76 / len(datasets))
    offsets = [(index - (len(datasets) - 1) / 2) * width for index in range(len(datasets))]
    for (dataset, _, _, _), color, hatch, offset in zip(
        datasets, colors_for_datasets(datasets), hatches_for_datasets(datasets), offsets
    ):
        values = [lookup[(dataset, dinuc)] for dinuc in SELECTED_DINUCS]
        ax.bar(
            [x + offset for x in x_positions],
            values,
            width=width,
            label=dataset,
            color=color,
            edgecolor=color,
            linewidth=0.65,
            hatch=hatch,
            zorder=3,
        )
    ax.set_xticks(x_positions)
    ax.set_xticklabels(SELECTED_DINUCS)
    ax.set_xlabel("Dinucleotide")
    ax.set_ylabel("Pooled dinucleotide frequency")
    ax.set_ylim(0, 0.70)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.6, zorder=0)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_figure(fig, output_stem)


def make_dinucleotide_boxplot(
    per_sequence: list[dict[str, object]],
    output_stem: Path,
    datasets: Sequence[DatasetSpec],
    dinucleotide: str = "GG",
) -> None:
    configure_plot_style()
    fig, ax = plt.subplots(figsize=(4.8, 4.25))
    dinucleotide_values = [
        [float(row[dinucleotide]) for row in per_sequence if row["dataset"] == dataset]
        for dataset, _, _, _ in datasets
    ]
    box = ax.boxplot(
        dinucleotide_values,
        patch_artist=True,
        widths=0.62,
        showfliers=False,
        showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 4},
        medianprops={"color": "black", "linewidth": 1.2},
        whiskerprops={"color": "black", "linewidth": 0.8},
        capprops={"color": "black", "linewidth": 0.8},
    )
    colors = colors_for_datasets(datasets)
    for index, (patch, color, hatch) in enumerate(
        zip(box["boxes"], colors, hatches_for_datasets(datasets))
    ):
        patch.set_facecolor(color)
        patch.set_edgecolor(color)
        patch.set_hatch(hatch)
        patch.set_linewidth(0.8)
        for artist in box["whiskers"][2 * index : 2 * index + 2]:
            artist.set_color(color)
        for artist in box["caps"][2 * index : 2 * index + 2]:
            artist.set_color(color)
        box["medians"][index].set_color(color)
        box["means"][index].set_color(color)
        box["means"][index].set_markerfacecolor(color)
        box["means"][index].set_markeredgecolor(color)
    ax.set_xticks(range(1, len(datasets) + 1))
    ax.set_xticklabels(
        [label if seed is None else label.replace("TACO ", "TACO\n") for label, _, _, seed in datasets]
    )
    ax.set_xlabel("Dataset / run")
    ax.set_ylabel(f"Per-sequence {dinucleotide} frequency")
    ax.set_ylim(0, 0.25 if dinucleotide == "CG" else 0.85)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.6, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_figure(fig, output_stem)


def main() -> None:
    args = parse_args()
    datasets = build_datasets(args)
    per_sequence, aggregate = calculate_frequencies(datasets)
    cell = args.cell_type
    write_csv(args.results_dir / f"{cell}_selected_dinucleotide_frequencies_per_sequence.csv", per_sequence)
    write_csv(args.results_dir / f"{cell}_selected_dinucleotide_frequencies_aggregate.csv", aggregate)
    seed_rows, summary_rows = make_taco_seed_composition_summaries(
        per_sequence,
        aggregate,
        datasets,
        cell,
    )
    write_csv(args.results_dir / f"{cell}_taco_seed_composition_values.csv", seed_rows)
    write_csv(args.results_dir / f"{cell}_taco_composition_summary.csv", summary_rows)
    make_barplot(
        aggregate,
        args.figure_dir / f"taco_{cell}_compositional_bias_barplot",
        datasets,
    )
    signature_dinucleotide = "CG" if cell == "hepg2" else "GG"
    make_dinucleotide_boxplot(
        per_sequence,
        args.figure_dir
        / f"taco_{cell}_compositional_bias_{signature_dinucleotide.lower()}_boxplot",
        datasets,
        signature_dinucleotide,
    )
    print(f"Analyzed {len(per_sequence):,} sequences from {len(datasets)} datasets")
    print(f"Wrote CSV data to {args.results_dir}")
    print(f"Wrote two plot panels to {args.figure_dir}")


if __name__ == "__main__":
    main()
