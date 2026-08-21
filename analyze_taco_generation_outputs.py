#!/usr/bin/env python3

# Analyzing Sequence Generation Outputs
# TACO comparison version
#
# Key idea:
# - Heatmaps that are compared in the same thesis figure must use the same color scale.
# - For TACO run comparisons, pass original + all TACO runs in one call and use
#   --dinuc_heatmap_scale shared.
# - This computes one shared vmin/vmax from all compared datasets.
# - Optional fixed scaling is still available for fully reproducible figures.

import argparse
import math
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from taco_final_pool import load_validated_final_pool


DNA_ALPHABET = ["A", "C", "G", "T"]
DINUCS = [a + b for a in DNA_ALPHABET for b in DNA_ALPHABET]
TACO_DATASET_PATTERN = re.compile(r"^taco_(hepg2|k562)_(\d+)$", re.IGNORECASE)


def load_strict_taco_dataset(csv_path, dataset_name, round_value):
    """Load a protocol-aligned pool through the shared thesis validator."""
    match = TACO_DATASET_PATTERN.fullmatch(dataset_name)
    if match is None:
        raise ValueError(
            "Strict TACO final-pool validation requires a dataset label such as "
            "taco_hepg2_0 or taco_k562_4."
        )
    if round_value is None:
        raise ValueError("--round_value is required with --strict_taco_final_pool.")
    cell_line, raw_seed = match.groups()
    pool = load_validated_final_pool(
        csv_path,
        expected_cell_line=cell_line.lower(),
        expected_seed=int(raw_seed),
        effective_round=round_value,
    )
    return list(pool.sequences), pd.DataFrame(pool.rows)


def clean_sequences(sequences):
    cleaned = []

    for seq in sequences:
        seq = str(seq).upper().replace("N", "")
        if seq and set(seq).issubset(set(DNA_ALPHABET)):
            cleaned.append(seq)

    if not cleaned:
        raise ValueError("No valid DNA sequences found.")

    return cleaned


def load_sequences_from_mbo(csv_path, sequence_column="sequence"):
    df = pd.read_csv(csv_path)

    if sequence_column not in df.columns:
        raise ValueError(
            f"Column '{sequence_column}' not found in {csv_path}. "
            f"Available columns: {list(df.columns)}"
        )

    return clean_sequences(df[sequence_column].dropna().astype(str).tolist()), df


def load_sequences_from_fasta(fasta_path):
    sequences = []
    current = []

    with open(fasta_path, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            if line.startswith(">"):
                if current:
                    sequences.append("".join(current))
                    current = []
            else:
                current.append(line)

        if current:
            sequences.append("".join(current))

    return clean_sequences(sequences), None


def load_dataset(path, sequence_column="sequence"):
    path = Path(path)

    if path.suffix.lower() == ".csv":
        return load_sequences_from_mbo(
            path,
            sequence_column=sequence_column,
        )

    if path.suffix.lower() in [".fa", ".fasta", ".fna"]:
        return load_sequences_from_fasta(path)

    raise ValueError(f"Unsupported file format: {path}")


def gc_content(seq):
    return (seq.count("G") + seq.count("C")) / len(seq)


def nucleotide_frequencies(seq):
    counts = Counter(seq)
    total = len(seq)
    return {base: counts[base] / total for base in DNA_ALPHABET}


def shannon_entropy(seq):
    counts = Counter(seq)
    total = len(seq)

    entropy = 0.0
    for base in DNA_ALPHABET:
        p = counts[base] / total
        if p > 0:
            entropy -= p * math.log2(p)

    return entropy


def summarize_dataset(dataset_name, sequences):
    sequence_rows = []
    nucleotide_sum = Counter()
    dinucleotide_sum = Counter()

    for i, seq in enumerate(sequences, start=1):
        nuc_freqs = nucleotide_frequencies(seq)

        sequence_rows.append({
            "dataset": dataset_name,
            "sequence_id": i,
            "sequence_length": len(seq),
            "gc_content": gc_content(seq),
            "entropy": shannon_entropy(seq),
            **{f"freq_{base}": nuc_freqs[base] for base in DNA_ALPHABET},
        })

        for base in DNA_ALPHABET:
            nucleotide_sum[base] += seq.count(base)

        for j in range(len(seq) - 1):
            dinucleotide_sum[seq[j:j + 2]] += 1

    total_nucleotides = sum(nucleotide_sum.values())
    total_dinucleotides = sum(dinucleotide_sum.values())

    nucleotide_rows = [
        {
            "dataset": dataset_name,
            "nucleotide": base,
            "frequency": nucleotide_sum[base] / total_nucleotides
        }
        for base in DNA_ALPHABET
    ]

    dinucleotide_rows = [
        {
            "dataset": dataset_name,
            "dinucleotide": dinuc,
            "frequency": dinucleotide_sum[dinuc] / total_dinucleotides
        }
        for dinuc in DINUCS
    ]

    gc_values = [gc_content(s) for s in sequences]
    entropy_values = [shannon_entropy(s) for s in sequences]
    length_values = [len(s) for s in sequences]

    summary_row = {
        "dataset": dataset_name,
        "n_sequences": len(sequences),
        "mean_sequence_length": pd.Series(length_values).mean(),
        "std_sequence_length": pd.Series(length_values).std(),
        "mean_gc_content": pd.Series(gc_values).mean(),
        "std_gc_content": pd.Series(gc_values).std(),
        "mean_entropy": pd.Series(entropy_values).mean(),
        "std_entropy": pd.Series(entropy_values).std(),
    }

    return sequence_rows, nucleotide_rows, dinucleotide_rows, summary_row


def summarize_scores(dataset_name, raw_df):
    if raw_df is None:
        return None

    score_columns = [
        col for col in ["surrogate_score", "oracle_score"]
        if col in raw_df.columns
    ]

    if not score_columns:
        return None

    row = {"dataset": dataset_name, "n_rows": len(raw_df)}

    if "round" in raw_df.columns:
        rounds = pd.to_numeric(raw_df["round"], errors="raise")
        row["min_round"] = rounds.min()
        row["max_round"] = rounds.max()

    for col in score_columns:
        scores = pd.to_numeric(raw_df[col], errors="raise")
        row[f"mean_{col}"] = scores.mean()
        row[f"std_{col}"] = scores.std()
        row[f"min_{col}"] = scores.min()
        row[f"max_{col}"] = scores.max()

    return row


def reformat_filename(name):
    return (
        str(name)
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
    )


def save_gc_content_boxplot(sequence_df, output_dir):
    plt.figure(figsize=(8, 5))
    sequence_df.boxplot(column="gc_content", by="dataset", grid=False)
    plt.title("GC Content Distribution")
    plt.suptitle("")
    plt.xlabel("Dataset")
    plt.ylabel("GC content")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "gc_content_boxplot.png", dpi=300)
    plt.close()


def save_entropy_boxplot(sequence_df, output_dir):
    plt.figure(figsize=(8, 5))
    sequence_df.boxplot(column="entropy", by="dataset", grid=False)
    plt.title("Shannon Entropy Distribution")
    plt.suptitle("")
    plt.xlabel("Dataset")
    plt.ylabel("Shannon entropy")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "entropy_boxplot.png", dpi=300)
    plt.close()


def save_nucleotide_frequency_barplot(nucleotide_df, output_dir):
    pivot = nucleotide_df.pivot(
        index="nucleotide",
        columns="dataset",
        values="frequency"
    )

    ax = pivot.plot(kind="bar", figsize=(8, 5))
    ax.set_title("Nucleotide Frequencies")
    ax.set_xlabel("Nucleotide")
    ax.set_ylabel("Frequency")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_dir / "nucleotide_frequencies_barplot.png", dpi=300)
    plt.close()


def save_dinucleotide_frequency_barplot(dinucleotide_df, output_dir):
    pivot = dinucleotide_df.pivot(
        index="dinucleotide",
        columns="dataset",
        values="frequency"
    )

    ax = pivot.plot(
        kind="bar",
        figsize=(12, 5),
        width=0.8
    )

    ax.set_title("Dinucleotide Frequencies")
    ax.set_xlabel("Dinucleotide")
    ax.set_ylabel("Frequency")

    plt.xticks(rotation=45, ha="right")

    ax.legend(
        title="Dataset",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=min(4, len(pivot.columns)),
        frameon=True
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.20)

    plt.savefig(
        output_dir / "dinucleotide_frequencies_barplot.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


def determine_heatmap_scale(dinucleotide_df, scale_mode, fixed_vmin, fixed_vmax, round_vmax_to):
    observed_min = float(dinucleotide_df["frequency"].min())
    observed_max = float(dinucleotide_df["frequency"].max())

    if scale_mode == "shared":
        # Shared data-driven scale across all datasets passed to this script call.
        # For TACO comparisons, pass original + all TACO runs together.
        vmin = observed_min
        vmax = observed_max

        if round_vmax_to is not None:
            vmax = math.ceil(vmax / round_vmax_to) * round_vmax_to

        return vmin, vmax, observed_min, observed_max

    if scale_mode == "fixed":
        if fixed_vmin is None or fixed_vmax is None:
            raise ValueError("--dinuc_heatmap_fixed_vmin and --dinuc_heatmap_fixed_vmax are required for fixed scale.")
        return fixed_vmin, fixed_vmax, observed_min, observed_max

    if scale_mode == "individual":
        # This mode is usually not suitable for thesis figures with side-by-side comparisons.
        # It is kept only for exploratory inspection.
        return None, None, observed_min, observed_max

    raise ValueError(f"Unsupported scale mode: {scale_mode}")


def save_dinucleotide_heatmaps(
    dinucleotide_df,
    output_dir,
    scale_mode="shared",
    fixed_vmin=None,
    fixed_vmax=None,
    round_vmax_to=0.01,
):
    shared_vmin, shared_vmax, observed_min, observed_max = determine_heatmap_scale(
        dinucleotide_df=dinucleotide_df,
        scale_mode=scale_mode,
        fixed_vmin=fixed_vmin,
        fixed_vmax=fixed_vmax,
        round_vmax_to=round_vmax_to,
    )

    if scale_mode in ["shared", "fixed"]:
        if shared_vmin >= shared_vmax:
            raise ValueError("Dinucleotide heatmap vmin must be smaller than vmax.")

        if observed_min < shared_vmin or observed_max > shared_vmax:
            print(
                "[warning] Dinucleotide frequencies outside fixed/shared heatmap scale: "
                f"observed range {observed_min:.4f}-{observed_max:.4f}, "
                f"scale {shared_vmin:.4f}-{shared_vmax:.4f}. "
                "Colors will be clipped, numeric labels remain unchanged."
            )

        threshold = (shared_vmin + shared_vmax) / 2

        print(
            "[info] Dinucleotide heatmap scale: "
            f"mode={scale_mode}, vmin={shared_vmin:.4f}, vmax={shared_vmax:.4f}, "
            f"observed={observed_min:.4f}-{observed_max:.4f}"
        )
    else:
        print(
            "[info] Dinucleotide heatmap scale: mode=individual. "
            "Each heatmap uses its own color scale."
        )

    scale_summary_rows = []

    for dataset in dinucleotide_df["dataset"].unique():
        subset = dinucleotide_df[dinucleotide_df["dataset"] == dataset]

        matrix = pd.DataFrame(
            index=DNA_ALPHABET,
            columns=DNA_ALPHABET,
            dtype=float
        )

        for _, row in subset.iterrows():
            dinuc = row["dinucleotide"]
            matrix.loc[dinuc[0], dinuc[1]] = row["frequency"]

        if scale_mode == "individual":
            vmin = float(matrix.values.min())
            vmax = float(matrix.values.max())
            text_threshold = (vmin + vmax) / 2
        else:
            vmin = shared_vmin
            vmax = shared_vmax
            text_threshold = threshold

        scale_summary_rows.append({
            "dataset": dataset,
            "heatmap_scale_mode": scale_mode,
            "heatmap_vmin": vmin,
            "heatmap_vmax": vmax,
            "dataset_min_frequency": float(matrix.values.min()),
            "dataset_max_frequency": float(matrix.values.max()),
        })

        fig, ax = plt.subplots(
            figsize=(6, 5),
            constrained_layout=True
        )

        im = ax.imshow(
            matrix.values,
            aspect="equal",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax
        )

        cbar = fig.colorbar(
            im,
            ax=ax,
            fraction=0.046,
            pad=0.04
        )
        cbar.set_label("Frequency")

        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.set_xticklabels(DNA_ALPHABET)
        ax.set_yticklabels(DNA_ALPHABET)

        ax.set_xlabel("Second nucleotide")
        ax.set_ylabel("First nucleotide")

        ax.set_title(
            f"Dinucleotide frequencies: {dataset}",
            pad=12
        )

        for i in range(4):
            for j in range(4):
                value = matrix.values[i, j]

                text_color = "white" if value < text_threshold else "black"

                ax.text(
                    j,
                    i,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=10
                )

        safe_dataset = reformat_filename(dataset)

        fig.savefig(
            output_dir / f"dinucleotide_heatmap_{safe_dataset}.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close(fig)

    pd.DataFrame(scale_summary_rows).to_csv(
        output_dir / "dinucleotide_heatmap_scale_summary.csv",
        index=False
    )


def save_combined_dinucleotide_heatmap_figure(
    dinucleotide_df,
    output_dir,
    scale_mode="shared",
    fixed_vmin=None,
    fixed_vmax=None,
    round_vmax_to=0.01,
    columns=4,
):
    # Optional convenience plot: one multi-panel figure containing all heatmaps.
    # This avoids accidentally comparing PNGs created with different scales.
    if columns < 1:
        raise ValueError("--combined_heatmap_columns must be at least 1.")

    dataset_names = list(dinucleotide_df["dataset"].unique())
    n_datasets = len(dataset_names)

    shared_vmin, shared_vmax, _, _ = determine_heatmap_scale(
        dinucleotide_df=dinucleotide_df,
        scale_mode=scale_mode,
        fixed_vmin=fixed_vmin,
        fixed_vmax=fixed_vmax,
        round_vmax_to=round_vmax_to,
    )

    if scale_mode == "individual":
        print("[warning] Combined heatmap figure requested with individual scaling. This is not recommended.")

    ncols = min(columns, n_datasets)
    nrows = math.ceil(n_datasets / ncols)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5.2 * ncols, 4.6 * nrows),
        constrained_layout=True
    )

    if n_datasets == 1:
        axes = [axes]
    else:
        axes = list(pd.Series(axes.ravel()))

    last_im = None

    for ax, dataset in zip(axes, dataset_names):
        subset = dinucleotide_df[dinucleotide_df["dataset"] == dataset]

        matrix = pd.DataFrame(
            index=DNA_ALPHABET,
            columns=DNA_ALPHABET,
            dtype=float
        )

        for _, row in subset.iterrows():
            dinuc = row["dinucleotide"]
            matrix.loc[dinuc[0], dinuc[1]] = row["frequency"]

        if scale_mode == "individual":
            vmin = float(matrix.values.min())
            vmax = float(matrix.values.max())
        else:
            vmin = shared_vmin
            vmax = shared_vmax

        text_threshold = (vmin + vmax) / 2

        last_im = ax.imshow(
            matrix.values,
            aspect="equal",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax
        )

        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.set_xticklabels(DNA_ALPHABET)
        ax.set_yticklabels(DNA_ALPHABET)

        ax.set_xlabel("Second nucleotide")
        ax.set_ylabel("First nucleotide")
        ax.set_title(str(dataset), pad=10)

        for i in range(4):
            for j in range(4):
                value = matrix.values[i, j]
                text_color = "white" if value < text_threshold else "black"

                ax.text(
                    j,
                    i,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=9
                )

    for ax in axes[n_datasets:]:
        ax.axis("off")

    if last_im is not None:
        cbar = fig.colorbar(
            last_im,
            ax=axes[:n_datasets],
            fraction=0.025,
            pad=0.02
        )
        cbar.set_label("Frequency")

    fig.savefig(
        output_dir / "dinucleotide_heatmaps_combined.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


def create_plots(
    sequence_df,
    nucleotide_df,
    dinucleotide_df,
    output_dir,
    dinuc_heatmap_scale="shared",
    dinuc_heatmap_fixed_vmin=None,
    dinuc_heatmap_fixed_vmax=None,
    dinuc_heatmap_round_vmax_to=0.01,
    save_combined_heatmap=True,
    combined_heatmap_columns=4,
):
    save_gc_content_boxplot(sequence_df, output_dir)
    save_entropy_boxplot(sequence_df, output_dir)
    save_nucleotide_frequency_barplot(nucleotide_df, output_dir)
    save_dinucleotide_frequency_barplot(dinucleotide_df, output_dir)

    save_dinucleotide_heatmaps(
        dinucleotide_df,
        output_dir,
        scale_mode=dinuc_heatmap_scale,
        fixed_vmin=dinuc_heatmap_fixed_vmin,
        fixed_vmax=dinuc_heatmap_fixed_vmax,
        round_vmax_to=dinuc_heatmap_round_vmax_to,
    )

    if save_combined_heatmap:
        save_combined_dinucleotide_heatmap_figure(
            dinucleotide_df,
            output_dir,
            scale_mode=dinuc_heatmap_scale,
            fixed_vmin=dinuc_heatmap_fixed_vmin,
            fixed_vmax=dinuc_heatmap_fixed_vmax,
            round_vmax_to=dinuc_heatmap_round_vmax_to,
            columns=combined_heatmap_columns,
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze real, custom, and protocol-aligned TACO DNA sequences. "
            "CSV inputs must contain a sequence column. Canonically labeled "
            "TACO inputs require strict final-pool validation."
        )
    )

    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help=(
            "Input datasets in the form name=path. "
            "Supported formats: .csv with sequence column or FASTA."
        )
    )

    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory for output CSV files and plots."
    )

    parser.add_argument(
        "--sequence_column",
        default="sequence",
        help="Name of the sequence column in CSV files. Default: sequence."
    )

    parser.add_argument(
        "--round_value",
        type=int,
        default=None,
        help="Effective round required for protocol-aligned TACO inputs."
    )

    parser.add_argument(
        "--strict_taco_final_pool",
        action="store_true",
        help=(
            "Validate protocol-aligned TACO inputs as 256-sequence B64/GA4 "
            "same-policy pools. Dataset labels must encode cell line and seed."
        ),
    )

    parser.add_argument(
        "--dinuc_heatmap_scale",
        choices=["shared", "fixed", "individual"],
        default="shared",
        help=(
            "Color scale for dinucleotide heatmaps. "
            "'shared' uses one data-driven scale across all inputs in this run. "
            "'fixed' uses --dinuc_heatmap_fixed_vmin and --dinuc_heatmap_fixed_vmax. "
            "'individual' gives each heatmap its own scale and is not recommended for direct comparison. "
            "Default: shared."
        )
    )

    parser.add_argument(
        "--dinuc_heatmap_fixed_vmin",
        type=float,
        default=None,
        help="Fixed vmin for dinucleotide heatmaps when --dinuc_heatmap_scale fixed."
    )

    parser.add_argument(
        "--dinuc_heatmap_fixed_vmax",
        type=float,
        default=None,
        help="Fixed vmax for dinucleotide heatmaps when --dinuc_heatmap_scale fixed."
    )

    parser.add_argument(
        "--dinuc_heatmap_round_vmax_to",
        type=float,
        default=0.01,
        help=(
            "For shared scale, round the shared vmax up to this step. "
            "Use 0 to disable rounding. Default: 0.01."
        )
    )

    parser.add_argument(
        "--no_combined_heatmap",
        action="store_true",
        help="Do not save dinucleotide_heatmaps_combined.png."
    )

    parser.add_argument(
        "--combined_heatmap_columns",
        type=int,
        default=4,
        help="Number of columns in dinucleotide_heatmaps_combined.png. Default: 4."
    )

    args = parser.parse_args()

    if args.dinuc_heatmap_round_vmax_to == 0:
        round_vmax_to = None
    elif args.dinuc_heatmap_round_vmax_to > 0:
        round_vmax_to = args.dinuc_heatmap_round_vmax_to
    else:
        raise ValueError("--dinuc_heatmap_round_vmax_to must be >= 0.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_sequence_rows = []
    all_nucleotide_rows = []
    all_dinucleotide_rows = []
    all_summary_rows = []
    all_score_rows = []

    for item in args.inputs:
        if "=" not in item:
            raise ValueError(f"Invalid input format: {item}. Use name=path.")

        dataset_name, dataset_path = item.split("=", 1)

        if dataset_name.lower().startswith("taco_"):
            if not args.strict_taco_final_pool:
                raise ValueError(
                    "TACO inputs require --strict_taco_final_pool and --round_value."
                )
            sequences, raw_df = load_strict_taco_dataset(
                dataset_path,
                dataset_name,
                args.round_value,
            )
        else:
            sequences, raw_df = load_dataset(
                dataset_path,
                sequence_column=args.sequence_column,
            )

        sequence_rows, nucleotide_rows, dinucleotide_rows, summary_row = summarize_dataset(
            dataset_name=dataset_name,
            sequences=sequences
        )

        score_row = summarize_scores(dataset_name, raw_df)

        all_sequence_rows.extend(sequence_rows)
        all_nucleotide_rows.extend(nucleotide_rows)
        all_dinucleotide_rows.extend(dinucleotide_rows)
        all_summary_rows.append(summary_row)

        if score_row is not None:
            all_score_rows.append(score_row)

        print(f"[{dataset_name}] analyzed {len(sequences)} sequences from {dataset_path}")

    sequence_df = pd.DataFrame(all_sequence_rows)
    nucleotide_df = pd.DataFrame(all_nucleotide_rows)
    dinucleotide_df = pd.DataFrame(all_dinucleotide_rows)
    summary_df = pd.DataFrame(all_summary_rows)

    sequence_df.to_csv(output_dir / "sequence_level_statistics.csv", index=False)
    nucleotide_df.to_csv(output_dir / "nucleotide_frequencies.csv", index=False)
    dinucleotide_df.to_csv(output_dir / "dinucleotide_frequencies.csv", index=False)
    summary_df.to_csv(output_dir / "dataset_summary_statistics.csv", index=False)

    if all_score_rows:
        score_df = pd.DataFrame(all_score_rows)
        score_df.to_csv(output_dir / "score_summary_statistics.csv", index=False)

    create_plots(
        sequence_df=sequence_df,
        nucleotide_df=nucleotide_df,
        dinucleotide_df=dinucleotide_df,
        output_dir=output_dir,
        dinuc_heatmap_scale=args.dinuc_heatmap_scale,
        dinuc_heatmap_fixed_vmin=args.dinuc_heatmap_fixed_vmin,
        dinuc_heatmap_fixed_vmax=args.dinuc_heatmap_fixed_vmax,
        dinuc_heatmap_round_vmax_to=round_vmax_to,
        save_combined_heatmap=not args.no_combined_heatmap,
        combined_heatmap_columns=args.combined_heatmap_columns,
    )

    print(f"\nResults written to: {output_dir}")
    print("Generated CSV files:")
    print("- sequence_level_statistics.csv")
    print("- nucleotide_frequencies.csv")
    print("- dinucleotide_frequencies.csv")
    print("- dataset_summary_statistics.csv")
    print("- dinucleotide_heatmap_scale_summary.csv")
    if all_score_rows:
        print("- score_summary_statistics.csv")
    print("Generated plots:")
    print("- gc_content_boxplot.png")
    print("- entropy_boxplot.png")
    print("- nucleotide_frequencies_barplot.png")
    print("- dinucleotide_frequencies_barplot.png")
    print("- dinucleotide_heatmap_<dataset>.png")
    if not args.no_combined_heatmap:
        print("- dinucleotide_heatmaps_combined.png")


if __name__ == "__main__":
    main()
