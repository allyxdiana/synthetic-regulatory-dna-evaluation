#!/usr/bin/env python3

import argparse
import math
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DNA_ALPHABET = ["A", "C", "G", "T"]
DINUCS = [a + b for a in DNA_ALPHABET for b in DNA_ALPHABET]


def clean_sequences(sequences):
    cleaned = []

    for seq in sequences:
        seq = seq.upper().replace("N", "")
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

    return clean_sequences(df[sequence_column].dropna().astype(str).tolist())


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

    return clean_sequences(sequences)


def load_dataset(path):
    path = Path(path)

    if path.suffix.lower() == ".csv":
        return load_sequences_from_mbo(path)

    if path.suffix.lower() in [".fa", ".fasta", ".fna"]:
        return load_sequences_from_fasta(path)

    raise ValueError(f"Unsupported file format: {path}")


def gc_content(seq):
    return (seq.count("G") + seq.count("C")) / len(seq)


def nucleotide_frequencies(seq):
    counts = Counter(seq)
    total = len(seq)
    return {base: counts[base] / total for base in DNA_ALPHABET}


def dinucleotide_frequencies(seq):
    counts = Counter(seq[i:i + 2] for i in range(len(seq) - 1))
    total = max(len(seq) - 1, 1)
    return {dinuc: counts[dinuc] / total for dinuc in DINUCS}


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

    summary_row = {
        "dataset": dataset_name,
        "n_sequences": len(sequences),
        "mean_gc_content": pd.Series(gc_values).mean(),
        "std_gc_content": pd.Series(gc_values).std(),
        "mean_entropy": pd.Series(entropy_values).mean(),
        "std_entropy": pd.Series(entropy_values).std(),
    }

    return sequence_rows, nucleotide_rows, dinucleotide_rows, summary_row


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

    color_cycle = [
        "#1f77b4",
        "#aec7e8",
        "#2ca02c",
        "#98df8a",
        "#ff7f0e",
        "#ffbb78",
        "#d62728",
        "#ff9896",
    ]

    colors = color_cycle[:len(pivot.columns)]

    ax = pivot.plot(
        kind="bar",
        figsize=(12, 5),
        color=colors,
        width=0.8
    )

    ax.set_title("Dinucleotide Frequencies")
    ax.set_xlabel("Dinucleotide")
    ax.set_ylabel("Frequency")

    plt.xticks(rotation=45, ha="right")

    ax.legend(
        title="Dataset",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=min(len(pivot.columns), 4),
        frameon=True
    )

    plt.tight_layout()
    plt.savefig(
        output_dir / "dinucleotide_frequencies_barplot.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


def get_dinucleotide_matrix(subset):
    matrix = pd.DataFrame(
        index=DNA_ALPHABET,
        columns=DNA_ALPHABET,
        dtype=float
    )

    for _, row in subset.iterrows():
        dinuc = row["dinucleotide"]
        matrix.loc[dinuc[0], dinuc[1]] = row["frequency"]

    return matrix


def determine_heatmap_scale(
    dinucleotide_df,
    scale_mode,
    fixed_vmin=None,
    fixed_vmax=None
):
    if scale_mode == "individual":
        return None, None

    if scale_mode == "shared":
        return (
            float(dinucleotide_df["frequency"].min()),
            float(dinucleotide_df["frequency"].max())
        )

    if scale_mode == "fixed":
        if fixed_vmin is None or fixed_vmax is None:
            raise ValueError(
                "For --dinuc_heatmap_scale fixed, both "
                "--dinuc_heatmap_fixed_vmin and --dinuc_heatmap_fixed_vmax "
                "must be provided."
            )

        if fixed_vmin >= fixed_vmax:
            raise ValueError(
                "--dinuc_heatmap_fixed_vmin must be smaller than "
                "--dinuc_heatmap_fixed_vmax."
            )

        return float(fixed_vmin), float(fixed_vmax)

    raise ValueError(f"Unsupported heatmap scale mode: {scale_mode}")


def save_dinucleotide_heatmap_scale_summary(
    dinucleotide_df,
    output_dir,
    scale_mode,
    global_vmin,
    global_vmax
):
    rows = []

    for dataset in dinucleotide_df["dataset"].unique():
        subset = dinucleotide_df[dinucleotide_df["dataset"] == dataset]

        rows.append({
            "dataset": dataset,
            "scale_mode": scale_mode,
            "dataset_min_frequency": subset["frequency"].min(),
            "dataset_max_frequency": subset["frequency"].max(),
            "used_vmin": (
                subset["frequency"].min()
                if scale_mode == "individual"
                else global_vmin
            ),
            "used_vmax": (
                subset["frequency"].max()
                if scale_mode == "individual"
                else global_vmax
            ),
        })

    pd.DataFrame(rows).to_csv(
        output_dir / "dinucleotide_heatmap_scale_summary.csv",
        index=False
    )


def save_dinucleotide_heatmaps(
    dinucleotide_df,
    output_dir,
    scale_mode="individual",
    fixed_vmin=None,
    fixed_vmax=None
):
    global_vmin, global_vmax = determine_heatmap_scale(
        dinucleotide_df=dinucleotide_df,
        scale_mode=scale_mode,
        fixed_vmin=fixed_vmin,
        fixed_vmax=fixed_vmax
    )

    save_dinucleotide_heatmap_scale_summary(
        dinucleotide_df=dinucleotide_df,
        output_dir=output_dir,
        scale_mode=scale_mode,
        global_vmin=global_vmin,
        global_vmax=global_vmax
    )

    for dataset in dinucleotide_df["dataset"].unique():
        subset = dinucleotide_df[dinucleotide_df["dataset"] == dataset]
        matrix = get_dinucleotide_matrix(subset)

        if scale_mode == "individual":
            vmin = float(matrix.values.min())
            vmax = float(matrix.values.max())
        else:
            vmin = global_vmin
            vmax = global_vmax

        threshold = (vmin + vmax) / 2

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

                text_color = (
                    "white"
                    if value < threshold
                    else "black"
                )

                ax.text(
                    j,
                    i,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=10
                )

        fig.savefig(
            output_dir / f"dinucleotide_heatmap_{dataset}.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close(fig)


def create_plots(
    sequence_df,
    nucleotide_df,
    dinucleotide_df,
    output_dir,
    dinuc_heatmap_scale="individual",
    dinuc_heatmap_fixed_vmin=None,
    dinuc_heatmap_fixed_vmax=None
):
    save_gc_content_boxplot(sequence_df, output_dir)
    save_entropy_boxplot(sequence_df, output_dir)
    save_nucleotide_frequency_barplot(nucleotide_df, output_dir)
    save_dinucleotide_frequency_barplot(dinucleotide_df, output_dir)

    save_dinucleotide_heatmaps(
        dinucleotide_df=dinucleotide_df,
        output_dir=output_dir,
        scale_mode=dinuc_heatmap_scale,
        fixed_vmin=dinuc_heatmap_fixed_vmin,
        fixed_vmax=dinuc_heatmap_fixed_vmax
    )


def main():
    parser = argparse.ArgumentParser(
        description="Analyze real and synthetic DNA sequences for statistical baseline evaluation."
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
        "--dinuc_heatmap_scale",
        choices=["individual", "shared", "fixed"],
        default="individual",
        help=(
            "Color scale mode for dinucleotide heatmaps. "
            "'individual' uses a separate scale for each dataset, "
            "'shared' uses the global min/max across all datasets, "
            "and 'fixed' uses user-provided vmin/vmax."
        )
    )

    parser.add_argument(
        "--dinuc_heatmap_fixed_vmin",
        type=float,
        default=None,
        help="Fixed minimum value for dinucleotide heatmap color scale."
    )

    parser.add_argument(
        "--dinuc_heatmap_fixed_vmax",
        type=float,
        default=None,
        help="Fixed maximum value for dinucleotide heatmap color scale."
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_sequence_rows = []
    all_nucleotide_rows = []
    all_dinucleotide_rows = []
    all_summary_rows = []

    for item in args.inputs:
        if "=" not in item:
            raise ValueError(f"Invalid input format: {item}. Use name=path.")

        dataset_name, dataset_path = item.split("=", 1)
        sequences = load_dataset(dataset_path)

        sequence_rows, nucleotide_rows, dinucleotide_rows, summary_row = summarize_dataset(
            dataset_name=dataset_name,
            sequences=sequences
        )

        all_sequence_rows.extend(sequence_rows)
        all_nucleotide_rows.extend(nucleotide_rows)
        all_dinucleotide_rows.extend(dinucleotide_rows)
        all_summary_rows.append(summary_row)

        print(f"[{dataset_name}] analyzed {len(sequences)} sequences from {dataset_path}")

    sequence_df = pd.DataFrame(all_sequence_rows)
    nucleotide_df = pd.DataFrame(all_nucleotide_rows)
    dinucleotide_df = pd.DataFrame(all_dinucleotide_rows)
    summary_df = pd.DataFrame(all_summary_rows)

    sequence_df.to_csv(output_dir / "sequence_level_statistics.csv", index=False)
    nucleotide_df.to_csv(output_dir / "nucleotide_frequencies.csv", index=False)
    dinucleotide_df.to_csv(output_dir / "dinucleotide_frequencies.csv", index=False)
    summary_df.to_csv(output_dir / "dataset_summary_statistics.csv", index=False)

    create_plots(
        sequence_df=sequence_df,
        nucleotide_df=nucleotide_df,
        dinucleotide_df=dinucleotide_df,
        output_dir=output_dir,
        dinuc_heatmap_scale=args.dinuc_heatmap_scale,
        dinuc_heatmap_fixed_vmin=args.dinuc_heatmap_fixed_vmin,
        dinuc_heatmap_fixed_vmax=args.dinuc_heatmap_fixed_vmax
    )

    print(f"\nResults written to: {output_dir}")
    print("Generated plots:")
    print("- gc_content_boxplot.png")
    print("- entropy_boxplot.png")
    print("- nucleotide_frequencies_barplot.png")
    print("- dinucleotide_frequencies_barplot.png")
    print("- dinucleotide_heatmap_<dataset>.png")
    print("- dinucleotide_heatmap_scale_summary.csv")


if __name__ == "__main__":
    main()
