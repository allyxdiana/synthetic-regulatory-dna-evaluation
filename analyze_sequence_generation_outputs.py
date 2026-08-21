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


def save_dinucleotide_heatmaps(dinucleotide_df, output_dir):
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

        local_min = matrix.values.min()
        local_max = matrix.values.max()
        threshold = (local_min + local_max) / 2

        fig, ax = plt.subplots(
            figsize=(6, 5),
            constrained_layout=True
        )

        im = ax.imshow(
            matrix.values,
            aspect="equal",
            cmap="viridis"
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

        safe_dataset = reformat_filename(dataset)

        fig.savefig(
            output_dir / f"dinucleotide_heatmap_{safe_dataset}.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close(fig)


def reformat_filename(name):
    return (
        str(name)
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
    )


def create_plots(sequence_df, nucleotide_df, dinucleotide_df, output_dir):
    save_gc_content_boxplot(sequence_df, output_dir)
    save_entropy_boxplot(sequence_df, output_dir)
    save_nucleotide_frequency_barplot(nucleotide_df, output_dir)
    save_dinucleotide_frequency_barplot(dinucleotide_df, output_dir)
    save_dinucleotide_heatmaps(dinucleotide_df, output_dir)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze real, baseline, or TFBS-guided DNA sequences. "
            "CSV inputs must contain a sequence column."
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

        sequences, _ = load_dataset(
            dataset_path,
            sequence_column=args.sequence_column,
        )

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

    create_plots(sequence_df, nucleotide_df, dinucleotide_df, output_dir)

    print(f"\nResults written to: {output_dir}")
    print("Generated CSV files:")
    print("- sequence_level_statistics.csv")
    print("- nucleotide_frequencies.csv")
    print("- dinucleotide_frequencies.csv")
    print("- dataset_summary_statistics.csv")
    print("Generated plots:")
    print("- gc_content_boxplot.png")
    print("- entropy_boxplot.png")
    print("- nucleotide_frequencies_barplot.png")
    print("- dinucleotide_frequencies_barplot.png")
    print("- dinucleotide_heatmap_<dataset>.png")


if __name__ == "__main__":
    main()
