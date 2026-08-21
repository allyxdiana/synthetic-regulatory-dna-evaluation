#!/usr/bin/env python3

import argparse
import random
import re
from collections import Counter
from pathlib import Path

import pandas as pd


DNA_ALPHABET = ["A", "C", "G", "T"]
DEFAULT_CELL_TYPES = ["hepg2", "k562"]

# Fixed JASPAR motif IDs selected for the two cell-type-specific designs.
# HepG2: CEBPA, FOXA2, HNF4A
# K562: GATA1, GATA1::TAL1, KLF1
CELL_TYPE_MOTIFS = {
    "hepg2": ["MA0102.5", "MA0047.4", "MA1494.2"],
    "k562": ["MA0035.5", "MA0140.3", "MA0493.3"],
}

COMPLEMENT = str.maketrans("ACGT", "TGCA")


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

    return clean_sequences(df[sequence_column].dropna().astype(str).tolist())


def estimate_initial_probabilities(sequences, pseudocount=1.0):
    counts = Counter({base: pseudocount for base in DNA_ALPHABET})

    for seq in sequences:
        counts[seq[0]] += 1

    total = sum(counts.values())
    return {base: counts[base] / total for base in DNA_ALPHABET}


def estimate_transition_probabilities(sequences, pseudocount=1.0):
    transition_counts = {
        base: Counter({next_base: pseudocount for next_base in DNA_ALPHABET})
        for base in DNA_ALPHABET
    }

    for seq in sequences:
        for i in range(len(seq) - 1):
            current_base = seq[i]
            next_base = seq[i + 1]
            transition_counts[current_base][next_base] += 1

    transition_probabilities = {}

    for base in DNA_ALPHABET:
        total = sum(transition_counts[base].values())
        transition_probabilities[base] = {
            next_base: transition_counts[base][next_base] / total
            for next_base in DNA_ALPHABET
        }

    return transition_probabilities


def weighted_choice(probabilities):
    bases = list(probabilities.keys())
    weights = list(probabilities.values())
    return random.choices(bases, weights=weights, k=1)[0]


def generate_markov_sequence(length, initial_probabilities, transition_probabilities):
    if length <= 0:
        raise ValueError("Sequence length must be greater than zero.")

    sequence = [weighted_choice(initial_probabilities)]

    while len(sequence) < length:
        previous_base = sequence[-1]
        next_base = weighted_choice(transition_probabilities[previous_base])
        sequence.append(next_base)

    return "".join(sequence)


def reverse_complement(seq):
    return seq.translate(COMPLEMENT)[::-1]


def normalize_ppm_row(values):
    if len(values) != 4:
        raise ValueError(f"PPM row must contain four values, got: {values}")

    values = [max(float(v), 0.0) for v in values]
    total = sum(values)

    if total == 0:
        return {base: 0.25 for base in DNA_ALPHABET}

    return {base: values[i] / total for i, base in enumerate(DNA_ALPHABET)}


def parse_meme_file(meme_path):
    motifs = {}
    current = None
    reading_matrix = False

    with open(meme_path, "r") as f:
        for raw_line in f:
            line = raw_line.strip()

            if not line:
                reading_matrix = False
                current = None
                continue

            if line.startswith("MOTIF "):
                parts = line.split(maxsplit=2)
                motif_id = parts[1]
                motif_name = parts[2] if len(parts) > 2 else motif_id
                current = {
                    "id": motif_id,
                    "name": motif_name,
                    "ppm": [],
                }
                motifs[motif_id] = current
                reading_matrix = False
                continue

            if line.startswith("letter-probability matrix"):
                if current is None:
                    raise ValueError("Found matrix before MOTIF declaration.")
                reading_matrix = True
                continue

            if reading_matrix:
                if line.startswith("URL") or line.startswith("MOTIF"):
                    reading_matrix = False
                    current = None
                    continue

                values = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", line)
                if len(values) >= 4:
                    current["ppm"].append(normalize_ppm_row(values[:4]))
                else:
                    reading_matrix = False
                    current = None

    motifs = {
        motif_id: motif
        for motif_id, motif in motifs.items()
        if motif["ppm"]
    }

    if not motifs:
        raise ValueError(f"No motifs parsed from MEME file: {meme_path}")

    return motifs


def sample_sequence_from_ppm(ppm):
    return "".join(weighted_choice(position_probs) for position_probs in ppm)


def choose_motif_positions(motif_lengths, sequence_length, min_spacing, max_spacing):
    total_motif_length = sum(motif_lengths)
    min_required = total_motif_length + min_spacing * (len(motif_lengths) - 1)

    if min_required > sequence_length:
        raise ValueError(
            "Motifs and minimum spacers do not fit into sequence length. "
            f"Required={min_required}, sequence_length={sequence_length}"
        )

    n_spacers = len(motif_lengths) - 1
    internal_spacers = [random.randint(min_spacing, max_spacing) for _ in range(n_spacers)]

    used_length = total_motif_length + sum(internal_spacers)

    # If randomly sampled spacers do not fit, resample until they do.
    # With the default parameters and short TFBSs this usually succeeds immediately.
    attempts = 0
    while used_length > sequence_length:
        attempts += 1
        if attempts > 1000:
            raise ValueError("Could not sample valid motif spacing after 1000 attempts.")
        internal_spacers = [random.randint(min_spacing, max_spacing) for _ in range(n_spacers)]
        used_length = total_motif_length + sum(internal_spacers)

    remaining = sequence_length - used_length
    left_flank = random.randint(0, remaining)

    positions = []
    cursor = left_flank

    for i, motif_length in enumerate(motif_lengths):
        positions.append(cursor)
        cursor += motif_length
        if i < n_spacers:
            cursor += internal_spacers[i]

    return positions, internal_spacers, left_flank, remaining - left_flank


def insert_motifs_into_background(
    background_sequence,
    motif_ids,
    motifs,
    min_spacing,
    max_spacing,
    allow_reverse_complement,
):
    sampled_motifs = []
    metadata_rows = []

    for motif_id in motif_ids:
        motif = motifs[motif_id]
        sampled_seq = sample_sequence_from_ppm(motif["ppm"])
        orientation = "forward"

        if allow_reverse_complement and random.random() < 0.5:
            sampled_seq = reverse_complement(sampled_seq)
            orientation = "reverse_complement"

        sampled_motifs.append(sampled_seq)
        metadata_rows.append({
            "motif_id": motif_id,
            "motif_name": motif["name"],
            "motif_length": len(sampled_seq),
            "orientation": orientation,
            "sampled_tfbs": sampled_seq,
        })

    positions, spacers, left_flank, right_flank = choose_motif_positions(
        motif_lengths=[len(m) for m in sampled_motifs],
        sequence_length=len(background_sequence),
        min_spacing=min_spacing,
        max_spacing=max_spacing,
    )

    sequence_list = list(background_sequence)

    for idx, (position, motif_seq) in enumerate(zip(positions, sampled_motifs)):
        sequence_list[position:position + len(motif_seq)] = list(motif_seq)
        metadata_rows[idx]["start_0_based"] = position
        metadata_rows[idx]["end_0_based_exclusive"] = position + len(motif_seq)
        metadata_rows[idx]["left_flank"] = left_flank
        metadata_rows[idx]["right_flank"] = right_flank
        metadata_rows[idx]["spacing_after"] = spacers[idx] if idx < len(spacers) else None

    return "".join(sequence_list), metadata_rows


def write_fasta(sequences, output_path, cell_type):
    with open(output_path, "w") as f:
        for i, seq in enumerate(sequences, start=1):
            f.write(f">tfbs_guided_{cell_type}_{i}\n")
            for j in range(0, len(seq), 80):
                f.write(seq[j:j + 80] + "\n")


def validate_motif_ids(motif_ids, motifs, cell_type):
    missing = [motif_id for motif_id in motif_ids if motif_id not in motifs]
    if missing:
        raise ValueError(
            f"Missing motif IDs for {cell_type}: {missing}. "
            "Check the JASPAR MEME file or update CELL_TYPE_MOTIFS."
        )


def generate_for_cell_type(
    cell_type,
    taco_data_dir,
    output_dir,
    motifs,
    seed,
    pseudocount,
    min_spacing,
    max_spacing,
    allow_reverse_complement,
):
    csv_path = taco_data_dir / cell_type / "mbo.csv"
    output_path = output_dir / f"tfbs_guided_{cell_type}.fasta"
    metadata_path = output_dir / f"tfbs_guided_{cell_type}_metadata.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Input file not found: {csv_path}")

    if cell_type not in CELL_TYPE_MOTIFS:
        raise ValueError(
            f"No motif configuration defined for cell type '{cell_type}'. "
            f"Available: {list(CELL_TYPE_MOTIFS.keys())}"
        )

    random.seed(seed)

    motif_ids = CELL_TYPE_MOTIFS[cell_type].copy()
    random.shuffle(motif_ids)

    validate_motif_ids(motif_ids, motifs, cell_type)

    real_sequences = load_sequences_from_mbo(csv_path)
    lengths = [len(seq) for seq in real_sequences]

    initial_probabilities = estimate_initial_probabilities(
        sequences=real_sequences,
        pseudocount=pseudocount,
    )
    transition_probabilities = estimate_transition_probabilities(
        sequences=real_sequences,
        pseudocount=pseudocount,
    )

    generated_sequences = []
    all_metadata_rows = []

    for sequence_index, length in enumerate(lengths, start=1):

        motif_ids_for_sequence = CELL_TYPE_MOTIFS[cell_type].copy()
        random.shuffle(motif_ids_for_sequence)

        background_sequence = generate_markov_sequence(
            length=length,
            initial_probabilities=initial_probabilities,
            transition_probabilities=transition_probabilities,
        )

        generated_sequence, metadata_rows = insert_motifs_into_background(
            background_sequence=background_sequence,
            motif_ids=motif_ids_for_sequence,
            motifs=motifs,
            min_spacing=min_spacing,
            max_spacing=max_spacing,
            allow_reverse_complement=allow_reverse_complement,
        )

        generated_sequences.append(generated_sequence)

        for motif_order, row in enumerate(metadata_rows, start=1):
            row = dict(row)
            row["cell_type"] = cell_type
            row["sequence_id"] = sequence_index
            row["motif_order"] = motif_order
            row["sequence_length"] = length
            all_metadata_rows.append(row)

    write_fasta(generated_sequences, output_path, cell_type)
    pd.DataFrame(all_metadata_rows).to_csv(metadata_path, index=False)

    motif_label = ", ".join(
        f"{motif_id} ({motifs[motif_id]['name']})" for motif_id in motif_ids
    )

    print(f"[{cell_type}] Input: {csv_path}")
    print(f"[{cell_type}] Real sequences: {len(real_sequences)}")
    print(f"[{cell_type}] Generated sequences: {len(generated_sequences)}")
    print(f"[{cell_type}] Lengths: min={min(lengths)}, max={max(lengths)}")
    print(f"[{cell_type}] Motifs: {motif_label}")
    print(f"[{cell_type}] Spacing: {min_spacing}-{max_spacing} bp")
    print(f"[{cell_type}] Reverse complement: {allow_reverse_complement}")
    print(f"[{cell_type}] FASTA output: {output_path}")
    print(f"[{cell_type}] Metadata output: {metadata_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate TFBS-guided synthetic DNA sequences by inserting "
            "cell-type-specific JASPAR motifs into fixed-length Markov "
            "background sequences."
        )
    )

    parser.add_argument(
        "--taco_data_dir",
        default="../TACO_MA/data",
        help="Path to TACO data directory containing <cell_type>/mbo.csv",
    )

    parser.add_argument(
        "--meme_file",
        default=(
            "local_assets/tfbs/human/"
            "20240913075738_JASPAR2024_combined_matrices_1210274_meme.txt"
        ),
        help="Path to JASPAR MEME file containing PPMs.",
    )

    parser.add_argument(
        "--output_dir",
        default="results/tfbs_guided_generation",
        help="Directory for generated FASTA files and metadata.",
    )

    parser.add_argument(
        "--cell_types",
        nargs="+",
        default=DEFAULT_CELL_TYPES,
        help="Cell types to process. Default: hepg2 k562",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=23,
        help="Random seed for reproducibility.",
    )

    parser.add_argument(
        "--pseudocount",
        type=float,
        default=1.0,
        help="Laplace smoothing pseudocount for Markov background. Default: 1.0",
    )

    parser.add_argument(
        "--min_spacing",
        type=int,
        default=10,
        help="Minimum spacer length between adjacent inserted TFBSs. Default: 10",
    )

    parser.add_argument(
        "--max_spacing",
        type=int,
        default=30,
        help="Maximum spacer length between adjacent inserted TFBSs. Default: 30",
    )

    parser.add_argument(
        "--no_reverse_complement",
        action="store_true",
        help="Disable random reverse-complement insertion of sampled TFBSs.",
    )

    args = parser.parse_args()

    taco_data_dir = Path(args.taco_data_dir)
    meme_file = Path(args.meme_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not meme_file.exists():
        raise FileNotFoundError(f"MEME file not found: {meme_file}")

    if args.min_spacing < 0 or args.max_spacing < args.min_spacing:
        raise ValueError("Require 0 <= min_spacing <= max_spacing.")

    motifs = parse_meme_file(meme_file)
    print(f"Parsed {len(motifs)} motifs from {meme_file}")

    for cell_type in args.cell_types:
        generate_for_cell_type(
            cell_type=cell_type.lower(),
            taco_data_dir=taco_data_dir,
            output_dir=output_dir,
            motifs=motifs,
            seed=args.seed,
            pseudocount=args.pseudocount,
            min_spacing=args.min_spacing,
            max_spacing=args.max_spacing,
            allow_reverse_complement=not args.no_reverse_complement,
        )


if __name__ == "__main__":
    main()
