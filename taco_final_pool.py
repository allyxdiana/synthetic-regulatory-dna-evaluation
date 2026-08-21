#!/usr/bin/env python3
"""Select and validate a protocol-aligned TACO effective-round pool.

The final thesis analysis uses four consecutive physical microbatches generated
under one unchanged policy state.  For effective round ``r`` these are physical
microbatches ``4r-3`` through ``4r``; the optimizer update happens only after
the fourth microbatch.  The annotated result package records the information
needed to validate this invariant.
"""

from __future__ import annotations

import csv
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence, Union


DNA_ALPHABET = frozenset("ACGT")
PROTOCOL_POOL_SIZE = 256
PROTOCOL_SEQUENCE_LENGTH = 200
PROTOCOL_PHYSICAL_BATCH_SIZE = 64
PROTOCOL_GRAD_ACCUM_STEPS = 4
PROTOCOL_TOTAL_CANDIDATES = 25_600
PROTOCOL_TOTAL_PHYSICAL_BATCHES = 400
PROTOCOL_TOTAL_EFFECTIVE_ROUNDS = 100

REQUIRED_COLUMNS = (
    "cell_line",
    "seed",
    "round",
    "effective_round",
    "physical_batch_size",
    "grad_accum_steps",
    "sequence",
    "surrogate_score",
    "oracle_score",
)


class FinalPoolValidationError(ValueError):
    """A TACO result cannot represent the requested protocol-aligned pool."""


@dataclass(frozen=True)
class FinalPool:
    """Validated rows and normalized sequences for one cell/seed/round."""

    source: Path
    cell_line: str
    seed: int
    effective_round: int
    physical_batches: tuple[int, ...]
    rows: tuple[Mapping[str, str], ...]
    sequences: tuple[str, ...]


def _parse_int(row: Mapping[str, str], column: str, path: Path) -> int:
    raw_value = row.get(column)
    if raw_value is None or str(raw_value).strip() == "":
        raise FinalPoolValidationError(f"Missing {column!r} value in {path}.")
    try:
        numeric_value = float(str(raw_value))
        if not math.isfinite(numeric_value) or not numeric_value.is_integer():
            raise ValueError
        return int(numeric_value)
    except (TypeError, ValueError) as exc:
        raise FinalPoolValidationError(
            f"Invalid {column!r} value {raw_value!r} in {path}."
        ) from exc


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FinalPoolValidationError(f"Missing TACO result: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        missing = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
        if missing:
            raise FinalPoolValidationError(
                f"Invalid protocol-aligned TACO result {path}; missing columns: "
                + ", ".join(missing)
            )
        return [dict(row) for row in reader]


def _normalized_sequence(row: Mapping[str, str], path: Path) -> str:
    raw_sequence = row.get("sequence")
    if raw_sequence is None or not str(raw_sequence).strip():
        raise FinalPoolValidationError(f"Missing DNA sequence in {path}.")
    return str(raw_sequence).strip().upper()


def _validate_sequence(sequence: str, expected_length: int, path: Path) -> None:
    invalid = set(sequence) - DNA_ALPHABET
    if invalid:
        raise FinalPoolValidationError(
            f"Non-ACGT symbols {sorted(invalid)} in TACO sequence from {path}."
        )
    if len(sequence) != expected_length:
        raise FinalPoolValidationError(
            f"TACO sequence length {len(sequence)} in {path}; expected {expected_length}."
        )


def expected_physical_batches(effective_round: int) -> tuple[int, ...]:
    """Return the four B64/GA4 microbatches forming an effective round."""
    if effective_round < 1:
        raise FinalPoolValidationError("Effective round must be at least 1.")
    final_batch = effective_round * PROTOCOL_GRAD_ACCUM_STEPS
    return tuple(
        range(final_batch - PROTOCOL_GRAD_ACCUM_STEPS + 1, final_batch + 1)
    )


def _validate_common_metadata(
    rows: Sequence[Mapping[str, str]],
    *,
    path: Path,
    expected_cell_line: str,
    expected_seed: int,
    expected_sequence_length: int,
) -> tuple[str, ...]:
    normalized_cell = expected_cell_line.strip().lower()
    observed_cells = {
        str(row.get("cell_line", "")).strip().lower() for row in rows
    }
    if observed_cells != {normalized_cell}:
        raise FinalPoolValidationError(
            f"Wrong cell_line metadata in {path}: {sorted(observed_cells)!r}; "
            f"expected {normalized_cell!r}."
        )

    observed_seeds = {_parse_int(row, "seed", path) for row in rows}
    if observed_seeds != {expected_seed}:
        raise FinalPoolValidationError(
            f"Wrong seed metadata in {path}: {sorted(observed_seeds)}; "
            f"expected {expected_seed}."
        )

    batch_sizes = {_parse_int(row, "physical_batch_size", path) for row in rows}
    if batch_sizes != {PROTOCOL_PHYSICAL_BATCH_SIZE}:
        raise FinalPoolValidationError(
            f"Wrong physical_batch_size in {path}: {sorted(batch_sizes)}; "
            f"expected {PROTOCOL_PHYSICAL_BATCH_SIZE}."
        )

    grad_accum_values = {_parse_int(row, "grad_accum_steps", path) for row in rows}
    if grad_accum_values != {PROTOCOL_GRAD_ACCUM_STEPS}:
        raise FinalPoolValidationError(
            f"Wrong grad_accum_steps in {path}: {sorted(grad_accum_values)}; "
            f"expected {PROTOCOL_GRAD_ACCUM_STEPS}."
        )

    sequences = tuple(_normalized_sequence(row, path) for row in rows)
    for sequence in sequences:
        _validate_sequence(sequence, expected_sequence_length, path)
    return sequences


def load_validated_final_pool(
    path: Union[str, Path],
    *,
    expected_cell_line: str,
    expected_seed: int,
    effective_round: int,
    expected_pool_size: int = PROTOCOL_POOL_SIZE,
    expected_sequence_length: int = PROTOCOL_SEQUENCE_LENGTH,
    require_complete_run: bool = False,
) -> FinalPool:
    """Load one effective round and fail on any protocol-invariant violation.

    ``require_complete_run`` additionally checks that the annotated source is
    the complete 25,600-candidate B64/GA4 run rather than a final-pool-only
    extract.  Composition and FIMO use the same function and therefore the same
    sequence-selection semantics.
    """
    source = Path(path)
    all_rows = _read_rows(source)
    if not all_rows:
        raise FinalPoolValidationError(f"No rows in TACO result: {source}")

    if require_complete_run:
        if len(all_rows) != PROTOCOL_TOTAL_CANDIDATES:
            raise FinalPoolValidationError(
                f"Incomplete TACO run {source}: {len(all_rows)} rows; "
                f"expected {PROTOCOL_TOTAL_CANDIDATES}."
            )
        all_sequences = _validate_common_metadata(
            all_rows,
            path=source,
            expected_cell_line=expected_cell_line,
            expected_seed=expected_seed,
            expected_sequence_length=expected_sequence_length,
        )
        if len(set(all_sequences)) != PROTOCOL_TOTAL_CANDIDATES:
            raise FinalPoolValidationError(
                f"Complete TACO run {source} contains duplicate DNA sequences."
            )
        all_physical = [_parse_int(row, "round", source) for row in all_rows]
        physical_counts = Counter(all_physical)
        if set(physical_counts) != set(range(1, PROTOCOL_TOTAL_PHYSICAL_BATCHES + 1)):
            raise FinalPoolValidationError(
                f"Complete TACO run {source} does not contain physical batches 1-"
                f"{PROTOCOL_TOTAL_PHYSICAL_BATCHES}."
            )
        if set(physical_counts.values()) != {PROTOCOL_PHYSICAL_BATCH_SIZE}:
            raise FinalPoolValidationError(
                f"Complete TACO run {source} must contain "
                f"{PROTOCOL_PHYSICAL_BATCH_SIZE} rows per physical batch."
            )
        all_effective = [_parse_int(row, "effective_round", source) for row in all_rows]
        effective_counts = Counter(all_effective)
        if set(effective_counts) != set(range(1, PROTOCOL_TOTAL_EFFECTIVE_ROUNDS + 1)):
            raise FinalPoolValidationError(
                f"Complete TACO run {source} does not contain effective rounds 1-"
                f"{PROTOCOL_TOTAL_EFFECTIVE_ROUNDS}."
            )
        if set(effective_counts.values()) != {PROTOCOL_POOL_SIZE}:
            raise FinalPoolValidationError(
                f"Complete TACO run {source} must contain {PROTOCOL_POOL_SIZE} rows "
                "per effective round."
            )

    selected_rows = tuple(
        row
        for row in all_rows
        if _parse_int(row, "effective_round", source) == effective_round
    )
    if len(selected_rows) != expected_pool_size:
        raise FinalPoolValidationError(
            f"Effective Round {effective_round} in {source} contains "
            f"{len(selected_rows)} rows; expected exactly {expected_pool_size}."
        )

    sequences = _validate_common_metadata(
        selected_rows,
        path=source,
        expected_cell_line=expected_cell_line,
        expected_seed=expected_seed,
        expected_sequence_length=expected_sequence_length,
    )
    if len(set(sequences)) != expected_pool_size:
        raise FinalPoolValidationError(
            f"Effective Round {effective_round} in {source} contains duplicate DNA sequences."
        )

    physical_batches = expected_physical_batches(effective_round)
    observed_physical = [_parse_int(row, "round", source) for row in selected_rows]
    physical_counts = Counter(observed_physical)
    if set(physical_counts) != set(physical_batches):
        raise FinalPoolValidationError(
            f"Effective Round {effective_round} in {source} uses physical batches "
            f"{sorted(physical_counts)}; expected {list(physical_batches)}."
        )
    if set(physical_counts.values()) != {PROTOCOL_PHYSICAL_BATCH_SIZE}:
        raise FinalPoolValidationError(
            f"Effective Round {effective_round} in {source} must contain "
            f"{PROTOCOL_PHYSICAL_BATCH_SIZE} rows in each physical batch; "
            f"observed {dict(sorted(physical_counts.items()))}."
        )

    observed_effective = {
        _parse_int(row, "effective_round", source) for row in selected_rows
    }
    if observed_effective != {effective_round}:
        raise FinalPoolValidationError(
            f"Wrong effective_round metadata in {source}: "
            f"{sorted(observed_effective)}; expected {effective_round}."
        )

    return FinalPool(
        source=source,
        cell_line=expected_cell_line.strip().lower(),
        seed=expected_seed,
        effective_round=effective_round,
        physical_batches=physical_batches,
        rows=selected_rows,
        sequences=sequences,
    )
