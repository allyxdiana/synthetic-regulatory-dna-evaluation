# Reproducing the thesis postprocessing analysis

This document describes how to reproduce the analysis represented by the final
master's thesis: statistical baselines, TFBS-guided generation, validation and
extraction of the protocol-aligned TACO Effective Round 100 pools, sequence
composition, FIMO scans, machine-readable tables, and final plots.

The result is **reproducible with documented external inputs**. Large source
data, the combined JASPAR export, FIMO, and the TACO candidate package are not
vendored in this repository. Their identities and required interfaces are
specified below. The external package supports deterministic postprocessing;
bit-for-bit regeneration of the TACO candidates remains limited by incomplete
immutable identities for generation-time assets, as summarized in Section 11.

## 1. Reproduction boundary

| Stage | Performed by this repository? | Required input |
|---|---|---|
| Uniform, GC-matched, and first-order Markov generation | Yes | Original HepG2/K562 `mbo.csv` files |
| TFBS-guided generation | Yes | Original data and combined JASPAR MEME file |
| TACO policy optimization and candidate generation | **No** | Separate TACO code, models, and GPU environment |
| Validation and extraction of TACO Effective Round 100 | Yes | External annotated protocol-aligned candidate CSVs |
| Composition analysis and final composition plots | Yes | Real and generated sequences |
| FIMO scans, aggregation, and final TFBS plots | Yes | Real/generated sequences, JASPAR MEME file, FIMO 5.5.9 |

`master_workflow.py` never invokes the TACO GPU runner. It treats the annotated
candidate package as an immutable external input, validates each complete run,
and passes the same validated 256-sequence pool to both composition and FIMO.
The external TACO revisions and candidate-package contract are specified in
Section 3.3.

## 2. Software environment

The final postprocessing environment used Python 3.9.23. The Python packages
needed by this repository are pinned in `requirements.txt`:

```text
matplotlib==3.9.4
numpy==1.26.4
pandas==2.3.3
```

From a clean clone:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python --version
```

FIMO is an external executable, not a Python dependency. Install MEME Suite
5.5.9 separately and verify that the intended executable is on `PATH`:

```bash
fimo --version
```

The thesis-scale baseline generation, composition analysis, and FIMO scans are
CPU-, memory-, and disk-intensive. No GPU is required for this repository's
postprocessing workflow. GPU requirements apply only to the external TACO
generation step.

## 3. External inputs

Use these path variables in the commands below:

```bash
export TACO_MA_ROOT=/path/to/TACO_MA
export TACO_DATA_DIR="$TACO_MA_ROOT/data"
export TACO_RESULTS_DIR="$TACO_MA_ROOT/results/protocol_aligned_20260810"
export JASPAR_MEME="$TACO_MA_ROOT/local_assets/tfbs/human/20240913075738_JASPAR2024_combined_matrices_1210274_meme.txt"
export THESIS_OUTPUT=/path/to/new/thesis_analysis_output
```

Use a new, empty output directory so the execution manifest and generated files
form one self-contained reproduction.

### 3.1 Original regulatory-sequence data

The exact thesis inputs are:

| Cell line | Required path | Rows | Sequence contract | SHA-256 |
|---|---|---:|---|---|
| HepG2 | `$TACO_DATA_DIR/hepg2/mbo.csv` | 317,862 | `sequence`; exactly 200 bp; A/C/G/T only | `c0e36c7b2c9a9e3dc1360dd07c08436c1cd873211fa5c0457340cfacb687da92` |
| K562 | `$TACO_DATA_DIR/k562/mbo.csv` | 317,862 | `sequence`; exactly 200 bp; A/C/G/T only | `b56bef39c7307dc0c1c036b5b705d1554128f11e50b251189c530f4030b04437` |

The files are the prepared offline-MBO datasets distributed through the
[official TACO dataset collection](https://huggingface.co/datasets/yangyz1230/TACO/tree/main).
Their data-processing lineage is the regLM resource archived at
[Zenodo record 12668907](https://zenodo.org/records/12668907). The upstream
collection is mutable and the exact download revision was not recorded at run
time; therefore, the SHA-256 values above, rather than a `main`-branch URL, are
the authoritative identity of the thesis copies.

Only the `sequence` column is required by this postprocessing repository. CSV
export-index columns such as `Unnamed: 0` do not carry scientific information
and are not required by the readers.

### 3.2 JASPAR motif input

The required combined JASPAR 2024 MEME file is:

```text
20240913075738_JASPAR2024_combined_matrices_1210274_meme.txt
SHA-256: 44a87ef7a303553128c981eb891582bbba9ba7beb88855d3699c6d456df518ad
```

It was obtained from the [JASPAR 2024 collection](https://jaspar.elixir.no/).
An immutable export URL or JASPAR release-file revision was not captured; the
filename and checksum identify the exact input. The workflow extracts only the
following six matrices from that combined file:

| Cell line | Motif | Matrix ID |
|---|---|---|
| HepG2 | CEBPA | `MA0102.5` |
| HepG2 | FOXA2 | `MA0047.4` |
| HepG2 | HNF4A | `MA1494.2` |
| K562 | GATA1 | `MA0035.5` |
| K562 | GATA1::TAL1 | `MA0140.3` |
| K562 | KLF1 | `MA0493.3` |

Do not substitute a newer matrix version with the same TF name.

### 3.3 Protocol-aligned TACO candidate package

The official upstream implementation is
[yangzhao1230/TACO](https://github.com/yangzhao1230/TACO). Final TACO generation
used the public fork [allyxdiana/TACO_MA](https://github.com/allyxdiana/TACO_MA)
with core-code revision `d9ee93aae5d882d2e201fecb582bcf3593e23460`, as
recorded in every run metadata file. The preserved protocol-aligned runner,
result snapshot, and annotated package are identified by commit
[`25484a5cb50725809ae1975378e7218dd4188752`](https://github.com/allyxdiana/TACO_MA/tree/25484a5cb50725809ae1975378e7218dd4188752/results/protocol_aligned_20260810).
The run metadata does not contain an execution-time hash of the runner, so its
byte identity at execution is not independently established.

One way to obtain the immutable package is:

```bash
git clone https://github.com/allyxdiana/TACO_MA.git "$TACO_MA_ROOT"
git -C "$TACO_MA_ROOT" checkout 25484a5cb50725809ae1975378e7218dd4188752
```

The original `mbo.csv` and JASPAR files are ignored external assets and must be
placed at the paths in Sections 3.1 and 3.2 after cloning.

The default annotated-result path for each cell line and seed is:

```text
$TACO_RESULTS_DIR/full_runs/{cell}_seed{seed}_b64_ga4/analysis/
  {cell}_seed{seed}_all_candidates_annotated.csv
```

The required columns are:

```text
cell_line,seed,round,effective_round,physical_batch_size,
grad_accum_steps,sequence,surrogate_score,oracle_score
```

`round` is the physical microbatch index. Each complete annotated run must
satisfy all of these invariants:

- the file's `cell_line` and `seed` match its requested analysis identity;
- 25,600 rows and 25,600 unique 200-bp A/C/G/T sequences;
- `physical_batch_size = 64` and `grad_accum_steps = 4` on every row;
- physical microbatches 1--400, with 64 rows in each;
- effective rounds 1--100, with 256 rows in each.

The final analysis selects Effective Round 100, which must contain exactly 256
unique sequences from physical microbatches 397, 398, 399, and 400, with 64
sequences from each. These four batches were sampled under one policy state
after 99 regular optimizer updates and before the update following microbatch
400. The same-policy interpretation follows the documented TACO runner's
gradient-accumulation control flow; the CSV validation establishes the matching
B64/GA4 membership and metadata.

Pool membership is determined solely by cell line, seed, effective-round, B64,
and GA4 metadata. The package retains `surrogate_score` and `oracle_score` as
measurements associated with each sequence.

## 4. Verify external inputs before analysis

The following cross-platform Python check verifies the three authoritative
SHA-256 values, the original-sequence invariants, and the required motif IDs.
Run it from any directory after exporting the variables in Section 3:

```bash
python - <<'PY'
import hashlib
import os
import re
from pathlib import Path

import pandas as pd

data_root = Path(os.environ["TACO_DATA_DIR"])
motif_file = Path(os.environ["JASPAR_MEME"])
expected_hashes = {
    data_root / "hepg2" / "mbo.csv": "c0e36c7b2c9a9e3dc1360dd07c08436c1cd873211fa5c0457340cfacb687da92",
    data_root / "k562" / "mbo.csv": "b56bef39c7307dc0c1c036b5b705d1554128f11e50b251189c530f4030b04437",
    motif_file: "44a87ef7a303553128c981eb891582bbba9ba7beb88855d3699c6d456df518ad",
}

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

for path, expected in expected_hashes.items():
    observed = sha256(path)
    print(f"{path}: {observed}")
    if observed != expected:
        raise SystemExit(f"SHA-256 mismatch for {path}")

for cell in ("hepg2", "k562"):
    path = data_root / cell / "mbo.csv"
    count = 0
    for chunk in pd.read_csv(path, usecols=["sequence"], chunksize=50_000):
        if chunk["sequence"].isna().any():
            raise SystemExit(f"Missing sequence in {path}")
        sequences = chunk["sequence"].astype(str).str.upper()
        if not sequences.str.fullmatch(r"[ACGT]{200}").all():
            raise SystemExit(f"Invalid sequence in {path}")
        count += len(sequences)
    if count != 317_862:
        raise SystemExit(f"Expected 317862 sequences in {path}; found {count}")

required_ids = {
    "MA0102.5", "MA0047.4", "MA1494.2",
    "MA0035.5", "MA0140.3", "MA0493.3",
}
motif_text = motif_file.read_text(encoding="utf-8")
found_ids = set(re.findall(r"^MOTIF\s+(\S+)", motif_text, flags=re.MULTILINE))
missing = required_ids - found_ids
if missing:
    raise SystemExit(f"Missing motif IDs: {sorted(missing)}")
print("Checksums, source-sequence invariants, and selected motif IDs: OK")
PY
```

From the root of this repository, validate all ten complete TACO files and
their final pools without running any analysis:

```bash
python - <<'PY'
import os
from pathlib import Path

from taco_final_pool import load_validated_final_pool

root = Path(os.environ["TACO_RESULTS_DIR"])
pattern = (
    "full_runs/{cell}_seed{seed}_b64_ga4/analysis/"
    "{cell}_seed{seed}_all_candidates_annotated.csv"
)
for cell in ("hepg2", "k562"):
    for seed in range(5):
        path = root / pattern.format(cell=cell, seed=seed)
        pool = load_validated_final_pool(
            path,
            expected_cell_line=cell,
            expected_seed=seed,
            effective_round=100,
            require_complete_run=True,
        )
        assert pool.physical_batches == (397, 398, 399, 400)
        print(cell, seed, len(pool.sequences), pool.physical_batches)
print("All ten complete runs and Effective Round 100 pools: OK")
PY
```

This validation requires the exact complete-run and Effective Round 100
contracts and terminates on any violation.

## 5. Exact thesis postprocessing command

Run from the root of `synthetic-regulatory-dna-evaluation` after completing the
setup and checksum checks above:

```bash
python master_workflow.py all \
  --cell-types hepg2 k562 \
  --taco-data-dir "$TACO_DATA_DIR" \
  --taco-results-dir "$TACO_RESULTS_DIR" \
  --motif-file "$JASPAR_MEME" \
  --taco-pattern 'full_runs/{cell}_seed{seed}_b64_ga4/analysis/{cell}_seed{seed}_all_candidates_annotated.csv' \
  --taco-seeds 0 1 2 3 4 \
  --taco-round 100 \
  --seed 23 \
  --fimo yes \
  --fimo-pvalue 1e-4 \
  --fimo-max-stored-scores 10000000 \
  --output-root "$THESIS_OUTPUT"
```

Do not add `--max-sequences`; that option is only a wiring/test convenience
and would not reproduce the thesis populations, so the master workflow skips
the final TFBS aggregation/figure step for such limited scans. `--fimo yes` is deliberate:
unlike `--fimo auto`, it fails if FIMO is unavailable instead of skipping the
scan. The output root and every external input root remain configurable; no
machine-specific path is required.

The workflow performs these stages in order:

1. generates one uniform, GC-matched, Markov, and TFBS-guided sequence per
   source sequence using seed 23;
2. calculates sequence-level GC content and mononucleotide Shannon entropy,
   pooled nucleotide/dinucleotide frequencies, and heatmaps;
3. validates all ten TACO source files and extracts Effective Round 100;
4. calculates TACO seed-level composition and creates the final TACO figures;
5. prepares FASTA inputs and runs the selected-motif FIMO scans; and
6. aggregates motif occurrence and burden metrics and creates the final TFBS
   tables and plots.

The Shannon entropy reported for each sequence is
`-sum(p * log2(p) for p > 0)` over A, C, G, and T. Across TACO seeds, reported
standard deviations are sample standard deviations (`ddof=1`); minimum and
maximum retain the observed seed range.

The workflow writes an execution manifest to
`$THESIS_OUTPUT/master_workflow_report.json`. A nonzero process exit status or
any failed step means the reproduction is incomplete.

## 6. Exact FIMO configuration and outputs

For each dataset, `scan_selected_tfbs.py` constructs and executes this command:

```text
fimo --oc OUTPUT_DIRECTORY --thresh 0.0001 --max-stored-scores 10000000 \
  SELECTED_MOTIFS.meme PREPARED_SEQUENCES.fasta
```

The thesis configuration is:

- MEME Suite/FIMO 5.5.9;
- both DNA strands (FIMO default; no `--norc`);
- p-value threshold `1e-4`;
- no q-value filtering for the primary analysis;
- no separate `--bgfile`; and
- FIMO's default zero-order NRDB background: A=T=0.275 and C=G=0.225.

FIMO calculates q-values by default and they remain available in `tfbs_hits.csv`,
but the workflow does not use them to filter the primary results. A completed
FIMO 5.5.9 run records `scan both strands=true`, `threshold type=p-value`,
`output threshold=0.0001`, and `background source=--nrdb--` in `fimo.xml`.

Each scan group writes:

```text
selected_motifs_{cell}.meme
selected_motifs.csv
prepared_fastas/
fimo_runs/{dataset}/fimo.tsv
fimo_runs/{dataset}/fimo.xml
tfbs_hits.csv
tfbs_per_sequence_counts.csv
tfbs_summary.csv
```

`tfbs_summary.csv` contains, per dataset and motif, the number and fraction of
sequences with at least one hit, total hits, and mean hits per sequence.
`plot_tfbs_rescan_results.py` derives `Any selected TFBS` and `All 3 selected
TFBS` from `tfbs_per_sequence_counts.csv`. The main thesis metric is the
percentage of sequences with at least one significant hit; TFBS-guided results
measure FIMO detectability/hit burden, not recovery of the inserted motif's
ground-truth orientation.

## 7. Output layout

The paths used below are relative to `$THESIS_OUTPUT`:

```text
statistical_baselines/
tfbs_guided_generation/
master_analysis/
  uniform/
  gc_matched/
  markov/
  tfbs_guided/
  taco/{cell}_seed_{seed}/
taco_thesis_heatmaps/
taco_hepg2_compositional_bias/
taco_k562_compositional_bias/
tfbs_analysis/
  rescan_maxstored10m/
  figures/
master_workflow_report.json
```

The analysis directories contain the machine-readable values used for the
thesis. LaTeX tables and the copying/assembly of generated panels into the
Overleaf figure tree are not automated by this repository. Consequently,
matching the generated CSV values and image files is the scientific
reproduction endpoint; publication formatting and displayed rounding remain a
small, explicit manual step.

## 8. Thesis table mapping

`<OUT>` below means `$THESIS_OUTPUT`; `<DATA>` means `$TACO_DATA_DIR`; and
`<TACO>` means `$TACO_RESULTS_DIR`.

| Thesis table | Generating code | Scientific input | Machine-readable output and value source |
|---|---|---|---|
| Table 4.1, real vs uniform HepG2 composition | `baseline_A_uniform_random.py` then `analyze_statistical_baseline_same_heatmap_scale.py` | `<DATA>/hepg2/mbo.csv`; generated `statistical_baselines/baseline_A_uniform_hepg2.fasta` | `<OUT>/master_analysis/uniform/dataset_summary_statistics.csv` (`mean_gc_content`) and `dinucleotide_frequencies.csv` (AA, TT, CG rows) |
| Table 4.2, real vs GC-matched HepG2 composition | `generate_gc_matched_random.py` then `analyze_statistical_baseline_same_heatmap_scale.py` | HepG2 original; generated `baseline_B_gc_matched_hepg2.fasta` | `<OUT>/master_analysis/gc_matched/dataset_summary_statistics.csv` and `dinucleotide_frequencies.csv` |
| Table 4.3, real vs Markov HepG2 composition | `baseline_C_markov_random.py` then `analyze_statistical_baseline_same_heatmap_scale.py` | HepG2 original; generated `baseline_C_markov_hepg2.fasta` | `<OUT>/master_analysis/markov/dataset_summary_statistics.csv` (`mean_gc_content`, `mean_entropy`) and `dinucleotide_frequencies.csv` |
| Table 4.4, GC content and entropy across real, statistical, and TFBS-guided datasets | All four generation scripts; the two composition analyzers | Both original cell lines and all generated baseline/TFBS-guided FASTAs | `dataset_summary_statistics.csv` in `<OUT>/master_analysis/{uniform,gc_matched,markov,tfbs_guided}/`; select each approach/cell row and columns `mean_gc_content`, `mean_entropy` |
| Table 4.5, real and TACO HepG2 selected dinucleotides | `plot_taco_compositional_bias.py --cell-type hepg2` | HepG2 original; each seed's annotated CSV, validated Round 100 | `<OUT>/taco_hepg2_compositional_bias/hepg2_selected_dinucleotide_frequencies_aggregate.csv`; seed GC/entropy values and mean/sample-SD/range are additionally in `hepg2_taco_seed_composition_values.csv` and `hepg2_taco_composition_summary.csv` |
| Table 4.6, real and TACO K562 selected dinucleotides | `plot_taco_compositional_bias.py --cell-type k562` | K562 original; each seed's annotated CSV, validated Round 100 | `<OUT>/taco_k562_compositional_bias/k562_selected_dinucleotide_frequencies_aggregate.csv`; corresponding seed values/summary CSVs are in the same directory |
| Table 4.7, HepG2 FIMO occurrence | `scan_selected_tfbs.py` then `plot_tfbs_rescan_results.py` | HepG2 real/baselines/TFBS-guided plus the five validated Round 100 pools and HepG2 motifs | `<OUT>/tfbs_analysis/rescan_maxstored10m/hepg2_tfbs_percentages_rescan.csv`; detailed seed occurrence and burden values are in `hepg2_taco_seed_summary.csv` and `hepg2_tfbs_dataset_summary.csv` |
| Table 4.8, K562 FIMO occurrence | `scan_selected_tfbs.py` then `plot_tfbs_rescan_results.py` | K562 real/baselines/TFBS-guided plus the five validated Round 100 pools and K562 motifs | `<OUT>/tfbs_analysis/rescan_maxstored10m/k562_tfbs_percentages_rescan.csv`; detailed seed occurrence and burden values are in `k562_taco_seed_summary.csv` and `k562_tfbs_dataset_summary.csv` |

Tables 4.1--4.6 display composition values to three decimals. Tables 4.7 and
4.8 display individual motif and `Any TFBS` percentages to two decimals. Their
rare and zero-valued `All 3 TFBS` entries are shown to four decimals. The
thesis LaTeX tables use two decimals for the much larger TFBS-guided
`All 3 TFBS` values (35.66 and 36.25); the machine-readable values are
35.656039... and 36.247176..., respectively. This display-precision choice does
not change the underlying result.
Values are rounded for display, not truncated.
In particular, the K562 TACO GATA1 and GATA1::TAL1 values are exact zero-hit
observations, not small values hidden by rounding. Boldface in Tables 4.7 and
4.8 is a presentation annotation identifying the generated value with the
smallest absolute difference from the real-data value; it does not change the
underlying CSV.

## 9. Thesis figure mapping

| Thesis figure | Generating code | Scientific input | Workflow output |
|---|---|---|---|
| Figure 4.1, real/uniform HepG2 heatmaps | `analyze_statistical_baseline_same_heatmap_scale.py` | HepG2 original and uniform FASTA | `<OUT>/master_analysis/uniform/dinucleotide_heatmap_real_hepg2.png` and `dinucleotide_heatmap_uniform_hepg2.png` |
| Figure 4.2, real/GC-matched HepG2 heatmaps | `analyze_statistical_baseline_same_heatmap_scale.py` | HepG2 original and GC-matched FASTA | `<OUT>/master_analysis/gc_matched/dinucleotide_heatmap_real_hepg2.png` and `dinucleotide_heatmap_gc_matched_hepg2.png` |
| Figure 4.3, real/Markov HepG2 heatmaps | `analyze_statistical_baseline_same_heatmap_scale.py` | HepG2 original and Markov FASTA | `<OUT>/master_analysis/markov/dinucleotide_heatmap_real_hepg2.png` and `dinucleotide_heatmap_markov_hepg2.png` |
| Figure 4.4, real/TFBS-guided HepG2 heatmaps | `analyze_sequence_generation_outputs.py` | HepG2 original and TFBS-guided FASTA | `<OUT>/master_analysis/tfbs_guided/dinucleotide_heatmap_real_hepg2.png` and `dinucleotide_heatmap_tfbs_hepg2.png` |
| Figure 4.5, HepG2 Original plus TACO seeds 0--4 | `plot_taco_thesis_heatmaps.py --cell-type hepg2` | HepG2 original plus five validated Round 100 pools | `<OUT>/taco_thesis_heatmaps/taco_hepg2_dinucleotide_heatmaps_five_seed.{pdf,png}` |
| Figure 4.6, HepG2 detailed composition | `plot_taco_compositional_bias.py --cell-type hepg2` | Same HepG2 populations as Figure 4.5 | `<OUT>/taco_hepg2_compositional_bias/figures/taco_hepg2_compositional_bias_barplot.{pdf,png}` and `taco_hepg2_compositional_bias_cg_boxplot.{pdf,png}` |
| Figure 4.7, K562 Original plus TACO seeds 0--4 | `plot_taco_thesis_heatmaps.py --cell-type k562` | K562 original plus five validated Round 100 pools | `<OUT>/taco_thesis_heatmaps/taco_k562_dinucleotide_heatmaps_five_seed.{pdf,png}` |
| Figure 4.8, K562 detailed composition | `plot_taco_compositional_bias.py --cell-type k562` | Same K562 populations as Figure 4.7 | `<OUT>/taco_k562_compositional_bias/figures/taco_k562_compositional_bias_barplot.{pdf,png}` and `taco_k562_compositional_bias_gg_boxplot.{pdf,png}` |
| Figure 4.9, HepG2 selected TFBS occurrence | `plot_tfbs_rescan_results.py` | HepG2 `tfbs_summary.csv` and `tfbs_per_sequence_counts.csv` files from baseline, TACO, and TFBS-guided scans | `<OUT>/tfbs_analysis/figures/hepg2_tfbs_percentages_motifs_only_rescan.{pdf,png}`; plotted data in `<OUT>/tfbs_analysis/rescan_maxstored10m/hepg2_tfbs_figure_data.csv` |
| Figure 4.10, K562 selected TFBS occurrence | `plot_tfbs_rescan_results.py` | K562 scan summaries/counts | `<OUT>/tfbs_analysis/figures/k562_tfbs_percentages_motifs_only_rescan.{pdf,png}`; plotted data in `<OUT>/tfbs_analysis/rescan_maxstored10m/k562_tfbs_figure_data.csv` |

The thesis source names the Figure 4.2 GC-matched synthetic panel
`dinucleotide_heatmap_baseline_hepg2.png`; this is a publication-copy name for
the workflow output `dinucleotide_heatmap_gc_matched_hepg2.png`. Figures
4.1--4.4 are assembled as two LaTeX subfigures. Figures 4.6 and 4.8 likewise
combine the two separately generated PDF panels. These filename/copy and LaTeX
assembly steps do not recalculate data.

Figures 4.5 and 4.7 use one fixed shared scale per cell line (0.00--0.20 for
HepG2 and 0.00--0.66 for K562) and contrast-aware annotations. Figures 4.9 and
4.10 use a logarithmic percentage axis; true zero values remain zero in the
CSV and are not replaced with plotting pseudocounts.

## 10. Completion checks

After the command in Section 5 returns successfully:

1. inspect `$THESIS_OUTPUT/master_workflow_report.json` and require
   `exit_status` to be zero with no failed step;
2. confirm that every TACO record in the report refers to seeds 0--4 and
   Effective Round 100;
3. confirm that each TACO FIMO prepared FASTA contains 256 records;
4. inspect at least one `fimo.xml` per cell line and confirm FIMO 5.5.9,
   `scan both strands=true`, p-value threshold 0.0001, and NRDB frequencies
   A/T=0.275 and C/G=0.225;
5. compare the machine-readable outputs in Sections 8 and 9 before applying
   the thesis's displayed rounding; and
6. retain the complete output root so that the report, intermediate counts,
   FIMO files, final CSVs, and figures remain auditable together.

## 11. Remaining limitations

- The exact thesis copies of the original `mbo.csv` files and combined JASPAR
  export are identified by checksums, but immutable upstream download revisions
  were not recorded. A user must obtain byte-identical copies from the stated
  external collections or another byte-identical source.
- The protocol-aligned annotated TACO package is immutable at the public TACO
  fork commit cited above and is sufficient for deterministic validation,
  extraction, composition analysis, and FIMO input preparation.
- Bit-for-bit regeneration of the ten TACO candidate runs is not fully
  specified because not every generation-time checkpoint/source asset has an
  execution-time checksum or immutable revision. This limitation does not
  alter reproducibility of postprocessing the documented candidate package.
- The repository produces the scientific CSVs and figure files, but does not
  write the thesis's LaTeX tables or copy files into an Overleaf project.

Subject to those documented external inputs, the public workflow reproduces
the final thesis analysis using seeds 0--4 and the validated 256-sequence
Effective Round 100 same-policy pool from each seed.
