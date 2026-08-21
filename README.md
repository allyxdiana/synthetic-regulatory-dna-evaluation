# Synthetic regulatory DNA evaluation

## 1. Purpose

This repository contains the public analysis workflow for the master's thesis
*Evaluating Generative Approaches for the Design of Synthetic DNA Sequences*.
It compares real HepG2 and K562 regulatory DNA with statistical baselines,
TFBS-guided sequences, and protocol-aligned TACO outputs.

The repository is reproducible with documented external inputs. Large source
datasets, the JASPAR collection, MEME Suite, TACO checkpoints, and generated
TACO candidates are intentionally not duplicated here. Exact input checksums,
commands, and the thesis-item mapping are in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## 2. Thesis analysis overview

The final primary comparison covers:

- real HepG2 and K562 sequences;
- uniform, GC-content-matched, and first-order Markov baselines;
- TFBS-guided generation using three cell-line-specific JASPAR motifs;
- TACO seeds 0, 1, 2, 3, and 4 for each cell line; and
- the validated Effective Round 100 final sampled same-policy pool from each
  TACO seed.

Every final TACO composition and FIMO analysis uses the same validated
Effective Round 100 pool of 256 unique sequences per seed.

## 3. Repository scope

`master_workflow.py` is the public entry point. It can generate the statistical
and TFBS-guided baselines, analyze sequence composition, validate and analyze
external TACO results, run selected-motif FIMO scans, aggregate seed-level
statistics, and create the final figures.

The workflow does not train reward models or generate TACO candidates. Those
GPU-intensive steps belong to the separate TACO repository and are outside this
workflow.

## 4. What is included and not included

Included:

- all Python code used by the public postprocessing workflow;
- the strict protocol-aligned TACO final-pool validator;
- baseline and TFBS-guided sequence generators;
- composition, FIMO, aggregation, and final figure code;
- synthetic unit tests; and
- documentation of required external inputs.

Not included:

- the 317,862-sequence HepG2 and K562 source CSVs;
- the protocol-aligned TACO candidate package;
- the JASPAR 2024 MEME export;
- MEME Suite/FIMO;
- TACO, regLM, guide, oracle, or HyenaDNA checkpoints; or
- the thesis/Overleaf source and thesis-ready output files.

## 5. External data and model requirements

Postprocessing requires:

- `hepg2/mbo.csv` and `k562/mbo.csv`, each with a `sequence` column;
- the annotated protocol-aligned TACO result CSV for every selected cell
  line/seed;
- the combined JASPAR 2024 MEME file containing the six selected matrices; and
- FIMO 5.5.9 for regenerating the final motif scans.

Full TACO candidate regeneration additionally requires the checkpoints, reward
files, TFBS inputs, HyenaDNA assets, and runner snapshot. These generation
assets are not needed when the annotated result package is supplied to this
repository; the external requirements are summarized in
[REPRODUCIBILITY.md](REPRODUCIBILITY.md).

The external resources retain their own licenses. This repository's MIT
license applies only to the code distributed here.

## 6. Environment setup

The postprocessing workflow was validated with Python 3.9.23. Python itself is
not pinned by `requirements.txt`; that file pins only the required Python
packages:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

FIMO is a separate executable and is not installed by `pip`. Confirm the
version used for the final thesis analysis with:

```bash
fimo --version
```

The TACO generation environment is separate: the final run metadata recorded
Python 3.9.25, PyTorch 2.8.0+cu128, and CUDA 12.8; `env_install.sh` in the
documented TACO snapshot pins PyTorch Lightning 1.8.6.

## 7. Input directory structure

All external roots are configurable. A convenient layout is:

```text
external/TACO_MA/
├── data/
│   ├── hepg2/mbo.csv
│   └── k562/mbo.csv
├── results/protocol_aligned_result_package/
│   └── full_runs/
│       ├── hepg2_seed0_b64_ga4/analysis/
│       │   └── hepg2_seed0_all_candidates_annotated.csv
│       └── ...
└── local_assets/tfbs/human/
    └── 20240913075738_JASPAR2024_combined_matrices_1210274_meme.txt
```

The default TACO result pattern is:

```text
full_runs/{cell}_seed{seed}_b64_ga4/analysis/
{cell}_seed{seed}_all_candidates_annotated.csv
```

Each annotated CSV must contain:

```text
cell_line,seed,round,effective_round,physical_batch_size,
grad_accum_steps,sequence,surrogate_score,oracle_score
```

The source files contain 25,600 candidate rows per run. The final thesis pool
is Effective Round 100: 256 unique 200-bp sequences from physical
microbatches 397--400 with physical batch size 64 and gradient accumulation 4.

## 8. Statistical baselines

The workflow creates:

- `baseline_A_uniform_random.py`: independent uniform A/C/G/T sampling;
- `generate_gc_matched_random.py`: one length- and GC-matched sequence per
  source sequence; and
- `baseline_C_markov_random.py`: first-order Markov sampling learned from the
  real cell-line data.

The default generation seed in `master_workflow.py` is 23 and can be changed
with `--seed`. A CSV export-index column such as `Unnamed: 0` is optional; the
scientific input requirement is the `sequence` column.

## 9. TFBS-guided generation

`tfbs_guided_generation.py` samples motif instances from the same JASPAR PPMs
used by FIMO and inserts them into first-order Markov backgrounds. The selected
matrices are:

| Cell line | Motif | Matrix ID |
|---|---|---|
| HepG2 | CEBPA | MA0102.5 |
| HepG2 | FOXA2 | MA0047.4 |
| HepG2 | HNF4A | MA1494.2 |
| K562 | GATA1 | MA0035.5 |
| K562 | GATA1::TAL1 | MA0140.3 |
| K562 | KLF1 | MA0493.3 |

The motif-file checksum is provided in
[REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## 10. External TACO generation

The ten final external runs comprise HepG2 and K562 seeds 0--4. Each run used
400 physical B64 microbatches, gradient accumulation over four microbatches,
100 regular optimizer updates, and 25,600 generated candidates. A separate
initialization optimizer step occurred before candidate generation.

The external code revisions and candidate-package identity are specified in
[REPRODUCIBILITY.md](REPRODUCIBILITY.md). The analysis repository never invokes
the GPU runner.

## 11. Protocol-aligned Effective Round 100 analysis

`taco_final_pool.py` is shared by the composition and FIMO paths. It fails
loudly unless the requested pool has the expected cell line and seed, exactly
256 rows and unique valid 200-bp DNA sequences, the four expected physical
microbatches, B64, and GA4 metadata. The master workflow additionally verifies
the complete 25,600-row, 400-microbatch, 100-effective-round source run.

For Effective Round `r`, same-policy membership is inferred from consecutive
microbatches `4r-3` through `4r`: all four are sampled before the next optimizer
step. Thus, Round 100 corresponds to microbatches 397--400. The thesis defaults
can be changed explicitly with `--taco-seeds` and `--taco-round`; no silent
fallback occurs.

## 12. Composition analysis

The workflow reports GC content, per-sequence mononucleotide Shannon entropy,
nucleotide frequencies, and dinucleotide frequencies. Final TACO outputs keep
the five seed-level values and report mean, sample SD (`ddof=1`), minimum, and
maximum across seeds.

The thesis-specific plots use:

- HepG2 six-panel heatmaps: Original plus seeds 0--4, shared 0.00--0.20 scale;
- K562 six-panel heatmaps: Original plus seeds 0--4, shared 0.00--0.66 scale;
- contrast-aware cell labels with threshold `vmax * 0.52`;
- HepG2 detail plot: selected dinucleotides plus per-sequence CG frequency; and
- K562 detail plot: selected dinucleotides plus per-sequence GG frequency.

Both detail barplots retain the established 0--70% scale.

## 13. FIMO analysis

The final scans were generated with MEME Suite/FIMO 5.5.9 using:

- both strands (FIMO default; no `--norc`);
- `p < 1e-4`;
- no primary q-value filtering;
- no `--bgfile`; and
- FIMO's default zero-order NRDB background, A=T=0.275 and C=G=0.225.

`scan_selected_tfbs.py` warns when the available FIMO version differs from
5.5.9. `master_workflow.py --fimo auto` skips scanning if FIMO is absent;
`--fimo yes` requires it. Existing final FIMO outputs can be aggregated
without rerunning TACO or FIMO. The final aggregator rejects TACO scan inputs
unless every selected seed contains exactly 256 sequences.

## 14. Final figures

The final workflow creates these thesis comparisons:

- Figures 4.1--4.4: statistical and TFBS-guided composition figures;
- Figure 4.5: `taco_hepg2_dinucleotide_heatmaps_five_seed.{pdf,png}`;
- Figure 4.6: HepG2 TACO barplot and CG boxplot;
- Figure 4.7: `taco_k562_dinucleotide_heatmaps_five_seed.{pdf,png}`;
- Figure 4.8: K562 TACO barplot and GG boxplot;
- Figures 4.9--4.10: log-scale selected-TFBS occurrence plots.

The final TFBS figures contain one TACO mean bar, sample-SD error bars, and five
individual seed points. True zeros remain zero in the data and are not replaced
with plotting pseudocounts. Exact script/input/output mappings are in
[REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## 15. Final tables and machine-readable outputs

The underlying baseline scripts write dataset summary and dinucleotide CSVs.
The final TACO detail analysis additionally writes:

```text
{cell}_taco_seed_composition_values.csv
{cell}_taco_composition_summary.csv
```

The final FIMO aggregation writes:

```text
{cell}_taco_seed_summary.csv
{cell}_tfbs_dataset_summary.csv
{cell}_tfbs_figure_data.csv
{cell}_tfbs_percentages_rescan.csv
```

These retain per-motif occurrence, total hits, mean hits per sequence, Any
selected TFBS, All 3 selected TFBS, and the TACO seed-level values. The mapping
for Tables 4.1--4.8 is in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## 16. Master workflow

Show all options and defaults:

```bash
python master_workflow.py --help
```

Reproduce the final five-seed Round-100 workflow with explicit external roots:

```bash
python master_workflow.py all \
  --taco-data-dir /path/to/TACO_MA/data \
  --taco-results-dir /path/to/protocol_aligned_taco_results \
  --motif-file /path/to/JASPAR2024_combined.meme \
  --taco-seeds 0 1 2 3 4 \
  --taco-round 100 \
  --fimo yes \
  --output-root /path/to/output
```

The defaults are seeds `0 1 2 3 4` and round `100`. Any available subset or
another requested effective round can be supplied, for example
`--taco-seeds 1 3 4 --taco-round 99`. Every table and plot adapts to the
selected seed count.

Useful modes:

```bash
python master_workflow.py generate
python master_workflow.py analyze
python master_workflow.py tfbs --fimo yes
python master_workflow.py --dry-run --fimo yes
```

All generated files and `master_workflow_report.json` are written below
`--output-root`. Custom FASTA/CSV datasets may be added with
`--custom-inputs label=/path/to/sequences.csv`; CSV inputs require `sequence`.
They are included in composition analyses and a separate FIMO scan, not in the
fixed six-group thesis TFBS figures.

## 17. Tests

The tests use small synthetic inputs and do not depend on private datasets:

```bash
python -m unittest discover -s tests -v
```

They cover CLI defaults/custom propagation, Effective Round 100 selection,
final-pool failures, identical composition/FIMO selection, known entropy and
dinucleotide examples, optional CSV export indices, sample-SD aggregation,
log-scale TFBS plots, true zeros, seed points, six-panel heatmaps, fixed scales,
and contrast-aware annotations.

## 18. Reproducibility notes

The exact source-data and motif checksums are authoritative when an upstream
location is mutable. Verify all inputs before running the workflow. The
complete commands, checksums, output mapping, and clean-room procedure are in
[REPRODUCIBILITY.md](REPRODUCIBILITY.md).

The final analysis logic is self-contained here, but bit-for-bit regeneration
of TACO candidates remains limited by external checkpoints and several
generation-time assets for which immutable execution-time hashes were not
recorded. This does not prevent deterministic extraction and postprocessing of
the documented annotated result package.

## 19. Known computational requirements

Baseline generation and composition analysis operate on two 317,862-sequence
datasets and therefore require substantial CPU time, memory, and disk space.
FIMO scans are also computationally expensive. The ten TACO generation runs
used NVIDIA RTX 3070 GPUs with 8 GiB VRAM, but GPU generation is outside this
repository's master workflow.

Use `--dry-run` to inspect every command without computation. For a quick FIMO
wiring check, `--max-sequences` can limit scan inputs, but such a limited run is
not the final thesis analysis and is therefore not passed to the final TFBS
aggregation/figure step.

## Citation

> Jury, Alina (2026). *Evaluating Generative Approaches for the Design of
> Synthetic DNA Sequences*. Master's thesis, FH JOANNEUM.

Machine-readable citation metadata is provided in `CITATION.cff`.

## License

The code in this repository is released under the MIT License; see `LICENSE`.
External datasets, motif files, model assets, software, and thesis text retain
their respective licenses.
