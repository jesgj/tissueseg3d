# Independent Python-vs-Rust validation

`tools/validate_migration.py` is a small, reproducible oracle check and
benchmark for the native port. It does not import Napari and does not use the
Rust implementation to derive expected values. The supplied input TIFF is the
only dataset it reads; the script never downloads a sample.

## Environment

Use the reference Python environment that contains the pinned packages:

```bash
/tmp/tissueseg3d-reference-env/bin/python -m pip install \
    -r tools/reference-requirements.txt
```

`tools/reference-requirements.txt` pins the exact versions used for the
reported check:

```text
numpy==2.5.4
scipy==1.18.1
scikit-image==0.25.2
pandas==2.3.1
tifffile==2026.10.15
```

Build the executable if needed:

```bash
cargo build --release --locked
```

## Reproducible command

```bash
/tmp/tissueseg3d-reference-env/bin/python tools/validate_migration.py \
    --input /tmp/tissueseg3d-parity/sample.tif \
    --binary target/release/tissueseg3d \
    --output-dir /tmp/tissueseg3d-validation \
    --repeats 3
```

The script exits `0` only when every compared output matches exactly. It exits
non-zero on a Rust failure, a shape difference, a single differing label voxel,
a differing CSV byte, or an input hash change.

For a quick single check, use `--repeats 1`. The three-repeat default is the
benchmark mode.

## What the oracle does

The Python side mirrors the original `main.py` pipeline and defaults:

1. Load the dual-channel `ZCYX` TIFF as `uint16`, with channel 0 as nuclei and
   channel 1 as membrane.
2. Detect candidates with `blob_dog` on the XY-prescaled, z-ratio-resampled
   nucleus channel (`min_sigma=4*0.5`, `max_sigma=12*0.5`, `sigma_ratio=1.6`,
   `overlap=0.5`, `threshold=1*1e-3`, no border exclusion).
3. Convert candidates back to full-resolution coordinates and run three
   membrane-ray-tracing merge rounds (`merge distance=100*0.5`, membrane delta
   `25`, z ratio `6.92`).
4. Gaussian-filter the membrane (`sigma=0.5`), impose minima at the kept seeds,
   run watershed (`compactness=0`), median-filter XY (`size=(1,5,5)`), remove
   labels outside `[2000, 100000]` voxels, remove border labels, fill holes and
   relabel.
5. Measure cells with `regionprops_table` and write the cell TIFF/CSV.
6. Gaussian-filter the nucleus channel (`sigma=(1.5/6.92, 1.5, 1.5)`), threshold
   at `>=500`, median-filter XY (`size=(1,5,5)`), intersect with the **full**
   cell mask, fill holes and write the nucleus TIFF/CSV.

The utility functions (`seedcoords2mask`, `distance_pt2pt`,
`interpolate_3d_line`, `remove_seeds`, `imposemin`, `remove_lbl_size`,
`remove_lbl_edge`, `fill_lbl_holes`, `relabel`) are extracted from the original
`utils.py` with `ast` and executed without importing the Napari module. No
other Python source or fixture is modified.

### Deliberate corrections in the oracle

Only the documented Python bugs are corrected:

- **First seed omission.** In `remove_seeds`, the source is changed from
  `range(1, len(seeds))` to `range(len(seeds))`. The script asserts that the
  original expression occurs exactly once and that the replacement was applied.
- **Nuclei use the full cell mask.** The original Napari display erosion
  (`minimum_filter` on `CellsLbl`) is not applied to the data used for nucleus
  segmentation, measurement or export.
- **Nucleus TIFF contains nuclei.** The original app measured `nucleus_lbl` but
  wrote `cell_lbl` to `_nuc_lbl.tif`; the oracle exports the actual nucleus
  mask, the same mask used for the nucleus CSV.

Everything else uses the original pipeline order and defaults. The script does
not assert general parity for all datasets.

## Comparison rules

The Rust executable is invoked directly with an argument list, never through a
shell, using `--batch`, the input path and a dedicated output directory. The
Python oracle and Rust outputs are written to separate run directories:

```text
<output-dir>/python-results/run-N/
<output-dir>/rust-results/run-N/
```

For each run:

- TIFFs are decoded with `tifffile` and compared by shape and every label
  voxel. A differing dtype alone is not a failure if all label values match.
- CSVs are compared byte-for-byte, including the header and float formatting.
- The input SHA-256 is recorded before and after all runs and must not change.

The script does not compare seed arrays because the batch CLI does not export
them.

## Benchmark method

Python imports and the AST extraction happen before timing. Each Python repeat
times TIFF load, seed detection, cell segmentation, nucleus segmentation,
measurement and TIFF/CSV export. Each Rust repeat times process startup through
batch load, both segmentations, measurement and export. Repeats run
sequentially. The JSON report lists every duration, the median for each
implementation, and `speed_ratio_python_over_rust = python_median /
rust_median`.

The speed ratio is descriptive for the recorded input, machine and build. It is
not a general performance claim.

## Report

`<output-dir>/validation-report.json` contains:

- input path, SHA-256 before/after, preservation flag and shape;
- all parameters and exact package versions;
- Python version, executable, platform and CPU description;
- per-run durations, medians and speed ratio;
- candidate/kept seed counts, cell and nucleus label counts and foreground
  voxel counts;
- exact comparison results for both TIFFs and both CSVs, including mismatch
  counts and the first differing voxel when applicable;
- all Python and Rust output paths.

## Verified result — 2026-11-01

The final release build was compared with the corrected Python oracle on the
public CAAXinjH2B_025 microscopy sample linked in the README. All three repeats
passed every full-volume and CSV comparison. The input hash was unchanged.

| Field | Value |
| --- | --- |
| Input SHA-256 | `c9ac014ebe84e97af410c8aaa05cc93ebf26f6c532043da84c096482e1b9ba15` |
| Input shape | `36 x 2 x 307 x 287` (ZCYX, uint16) |
| Machine | AMD Ryzen 5 7600X, 8 logical CPUs available, Linux x86-64 |
| Build | Rust release, thin LTO, no native-CPU compiler flag |
| Python | 3.13.17; package versions pinned in `tools/reference-requirements.txt` |
| Repeats | `3`, sequential |
| Python pipeline median | `4.493 s` |
| Rust batch median, including startup | `2.629 s` |
| Measured speed ratio | `1.71x` for this sample and machine |
| Cell TIFF | Same shape and labels, `0` differing voxels in every repeat |
| Cell CSV | Byte-identical in every repeat |
| Nucleus TIFF | Same shape and labels, `0` differing voxels in every repeat |
| Nucleus CSV | Byte-identical in every repeat |
| Cell labels / foreground | `121` / `1,893,976` voxels, both implementations |
| Nucleus labels / foreground | `121` / `665,696` voxels, both implementations |

| Repeat | Python pipeline | Rust batch |
| --- | --- | --- |
| 1 | 4.791 s | 2.620 s |
| 2 | 4.493 s | 2.629 s |
| 3 | 4.488 s | 2.646 s |

The local full report is `target/validation/final/validation-report.json`.
This measures one real input and the documented parameters; it does not claim
that every dataset has identical timing or establish universal numerical parity.

## Additional regression checks

The Rust suite has 88 passing tests, covering processing, image IO, exports,
CLI parsing, viewer geometry and editing state. The tests include:

- independent SciPy/scikit-image fixtures for interpolation, Gaussian and
  median filtering, reconstruction, watershed, hole filling, seeds and both
  segmentation pipelines;
- even-sized median windows and adjacent-scale DoG maxima;
- 264 exact Gaussian quantisation and threshold cases, including constant
  values immediately at a threshold;
- bit-identical Gaussian kernel weights for 73 sigma values, using a portable
  translation of the pinned NumPy reference's exponential and reduction order;
- nucleus TIFF export, retaining the first seed, safe dotted filenames,
  Python-style CSV floats, write errors and seed-label shape validation.

The numerical helper is pure Rust and carries the upstream notices in
`THIRD_PARTY_NOTICES.md`. It fixes the reference arithmetic in software, so
Gaussian kernel construction does not depend on the host CPU's vector math
library. The application has no Python or NumPy runtime dependency.

Formatting and `cargo clippy --locked --all-targets -- -D warnings` also pass.
Native platform build and test results are tracked by the repository's Rust
workflow; those results are separate from this Linux reference benchmark.
