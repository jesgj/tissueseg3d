# Migrating from the Python/Napari version

This document explains what changes when moving from the original Python
TissueSeg3D app to the native Rust app. The scientific workflow and parameter
defaults are intended to match. The differences listed here are deliberate and
are limited to packaging, output safety and known Python bugs.

## 1. What changes for users

| | Python version | Rust version |
| --- | --- | --- |
| Runtime | Python 3.10, Napari, scikit-image, SciPy | One native executable |
| Install | Conda environment + `pip install -r requirements.txt` | Download a build or `cargo build --release --locked` |
| GUI | Napari viewer with dock widgets | Native desktop window with XY/XZ/YZ views |
| Run | `python main.py` | `./target/release/tissueseg3d` |
| Batch | Not available | `--batch` and `--batch-demo` |
| Exports | Written next to the input file | Written to a chosen output directory |
| Label editing | Napari layers and two dock widgets | Built-in add/remove seeds, remove and merge labels |
| Seed persistence | Reload `_nuc_lbl.tif` through a checkbox | Load a label TIFF as seeds, or pass seeds as JSON |

The Python app still works and remains in the repository (`main.py`,
`utils.py`, `requirements.txt`). It is useful for comparison and for users who
depend on Napari. It is not required by the Rust app.

## 2. Build and installation

Follow the normal stable Rust installation at <https://rustup.rs>. Linux needs
the native windowing and GTK/GL development packages used by eframe and the
file dialog; the CI workflow installs the Debian/Ubuntu equivalents. Then:

```bash
cargo build --release --locked
./target/release/tissueseg3d --help
```

`Cargo.lock` is committed and the documented commands use `--locked`, so a
build uses the reviewed dependency versions. There are no signing secrets,
accounts or network services needed for a normal build.

The Python environment is unchanged:

```bash
conda create -n tissueseg3d python=3.10
conda activate tissueseg3d
pip install -r requirements.txt
python main.py
```

## 3. Workflow and parameter mapping

The Rust app follows the same order as the Python dock widgets. Defaults are
the Python defaults.

| Python widget and argument | Default | Rust GUI | Batch flag |
| --- | --- | --- | --- |
| `load_image_tiff.imagefile` | | Open TIFF | `--batch <file>` |
| `load_image_tiff.zratio` | `6.92` | Z ratio | `--z-ratio` |
| `load_image_tiff.dualchan` | `True` | Two-channel setting | `--single-channel` disables it |
| `seed_nuclei.nuc_scale_minmax` | `(4, 12)` | Seed min/max sigma | `--seed-min-sigma`, `--seed-max-sigma` |
| `seed_nuclei.nuc_det_thr` | `1` | Seed threshold | `--seed-threshold` |
| `seed_nuclei.nuc_merge_maxdst` | `100` | Merge distance | `--merge-distance` |
| `seed_nuclei.memb_mindelta` | `25` | Membrane delta | `--membrane-delta` |
| `seed_nuclei.load_labels` | `False` | Load seeds from labels | `--seeds <json>` |
| `segment_cells.cell_gaussrad` | `0.5` | Cell sigma | `--cell-sigma` |
| `segment_cells.cell_regrad` | `5` | Cell median window | `--cell-median` |
| `segment_cells.cell_minvol` | `2000` | Minimum volume | `--min-volume` |
| `segment_cells.cell_maxvol` | `100000` | Maximum volume | `--max-volume` |
| `segment_nuclei.nucleus_gaussrad` | `1.5` | Nucleus sigma | `--nucleus-sigma` |
| `segment_nuclei.nucleus_thr` | `500` | Nucleus threshold | `--nucleus-threshold` |
| `segment_nuclei.nucleus_regrad` | `5` | Nucleus median window | `--nucleus-median` |

The shared z ratio is used consistently for seed detection, anisotropic ray
tracing, displayed geometry and nucleus Gaussian smoothing. It is the physical
z step divided by the XY pixel size; for isotropic stacks use `1`.

`--cell-median` and `--nucleus-median` take an XY window size, not a radius:
`5` applies a `5 x 5` median filter slice by slice. The Python widget fields
are named `cell_regrad` and `nucleus_regrad`, but they feed the same window-size
filter, so the defaults map one to one.

The Python workflow's implicit actions become explicit in the Rust app:

- Seed detection only creates candidate and kept seeds. Cell labeling is a
  separate action.
- Nucleus labeling is optional and can be skipped.
- Export is explicit in the GUI; batch mode exports at the end of the run.
- Loading a new stack clears downstream results, as changing a channel or the
  seed set invalidates labels computed from it.

### Batch equivalents

```bash
# Same defaults as the Python widgets
tissueseg3d --batch input.tif --output-dir results

# Membrane only, no nucleus channel and no nucleus segmentation. Automatic
# seed detection needs the nucleus channel, so seeds must be supplied.
tissueseg3d --batch input.tif --output-dir results --single-channel \
  --seeds seeds.json --skip-nuclei

# Reproduce a manual seed set; coordinates are z, y, x voxel indices
tissueseg3d --batch input.tif --output-dir results --seeds seeds.json \
  --skip-nuclei

# Built-in synthetic stack
tissueseg3d --batch-demo --output-dir demo-results
```

A seeds file is a JSON array such as `[[12, 140, 96], [18, 210, 130]]`. The
batch command uses these coordinates instead of running automatic seed
detection. It returns a non-zero status for bad arguments, invalid seeds and
failed reads or writes. Combining `--skip-nuclei` with a seed file is the
normal cells-only workflow: nuclei cannot be segmented without a nucleus
channel, and seeds cannot be detected without one either.

## 4. Outputs

For `sample.tif`, both versions write the same kind of result, but the Rust
version writes into `--output-dir` rather than beside the input:

| Rust output | Python output | Content |
| --- | --- | --- |
| `sample_lbl.tif` | `sample_lbl.tif` | Cell label mask |
| `sample_lbl.csv` | `sample_lbl.csv` | Cell measurements |
| `sample_nuc_lbl.tif` | `sample_nuc_lbl.tif` | Nucleus label mask |
| `sample_nuc_lbl.csv` | `sample_nuc_lbl.csv` | Nucleus measurements |

`sample.tiff` is handled correctly; the Rust exporter derives names from the
file stem instead of assuming the suffix `.tif`.

Both CSV files have exactly this header, preserving the Python column order:

```text
Cell,CZ,CY,CX,Volume (vox)
```

- `Cell`: label ID in the exported mask.
- `CZ`, `CY`, `CX`: centroid in voxel coordinates, in z, y, x order.
- `Volume (vox)`: number of voxels carrying that label.

Nucleus labels reuse the parent cell label ID. Measurements are computed from
the exported mask.

The Rust app does not copy or overwrite the source image. Batch mode requires
an output directory; the GUI asks where to export. The Python app's
`chmod 0o666` on exports is not reproduced.

## 5. Intentional bug fixes

### 5.1 First seed is no longer lost

Python `utils.py` builds the indices of kept seeds as:

```python
idx = set(range(1, len(seeds))) - set(sum(mergelst, []))
```

Index `0` is never in the keep set, so the first candidate seed is discarded on
every call, even when no merge was applied. The Rust implementation keeps every
seed that survives the three membrane-ray-tracing rounds. The cluster
representative is the rounded mean position of the merged candidates, matching
the Python intent.

Effect: seed counts can be higher than the Python result, especially when the
first candidate is a real nucleus. This is a correction, not a parameter
change.

### 5.2 Nucleus TIFF export contains nuclei

Python `segment_nuclei` measures `nucleus_lbl` correctly, but saves
`cell_lbl` to the nucleus TIFF:

```python
nucleus_lbl = cell_lbl * nuclei_thr
nucleus_lbl = fill_lbl_holes(nucleus_lbl)
properties = regionprops_table(nucleus_lbl,
                               properties=['label', 'centroid', 'area'])
...
imsave(..., '_nuc_lbl.tif', np.uint16(cell_lbl), ...)
```

The Python `_nuc_lbl.csv` therefore already describes nuclei, while the Python
`_nuc_lbl.tif` contains whole cells. The Rust implementation exports
`nucleus_lbl` to the TIFF: the same mask that Python measures for its nucleus
CSV.

### 5.3 Nuclei use the full cell mask

Python `segment_cells` exports and measures the full cell label mask, then
erodes that mask for display before storing it in the Napari `CellsLbl` layer:

```python
imsave(..., '_lbl.tif', np.uint16(cell_lbl), ...)
...
cell_lbl = cell_lbl * (cell_lbl == minimum_filter(cell_lbl, size=(1, 3, 3)))
vw.layers["CellsLbl"].data = cell_lbl
```

`segment_nuclei` then reads the `CellsLbl` layer, so Python nucleus
segmentation is restricted by the display erosion. The Rust app keeps that
erosion display-only and uses the complete cell mask for nucleus segmentation,
measurement and export. This is a deliberate correction, so nucleus masks can
include voxels that the display-eroded Python mask excluded.

## 6. Other deliberate differences

These are robustness changes, not changes to the intended segmentation:

- **32-bit labels.** Label arrays are 32-bit internally. Export rejects values
  that cannot be represented safely instead of truncating to 16 bits as Python
  does with `np.uint16`. A 32-bit TIFF is used when the label range requires it.
- **TIFF validation.** Dual-channel input expects interleaved grayscale pages
  with the nucleus channel first and the membrane channel second, in the same
  ZCYX layout as the Python reader. Ordinary single-channel ZYX, ImageJ
  hyperstacks and OME metadata are recognized where the axis layout is
  unambiguous. Ambiguous layouts are rejected rather than silently mixed.
- **File names.** `.tif` and `.tiff` inputs both produce `_lbl` / `_nuc_lbl`
  outputs using the file stem. The source path is never used as an output
  path.
- **Errors.** Failed loads keep the previous GUI data where possible, invalid
  batch input exits non-zero, and unsupported formats produce a clear error.
- **Memory and threading.** Filtering and segmentation run off the UI thread.
  Batch mode avoids opening a window. This does not by itself imply a speedup
  over Python.

## 7. Scientific behavior that is preserved

The Rust core follows the Python pipeline:

1. Load uint16 data and split the channels (nucleus first, membrane second) or
   use a single membrane channel.
2. Resample for nucleus detection with the z ratio and XY prescale `0.5`, using
   first-order interpolation with endpoint alignment and rounding.
3. Detect candidates with multiscale difference-of-Gaussians: sigma ratio
   `1.6`, overlap `0.5`, threshold `nuc_det_thr * 1e-3`, no border exclusion.
4. Convert candidate coordinates back to the full-resolution grid and merge
   nearby candidates when the membrane ray profile is too flat: three rounds,
   anisotropic distances, distance and delta scaled as in Python.
5. Gaussian-filter the membrane with the configured sigma, convert to uint16
   with truncation and preserve-range semantics, impose regional minima at the
   seed markers by erosion reconstruction, and run marker-controlled watershed
   with six-connectivity and FIFO tie handling.
6. Median filter in XY with window size `(1, r, r)`, remove labels outside the
   inclusive volume range, remove labels near the XY edges using the Python
   bounding-box rule, fill enclosed 3D holes while leaving background
   connected to any outer face as background, assign the maximum neighbouring
   label under a `(1, 3, 3)` maximum filter, and compact labels while keeping
   background `0`.
7. For nuclei: Gaussian with sigma `(nucleus_sigma / z_ratio, nucleus_sigma,
   nucleus_sigma)`, threshold with `>=`, median filter in XY, intersect with
   the cell labels and fill enclosed holes.

Seed markers use label `1` for the XY border background; the stack's Z faces
are not treated as borders, matching Python.

## 8. Validation status and claims

- Unit and integration tests cover volume construction and indexing, seed and
  label edge cases, filter/segmentation behavior and export validation.
- Where reference fixtures were generated independently with Python, they were
  compared against the Rust stages. Fixtures and generation/validation scripts
  are part of the repository.
- Native CI has passed the reference tests, release build and headless demo on
  Linux x86-64, Windows x86-64, macOS Apple Silicon and macOS Intel. Linux/macOS
  run 88 tests; Windows runs 87 because a Unix-only write-error test is
  excluded. Interactive GUI rendering has been checked locally on Linux;
  this does not claim manual GUI testing on the other platforms.
- No complete numerical parity claim is made for every possible input. Output
  parity is asserted only for the cases measured by the reference fixtures and
  reported alongside the commands that produced them. The reproducible
  comparison procedure and the recorded result are in
  [VALIDATION.md](VALIDATION.md).
- No speedup claim is made. Performance statements, if any, will include the
  hardware, dataset, build profile and exact command. A benchmark is not
  evidence that all datasets are faster.
- Segmentation masks should be reviewed visually, as with the Python version.

## 9. Known limitations

- Prebuilt binaries are unsigned. macOS Gatekeeper and Windows SmartScreen can
  warn on first launch; see the README for the factual workaround.
- There is no installer, auto-updater or code-signing pipeline yet.
- The GUI is a new implementation, not a Napari plugin host. Napari-specific
  plugins, layer types and Python scripting inside the viewer are not
  available.
- Batch seeds are plain coordinates. Loading an existing label mask as seeds is
  available in the GUI; in batch mode, provide coordinates.
- Large stacks are limited by available memory, as in the Python version.
- The Python version remains the reference implementation for any behavior not
  covered by the Rust tests. If a discrepancy matters for your data, compare
  both outputs and report it with the input characteristics and parameters.

## 10. License

The Rust port keeps the original BSD 3-Clause license. See `LICENSE`.
