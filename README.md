# TissueSeg3D

TissueSeg3D segments cells and nuclei in 3D two-channel fluorescence image
stacks: one membrane channel and one nucleus channel. This branch is a native
Rust replacement for the original Python/Napari application. It provides the
same workflow as a desktop app and as a headless batch command, with no Python
runtime or Napari installation required.

Workflow: load a TIFF stack, detect nucleus seeds, label cells from the
membrane signal, label nuclei inside the cells, review or edit the result, and
export label masks plus a CSV measurement table.

## Build and run

Use a normal stable Rust installation from <https://rustup.rs>. On Linux,
install the native GUI libraries first (Debian/Ubuntu names):

```bash
sudo apt-get install -y libgtk-3-dev libxkbcommon-dev \
  libxcb-render0-dev libxcb-shape0-dev libxcb-xfixes0-dev libgl1-mesa-dev
```

Build and start the desktop app:

```bash
cargo build --release --locked
./target/release/tissueseg3d
```

On Windows the executable is `target\release\tissueseg3d.exe`. On macOS,
building from source avoids the quarantine applied to downloaded binaries.

The published Windows release build links the C runtime statically (MSVC
`+crt-static`), so the archive runs without installing the Visual C++
redistributable. A plain `cargo build --release` is still valid for local use
and uses the toolchain's default runtime linkage. To
build a portable static-runtime binary locally in PowerShell:

```powershell
$env:CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_RUSTFLAGS = "-C target-feature=+crt-static"
cargo build --release --locked --target x86_64-pc-windows-msvc
.\target\x86_64-pc-windows-msvc\release\tissueseg3d.exe --help
```

When release archives are published by the workflow, each platform gets one
archive named after it (`tissueseg3d-<platform>.tar.gz` on Linux and macOS,
`tissueseg3d-<platform>.zip` on Windows). An archive contains the executable,
`LICENSE`, `README.md` and, when the repository provides it,
`THIRD_PARTY_NOTICES`; the tar archives keep the executable permission, so
extract rather than copying the bare binary. The Windows archive needs no
separate runtime installation.

The app accepts:

```bash
./target/release/tissueseg3d                 # normal GUI
./target/release/tissueseg3d --demo          # GUI with a synthetic demo stack
./target/release/tissueseg3d --help          # command-line help
```

### Unsigned builds

Release binaries built by this project are not code-signed or notarized, and
the build does not require any signing secrets. Windows SmartScreen and macOS
Gatekeeper may therefore show a warning for downloaded binaries. On macOS you
can open the app once with right-click > Open, or build it locally with
`cargo build --release`. This is a distribution limitation of unsigned desktop
software, not an indication of a build failure.

## Batch mode

Batch mode runs the same processing and export code without opening a window.
It is useful for scripted processing, parity checks and benchmarking.

```bash
# Automatic seed detection, cell and nucleus segmentation
./target/release/tissueseg3d --batch /data/stack.tif --output-dir results

# Single membrane channel (no nucleus channel in the input). Automatic seed
# detection needs the nucleus channel, so a seed file is required here, and
# nucleus segmentation has to be skipped.
./target/release/tissueseg3d --batch /data/stack.tif --output-dir results \
  --single-channel --seeds seeds.json --skip-nuclei

# Start from seeds stored as a JSON array of [z, y, x] voxel coordinates
./target/release/tissueseg3d --batch /data/stack.tif --output-dir results \
  --seeds seeds.json --skip-nuclei

# Headless run on the built-in synthetic demo
./target/release/tissueseg3d --batch-demo --output-dir demo-results
```

`--single-channel` alone is not enough: without a nucleus channel there is
nothing to detect seeds from and nothing to segment nuclei in, so the app
exits with an error telling you to pass `--seeds` and `--skip-nuclei`.

Input `stack.tif` produces `stack_lbl.tif` and `stack_lbl.csv`, and, when nucleus
segmentation runs, `stack_nuc_lbl.tif` and `stack_nuc_lbl.csv`, in the output
directory. Existing input files are never overwritten. Invalid arguments,
unreadable seeds, unsupported TIFF layouts and failed writes return a non-zero
exit status.

Run `--help` for the complete flag list. The processing flags are the same
parameters exposed by the GUI:

| Purpose | Flag | Default |
| --- | --- | --- |
| Input stack (required in batch mode) | `--batch <TIFF>` | |
| Output directory (required in batch mode) | `--output-dir <DIR>` | |
| Single membrane channel instead of nucleus + membrane | `--single-channel` | two channels |
| Physical z spacing / XY pixel size | `--z-ratio <N>` | `6.92` |
| Seed detection scale range | `--seed-min-sigma`, `--seed-max-sigma` | `4`, `12` |
| Seed detection threshold | `--seed-threshold <N>` | `1` |
| Seed merge distance | `--merge-distance <N>` | `100` |
| Minimum membrane intensity change between seeds | `--membrane-delta <N>` | `25` |
| Seeds to use instead of automatic detection | `--seeds <JSON>` | |
| Cell Gaussian sigma | `--cell-sigma <N>` | `0.5` |
| Cell median filter XY window size | `--cell-median <N>` | `5` |
| Minimum cell volume in voxels (inclusive) | `--min-volume <N>` | `2000` |
| Maximum cell volume in voxels (inclusive) | `--max-volume <N>` | `100000` |
| Nucleus Gaussian sigma | `--nucleus-sigma <N>` | `1.5` |
| Nucleus intensity threshold | `--nucleus-threshold <N>` | `500` |
| Nucleus median filter XY window size | `--nucleus-median <N>` | `5` |
| Skip nucleus segmentation | `--skip-nuclei` | nuclei enabled |

A seeds file is a JSON array of voxel coordinates using z, y, x order, for
example `[[12, 140, 96], [18, 210, 130]]`.

`--cell-median` and `--nucleus-median` are XY window sizes, not radii: `5`
means a `5 x 5` window applied slice by slice. The Python widget names are
`cell_regrad` and `nucleus_regrad`, but they are passed to the same window-size
median filter.

## GUI workflow and Python parameter mapping

The left panel follows the original Napari widgets: load the stack, seed
nuclei, label cells, label nuclei, then edit and export. The central viewer
shows orthogonal XY, XZ and YZ views with the z ratio applied, so the stack is
displayed with its physical proportions.

| Step | Original Python control | Rust app | Batch equivalent |
| --- | --- | --- | --- |
| Load | image file, zratio, two-channel checkbox | Open TIFF, Z ratio, two-channel setting | `--batch`, `--z-ratio`, `--single-channel` |
| Seed nuclei | scale range `(4, 12)` | seed min/max sigma | `--seed-min-sigma`, `--seed-max-sigma` |
| Seed nuclei | detection threshold `1` | seed threshold | `--seed-threshold` |
| Seed nuclei | merge distance `100` | merge distance | `--merge-distance` |
| Seed nuclei | membrane minimum delta `25` | membrane delta | `--membrane-delta` |
| Seed nuclei | load labels checkbox | Load seeds from label TIFF | `--seeds` |
| Label cells | Gaussian radius `0.5` | cell sigma | `--cell-sigma` |
| Label cells | region radius `5` | cell median window | `--cell-median` |
| Label cells | min/max volume `2000` / `100000` | min/max volume | `--min-volume`, `--max-volume` |
| Label nuclei | Gaussian radius `1.5` | nucleus sigma | `--nucleus-sigma` |
| Label nuclei | threshold `500` | nucleus threshold | `--nucleus-threshold` |
| Label nuclei | region radius `5` | nucleus median window | `--nucleus-median` |
| Edit | Remove Label, Merge Labels | remove/merge selected labels | GUI only |
| Export | implicit on Label Cells / Label Nuclei | explicit Export | automatic in batch |

Seeds can be added or removed interactively before cell segmentation; the
membrane-only workflow (single channel, manual seeds, cells only) is supported.
Cell edits are applied to the cell and nucleus masks together, as in Python.

## Outputs

For an input named `sample.tif` (or `sample.tiff`), the output directory
contains:

| File | Content |
| --- | --- |
| `sample_lbl.tif` | Cell label mask; background is `0`, each cell has a positive label |
| `sample_lbl.csv` | Cell measurements |
| `sample_nuc_lbl.tif` | Nucleus label mask, using the corresponding cell label IDs |
| `sample_nuc_lbl.csv` | Nucleus measurements |

Both CSV files use exactly this header:

```text
Cell,CZ,CY,CX,Volume (vox)
```

`Cell` is the label ID, `CZ`, `CY`, `CX` are centroid coordinates in voxels,
and `Volume (vox)` is the number of voxels with that label. Measurements are
taken from the exported label mask, not from the on-screen overlay.

Label masks are 32-bit integer data internally. The exporter rejects label
values that cannot be stored safely instead of silently truncating them, and
uses a 32-bit TIFF when values exceed the 16-bit range. The original Python
version casts to `uint16`, which can wrap large label counts.

### Display masks versus saved masks

The viewer may draw label boundaries or partially eroded labels so that
individual cells remain visible. That display decoration is not the
segmentation result. The exported masks are the complete, unmodified label
volumes. Nucleus segmentation also uses the complete cell mask, never the
display version.

## Intentional corrections to the Python version

The Rust implementation preserves the scientific pipeline but fixes three
known Python bugs:

1. **First seed loss.** The Python `remove_seeds` function starts its keep-set
   at index 1, so candidate seed 0 is always discarded. The Rust version keeps
   every seed that survives membrane ray tracing.
2. **Nucleus TIFF export.** The Python `segment_nuclei` function measures
   `nucleus_lbl` correctly but writes `cell_lbl` to `_nuc_lbl.tif`. The Rust
   version exports the actual nucleus mask, the same mask that Python measures
   for its nucleus CSV.
3. **Nuclei use the full cell mask.** Python erodes the `CellsLbl` layer for
   display and then feeds that eroded display mask into nucleus segmentation.
   The Rust app keeps the erosion display-only and segments nuclei from the
   complete cell mask.

Additional deliberate robustness changes:

- Export names are derived with proper file-stem handling (`.tif` and `.tiff`)
  and output goes to a chosen output directory; the source image is never
  overwritten.
- Ambiguous or unsupported TIFF channel/axis layouts are rejected rather than
  silently interpreted as the wrong channel.
- The Python code makes exported files world-writable on POSIX systems; the
  Rust version uses normal file permissions.

## Scientific behavior

The processing follows the Python pipeline: uint16 data, z-ratio-aware
geometry, half-resolution nucleus seed detection with a multiscale
difference-of-Gaussians (sigma ratio 1.6, overlap 0.5, threshold scaled by
1e-3), three rounds of anisotropic membrane ray tracing, Gaussian filtering,
erosion reconstruction to impose regional minima, marker-controlled watershed,
median filtering in XY, inclusive volume filtering, XY-edge exclusion, filling
of enclosed 3D holes and compact relabeling. Nuclei are thresholded with `>=`,
median filtered in XY, masked by the cell labels and hole-filled.

## Verification status

The test suite covers the volume primitives, filtering and segmentation
behavior, IO validation and export. Where independent reference data is used,
it was generated with the Python implementation and compared stage by stage.

Native CI has passed the reference tests, release build and headless demo on
Linux x86-64, Windows x86-64, macOS Apple Silicon and macOS Intel. There are 88
passing tests on Linux/macOS and 87 on Windows, where a Unix-only write-error
test is excluded. Interactive GUI rendering has been checked locally on Linux;
the other systems have native build and batch-runtime coverage.

See the [Rust workflow](https://github.com/jesgj/tissueseg3d/actions/workflows/rust.yml)
for per-commit build results and downloadable archives.

The reproducible Python-vs-Rust comparison is documented in
[docs/VALIDATION.md](docs/VALIDATION.md): it lists the reference environment,
the exact command, the comparison rules and the recorded single-input result.

Parity and performance are only claimed where measured. No general speedup is
asserted, and no benchmark numbers are promised without stating the hardware,
dataset and command used. See [docs/MIGRATION.md](docs/MIGRATION.md) for the
migration details and known limitations.

## Original Python version

The original Python/Napari implementation remains in this repository as
`main.py`, `utils.py` and `requirements.txt`. Its setup is unchanged:

```bash
conda create -n tissueseg3d python=3.10
conda activate tissueseg3d
pip install -r requirements.txt
python main.py
```

It requires Napari and a display. It writes labels next to the input image and
retains the known bugs listed above. Use the Rust app for new work.

User manual and sample images are linked from the original project:
[User Manual](https://drive.google.com/file/d/1T3oVBuXkCHBVpNVjMfHAGD4-xAkIjbwq/view?usp=drive_link),
[CAAXinjH2B_025 sample](https://drive.google.com/file/d/12rfaq3lCsqYb4iTMKZ8yaRM1NiDgfDTX/view?usp=drive_link).

## License

BSD 3-Clause. See [LICENSE](LICENSE).
