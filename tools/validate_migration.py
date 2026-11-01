#!/usr/bin/env python3
"""Independent Python-vs-Rust validation and benchmark for TissueSeg3D.

The oracle is the original Python algorithm from ``utils.py`` and ``main.py``.
Utility functions are extracted with :mod:`ast` so Napari is never imported.
Only the documented Python seed-index omission is corrected.  The pipeline
mirrors the original dock-widget order and exports the actual nucleus mask
(the original Python app measured the nucleus mask but wrote the cell mask).

The Rust executable is run as a child process with an argument list; no path is
ever passed through a shell.  The input file is only read, and the input hash
is checked again after the runs.

Example:
    /tmp/tissueseg3d-reference-env/bin/python tools/validate_migration.py \
        --input sample.tif \
        --binary target/release/tissueseg3d \
        --output-dir /tmp/tissueseg3d-validation \
        --repeats 3
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy.ndimage import binary_fill_holes, maximum_filter, median_filter, zoom
from skimage.feature import blob_dog
from skimage.filters import gaussian
from skimage.measure import label, regionprops, regionprops_table
from skimage.morphology import reconstruction
from skimage.segmentation import watershed

REPO_ROOT = Path(__file__).resolve().parent.parent
UTILS_PATH = REPO_ROOT / "utils.py"

# Defaults from the original main.py widget signatures.
DEFAULTS = {
    "z_ratio": 6.92,
    "prescale": 0.5,
    "nuc_scale_minmax": (4, 12),
    "nuc_det_thr": 1.0,
    "nuc_merge_maxdst": 100.0,
    "memb_mindelta": 25.0,
    "cell_gaussrad": 0.5,
    "cell_regrad": 5,
    "cell_minvol": 2000,
    "cell_maxvol": 100000,
    "nucleus_gaussrad": 1.5,
    "nucleus_thr": 500,
    "nucleus_regrad": 5,
    "load_labels": False,
    "dual_channel": True,
}

UTILITY_NAMES = {
    "seedcoords2mask",
    "distance_pt2pt",
    "interpolate_3d_line",
    "remove_seeds",
    "imposemin",
    "remove_lbl_size",
    "remove_lbl_edge",
    "fill_lbl_holes",
    "relabel",
}


def extract_utils(utils_path: Path = UTILS_PATH) -> dict:
    """Extract the pure utility functions from utils.py without importing it.

    The only source correction is the documented first-index omission in
    ``remove_seeds``.  The replacement is asserted so a future edit to
    ``utils.py`` cannot silently make the oracle wrong.
    """
    tree = ast.parse(utils_path.read_text(encoding="utf-8"))
    namespace = {
        "np": np,
        "label": label,
        "regionprops": regionprops,
        "binary_fill_holes": binary_fill_holes,
        "maximum_filter": maximum_filter,
        "reconstruction": reconstruction,
    }
    found = set()
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or node.name not in UTILITY_NAMES:
            continue
        source = ast.unparse(node)
        if node.name == "remove_seeds":
            original = "range(1, len(seeds))"
            corrected = "range(len(seeds))"
            assert source.count(original) == 1, (
                "expected exactly one first-index omission in utils.remove_seeds"
            )
            source = source.replace(original, corrected)
            assert original not in source, "remove_seeds correction was not applied"
            assert corrected in source, "remove_seeds correction produced unexpected source"
        exec(compile(source, str(utils_path), "exec"), namespace)
        found.add(node.name)
    missing = UTILITY_NAMES - found
    if missing:
        raise RuntimeError(f"utils.py is missing expected functions: {sorted(missing)}")
    return namespace


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cpu_description() -> str:
    model = None
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break
    count = os.cpu_count() or 0
    if model:
        return f"{model} ({count} logical CPUs)"
    return f"{platform.processor() or platform.machine()} ({count} logical CPUs)"


def package_versions() -> dict:
    versions = {}
    for distribution in ("numpy", "scipy", "scikit-image", "pandas", "tifffile"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def detect_seeds(nuclei: np.ndarray, membrane: np.ndarray, utils: dict, params: dict):
    """Mirror seed_nuclei from main.py, including three ray-tracing rounds."""
    zratio = params["z_ratio"]
    prescale = params["prescale"]
    blobs = blob_dog(
        zoom(nuclei, (zratio * prescale, prescale, prescale), order=1),
        min_sigma=params["nuc_scale_minmax"][0] * prescale,
        max_sigma=params["nuc_scale_minmax"][1] * prescale,
        sigma_ratio=1.6,
        overlap=0.5,
        threshold=params["nuc_det_thr"] * 1e-3,
        exclude_border=False,
    )
    coords = [
        (
            int(blob[0] / (zratio * prescale)),
            int(blob[1] / prescale),
            int(blob[2] / prescale),
        )
        for blob in blobs
    ]
    kept = coords
    for _ in range(3):
        kept = utils["remove_seeds"](
            membrane,
            kept,
            params["nuc_merge_maxdst"] * prescale,
            params["memb_mindelta"],
            zratio,
        )
    return coords, kept


def segment_cells(membrane: np.ndarray, coords, utils: dict, params: dict) -> np.ndarray:
    """Mirror segment_cells from main.py, up to the exported label mask."""
    sigma = params["cell_gaussrad"]
    membrane_flt = gaussian(
        membrane.astype(float),
        sigma=(sigma, sigma, sigma),
        preserve_range=True,
    ).astype("uint16")
    seeds = utils["seedcoords2mask"](coords, membrane.shape)
    membrane_imp = utils["imposemin"](membrane_flt, seeds > 0)
    cell_lbl = watershed(membrane_imp, seeds, compactness=0)
    cell_lbl = median_filter(cell_lbl, size=(1, params["cell_regrad"], params["cell_regrad"]))
    cell_lbl = utils["remove_lbl_size"](
        cell_lbl, params["cell_minvol"], params["cell_maxvol"]
    )
    cell_lbl = utils["remove_lbl_edge"](cell_lbl)
    cell_lbl = utils["fill_lbl_holes"](cell_lbl)
    cell_lbl = utils["relabel"](cell_lbl)
    return cell_lbl


def segment_nuclei(
    nuclei: np.ndarray, cell_lbl: np.ndarray, utils: dict, params: dict
) -> np.ndarray:
    """Mirror segment_nuclei from main.py using the complete cell mask.

    The original app's display erosion is intentionally not applied here: it
    changed the Napari layer, not the exported cell mask.
    """
    sigma = params["nucleus_gaussrad"]
    zratio = params["z_ratio"]
    nuclei_thr = gaussian(
        nuclei.astype(float),
        sigma=(sigma / zratio, sigma, sigma),
        preserve_range=True,
    ).astype("uint16") >= params["nucleus_thr"]
    nuclei_thr = median_filter(
        nuclei_thr, size=(1, params["nucleus_regrad"], params["nucleus_regrad"])
    )
    nucleus_lbl = cell_lbl * nuclei_thr
    return utils["fill_lbl_holes"](nucleus_lbl)


def measure_labels(mask: np.ndarray) -> pd.DataFrame:
    properties = regionprops_table(mask, properties=["label", "centroid", "area"])
    frame = pd.DataFrame(properties)
    frame.columns = ["Cell", "CZ", "CY", "CX", "Volume (vox)"]
    return frame


def export_label_mask(mask: np.ndarray, output_dir: Path, stem: str, nuclei: bool):
    suffix = "_nuc_lbl" if nuclei else "_lbl"
    tif_path = output_dir / f"{stem}{suffix}.tif"
    csv_path = output_dir / f"{stem}{suffix}.csv"
    maximum = int(mask.max()) if mask.size else 0
    dtype = np.uint16 if maximum <= np.iinfo(np.uint16).max else np.uint32
    tifffile.imwrite(tif_path, mask.astype(dtype), photometric="minisblack")
    measure_labels(mask).to_csv(csv_path, index=False)
    return tif_path, csv_path


def mask_counts(mask: np.ndarray) -> dict:
    labels = np.unique(mask)
    return {
        "labels": int(np.count_nonzero(labels)),
        "foreground": int(np.count_nonzero(mask)),
        "max_label": int(mask.max()) if mask.size else 0,
    }


def run_python_pipeline(input_path: Path, output_dir: Path, utils: dict, params: dict) -> dict:
    """Run the independent Python oracle and time load through export."""
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    image = tifffile.imread(input_path)
    image = image.astype(np.uint16)
    if image.ndim != 4 or image.shape[1] < 2:
        raise RuntimeError(
            f"expected a dual-channel ZCYX TIFF, got shape {image.shape}"
        )
    nuclei = image[:, 0]
    membrane = image[:, 1]
    coords, kept = detect_seeds(nuclei, membrane, utils, params)
    cells = segment_cells(membrane, kept, utils, params)
    cell_tif, cell_csv = export_label_mask(cells, output_dir, input_path.stem, nuclei=False)
    nuclei_mask = segment_nuclei(nuclei, cells, utils, params)
    nucleus_tif, nucleus_csv = export_label_mask(
        nuclei_mask, output_dir, input_path.stem, nuclei=True
    )

    elapsed = time.perf_counter() - started
    return {
        "seconds": elapsed,
        "shape": list(image.shape),
        "outputs": {
            "cells_tif": str(cell_tif),
            "cells_csv": str(cell_csv),
            "nuclei_tif": str(nucleus_tif),
            "nuclei_csv": str(nucleus_csv),
        },
        "counts": {
            "candidate_seeds": len(coords),
            "kept_seeds": len(kept),
            "cells": mask_counts(cells),
            "nuclei": mask_counts(nuclei_mask),
        },
    }


def rust_argv(binary: Path, input_path: Path, output_dir: Path, params: dict) -> list[str]:
    """Build an argument list with the Python defaults made explicit."""
    min_sigma, max_sigma = params["nuc_scale_minmax"]
    return [
        str(binary),
        "--batch",
        str(input_path),
        "--output-dir",
        str(output_dir),
        "--z-ratio",
        str(params["z_ratio"]),
        "--seed-min-sigma",
        str(min_sigma),
        "--seed-max-sigma",
        str(max_sigma),
        "--seed-threshold",
        str(params["nuc_det_thr"]),
        "--merge-distance",
        str(params["nuc_merge_maxdst"]),
        "--membrane-delta",
        str(params["memb_mindelta"]),
        "--cell-sigma",
        str(params["cell_gaussrad"]),
        "--cell-median",
        str(params["cell_regrad"]),
        "--min-volume",
        str(params["cell_minvol"]),
        "--max-volume",
        str(params["cell_maxvol"]),
        "--nucleus-sigma",
        str(params["nucleus_gaussrad"]),
        "--nucleus-threshold",
        str(params["nucleus_thr"]),
        "--nucleus-median",
        str(params["nucleus_regrad"]),
    ]


def run_rust_binary(
    binary: Path, input_path: Path, output_dir: Path, params: dict
) -> dict:
    """Run the Rust batch executable and time process startup through export."""
    output_dir.mkdir(parents=True, exist_ok=True)
    argv = rust_argv(binary, input_path, output_dir, params)
    started = time.perf_counter()
    completed = subprocess.run(argv, capture_output=True, text=True, check=False)
    elapsed = time.perf_counter() - started
    stem = input_path.stem
    outputs = {
        "cells_tif": str(output_dir / f"{stem}_lbl.tif"),
        "cells_csv": str(output_dir / f"{stem}_lbl.csv"),
        "nuclei_tif": str(output_dir / f"{stem}_nuc_lbl.tif"),
        "nuclei_csv": str(output_dir / f"{stem}_nuc_lbl.csv"),
    }
    return {
        "seconds": elapsed,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "outputs": outputs,
        "counts": {
            "cells": mask_counts(tifffile.imread(outputs["cells_tif"]))
            if completed.returncode == 0
            else None,
            "nuclei": mask_counts(tifffile.imread(outputs["nuclei_tif"]))
            if completed.returncode == 0
            else None,
        },
    }


def compare_tiff_arrays(expected_path: Path, actual_path: Path) -> dict:
    """Compare decoded TIFF shape and every label voxel."""
    expected = tifffile.imread(expected_path)
    actual = tifffile.imread(actual_path)
    result = {
        "expected_path": str(expected_path),
        "actual_path": str(actual_path),
        "expected_shape": list(expected.shape),
        "actual_shape": list(actual.shape),
        "expected_dtype": str(expected.dtype),
        "actual_dtype": str(actual.dtype),
        "shape_match": expected.shape == actual.shape,
        "values_match": False,
        "exact": False,
        "mismatch_count": None,
        "first_mismatch_index": None,
        "first_mismatch_expected": None,
        "first_mismatch_actual": None,
    }
    if expected.shape != actual.shape:
        return result
    mismatch = expected != actual
    count = int(np.count_nonzero(mismatch))
    result["values_match"] = count == 0
    result["exact"] = count == 0
    result["mismatch_count"] = count
    if count:
        index = tuple(int(v) for v in np.argwhere(mismatch)[0])
        result["first_mismatch_index"] = list(index)
        result["first_mismatch_expected"] = int(expected[index])
        result["first_mismatch_actual"] = int(actual[index])
    return result


def compare_csv_bytes(expected_path: Path, actual_path: Path) -> dict:
    """Require exact CSV byte identity."""
    expected = expected_path.read_bytes()
    actual = actual_path.read_bytes()
    result = {
        "expected_path": str(expected_path),
        "actual_path": str(actual_path),
        "expected_bytes": len(expected),
        "actual_bytes": len(actual),
        "expected_sha256": hashlib.sha256(expected).hexdigest(),
        "actual_sha256": hashlib.sha256(actual).hexdigest(),
        "exact": expected == actual,
        "first_difference_offset": None,
    }
    if expected != actual:
        limit = min(len(expected), len(actual))
        offset = next((i for i in range(limit) if expected[i] != actual[i]), limit)
        result["first_difference_offset"] = offset
    return result


def compare_run(python_run: dict, rust_run: dict) -> dict:
    cells_tif = compare_tiff_arrays(
        Path(python_run["outputs"]["cells_tif"]), Path(rust_run["outputs"]["cells_tif"])
    )
    cells_csv = compare_csv_bytes(
        Path(python_run["outputs"]["cells_csv"]), Path(rust_run["outputs"]["cells_csv"])
    )
    nuclei_tif = compare_tiff_arrays(
        Path(python_run["outputs"]["nuclei_tif"]), Path(rust_run["outputs"]["nuclei_tif"])
    )
    nuclei_csv = compare_csv_bytes(
        Path(python_run["outputs"]["nuclei_csv"]), Path(rust_run["outputs"]["nuclei_csv"])
    )
    exact = all(
        item["exact"] for item in (cells_tif, cells_csv, nuclei_tif, nuclei_csv)
    )
    return {
        "exact": exact,
        "cells_tif": cells_tif,
        "cells_csv": cells_csv,
        "nuclei_tif": nuclei_tif,
        "nuclei_csv": nuclei_csv,
    }


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Independently validate and benchmark the Rust TissueSeg3D port."
    )
    parser.add_argument("--input", required=True, type=Path, help="dual-channel ZCYX TIFF")
    parser.add_argument("--binary", required=True, type=Path, help="Rust executable")
    parser.add_argument("--output-dir", required=True, type=Path, help="dedicated output directory")
    parser.add_argument(
        "--repeats", type=int, default=3, help="benchmark repeats (default 3)"
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    input_path = args.input.resolve()
    binary = args.binary.resolve()
    output_dir = args.output_dir.resolve()

    if args.repeats < 1:
        print("--repeats must be at least 1", file=sys.stderr)
        return 2
    if not input_path.is_file():
        print(f"input file not found: {input_path}", file=sys.stderr)
        return 2
    if not binary.is_file():
        print(f"Rust executable not found: {binary}", file=sys.stderr)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    python_root = output_dir / "python-results"
    rust_root = output_dir / "rust-results"

    input_hash_before = sha256_file(input_path)
    utils = extract_utils()
    params = dict(DEFAULTS)
    params["nuc_scale_minmax"] = list(params["nuc_scale_minmax"])

    python_runs = []
    rust_runs = []
    comparisons = []
    for repeat in range(1, args.repeats + 1):
        python_run = run_python_pipeline(
            input_path, python_root / f"run-{repeat}", utils, params
        )
        python_runs.append(python_run)
        rust_run = run_rust_binary(
            binary, input_path, rust_root / f"run-{repeat}", params
        )
        rust_runs.append(rust_run)
        print(
            f"repeat {repeat}: python {python_run['seconds']:.3f} s, "
            f"rust {rust_run['seconds']:.3f} s",
            flush=True,
        )
        if rust_run["returncode"] != 0:
            print(
                f"Rust batch failed with exit code {rust_run['returncode']}\n"
                f"stdout:\n{rust_run['stdout']}\nstderr:\n{rust_run['stderr']}",
                file=sys.stderr,
            )
            comparisons.append({"repeat": repeat, "exact": False, "rust_failed": True})
            continue
        comparisons.append({"repeat": repeat, **compare_run(python_run, rust_run)})

    input_hash_after = sha256_file(input_path)
    input_preserved = input_hash_before == input_hash_after

    python_times = [run["seconds"] for run in python_runs]
    rust_times = [run["seconds"] for run in rust_runs]
    python_median = statistics.median(python_times)
    rust_median = statistics.median(rust_times)
    speed_ratio = python_median / rust_median if rust_median > 0 else None

    exact = (
        input_preserved
        and len(comparisons) == args.repeats
        and all(item.get("exact", False) for item in comparisons)
    )
    first_python = python_runs[0]
    first_rust = rust_runs[0]

    report = {
        "input": {
            "path": str(input_path),
            "sha256": input_hash_before,
            "sha256_after": input_hash_after,
            "preserved": input_preserved,
            "shape": first_python["shape"],
        },
        "params": params,
        "package_versions": package_versions(),
        "environment": {
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "cpu": cpu_description(),
            "rust_binary": str(binary),
        },
        "benchmark": {
            "repeats": args.repeats,
            "python_seconds": python_times,
            "rust_seconds": rust_times,
            "python_median_seconds": python_median,
            "rust_median_seconds": rust_median,
            "speed_ratio_python_over_rust": speed_ratio,
            "note": (
                "Descriptive timing for this input, machine and build only; "
                "it is not a general speed claim."
            ),
        },
        "counts": {
            "python": {
                "candidate_seeds": first_python["counts"]["candidate_seeds"],
                "kept_seeds": first_python["counts"]["kept_seeds"],
                "cells": first_python["counts"]["cells"],
                "nuclei": first_python["counts"]["nuclei"],
            },
            "rust": first_rust["counts"],
        },
        "comparisons": comparisons,
        "exact": exact,
        "output_paths": {
            "python_root": str(python_root),
            "rust_root": str(rust_root),
            "python_runs": [run["outputs"] for run in python_runs],
            "rust_runs": [run["outputs"] for run in rust_runs],
        },
        "corrections": {
            "remove_seeds_first_index": "range(1, len(seeds)) -> range(len(seeds))",
            "nuclei_use_full_cell_mask": True,
            "nuclei_export_actual_mask": True,
        },
    }

    report_path = output_dir / "validation-report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"exact comparison: {'PASS' if exact else 'FAIL'}")
    print(f"report: {report_path}")
    if not exact:
        for item in comparisons:
            if item.get("exact"):
                continue
            for name in ("cells_tif", "cells_csv", "nuclei_tif", "nuclei_csv"):
                detail = item.get(name)
                if detail and not detail["exact"]:
                    print(f"  repeat {item['repeat']} {name}: mismatch", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
