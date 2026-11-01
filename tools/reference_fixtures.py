"""Pure NumPy/SciPy/scikit-image reference implementation.

This mirrors main.py + utils.py but avoids importing napari.  It is used only
to generate and validate committed fixtures; the Rust test suite never needs a
Python runtime.

Known Python bugs corrected here (and in the Rust core):
  * remove_seeds always dropped seed index 0 (even when it was not merged) and
    divided by len(seeds) when reporting the keep ratio.
  * segment_nuclei exported cell_lbl instead of nucleus_lbl to *_nuc_lbl.tif.
"""
from __future__ import annotations

import json
import math
import struct

import numpy as np
from scipy.ndimage import zoom, median_filter, maximum_filter, binary_fill_holes, center_of_mass
from skimage.feature import blob_dog, peak_local_max
from skimage.filters import gaussian
from skimage.segmentation import watershed
from skimage.morphology import reconstruction
from skimage.measure import label, regionprops


# ---------------------------------------------------------------- utils.py
def seedcoords2mask(coords, shape):
    seeds = np.zeros(shape, dtype=np.uint32)
    seeds[:, 0, :] = 1
    seeds[:, :, 0] = 1
    seeds[:, -1, :] = 1
    seeds[:, :, -1] = 1
    for i, coord in enumerate(coords):
        seeds[tuple(coord)] = i + 2
    return seeds


def distance_pt2pt(pt1, pt2, zratio):
    dst = np.sqrt(((np.array(pt1) * np.array((zratio, 1, 1)) - np.array(pt2) * np.array((zratio, 1, 1))) ** 2).sum())
    return dst


def interpolate_3d_line(start, end):
    start = np.array(start)
    end = np.array(end)
    vector = end - start
    num_steps = int(np.ceil(np.linalg.norm(vector)))
    if num_steps == 0:
        return np.round(start)[None, :].astype(int)
    t = np.linspace(0, 1, num_steps)
    points = start[np.newaxis, :] + t[:, np.newaxis] * vector[np.newaxis, :]
    return np.round(points).astype(int)


def remove_seeds(img, seeds, dstthr, deltamin, zratio, correct_seed_zero=True):
    mergelst = [[] for _ in range(len(seeds))]
    for i, seed1 in enumerate(seeds):
        for j, seed2 in enumerate(seeds[i + 1:], start=i + 1):
            if distance_pt2pt(seed1, seed2, zratio) < dstthr:
                profile = np.array([img[tuple(point)] for point in interpolate_3d_line(seed1, seed2)])
                delta = profile.max() - profile.min()
                if delta < deltamin:
                    mergelst[i].append(j)

    merged_away = set(sum(mergelst, []))
    if correct_seed_zero:
        idx = sorted(set(range(len(seeds))) - merged_away)
    else:
        idx = sorted(set(range(1, len(seeds))) - merged_away)

    seeds = [tuple(np.round(np.mean(np.array(seeds)[[i] + lst], axis=0)).astype(int))
             for i, lst in enumerate(mergelst)]

    return [seeds[i] for i in idx]


def imposemin(img, mask):
    marker = np.full(img.shape, np.inf)
    marker[mask == 1] = 0
    mask = np.minimum((img + 1), marker)
    return reconstruction(marker, mask, method='erosion')


def imposemax(img, mask):
    marker = np.full(img.shape, -np.inf)
    marker[mask == 1] = 0
    mask = np.maximum((img + 1), marker)
    return reconstruction(marker, mask, method='dilation')


def remove_lbl_size(lbl, minvol, maxvol):
    labels, counts = np.unique(lbl, return_counts=True)
    mask = (counts >= minvol) & (counts <= maxvol)
    mapping = np.zeros(labels.max() + 1, dtype=lbl.dtype)
    mapping[labels[mask]] = labels[mask]
    return mapping[lbl]


def remove_lbl_edge(lbl):
    regions = regionprops(lbl)
    for region in regions:
        bbox = region.bbox
        if bbox[1] <= 1 or bbox[4] >= (lbl.shape[1] - 2) or bbox[2] <= 1 or bbox[5] >= (lbl.shape[2] - 2):
            x, y, z = zip(*region.coords)
            lbl[x, y, z] = 0
    return lbl


def fill_lbl_holes(lbl):
    lbl_holes = label(binary_fill_holes(lbl > 0) ^ (lbl > 0))
    regions = regionprops(lbl_holes, intensity_image=maximum_filter(lbl, size=(1, 3, 3)))
    for region in regions:
        reglbl = region.intensity_max
        x, y, z = zip(*region.coords)
        lbl[x, y, z] = reglbl
    return lbl


def relabel(lbl):
    unique_labels, inverse = np.unique(lbl, return_inverse=True)
    new_labels = np.arange(len(unique_labels))
    new_labels[unique_labels == 0] = 0
    return new_labels[inverse].reshape(lbl.shape)


# ---------------------------------------------------------------- main.py steps
def detect_seeds(nuclei, membrane, zratio=6.92, scale=(4, 12), threshold=1.0,
                 merge_distance=100, membrane_delta=25):
    prescale = 0.5
    blobs = blob_dog(zoom(nuclei, (zratio * prescale, prescale, prescale), order=1),
                     min_sigma=scale[0] * prescale, max_sigma=scale[1] * prescale,
                     sigma_ratio=1.6, overlap=0.5, threshold=threshold * 1e-3,
                     exclude_border=False)
    coords = [(int(blob[0] / (zratio * prescale)), int(blob[1] / prescale), int(blob[2] / prescale))
              for blob in blobs]
    kept = remove_seeds(membrane, coords, merge_distance * prescale, membrane_delta, zratio)
    kept = remove_seeds(membrane, kept, merge_distance * prescale, membrane_delta, zratio)
    kept = remove_seeds(membrane, kept, merge_distance * prescale, membrane_delta, zratio)
    return coords, kept


def segment_cells(membrane, coords, gaussrad=0.5, regrad=5, minvol=2000, maxvol=100000):
    membrane_flt = gaussian(membrane.astype(float), sigma=(gaussrad, gaussrad, gaussrad),
                            preserve_range=True).astype('uint16')
    seeds = seedcoords2mask(coords, membrane.shape)
    membrane_imp = imposemin(membrane_flt, seeds > 0)
    cell_lbl = watershed(membrane_imp, seeds, compactness=0)
    cell_lbl = median_filter(cell_lbl, size=(1, regrad, regrad))
    cell_lbl = remove_lbl_size(cell_lbl, minvol, maxvol)
    cell_lbl = remove_lbl_edge(cell_lbl)
    cell_lbl = fill_lbl_holes(cell_lbl)
    cell_lbl = relabel(cell_lbl)
    return cell_lbl


def segment_nuclei(nuclei, cell_lbl, gaussrad=1.5, threshold=500, regrad=5, zratio=6.92):
    nuclei_thr = gaussian(nuclei.astype(float), sigma=(gaussrad / zratio, gaussrad, gaussrad),
                          preserve_range=True).astype('uint16') >= threshold
    nuclei_thr = median_filter(nuclei_thr, size=(1, regrad, regrad))
    nucleus_lbl = cell_lbl * nuclei_thr
    nucleus_lbl = fill_lbl_holes(nucleus_lbl)
    return nucleus_lbl


# ---------------------------------------------------------------- fixtures
MAGIC = b'TS3D'
DTYPES = {'u16': (b'u', '<u2'), 'u32': (b'U', '<u4'), 'f64': (b'f', '<f8'), 'u8': (b'b', '<u1'), 'i64': (b'i', '<i8')}


def write_fixture(path, arr, kind):
    code, dt = DTYPES[kind]
    arr = np.ascontiguousarray(arr.astype(dt))
    if arr.ndim > 3:
        raise ValueError('fixture storage supports <=3D')
    with open(path, 'wb') as f:
        f.write(MAGIC)
        f.write(b'\x01')
        f.write(code)
        f.write(bytes([arr.ndim]))
        for dim in arr.shape:
            f.write(struct.pack('<Q', dim))
        f.write(arr.tobytes(order='C'))



# ---------------------------------------------------------------- TIFF fixtures
def write_tiff_fixtures(out):
    """Write the small TIFF metadata/encoding fixtures used by tests/io.rs.

    Plain stacks are written as ordinary multi-page grayscale TIFFs.  Hyperstack
    fixtures use the native ImageJ, OME and tifffile-JSON conventions so the Rust
    loader is exercised against real-world metadata rather than ad-hoc tags.
    """
    import tifffile

    # Ordinary single-channel ZYX stack, no hyperstack metadata.
    zyx = (np.arange(4 * 6 * 7, dtype=np.uint32) * 311).astype(np.uint16).reshape(4, 6, 7)
    tifffile.imwrite(out / 'single_zyx.tif', zyx, photometric='minisblack', metadata=None)

    u8 = (np.arange(4 * 6 * 7, dtype=np.uint16) % 251).astype(np.uint8).reshape(4, 6, 7)
    tifffile.imwrite(out / 'single_u8.tif', u8, photometric='minisblack', metadata=None)

    labels16 = np.zeros((3, 4, 5), dtype=np.uint16)
    labels16[0, :2, :2] = 2
    labels16[1, 1:3, 1:4] = 5
    labels16[2, 3, 4] = 9
    tifffile.imwrite(out / 'labels_u16.tif', labels16, photometric='minisblack', metadata=None)

    labels32 = np.zeros((2, 3, 4), dtype=np.uint32)
    labels32[0, 0, 0] = 70000
    labels32[1, 2, 3] = 70001
    tifffile.imwrite(out / 'labels_u32.tif', labels32, photometric='minisblack', metadata=None)

    float_img = (np.arange(4 * 6 * 7, dtype=np.float32) * 0.5 + 3.25).reshape(4, 6, 7)
    tifffile.imwrite(out / 'float_image.tif', float_img, photometric='minisblack', metadata=None)

    # Dual-channel ZCYX fixture.  Channel 0 is nuclei at offset 0 and channel 1
    # is membrane at +500; Z planes contribute +1000 each.  The base plane is
    # fixed so the tests can pin channel and page-order separation.
    base = np.zeros((4, 5), dtype=np.uint16)
    base[:] = np.array([
        [0, 1, 2, 3, 4],
        [17, 18, 21, 22, 23],
        [5, 6, 7, 8, 9],
        [10, 11, 12, 13, 14],
    ], dtype=np.uint16)
    dual = np.empty((3, 2, 4, 5), dtype=np.uint16)
    for z in range(3):
        for c in range(2):
            dual[z, c] = base + np.uint16(z * 1000 + c * 500)

    tifffile.imwrite(out / 'dual_imagej.tif', dual, imagej=True, metadata={'axes': 'ZCYX'})
    tifffile.imwrite(out / 'dual_ome.tif', dual, ome=True, metadata={'axes': 'ZCYX'})
    tifffile.imwrite(out / 'dual_json.tif', dual, photometric='minisblack')
    tifffile.imwrite(out / 'dual_nometa.tif', dual.reshape(-1, 4, 5),
                     photometric='minisblack', metadata=None)
    tifffile.imwrite(out / 'dual_metaempty.tif', dual.reshape(-1, 4, 5),
                     description='{}', photometric='minisblack')

    three = np.repeat(dual[:, :1], 3, axis=1)
    tifffile.imwrite(out / 'dual_three_channels.tif', three, imagej=True,
                     metadata={'axes': 'ZCYX'})


def main():
    import pathlib
    out = pathlib.Path(__file__).resolve().parent.parent / 'tests' / 'fixtures'
    out.mkdir(parents=True, exist_ok=True)

    write_tiff_fixtures(out)

    # --- zoom (uint16), including scipy's ties-to-even output sizing -------------
    zshape = (5, 7, 9)
    z = (np.arange(np.prod(zshape), dtype=np.uint32).reshape(zshape) * 997 % 65536).astype(np.uint16)
    zoomed = zoom(z, (6.92 * 0.5, 0.5, 0.5), order=1)
    write_fixture(out / 'zoom_input.u16', z, 'u16')
    write_fixture(out / 'zoom_expected.u16', zoomed, 'u16')

    # --- separable Gaussian, nearest (segmentation) and reflect (DoG) ------------
    gshape = (4, 7, 9)
    zz, yy, xx = np.meshgrid(np.arange(gshape[0]), np.arange(gshape[1]), np.arange(gshape[2]), indexing='ij')
    g = (np.sin(zz * 0.7) * 1000 + np.cos(yy * 0.4) * 300 + xx * 17 + 5000).astype(np.float64)
    write_fixture(out / 'gauss_input.f64', g, 'f64')
    write_fixture(out / 'gauss_nearest.f64',
                  gaussian(g, sigma=(0.5, 0.5, 0.5), preserve_range=True, mode='nearest'), 'f64')
    write_fixture(out / 'gauss_reflect.f64',
                  gaussian(g, sigma=(2.0, 3.2, 5.12), preserve_range=True, mode='reflect'), 'f64')

    # --- median filter on labels (1,r,r), reflect mode ---------------------------
    mshape = (3, 9, 11)
    rng = np.random.default_rng(12345)
    m = rng.integers(0, 8, size=mshape, dtype=np.uint32)
    write_fixture(out / 'median_input.u32', m, 'u32')
    write_fixture(out / 'median_expected.u32', median_filter(m, size=(1, 5, 5)), 'u32')

    # --- maximum filter on labels (1,3,3), reflect mode --------------------------
    xshape = (3, 7, 9)
    x = rng.integers(0, 6, size=xshape, dtype=np.uint32)
    write_fixture(out / 'max_input.u32', x, 'u32')
    write_fixture(out / 'max_expected.u32', maximum_filter(x, size=(1, 3, 3)), 'u32')

    # --- watershed on an imposed-minima landscape -------------------------------
    wshape = (5, 9, 11)
    wz, wy, wx = np.meshgrid(np.arange(wshape[0]), np.arange(wshape[1]), np.arange(wshape[2]), indexing='ij')
    base = ((wz * 37 + wy * 13 + wx * 7) % 101).astype(np.uint16)
    seed_coords = [(2, 4, 5), (2, 6, 5), (3, 2, 8)]
    markers = seedcoords2mask(seed_coords, wshape)
    imp = imposemin(base, markers > 0)
    wshed = watershed(imp, markers, compactness=0)
    write_fixture(out / 'watershed_input.u16', base, 'u16')
    write_fixture(out / 'watershed_markers.u32', markers, 'u32')
    write_fixture(out / 'watershed_expected.f64', imp, 'f64')
    write_fixture(out / 'watershed_expected.u32', wshed.astype(np.uint32), 'u32')

    # --- hole filling on labels ---------------------------------------------------
    hshape = (6, 12, 14)
    h = np.zeros(hshape, dtype=np.uint32)
    h[1:5, 1:7, 1:7] = 1
    h[2:4, 2:6, 2:6] = 0          # enclosed hole inside label 1
    h[1:5, 8:11, 8:11] = 2
    h[1:3, 9:10, 9:10] = 0        # hole open to z=0? (z=1 top face removed -> open)
    h[6 - 1, 8:11, 8:11] = 0
    # diagonal enclosed cavity across labels 1 and 2
    h[2:4, 7, 7] = 3
    h[2, 7, 7] = 0
    h[3, 8, 8] = 2
    h[3, 8, 8] = 0
    # Add one truly open vertical shaft through label 2's top
    h[1, 9, 9] = 0
    write_fixture(out / 'holes_input.u32', h, 'u32')
    write_fixture(out / 'holes_expected.u32', fill_lbl_holes(h.copy()), 'u32')

    # --- cell segmentation pipeline ----------------------------------------------
    cshape = (14, 88, 88)
    membrane = np.zeros(cshape, dtype=np.uint16)
    cz, cy, cx = np.meshgrid(np.arange(cshape[0]), np.arange(cshape[1]), np.arange(cshape[2]), indexing='ij')
    cells = [
        ((7, 26, 26), (4.0, 14.0, 14.0)),
        ((7, 60, 60), (4.0, 14.0, 14.0)),
        ((7, 70, 20), (2.0, 5.0, 5.0)),
        ((7, 8, 50), (4.0, 9.0, 9.0)),
    ]
    for center, radii in cells:
        d = np.sqrt(((cz - center[0]) / radii[0]) ** 2 + ((cy - center[1]) / radii[1]) ** 2 + ((cx - center[2]) / radii[2]) ** 2)
        membrane[(d >= 0.80) & (d <= 1.05)] = 50000
    coords = [center for center, _ in cells]
    cell_lbl = segment_cells(membrane, coords)
    write_fixture(out / 'cells_input.u16', membrane, 'u16')
    # store seed coordinates in a u32 array with three columns
    write_fixture(out / 'cells_seeds.u32', np.asarray(coords, dtype=np.uint32), 'u32')
    write_fixture(out / 'cells_expected.u32', cell_lbl.astype(np.uint32), 'u32')

    # --- nuclei segmentation ------------------------------------------------------
    nuc = np.zeros(cshape, dtype=np.uint16)
    for center, radii in cells[:3]:
        d = np.sqrt(((cz - center[0]) / (radii[0] * 0.5)) ** 2 + ((cy - center[1]) / (radii[1] * 0.45)) ** 2 + ((cx - center[2]) / (radii[2] * 0.45)) ** 2)
        nuc[d <= 1.0] = np.maximum(nuc[d <= 1.0], np.uint16(4000))
        nuc[d <= 0.75] = 12000
    # one nucleus hollow centre, one just below threshold
    nuc[7 - 1:7 + 2, 26 - 1:26 + 2, 26 - 1:26 + 2] = 100
    nuc[7, 70, 20] = 499
    nucleus_lbl = segment_nuclei(nuc, cell_lbl)
    write_fixture(out / 'nuclei_input.u16', nuc, 'u16')
    write_fixture(out / 'nuclei_cells.u32', cell_lbl.astype(np.uint32), 'u32')
    write_fixture(out / 'nuclei_expected.u32', nucleus_lbl.astype(np.uint32), 'u32')

    # --- seed detection -----------------------------------------------------------
    sshape = (16, 112, 112)
    sz, sy, sx = np.meshgrid(np.arange(sshape[0]), np.arange(sshape[1]), np.arange(sshape[2]), indexing='ij')
    nuclei = np.zeros(sshape, dtype=np.uint16)
    smembrane = np.zeros(sshape, dtype=np.uint16)
    seed_centers = [(8, 24, 24), (8, 46, 42), (8, 85, 78)]
    for center in seed_centers:
        d2 = ((sz - center[0]) / 1.6) ** 2 + ((sy - center[1]) / 6.0) ** 2 + ((sx - center[2]) / 6.0) ** 2
        nuclei = np.maximum(nuclei, (60000.0 * np.exp(-0.5 * d2)).astype(np.uint16))
    # Bright closed membrane shells: first two nuclei are in one cell and are
    # separated only by dim cytoplasm, so their seeds merge; the third is behind
    # a bright membrane.
    for center, radii in [((8, 35, 33), (5.0, 22.0, 20.0)), ((8, 85, 78), (5.0, 12.0, 12.0))]:
        d = np.sqrt(((sz - center[0]) / radii[0]) ** 2 + ((sy - center[1]) / radii[1]) ** 2 + ((sx - center[2]) / radii[2]) ** 2)
        smembrane[(d >= 0.90) & (d <= 1.02)] = 50000
    # bright wall directly between nucleus 2 and 3
    smembrane[1:15, 66:69, 58:98] = 50000
    write_fixture(out / 'seed_nuclei.u16', nuclei, 'u16')
    write_fixture(out / 'seed_membrane.u16', smembrane, 'u16')
    coords, kept = detect_seeds(nuclei, smembrane)
    write_fixture(out / 'seed_candidates.u32', np.asarray(coords, dtype=np.uint32), 'u32')
    write_fixture(out / 'seed_expected.u32', np.asarray(kept, dtype=np.uint32), 'u32')

    manifest = {
        'note': 'Generated by tools/reference_fixtures.py. Binary header: TS3D, version 1, dtype code, ndim, u64 dims, C-order payload.',
        'seed_candidates': [[int(v) for v in p] for p in coords],
        'seed_kept': [[int(v) for v in p] for p in kept],
        'cell_counts': {int(k): int(v) for k, v in zip(*np.unique(cell_lbl, return_counts=True))},
        'nucleus_counts': {int(k): int(v) for k, v in zip(*np.unique(nucleus_lbl, return_counts=True))},
    }
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
