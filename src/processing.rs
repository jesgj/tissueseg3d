//! Scientific processing core.  Public signatures are kept stable for the GUI.
//!
//! The implementation deliberately mirrors the NumPy/SciPy/scikit-image
//! behaviour used by the original Python application.  See `docs/rust-core-notes.md`
//! for the two intentional corrections (seed zero and nuclei export).

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

use anyhow::{bail, Result};
use rayon::prelude::*;

use crate::numpy_compat;
use crate::volume::{ImageStack, Volume};

pub type Seed = [usize; 3];

#[derive(Clone, Debug)]
pub struct SeedParams {
    pub min_sigma: f64,
    pub max_sigma: f64,
    pub threshold: f64,
    pub merge_distance: f64,
    pub membrane_delta: f64,
    pub z_ratio: f64,
}

impl Default for SeedParams {
    fn default() -> Self {
        Self {
            min_sigma: 4.0,
            max_sigma: 12.0,
            threshold: 1.0,
            merge_distance: 100.0,
            membrane_delta: 25.0,
            z_ratio: 6.92,
        }
    }
}

#[derive(Clone, Debug)]
pub struct CellParams {
    pub gaussian_sigma: f64,
    pub median_size: usize,
    pub min_volume: usize,
    pub max_volume: usize,
}

impl Default for CellParams {
    fn default() -> Self {
        Self {
            gaussian_sigma: 0.5,
            median_size: 5,
            min_volume: 2000,
            max_volume: 100000,
        }
    }
}

#[derive(Clone, Debug)]
pub struct NucleusParams {
    pub gaussian_sigma: f64,
    pub threshold: u16,
    pub median_size: usize,
    pub z_ratio: f64,
}

impl Default for NucleusParams {
    fn default() -> Self {
        Self {
            gaussian_sigma: 1.5,
            threshold: 500,
            median_size: 5,
            z_ratio: 6.92,
        }
    }
}

// ---------------------------------------------------------------------------
// Public operations
// ---------------------------------------------------------------------------

pub fn detect_seeds(stack: &ImageStack, p: &SeedParams) -> Result<Vec<Seed>> {
    let nuclei = stack
        .nuclei
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("seed detection requires a nuclei channel"))?;
    if nuclei.shape != stack.membrane.shape {
        bail!(
            "nuclei shape {:?} does not match membrane shape {:?}",
            nuclei.shape,
            stack.membrane.shape
        );
    }
    let prescale = 0.5;
    let mut seeds = detect_seed_candidates(nuclei, &stack.membrane, p)?;
    let distance = p.merge_distance * prescale;
    for _ in 0..3 {
        seeds = remove_seeds_round(
            &stack.membrane,
            &seeds,
            distance,
            p.membrane_delta,
            p.z_ratio,
        );
    }
    Ok(seeds)
}

pub fn segment_cells(
    membrane: &Volume<u16>,
    seeds: &[Seed],
    p: &CellParams,
) -> Result<Volume<u32>> {
    validate_params(p)?;
    let shape = membrane.shape;
    let markers = seed_mask(seeds, shape)?;

    let membrane_f = Volume::new(shape, membrane.data.iter().map(|&v| v as f64).collect())?;
    let sigma = [p.gaussian_sigma; 3];
    let blurred = gaussian_filter(&membrane_f, sigma, 4.0, BorderMode::Nearest);
    // skimage.filters.gaussian(...).astype('uint16') truncates toward zero.
    let blurred_u16 = Volume::new(shape, blurred.data.iter().map(|&v| trunc_u16(v)).collect())?;

    let imposed = impose_minima(&blurred_u16, &markers);
    let mut cells = watershed_6(&imposed, &markers)?;

    if p.median_size > 1 {
        cells = median_filter_generic(&cells, [1, p.median_size, p.median_size]);
    }
    remove_labels_by_size(&mut cells, p.min_volume, p.max_volume);
    remove_labels_touching_xy_edges(&mut cells);
    cells = fill_label_holes(cells);
    let cells = compact_labels(&cells);
    Ok(cells)
}

pub fn segment_nuclei(
    nuclei: &Volume<u16>,
    cells: &Volume<u32>,
    p: &NucleusParams,
) -> Result<Volume<u32>> {
    if nuclei.shape != cells.shape {
        bail!(
            "nuclei shape {:?} does not match cells shape {:?}",
            nuclei.shape,
            cells.shape
        );
    }
    if !p.gaussian_sigma.is_finite() || p.gaussian_sigma < 0.0 {
        bail!("nucleus gaussian sigma must be finite and non-negative");
    }
    if !p.z_ratio.is_finite() || p.z_ratio <= 0.0 {
        bail!("nucleus z ratio must be finite and positive");
    }
    if p.median_size == 0 {
        bail!("nucleus median size must be non-zero");
    }
    let shape = nuclei.shape;
    let f = Volume::new(shape, nuclei.data.iter().map(|&v| v as f64).collect())?;
    let sigma = [
        p.gaussian_sigma / p.z_ratio,
        p.gaussian_sigma,
        p.gaussian_sigma,
    ];
    let blurred = gaussian_filter(&f, sigma, 4.0, BorderMode::Nearest);
    let threshold = Volume::new(
        shape,
        blurred
            .data
            .iter()
            .map(|&v| trunc_u16(v) >= p.threshold)
            .collect(),
    )?;
    let smoothed = if p.median_size > 1 {
        median_filter_generic(&threshold, [1, p.median_size, p.median_size])
    } else {
        threshold
    };

    let mut labels = Volume::new(
        shape,
        cells
            .data
            .iter()
            .zip(&smoothed.data)
            .map(|(&cell, &on)| if on { cell } else { 0 })
            .collect(),
    )?;
    labels = fill_label_holes(labels);
    Ok(labels)
}

pub fn label_centroids(labels: &Volume<u32>) -> Vec<Seed> {
    // Equivalent to scipy.ndimage.center_of_mass(np.ones_like(labels), ...)
    // followed by the Python loader's `.astype(int)` (truncation).
    struct Acc {
        count: u64,
        sz: u64,
        sy: u64,
        sx: u64,
    }
    let mut acc: HashMap<u32, Acc> = HashMap::new();
    let [sy, sz] = [labels.shape[1], labels.shape[2]];
    for z in 0..labels.shape[0] {
        for y in 0..sy {
            for x in 0..sz {
                let label = labels.data[(z * sy + y) * sz + x];
                if label != 0 {
                    let a = acc.entry(label).or_insert(Acc {
                        count: 0,
                        sz: 0,
                        sy: 0,
                        sx: 0,
                    });
                    a.count += 1;
                    a.sz += z as u64;
                    a.sy += y as u64;
                    a.sx += x as u64;
                }
            }
        }
    }
    let mut items: Vec<(u32, Acc)> = acc.into_iter().collect();
    items.sort_by_key(|&(label, _)| label);
    items
        .into_iter()
        .map(|(_, a)| {
            let n = a.count as f64;
            [
                (a.sz as f64 / n) as usize,
                (a.sy as f64 / n) as usize,
                (a.sx as f64 / n) as usize,
            ]
        })
        .collect()
}

pub fn remove_label(cells: &mut Volume<u32>, nuclei: Option<&mut Volume<u32>>, label: u32) {
    for v in &mut cells.data {
        if *v == label {
            *v = 0;
        }
    }
    if let Some(nuc) = nuclei {
        for v in &mut nuc.data {
            if *v == label {
                *v = 0;
            }
        }
    }
}

pub fn merge_labels(
    cells: &mut Volume<u32>,
    nuclei: Option<&mut Volume<u32>>,
    keep: u32,
    remove: u32,
) {
    if keep == remove || keep == 0 {
        return;
    }
    for v in &mut cells.data {
        if *v == remove {
            *v = keep;
        }
    }
    if let Some(nuc) = nuclei {
        for v in &mut nuc.data {
            if *v == remove {
                *v = keep;
            }
        }
    }
}

fn validate_params(p: &CellParams) -> Result<()> {
    if !p.gaussian_sigma.is_finite() || p.gaussian_sigma < 0.0 {
        bail!("cell gaussian sigma must be finite and non-negative");
    }
    if p.median_size == 0 {
        bail!("cell median size must be non-zero");
    }
    if p.min_volume > p.max_volume {
        bail!("cell min volume must not exceed max volume");
    }
    Ok(())
}

/// A tiny binary heap whose ordering matches skimage's watershed heap:
/// lower value first, then lower insertion age.  Equal values/ages are left
/// in heap-array order, exactly like the Cython implementation.
#[derive(Clone, Copy, Debug)]
struct HeapItem {
    value: f64,
    age: u64,
    index: usize,
    label: u32,
}

#[derive(Default)]
struct WatershedHeap {
    items: Vec<HeapItem>,
}

impl WatershedHeap {
    #[inline]
    fn smaller(a: &HeapItem, b: &HeapItem) -> bool {
        if a.value != b.value {
            a.value < b.value
        } else {
            a.age < b.age
        }
    }

    fn push(&mut self, item: HeapItem) {
        let mut child = self.items.len();
        self.items.push(item);
        while child > 0 {
            let parent = child.div_ceil(2) - 1;
            if Self::smaller(&self.items[child], &self.items[parent]) {
                self.items.swap(parent, child);
                child = parent;
            } else {
                break;
            }
        }
    }

    fn pop(&mut self) -> Option<HeapItem> {
        if self.items.is_empty() {
            return None;
        }
        let top = self.items[0];
        self.items.swap_remove(0);
        if self.items.is_empty() {
            return Some(top);
        }
        let mut i = 0;
        loop {
            let l = 2 * i + 1;
            let r = 2 * i + 2;
            if l >= self.items.len() {
                break;
            }
            let mut smallest = i;
            if Self::smaller(&self.items[l], &self.items[i]) {
                smallest = l;
            }
            if r < self.items.len() && Self::smaller(&self.items[r], &self.items[smallest]) {
                smallest = r;
            }
            if smallest == i {
                break;
            }
            self.items.swap(i, smallest);
            i = smallest;
        }
        Some(top)
    }
}

fn watershed_6(image: &Volume<f64>, markers: &Volume<u32>) -> Result<Volume<u32>> {
    if image.shape != markers.shape {
        bail!(
            "watershed image shape {:?} does not match markers shape {:?}",
            image.shape,
            markers.shape
        );
    }
    let [nz, ny, nx] = image.shape;
    let ys = nx;
    let zs = ny * nx;
    let mut output = markers.clone();
    let mut heap = WatershedHeap::default();

    for index in 0..output.data.len() {
        let label = output.data[index];
        if label != 0 {
            heap.push(HeapItem {
                value: image.data[index],
                age: 0,
                index,
                label,
            });
        }
    }

    let strides: [isize; 3] = [zs as isize, ys as isize, 1];
    let offsets: [isize; 6] = [-strides[0], -strides[1], -1, 1, strides[1], strides[0]];
    let mut age = 0u64;
    while let Some(elem) = heap.pop() {
        for &off in &offsets {
            let ni = elem.index as isize + off;
            if ni < 0 || ni as usize >= output.data.len() {
                continue;
            }
            let ni = ni as usize;
            // A raveled offset can wrap between rows/planes.  Reject those
            // rather than treating the wrapped voxel as a neighbour.
            let (z, y, x) = unravel(ni, image.shape);
            let (cz, cy, cx) = unravel(elem.index, image.shape);
            let dz = (z as isize - cz as isize).abs();
            let dy = (y as isize - cy as isize).abs();
            let dx = (x as isize - cx as isize).abs();
            if dz + dy + dx != 1 {
                continue;
            }
            if output.data[ni] != 0 {
                continue;
            }
            age += 1;
            output.data[ni] = elem.label;
            let mut value = image.data[ni];
            if value < elem.value {
                value = elem.value;
            }
            heap.push(HeapItem {
                value,
                age,
                index: ni,
                label: elem.label,
            });
        }
        let _ = (nz, ny, nx);
    }
    Ok(output)
}

#[inline]
fn unravel(index: usize, shape: [usize; 3]) -> (usize, usize, usize) {
    let z = index / (shape[1] * shape[2]);
    let rem = index % (shape[1] * shape[2]);
    (z, rem / shape[2], rem % shape[2])
}

// ---------------------------------------------------------------------------
// Interpolation and filtering primitives
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
enum BorderMode {
    Nearest,
    Reflect,
}

#[inline]
fn reflect_index(i: isize, n: usize) -> usize {
    if n == 1 {
        return 0;
    }
    let period = 2 * n;
    let mut m = i.rem_euclid(period as isize) as usize;
    if m >= n {
        m = period - 1 - m;
    }
    m
}

#[inline]
fn border_index(i: isize, n: usize, mode: BorderMode) -> usize {
    match mode {
        BorderMode::Nearest => i.clamp(0, n as isize - 1) as usize,
        BorderMode::Reflect => reflect_index(i, n),
    }
}

fn zoom_linear_u16(input: &Volume<u16>, factors: [f64; 3]) -> Result<Volume<u16>> {
    if factors.iter().any(|f| !f.is_finite() || *f <= 0.0) {
        bail!("zoom factors must be finite and positive, got {factors:?}");
    }
    // scipy.ndimage.zoom: output_shape = int(round(input_shape * zoom)).
    let out_shape = [
        (input.shape[0] as f64 * factors[0]).round_ties_even() as usize,
        (input.shape[1] as f64 * factors[1]).round_ties_even() as usize,
        (input.shape[2] as f64 * factors[2]).round_ties_even() as usize,
    ];
    if out_shape.contains(&0) {
        bail!(
            "zoom of shape {:?} by {factors:?} produces an empty volume",
            input.shape
        );
    }
    let [iz, iy, ix] = input.shape;
    let [oz, oy, ox] = out_shape;
    let zs = iy * ix;
    let mut data = vec![0u16; oz * oy * ox];

    let coord = |out_len: usize, in_len: usize, i: usize| -> (usize, usize, f64) {
        if out_len <= 1 {
            (0, 0, 0.0)
        } else {
            let src = i as f64 * (in_len - 1) as f64 / (out_len - 1) as f64;
            let i0 = src.floor() as usize;
            let i1 = (i0 + 1).min(in_len - 1);
            (i0, i1, src - i0 as f64)
        }
    };

    data.par_chunks_mut(oy * ox)
        .enumerate()
        .for_each(|(z, plane)| {
            let (z0, z1, tz) = coord(oz, iz, z);
            for y in 0..oy {
                let (y0, y1, ty) = coord(oy, iy, y);
                for x in 0..ox {
                    let (x0, x1, tx) = coord(ox, ix, x);
                    let at = |zz: usize, yy: usize, xx: usize| -> f64 {
                        input.data[zz * zs + yy * ix + xx] as f64
                    };
                    let c00 = at(z0, y0, x0) * (1.0 - tx) + at(z0, y0, x1) * tx;
                    let c01 = at(z0, y1, x0) * (1.0 - tx) + at(z0, y1, x1) * tx;
                    let c10 = at(z1, y0, x0) * (1.0 - tx) + at(z1, y0, x1) * tx;
                    let c11 = at(z1, y1, x0) * (1.0 - tx) + at(z1, y1, x1) * tx;
                    let c0 = c00 * (1.0 - ty) + c01 * ty;
                    let c1 = c10 * (1.0 - ty) + c11 * ty;
                    let v = c0 * (1.0 - tz) + c1 * tz;
                    // scipy's integer output rounds half away from zero.
                    plane[y * ox + x] = (v + 0.5).floor().clamp(0.0, 65535.0) as u16;
                }
            }
        });
    Volume::new(out_shape, data)
}

/// The 1-D Gaussian kernel scipy's `gaussian_filter1d` builds for `sigma`.
///
/// `scipy.ndimage._filters._gaussian_kernel1d` normalises with `numpy.sum` and
/// exponent with `numpy.exp`, so both steps go through [`crate::numpy_compat`]
/// to keep the weights bit-identical to the reference stack.
fn gaussian_weights(sigma: f64, radius: isize) -> Vec<f64> {
    let sigma2 = sigma * sigma;
    let scale = -0.5 / sigma2;
    let mut weights: Vec<f64> = (-radius..=radius)
        .map(|x| numpy_compat::exp(scale * (x * x) as f64))
        .collect();
    let sum = numpy_compat::pairwise_sum(&weights);
    for w in &mut weights {
        *w /= sum;
    }
    weights
}

fn gaussian_filter(
    input: &Volume<f64>,
    sigma: [f64; 3],
    truncate: f64,
    mode: BorderMode,
) -> Volume<f64> {
    let [nz, ny, nx] = input.shape;
    let dims = [nz, ny, nx];
    let mut current = input.data.clone();
    let mut next = vec![0.0f64; current.len()];
    let row = nx;

    for axis in 0..3 {
        let s = sigma[axis];
        if s <= 1e-15 || s.is_nan() {
            continue;
        }
        let radius = (truncate * s + 0.5) as isize;
        if radius < 0 {
            continue;
        }
        let weights = gaussian_weights(s, radius);
        let n = dims[axis];
        let r = radius;
        next.par_chunks_mut(row)
            .enumerate()
            .for_each(|(row_idx, dst)| {
                let z = row_idx / ny;
                let y = row_idx % ny;
                for x in 0..nx {
                    let at = |offset: isize| -> f64 {
                        let p = match axis {
                            0 => z as isize + offset,
                            1 => y as isize + offset,
                            _ => x as isize + offset,
                        };
                        let p = border_index(p, n, mode);
                        match axis {
                            0 => current[p * ny * nx + y * nx + x],
                            1 => current[z * ny * nx + p * nx + x],
                            _ => current[z * ny * nx + y * nx + p],
                        }
                    };
                    // scipy's symmetric correlate1d accumulates the centre sample and
                    // then each mirrored pair of samples from the outside in.
                    let mut acc = weights[r as usize] * at(0);
                    for k in 0..r {
                        acc += (at(k - r) + at(r - k)) * weights[k as usize];
                    }
                    dst[x] = acc;
                }
            });
        std::mem::swap(&mut current, &mut next);
    }
    Volume::new(input.shape, current).expect("gaussian output has input shape")
}

/// Window offsets scipy.ndimage uses for a filter `size` samples wide on one axis.
fn offsets(size: usize) -> std::ops::Range<isize> {
    -((size / 2) as isize)..((size - size / 2) as isize)
}

fn median_filter_generic<T: Copy + Ord + Send + Sync>(
    input: &Volume<T>,
    size: [usize; 3],
) -> Volume<T> {
    let shape = input.shape;
    let [nz, ny, nx] = shape;
    // scipy.ndimage places an even-sized window at -(size/2)..(size-size/2), so
    // the window always holds exactly `size` samples and the upper median wins.
    let (dz, dy, dx) = (offsets(size[0]), offsets(size[1]), offsets(size[2]));
    let window = size[0] * size[1] * size[2];
    let planes: Vec<Vec<T>> = (0..nz)
        .into_par_iter()
        .map(|z| {
            let mut plane = vec![input.data[0]; ny * nx];
            let mut values = Vec::with_capacity(window);
            for y in 0..ny {
                for x in 0..nx {
                    values.clear();
                    for oz in dz.clone() {
                        let zz = border_index(z as isize + oz, nz, BorderMode::Reflect);
                        for oy in dy.clone() {
                            let yy = border_index(y as isize + oy, ny, BorderMode::Reflect);
                            for ox in dx.clone() {
                                let xx = border_index(x as isize + ox, nx, BorderMode::Reflect);
                                values.push(input.data[(zz * ny + yy) * nx + xx]);
                            }
                        }
                    }
                    let mid = values.len() / 2;
                    values.select_nth_unstable(mid);
                    plane[y * nx + x] = values[mid];
                }
            }
            plane
        })
        .collect();
    Volume::new(shape, planes.into_iter().flatten().collect()).expect("median shape")
}

#[cfg(test)]
fn median_filter_u32(input: &mut Volume<u32>, size: [usize; 3]) {
    *input = median_filter_generic(input, size);
}

fn maximum_filter_u32(input: &Volume<u32>, size: [usize; 3]) -> Volume<u32> {
    let shape = input.shape;
    let [nz, ny, nx] = shape;
    let rz = size[0] / 2;
    let ry = size[1] / 2;
    let rx = size[2] / 2;
    let planes: Vec<Vec<u32>> = (0..nz)
        .into_par_iter()
        .map(|z| {
            let mut plane = vec![0u32; ny * nx];
            for y in 0..ny {
                for x in 0..nx {
                    let mut best = 0;
                    for dz in -(rz as isize)..=rz as isize {
                        let zz = border_index(z as isize + dz, nz, BorderMode::Reflect);
                        for dy in -(ry as isize)..=ry as isize {
                            let yy = border_index(y as isize + dy, ny, BorderMode::Reflect);
                            for dx in -(rx as isize)..=rx as isize {
                                let xx = border_index(x as isize + dx, nx, BorderMode::Reflect);
                                best = best.max(input.data[(zz * ny + yy) * nx + xx]);
                            }
                        }
                    }
                    plane[y * nx + x] = best;
                }
            }
            plane
        })
        .collect();
    Volume::new(shape, planes.into_iter().flatten().collect()).expect("maximum filter shape")
}

#[inline]
fn trunc_u16(v: f64) -> u16 {
    v.trunc().clamp(0.0, 65535.0) as u16
}

#[derive(Clone, Copy, Debug)]
struct ReconstructionItem {
    value: f64,
    index: usize,
}

impl PartialEq for ReconstructionItem {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value && self.index == other.index
    }
}
impl Eq for ReconstructionItem {}
impl PartialOrd for ReconstructionItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for ReconstructionItem {
    fn cmp(&self, other: &Self) -> Ordering {
        // BinaryHeap is a max-heap; reverse the value so the smallest
        // reconstruction level is popped first.
        other
            .value
            .partial_cmp(&self.value)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.index.cmp(&self.index))
    }
}

fn impose_minima(input: &Volume<u16>, markers: &Volume<u32>) -> Volume<f64> {
    // skimage.morphology.reconstruction(marker=+inf except seeds=0,
    // mask=minimum(img+1, marker), method='erosion').
    //
    // Flood the unseeded valleys from the markers; each voxel rises to the
    // lowest saddle that connects it to a seeded minimum.  This is the
    // erosion-reconstruction fixed point and cannot be replaced by img+1.
    let shape = input.shape;
    let seed: Vec<f64> = markers
        .data
        .iter()
        .map(|&m| if m != 0 { 0.0 } else { f64::INFINITY })
        .collect();
    let mask: Vec<f64> = input
        .data
        .iter()
        .zip(&markers.data)
        .map(|(&v, &m)| {
            if m != 0 {
                0.0
            } else {
                v.wrapping_add(1) as f64
            }
        })
        .collect();
    let [nz, ny, nx] = shape;
    let mut out = seed;
    let mut heap = BinaryHeap::new();
    for (i, &value) in out.iter().enumerate() {
        if value.is_finite() {
            heap.push(ReconstructionItem { value, index: i });
        }
    }
    while let Some(item) = heap.pop() {
        if item.value != out[item.index] {
            continue; // stale queue entry
        }
        let (z, y, x) = unravel(item.index, shape);
        // skimage's default reconstruction footprint is the full 3x3x3
        // window with only the centre removed, i.e. 26-connectivity.
        for dz in -1isize..=1 {
            let zz = z as isize + dz;
            if zz < 0 || zz >= nz as isize {
                continue;
            }
            for dy in -1isize..=1 {
                let yy = y as isize + dy;
                if yy < 0 || yy >= ny as isize {
                    continue;
                }
                for dx in -1isize..=1 {
                    if dz == 0 && dy == 0 && dx == 0 {
                        continue;
                    }
                    let xx = x as isize + dx;
                    if xx < 0 || xx >= nx as isize {
                        continue;
                    }
                    let j = (zz as usize * ny + yy as usize) * nx + xx as usize;
                    if out[j] > item.value && out[j] > mask[j] {
                        let candidate = mask[j].max(item.value);
                        if candidate < out[j] {
                            out[j] = candidate;
                            heap.push(ReconstructionItem {
                                value: candidate,
                                index: j,
                            });
                        }
                    }
                }
            }
        }
    }
    Volume::new(shape, out).expect("impose minima shape")
}

// ---------------------------------------------------------------------------
// Labels
// ---------------------------------------------------------------------------

fn seed_mask(seeds: &[Seed], shape: [usize; 3]) -> Result<Volume<u32>> {
    let [nz, ny, nx] = shape;
    let mut markers = Volume::filled(shape, 0u32)?;
    for z in 0..nz {
        for x in 0..nx {
            markers.data[(z * ny) * nx + x] = 1; // y = 0
            markers.data[(z * ny + ny - 1) * nx + x] = 1; // y = -1
        }
        for y in 0..ny {
            markers.data[(z * ny + y) * nx] = 1; // x = 0
            markers.data[(z * ny + y) * nx + nx - 1] = 1; // x = -1
        }
    }
    for (i, &[z, y, x]) in seeds.iter().enumerate() {
        if z >= nz || y >= ny || x >= nx {
            bail!("seed {i} at ({z}, {y}, {x}) is outside volume shape {shape:?}");
        }
        let label = u32::try_from(i + 2)
            .map_err(|_| anyhow::anyhow!("too many seeds for u32 marker labels"))?;
        let idx = (z * ny + y) * nx + x;
        markers.data[idx] = label;
    }
    Ok(markers)
}

fn remove_labels_by_size(labels: &mut Volume<u32>, min_volume: usize, max_volume: usize) {
    let mut counts: Vec<usize> = Vec::new();
    for &label in &labels.data {
        let i = label as usize;
        if counts.len() <= i {
            counts.resize(i + 1, 0);
        }
        counts[i] += 1;
    }
    for v in &mut labels.data {
        if *v == 0 {
            continue;
        }
        let count = counts.get(*v as usize).copied().unwrap_or(0);
        if count < min_volume || count > max_volume {
            *v = 0;
        }
    }
}

fn remove_labels_touching_xy_edges(labels: &mut Volume<u32>) {
    let [_, ny, nx] = labels.shape;
    #[derive(Clone, Copy)]
    struct Bb {
        min_y: usize,
        max_y: usize,
        min_x: usize,
        max_x: usize,
    }
    let mut boxes: Vec<Option<Bb>> = Vec::new();
    for z in 0..labels.shape[0] {
        for y in 0..ny {
            for x in 0..nx {
                let label = labels.data[(z * ny + y) * nx + x];
                if label == 0 {
                    continue;
                }
                let i = label as usize;
                if boxes.len() <= i {
                    boxes.resize(i + 1, None);
                }
                match &mut boxes[i] {
                    None => {
                        boxes[i] = Some(Bb {
                            min_y: y,
                            max_y: y,
                            min_x: x,
                            max_x: x,
                        })
                    }
                    Some(b) => {
                        b.min_y = b.min_y.min(y);
                        b.max_y = b.max_y.max(y);
                        b.min_x = b.min_x.min(x);
                        b.max_x = b.max_x.max(x);
                    }
                }
            }
        }
    }
    let bad = |b: Bb| -> bool {
        // regionprops bbox maxima are exclusive.
        b.min_y <= 1
            || (b.max_y + 1) as isize >= ny as isize - 2
            || b.min_x <= 1
            || (b.max_x + 1) as isize >= nx as isize - 2
    };
    for v in &mut labels.data {
        if *v == 0 {
            continue;
        }
        let is_bad = boxes
            .get(*v as usize)
            .and_then(|b| *b)
            .map(bad)
            .unwrap_or(false);
        if is_bad {
            *v = 0;
        }
    }
}

fn fill_label_holes(labels: Volume<u32>) -> Volume<u32> {
    let shape = labels.shape;
    let [nz, ny, nx] = shape;
    let len = labels.data.len();
    let mut outside = vec![false; len];
    let mut stack: Vec<usize> = Vec::new();

    // 6-connected background flood from all six outer faces.
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                if z == 0 || z == nz - 1 || y == 0 || y == ny - 1 || x == 0 || x == nx - 1 {
                    let i = (z * ny + y) * nx + x;
                    if labels.data[i] == 0 && !outside[i] {
                        outside[i] = true;
                        stack.push(i);
                    }
                }
            }
        }
    }
    while let Some(i) = stack.pop() {
        let (z, y, x) = unravel(i, shape);
        let mut visit = |zz: usize, yy: usize, xx: usize, stack: &mut Vec<usize>| {
            let j = (zz * ny + yy) * nx + xx;
            if labels.data[j] == 0 && !outside[j] {
                outside[j] = true;
                stack.push(j);
            }
        };
        if z > 0 {
            visit(z - 1, y, x, &mut stack);
        }
        if z + 1 < nz {
            visit(z + 1, y, x, &mut stack);
        }
        if y > 0 {
            visit(z, y - 1, x, &mut stack);
        }
        if y + 1 < ny {
            visit(z, y + 1, x, &mut stack);
        }
        if x > 0 {
            visit(z, y, x - 1, &mut stack);
        }
        if x + 1 < nx {
            visit(z, y, x + 1, &mut stack);
        }
    }

    let mut holes = Volume::filled(shape, 0u32).expect("hole shape");
    for ((&label, &out), hole) in labels.data.iter().zip(&outside).zip(&mut holes.data) {
        if label == 0 && !out {
            *hole = 1;
        }
    }

    // skimage.measure.label defaults to full (26-) connectivity.
    let mut comp = vec![0u32; len];
    let mut next_label = 1u32;
    let mut stack: Vec<usize> = Vec::new();
    for (i, &hole) in holes.data.iter().enumerate() {
        if hole == 0 || comp[i] != 0 {
            continue;
        }
        comp[i] = next_label;
        stack.push(i);
        while let Some(p) = stack.pop() {
            let (z, y, x) = unravel(p, shape);
            for dz in -1isize..=1 {
                let zz = z as isize + dz;
                if zz < 0 || zz >= nz as isize {
                    continue;
                }
                for dy in -1isize..=1 {
                    let yy = y as isize + dy;
                    if yy < 0 || yy >= ny as isize {
                        continue;
                    }
                    for dx in -1isize..=1 {
                        if dz == 0 && dy == 0 && dx == 0 {
                            continue;
                        }
                        let xx = x as isize + dx;
                        if xx < 0 || xx >= nx as isize {
                            continue;
                        }
                        let j = (zz as usize * ny + yy as usize) * nx + xx as usize;
                        if holes.data[j] != 0 && comp[j] == 0 {
                            comp[j] = next_label;
                            stack.push(j);
                        }
                    }
                }
            }
        }
        next_label += 1;
    }

    let max_filtered = maximum_filter_u32(&labels, [1, 3, 3]);
    let ncomp = next_label as usize;
    let mut assign = vec![0u32; ncomp];
    for (&c, &v) in comp.iter().zip(&max_filtered.data) {
        if c != 0 && v > assign[c as usize] {
            assign[c as usize] = v;
        }
    }
    let mut out = labels;
    for (&c, value) in comp.iter().zip(&mut out.data) {
        if c != 0 {
            *value = assign[c as usize];
        }
    }
    out
}

fn compact_labels(labels: &Volume<u32>) -> Volume<u32> {
    let max = labels.data.iter().copied().max().unwrap_or(0);
    if max == 0 {
        return labels.clone();
    }
    let mut mapping = vec![0u32; max as usize + 1];
    let mut next = 1u32;
    let mut present = vec![false; max as usize + 1];
    for &v in &labels.data {
        present[v as usize] = true;
    }
    for label in 1..=max as usize {
        if present[label] {
            mapping[label] = next;
            next += 1;
        }
    }
    let data = labels.data.iter().map(|&v| mapping[v as usize]).collect();
    Volume::new(labels.shape, data).expect("compact labels shape")
}

// ---------------------------------------------------------------------------
// Nucleus seeds: zoom + multiscale DoG + membrane ray tracing
// ---------------------------------------------------------------------------

fn remove_seeds_round(
    img: &Volume<u16>,
    seeds: &[Seed],
    dstthr: f64,
    deltamin: f64,
    zratio: f64,
) -> Vec<Seed> {
    let n = seeds.len();
    if n == 0 {
        return Vec::new();
    }
    let mut mergelst: Vec<Vec<usize>> = vec![Vec::new(); n];
    for i in 0..n {
        for j in (i + 1)..n {
            let s1 = seeds[i];
            let s2 = seeds[j];
            let dz = (s1[0] as f64 - s2[0] as f64) * zratio;
            let dy = s1[1] as f64 - s2[1] as f64;
            let dx = s1[2] as f64 - s2[2] as f64;
            if (dz * dz + dy * dy + dx * dx).sqrt() >= dstthr {
                continue;
            }
            let points = interpolate_line(s1, s2);
            let mut lo = u16::MAX;
            let mut hi = 0u16;
            for p in &points {
                let v = img.data[p[0] * img.shape[1] * img.shape[2] + p[1] * img.shape[2] + p[2]];
                lo = lo.min(v);
                hi = hi.max(v);
            }
            if ((hi - lo) as f64) < deltamin {
                mergelst[i].push(j);
            }
        }
    }
    let mut merged_away = vec![false; n];
    for list in &mergelst {
        for &j in list {
            merged_away[j] = true;
        }
    }
    let mut centres: Vec<Seed> = Vec::with_capacity(n);
    for (i, list) in mergelst.iter().enumerate() {
        let mut sz = seeds[i][0] as f64;
        let mut sy = seeds[i][1] as f64;
        let mut sx = seeds[i][2] as f64;
        for &j in list {
            sz += seeds[j][0] as f64;
            sy += seeds[j][1] as f64;
            sx += seeds[j][2] as f64;
        }
        let count = (1 + list.len()) as f64;
        centres.push([
            (sz / count).round_ties_even() as usize,
            (sy / count).round_ties_even() as usize,
            (sx / count).round_ties_even() as usize,
        ]);
    }
    (0..n)
        .filter(|&i| !merged_away[i])
        .map(|i| centres[i])
        .collect()
}

fn interpolate_line(start: Seed, end: Seed) -> Vec<Seed> {
    let v = [
        end[0] as f64 - start[0] as f64,
        end[1] as f64 - start[1] as f64,
        end[2] as f64 - start[2] as f64,
    ];
    let norm = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    let steps = norm.ceil() as usize;
    if steps == 0 {
        return vec![start];
    }
    let mut out = Vec::with_capacity(steps);
    for i in 0..steps {
        if i == 0 {
            out.push(start);
        } else if i + 1 == steps {
            out.push(end);
        } else {
            let t = i as f64 / (steps - 1) as f64;
            out.push([
                (start[0] as f64 + t * v[0]).round_ties_even() as usize,
                (start[1] as f64 + t * v[1]).round_ties_even() as usize,
                (start[2] as f64 + t * v[2]).round_ties_even() as usize,
            ]);
        }
    }
    out
}

#[derive(Clone, Copy, Debug)]
struct Blob {
    z: f64,
    y: f64,
    x: f64,
    sigma: f64,
}

fn detect_seed_candidates(
    nuclei: &Volume<u16>,
    _membrane: &Volume<u16>,
    p: &SeedParams,
) -> Result<Vec<Seed>> {
    let prescale = 0.5;
    if !p.z_ratio.is_finite() || p.z_ratio <= 0.0 {
        bail!("seed z ratio must be finite and positive");
    }
    if !p.min_sigma.is_finite()
        || !p.max_sigma.is_finite()
        || p.min_sigma <= 0.0
        || p.max_sigma <= 0.0
    {
        bail!("seed sigma range must be finite and positive");
    }
    if p.max_sigma < p.min_sigma {
        bail!("seed max sigma must not be smaller than min sigma");
    }
    if !p.threshold.is_finite() || p.threshold < 0.0 {
        bail!("seed threshold must be finite and non-negative");
    }

    let zoomed = zoom_linear_u16(nuclei, [p.z_ratio * prescale, prescale, prescale])?;
    let image = Volume::new(
        zoomed.shape,
        zoomed.data.iter().map(|&v| v as f64 / 65535.0).collect(),
    )?;

    let ratio = 1.6f64;
    let min_sigma = p.min_sigma * prescale;
    let max_sigma = p.max_sigma * prescale;
    let k = ((max_sigma / min_sigma).ln() / ratio.ln() + 1.0) as usize;
    if k == 0 {
        return Ok(Vec::new());
    }
    let sigmas = dog_sigmas(min_sigma, max_sigma, ratio);
    let dogs = dog_layers(&image, &sigmas, ratio);

    let threshold = p.threshold * 1e-3;
    let peaks = dog_peaks(&dogs, threshold);
    let mut blobs: Vec<Blob> = peaks
        .into_iter()
        .map(|(_, z, y, x, scale)| Blob {
            z: z as f64,
            y: y as f64,
            x: x as f64,
            sigma: sigmas[scale],
        })
        .collect();
    blobs = prune_blobs(blobs, 0.5);

    Ok(blobs
        .into_iter()
        .map(|b| {
            [
                (b.z / (p.z_ratio * prescale)) as usize,
                (b.y / prescale) as usize,
                (b.x / prescale) as usize,
            ]
        })
        .collect())
}

/// Geometric progression of DoG sigmas, as `blob_dog` sizes its scale space.
fn dog_sigmas(min_sigma: f64, max_sigma: f64, ratio: f64) -> Vec<f64> {
    let k = ((max_sigma / min_sigma).ln() / ratio.ln() + 1.0) as usize;
    (0..=k).map(|i| min_sigma * ratio.powi(i as i32)).collect()
}

/// Difference-of-Gaussian layers between consecutive sigma levels.
fn dog_layers(image: &Volume<f64>, sigmas: &[f64], ratio: f64) -> Vec<Volume<f64>> {
    let normalization = 1.0 / (ratio - 1.0);
    let mut dogs = Vec::with_capacity(sigmas.len().saturating_sub(1));
    let mut previous = gaussian_filter_raw(image, sigmas[0]);
    for &sigma in &sigmas[1..] {
        let current = gaussian_filter_raw(image, sigma);
        let data: Vec<f64> = previous
            .data
            .iter()
            .zip(&current.data)
            .map(|(&a, &b)| (a - b) * normalization)
            .collect();
        dogs.push(Volume::new(image.shape, data).expect("DoG layer has the image shape"));
        previous = current;
    }
    dogs
}

/// Scale-space maxima, ordered like `np.argsort(-intensities, kind='stable')`.
fn dog_peaks(dogs: &[Volume<f64>], threshold: f64) -> Vec<(f64, usize, usize, usize, usize)> {
    let [nz, ny, nx] = dogs[0].shape;
    let scales = dogs.len();
    let mut peaks: Vec<(f64, usize, usize, usize, usize)> = Vec::new();
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let mut spatial = vec![f64::NEG_INFINITY; scales];
                for dz in -1isize..=1 {
                    let zz = border_index(z as isize + dz, nz, BorderMode::Nearest);
                    for dy in -1isize..=1 {
                        let yy = border_index(y as isize + dy, ny, BorderMode::Nearest);
                        for dx in -1isize..=1 {
                            let xx = border_index(x as isize + dx, nx, BorderMode::Nearest);
                            let idx = (zz * ny + yy) * nx + xx;
                            for (scale, slot) in spatial.iter_mut().enumerate() {
                                *slot = slot.max(dogs[scale].data[idx]);
                            }
                        }
                    }
                }
                for (scale, dog) in dogs.iter().enumerate() {
                    // peak_local_max's footprint spans scale-1..scale+1 only, with
                    // the scale axis clamped at its ends like mode='nearest'.
                    let lo = scale.saturating_sub(1);
                    let hi = (scale + 1).min(scales - 1);
                    let best = spatial[lo..=hi]
                        .iter()
                        .fold(f64::NEG_INFINITY, |m, &v| m.max(v));
                    let value = dog.data[(z * ny + y) * nx + x];
                    if value > threshold && value == best {
                        peaks.push((value, z, y, x, scale));
                    }
                }
            }
        }
    }
    peaks.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(Ordering::Equal)
            .then_with(|| (a.1, a.2, a.3, a.4).cmp(&(b.1, b.2, b.3, b.4)))
    });
    peaks
}

fn gaussian_filter_raw(input: &Volume<f64>, sigma: f64) -> Volume<f64> {
    gaussian_filter(input, [sigma; 3], 4.0, BorderMode::Reflect)
}

fn prune_blobs(mut blobs: Vec<Blob>, overlap: f64) -> Vec<Blob> {
    if blobs.is_empty() {
        return blobs;
    }
    let max_sigma = blobs.iter().map(|b| b.sigma).fold(0.0f64, f64::max);
    let radius = 2.0 * max_sigma * 3.0f64.sqrt();
    let r2 = radius * radius;
    for i in 0..blobs.len() {
        for j in (i + 1)..blobs.len() {
            let d2 = (blobs[i].z - blobs[j].z).powi(2)
                + (blobs[i].y - blobs[j].y).powi(2)
                + (blobs[i].x - blobs[j].x).powi(2);
            if d2 > r2 {
                continue;
            }
            if blob_overlap(&blobs[i], &blobs[j]) > overlap {
                if blobs[i].sigma > blobs[j].sigma {
                    blobs[j].sigma = 0.0;
                } else {
                    blobs[i].sigma = 0.0;
                }
            }
        }
    }
    blobs.into_iter().filter(|b| b.sigma > 0.0).collect()
}

fn blob_overlap(a: &Blob, b: &Blob) -> f64 {
    let ndim = 3u32;
    let root_ndim = (ndim as f64).sqrt();
    if a.sigma == 0.0 && b.sigma == 0.0 {
        return 0.0;
    }
    let (max_sigma, r1, r2) = if a.sigma > b.sigma {
        (a.sigma, 1.0, b.sigma / a.sigma)
    } else {
        (b.sigma, a.sigma / b.sigma, 1.0)
    };
    let scale = max_sigma * root_ndim;
    let p1 = [a.z / scale, a.y / scale, a.x / scale];
    let p2 = [b.z / scale, b.y / scale, b.x / scale];
    let d = ((p2[0] - p1[0]).powi(2) + (p2[1] - p1[1]).powi(2) + (p2[2] - p1[2]).powi(2)).sqrt();
    if d > r1 + r2 {
        return 0.0;
    }
    if d <= (r1 - r2).abs() {
        return 1.0;
    }
    let r1sq = r1 * r1;
    let r2sq = r2 * r2;
    let volume = std::f64::consts::PI / (12.0 * d)
        * (r1 + r2 - d).powi(2)
        * (d * d + 2.0 * d * (r1 + r2) - 3.0 * (r1sq + r2sq) + 6.0 * r1 * r2);
    volume / (4.0 / 3.0 * std::f64::consts::PI * r1.min(r2).powi(3))
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    const TOL: f64 = 1e-10;

    enum Fixture {
        U16(Volume<u16>),
        U32(Volume<u32>),
        F64(Volume<f64>),
    }

    impl Fixture {
        fn u16(self) -> Volume<u16> {
            match self {
                Fixture::U16(v) => v,
                _ => panic!("fixture is not u16"),
            }
        }
        fn u32(self) -> Volume<u32> {
            match self {
                Fixture::U32(v) => v,
                _ => panic!("fixture is not u32"),
            }
        }
        fn f64(self) -> Volume<f64> {
            match self {
                Fixture::F64(v) => v,
                _ => panic!("fixture is not f64"),
            }
        }
    }

    fn read_fixture(name: &str) -> Fixture {
        let path: PathBuf = [env!("CARGO_MANIFEST_DIR"), "tests", "fixtures", name]
            .iter()
            .collect();
        let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        assert_eq!(
            &bytes[0..4],
            b"TS3D",
            "bad fixture magic in {}",
            path.display()
        );
        assert_eq!(bytes[4], 1, "unsupported fixture version");
        let code = bytes[5];
        let ndim = bytes[6] as usize;
        assert!(
            ndim == 1 || ndim == 2 || ndim == 3,
            "unsupported fixture rank {ndim}"
        );
        let mut dims = [1usize; 3];
        let mut off = 7;
        for i in 0..ndim {
            let mut raw = [0u8; 8];
            raw.copy_from_slice(&bytes[off..off + 8]);
            dims[3 - ndim + i] = u64::from_le_bytes(raw) as usize;
            off += 8;
        }
        let payload = &bytes[off..];
        let count = dims[0] * dims[1] * dims[2];
        match code {
            b'u' => {
                assert_eq!(payload.len(), count * 2);
                let data = payload
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|c| u16::from_le_bytes(*c))
                    .collect();
                Fixture::U16(Volume::new(dims, data).unwrap())
            }
            b'U' => {
                assert_eq!(payload.len(), count * 4);
                let data = payload
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| u32::from_le_bytes(*c))
                    .collect();
                Fixture::U32(Volume::new(dims, data).unwrap())
            }
            b'f' => {
                assert_eq!(payload.len(), count * 8);
                let data = payload
                    .as_chunks::<8>()
                    .0
                    .iter()
                    .map(|c| f64::from_le_bytes(*c))
                    .collect();
                Fixture::F64(Volume::new(dims, data).unwrap())
            }
            other => panic!("unsupported fixture dtype {other}"),
        }
    }

    fn assert_f64_close(a: &Volume<f64>, b: &Volume<f64>) {
        assert_eq!(a.shape, b.shape);
        let mut worst = 0.0f64;
        let mut at = 0;
        for (i, (x, y)) in a.data.iter().zip(&b.data).enumerate() {
            let d = (x - y).abs();
            if d > worst {
                worst = d;
                at = i;
            }
        }
        assert!(
            worst <= TOL,
            "max |diff| {worst:e} at {at} (ours {}, ref {})",
            a.data[at],
            b.data[at]
        );
    }

    #[test]
    fn zoom_linear_matches_scipy_order1_uint16() {
        let input = read_fixture("zoom_input.u16").u16();
        let expected = read_fixture("zoom_expected.u16").u16();
        let got = super::zoom_linear_u16(&input, [6.92 * 0.5, 0.5, 0.5]).unwrap();
        assert_eq!(got, expected);
    }

    #[test]
    fn gaussian_filter_matches_scipy_nearest_mode() {
        let input = read_fixture("gauss_input.f64").f64();
        let expected = read_fixture("gauss_nearest.f64").f64();
        let got = super::gaussian_filter(&input, [0.5, 0.5, 0.5], 4.0, super::BorderMode::Nearest);
        assert_f64_close(&got, &expected);
    }

    #[test]
    fn gaussian_filter_matches_scipy_reflect_mode() {
        let input = read_fixture("gauss_input.f64").f64();
        let expected = read_fixture("gauss_reflect.f64").f64();
        let got = super::gaussian_filter(&input, [2.0, 3.2, 5.12], 4.0, super::BorderMode::Reflect);
        assert_f64_close(&got, &expected);
    }

    #[test]
    fn median_filter_matches_scipy_reflect_mode() {
        let input = read_fixture("median_input.u32").u32();
        let expected = read_fixture("median_expected.u32").u32();
        let mut got = input.clone();
        super::median_filter_u32(&mut got, [1, 5, 5]);
        assert_eq!(got, expected);
    }

    #[test]
    fn median_filter_even_size_uses_upper_median_of_exact_window() {
        // scipy.ndimage.median_filter([1, 2, 3, 4, 5], size=4) == [2, 2, 3, 4, 5]:
        // reflect borders, window x-2..x+1, and the *upper* median of four.
        let line = || Volume::new([1, 1, 5], vec![1u32, 2, 3, 4, 5]).unwrap();
        assert_eq!(
            super::median_filter_generic(&line(), [1, 1, 4]).data,
            vec![2, 2, 3, 4, 5]
        );
        // scipy.ndimage.median_filter([1, 2, 3, 4, 5], size=2) == [1, 2, 3, 4, 5].
        assert_eq!(
            super::median_filter_generic(&line(), [1, 1, 2]).data,
            vec![1, 2, 3, 4, 5]
        );
    }

    #[test]
    fn median_filter_even_sizes_match_scipy_reflect_mode() {
        let input = read_fixture("median_even_input.u32").u32();
        for (name, size) in [
            ("median_size2_expected.u32", [1, 2, 2]),
            ("median_size4_expected.u32", [1, 4, 4]),
            ("median_size_2x4x2_expected.u32", [2, 4, 2]),
        ] {
            let expected = read_fixture(name).u32();
            let got = super::median_filter_generic(&input, size);
            assert_eq!(got, expected, "median size {size:?} against scipy");
        }
    }

    #[test]
    fn gaussian_filter_truncates_to_uint16_like_scipy_at_integers() {
        // SciPy accumulates the symmetric kernel centre-first, then sample pairs
        // outside-in.  That order decides whether a near-integer result lands on
        // the integer or one ULP below it, and `astype('uint16')` truncates.
        for (name, sigma, value) in [
            ("gauss_const_sigma_0p5_1000_expected.u16", 0.5, 1000.0),
            ("gauss_const_sigma_0p75_max_expected.u16", 0.75, 65535.0),
            ("gauss_const_sigma_0p26_one_expected.u16", 0.26, 1.0),
        ] {
            let expected = read_fixture(name).u16();
            let input = Volume::filled(expected.shape, value).unwrap();
            let got = super::gaussian_filter(&input, [sigma; 3], 4.0, super::BorderMode::Nearest);
            let got: Vec<u16> = got.data.iter().map(|&v| super::trunc_u16(v)).collect();
            assert_eq!(got, expected.data, "constant {value} at sigma {sigma}");
        }
    }

    #[test]
    fn gaussian_kernel_weights_match_numpy_bits() {
        // scipy normalises the kernel with numpy.sum (pairwise) and exponent
        // with numpy.exp (SVML), so every weight has to match bit for bit.
        let sigmas = read_fixture("gauss_kernel_sigmas.f64").f64();
        let expected = read_fixture("gauss_kernel_weights.f64").f64();
        let mut at = 0usize;
        for (i, &sigma) in sigmas.data.iter().enumerate() {
            let radius = (4.0 * sigma + 0.5) as isize;
            let got = super::gaussian_weights(sigma, radius);
            assert_eq!(
                got.len(),
                (2 * radius + 1) as usize,
                "sigma {sigma} kernel size"
            );
            for (k, &w) in got.iter().enumerate() {
                let want = expected.data[at];
                at += 1;
                assert_eq!(
                    w.to_bits(),
                    want.to_bits(),
                    "sigma {sigma} [{i}] weight {k}: {w:?} but numpy gives {want:?}"
                );
            }
        }
        assert_eq!(
            at,
            expected.data.len(),
            "fixture covers every recorded weight"
        );
    }

    #[test]
    fn nuclei_constant_volume_threshold_matches_scipy_quantisation() {
        // 264-case probe: constant [3, 5, 7] volumes, all-ones cell labels,
        // median_size 1, z_ratio 1, and threshold == the constant value.  The
        // recorded uint16 volumes are the truncated Gaussian output of the
        // pinned scipy/scikit-image stack, so a single ULP of kernel error
        // shows up as a flipped voxel.
        let quantised = read_fixture("gauss_const_blurred.u16").u16();
        let sigmas = read_fixture("gauss_const_sigmas.f64").f64();
        let values = read_fixture("gauss_const_values.u16").u16();
        let cases = sigmas.data.len();
        assert_eq!(values.data.len(), cases);
        assert_eq!(quantised.data.len(), cases * 3 * 5 * 7);
        let cells = Volume::filled([3, 5, 7], 1u32).unwrap();

        for case in 0..cases {
            let sigma = sigmas.data[case];
            let value = values.data[case];
            let expected = &quantised.data[case * 105..(case + 1) * 105];

            let input = Volume::filled([3, 5, 7], value).unwrap();
            let blurred = super::gaussian_filter(
                &Volume::new(input.shape, input.data.iter().map(|&v| v as f64).collect()).unwrap(),
                [sigma; 3],
                4.0,
                super::BorderMode::Nearest,
            );
            let ours: Vec<u16> = blurred.data.iter().map(|&v| super::trunc_u16(v)).collect();
            assert_eq!(
                ours, expected,
                "sigma {sigma} constant {value}: uint16 Gaussian output"
            );

            let mask: Vec<u32> = expected.iter().map(|&q| u32::from(q >= value)).collect();
            let params = NucleusParams {
                gaussian_sigma: sigma,
                threshold: value,
                median_size: 1,
                z_ratio: 1.0,
            };
            let got = super::segment_nuclei(&input, &cells, &params).unwrap();
            assert_eq!(
                got.data, mask,
                "sigma {sigma} constant {value}: thresholded foreground"
            );
        }
    }

    #[test]
    fn dog_peaks_compare_only_adjacent_scale_layers() {
        // skimage's peak_local_max runs on the (z, y, x, scale) cube with a 3-wide
        // footprint, so a layer competes with scale-1..scale+1, not every layer.
        let values = [5.0, 10.0, 10.0, 9.0, 12.0, 12.0];
        let dogs: Vec<Volume<f64>> = values
            .iter()
            .map(|&v| Volume::new([1, 1, 1], vec![v]).unwrap())
            .collect();
        let peaks = super::dog_peaks(&dogs, 1.0);
        assert_eq!(
            peaks,
            vec![
                (12.0, 0, 0, 0, 4),
                (12.0, 0, 0, 0, 5),
                (10.0, 0, 0, 0, 1),
                (10.0, 0, 0, 0, 2)
            ]
        );
    }

    #[test]
    fn dog_peaks_match_skimage_peak_local_max_on_scale_space() {
        let input = read_fixture("dog_peaks_input.f64").f64();
        let expected = read_fixture("dog_peaks_expected.u32").u32();
        let sigmas = super::dog_sigmas(1.0, 8.0, 1.6);
        let dogs = super::dog_layers(&input, &sigmas, 1.6);
        let got: Vec<u32> = super::dog_peaks(&dogs, 0.001)
            .into_iter()
            .flat_map(|(_, z, y, x, s)| [z as u32, y as u32, x as u32, s as u32])
            .collect();
        assert_eq!(got, expected.data);
    }

    #[test]
    fn maximum_filter_matches_scipy_reflect_mode() {
        let input = read_fixture("max_input.u32").u32();
        let expected = read_fixture("max_expected.u32").u32();
        let got = super::maximum_filter_u32(&input, [1, 3, 3]);
        assert_eq!(got, expected);
    }

    #[test]
    fn impose_minima_matches_skimage_reconstruction() {
        let input = read_fixture("watershed_input.u16").u16();
        let markers = read_fixture("watershed_markers.u32").u32();
        let expected = read_fixture("watershed_expected.f64").f64();
        let got = super::impose_minima(&input, &markers);
        assert_f64_close(&got, &expected);
    }

    #[test]
    fn watershed_matches_skimage_fifo_tie_handling() {
        let image = read_fixture("watershed_expected.f64").f64();
        let markers = read_fixture("watershed_markers.u32").u32();
        let expected = read_fixture("watershed_expected.u32").u32();
        let got = super::watershed_6(&image, &markers).unwrap();
        assert_eq!(got, expected);
    }

    #[test]
    fn fill_label_holes_matches_python_reference() {
        let input = read_fixture("holes_input.u32").u32();
        let expected = read_fixture("holes_expected.u32").u32();
        let got = super::fill_label_holes(input);
        assert_eq!(got, expected);
    }

    #[test]
    fn remove_seeds_keeps_an_unmerged_seed_zero() {
        let membrane = Volume::filled([3, 3, 3], 0u16).unwrap();
        let seeds = vec![[1, 1, 1]];
        let kept = super::remove_seeds_round(&membrane, &seeds, 50.0, 25.0, 6.92);
        assert_eq!(kept, seeds);
    }

    #[test]
    fn seed_candidates_match_corrected_python_reference() {
        let nuclei = read_fixture("seed_nuclei.u16").u16();
        let membrane = read_fixture("seed_membrane.u16").u16();
        let expected = read_fixture("seed_candidates.u32").u32();
        let p = SeedParams::default();
        let got = super::detect_seed_candidates(&nuclei, &membrane, &p).unwrap();
        let expected: Vec<Seed> = expected
            .data
            .as_chunks::<3>()
            .0
            .iter()
            .map(|c| [c[0] as usize, c[1] as usize, c[2] as usize])
            .collect();
        assert_eq!(got, expected);
    }

    #[test]
    fn detect_seeds_matches_corrected_python_reference() {
        let nuclei = read_fixture("seed_nuclei.u16").u16();
        let membrane = read_fixture("seed_membrane.u16").u16();
        let expected = read_fixture("seed_expected.u32").u32();
        let stack = ImageStack {
            membrane,
            nuclei: Some(nuclei),
        };
        let got = detect_seeds(&stack, &SeedParams::default()).unwrap();
        let expected: Vec<Seed> = expected
            .data
            .as_chunks::<3>()
            .0
            .iter()
            .map(|c| [c[0] as usize, c[1] as usize, c[2] as usize])
            .collect();
        assert_eq!(got, expected);
    }

    #[test]
    fn segment_cells_matches_python_reference_pipeline() {
        let membrane = read_fixture("cells_input.u16").u16();
        let seeds_raw = read_fixture("cells_seeds.u32").u32();
        let expected = read_fixture("cells_expected.u32").u32();
        let seeds: Vec<Seed> = seeds_raw
            .data
            .as_chunks::<3>()
            .0
            .iter()
            .map(|c| [c[0] as usize, c[1] as usize, c[2] as usize])
            .collect();
        let got = segment_cells(&membrane, &seeds, &CellParams::default()).unwrap();
        assert_eq!(got, expected);
    }

    #[test]
    fn segment_nuclei_matches_python_reference_pipeline() {
        let nuclei = read_fixture("nuclei_input.u16").u16();
        let cells = read_fixture("nuclei_cells.u32").u32();
        let expected = read_fixture("nuclei_expected.u32").u32();
        let got = segment_nuclei(&nuclei, &cells, &NucleusParams::default()).unwrap();
        assert_eq!(got, expected);
    }
}
