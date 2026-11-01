//! TIFF input/output and demo data.
//!
//! The loader accepts ordinary single-channel Z-Y-X grayscale stacks as well
//! as ImageJ, OME-TIFF and tifffile-JSON hyperstacks with a `Z` and `C` axis.
//! Multi-sample (RGB/RGBA) pages are deliberately rejected: silently choosing a
//! sample plane would mix unrelated data.  Channel 0 is nuclei and channel 1 is
//! membrane, matching the original Python application.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Seek, Write};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use tiff::decoder::{Decoder, DecodingResult};
use tiff::encoder::colortype::{Gray16, Gray32};
use tiff::encoder::TiffEncoder;
use tiff::tags::Tag;
use tiff::ColorType;

use crate::volume::{ImageStack, Volume};

// ---------------------------------------------------------------------------
// TIFF decoding
// ---------------------------------------------------------------------------

enum RawPlane {
    U8(Vec<u8>),
    U16(Vec<u16>),
    U32(Vec<u32>),
}

impl RawPlane {
    fn len(&self) -> usize {
        match self {
            RawPlane::U8(v) => v.len(),
            RawPlane::U16(v) => v.len(),
            RawPlane::U32(v) => v.len(),
        }
    }

    fn to_u16(&self) -> Result<Vec<u16>> {
        match self {
            RawPlane::U8(v) => Ok(v.iter().map(|&x| x as u16).collect()),
            RawPlane::U16(v) => Ok(v.clone()),
            RawPlane::U32(_) => {
                bail!("unsupported 32-bit TIFF image data (expected 8 or 16-bit grayscale)")
            }
        }
    }

    fn to_u32(&self) -> Result<Vec<u32>> {
        match self {
            RawPlane::U8(v) => Ok(v.iter().map(|&x| x as u32).collect()),
            RawPlane::U16(v) => Ok(v.iter().map(|&x| x as u32).collect()),
            RawPlane::U32(v) => Ok(v.clone()),
        }
    }
}

struct RawStack {
    width: usize,
    height: usize,
    planes: Vec<RawPlane>,
    description: Option<String>,
}

fn read_raw_stack(path: &Path) -> Result<RawStack> {
    let file =
        File::open(path).with_context(|| format!("failed to open TIFF file {}", path.display()))?;
    let mut decoder = Decoder::new(BufReader::new(file))
        .with_context(|| format!("failed to read TIFF header from {}", path.display()))?;

    let mut planes = Vec::new();
    let mut width = None;
    let mut height = None;
    let mut description: Option<String> = None;

    loop {
        let (w, h) = decoder
            .dimensions()
            .with_context(|| format!("failed to read TIFF dimensions from {}", path.display()))?;
        let (w, h) = (w as usize, h as usize);
        match (width, height) {
            (None, None) => {
                width = Some(w);
                height = Some(h);
            }
            (Some(ew), Some(eh)) if (ew, eh) != (w, h) => {
                bail!(
                    "TIFF pages have inconsistent spatial dimensions: {}x{} and {}x{}",
                    ew,
                    eh,
                    w,
                    h
                );
            }
            _ => {}
        }

        if description.is_none() {
            if let Ok(text) = decoder.get_tag_ascii_string(Tag::ImageDescription) {
                if !text.trim().is_empty() {
                    description = Some(text);
                }
            }
        }

        let color = decoder
            .colortype()
            .with_context(|| format!("failed to read TIFF colour type from {}", path.display()))?;
        match color {
            ColorType::Gray(bits @ (8 | 16 | 32)) => {
                let decoded = decoder.read_image().with_context(|| {
                    format!("failed to decode TIFF image data from {}", path.display())
                })?;
                let plane = match (bits, decoded) {
                    (8, DecodingResult::U8(v)) => RawPlane::U8(v),
                    (16, DecodingResult::U16(v)) => RawPlane::U16(v),
                    (32, DecodingResult::U32(v)) => RawPlane::U32(v),
                    (_, DecodingResult::I8(_))
                    | (_, DecodingResult::I16(_))
                    | (_, DecodingResult::I32(_))
                    | (_, DecodingResult::I64(_)) => {
                        bail!(
                            "unsupported signed-integer TIFF image data in {}; expected unsigned grayscale",
                            path.display()
                        )
                    }
                    (_, DecodingResult::F16(_))
                    | (_, DecodingResult::F32(_))
                    | (_, DecodingResult::F64(_)) => {
                        bail!(
                            "unsupported float TIFF image data in {}; only integer grayscale TIFF files are supported",
                            path.display()
                        )
                    }
                    (_, DecodingResult::U64(_)) => {
                        bail!(
                            "unsupported 64-bit TIFF image data in {}; only 8/16-bit images and 16/32-bit labels are supported",
                            path.display()
                        )
                    }
                    (_, other) => {
                        bail!(
                            "unsupported TIFF sample type {:?} in {} (expected unsigned 8/16/32-bit grayscale)",
                            sample_kind(&other),
                            path.display()
                        )
                    }
                };
                let expected = w.checked_mul(h);
                if expected != Some(plane.len()) {
                    bail!(
                        "TIFF page in {} decodes to {} values but dimensions {}x{} need {}",
                        path.display(),
                        plane.len(),
                        w,
                        h,
                        expected.unwrap_or(0)
                    );
                }
                planes.push(plane);
            }
            ColorType::Gray(bits) => {
                bail!(
                    "unsupported {}-bit grayscale TIFF in {} (expected 8 or 16-bit images)",
                    bits,
                    path.display()
                )
            }
            other => {
                bail!(
                    "unsupported TIFF pixel layout {:?} in {}: multi-sample or colour pages are ambiguous; expected single-channel grayscale",
                    other,
                    path.display()
                )
            }
        }

        if !decoder.more_images() {
            break;
        }
        decoder
            .next_image()
            .with_context(|| format!("failed to read next TIFF page from {}", path.display()))?;
    }

    if planes.is_empty() {
        bail!("TIFF file {} contains no image pages", path.display());
    }
    Ok(RawStack {
        width: width.unwrap_or(0),
        height: height.unwrap_or(0),
        planes,
        description,
    })
}

fn sample_kind(result: &DecodingResult) -> &'static str {
    match result {
        DecodingResult::U8(_) => "u8",
        DecodingResult::U16(_) => "u16",
        DecodingResult::U32(_) => "u32",
        DecodingResult::U64(_) => "u64",
        DecodingResult::F16(_) => "f16",
        DecodingResult::F32(_) => "f32",
        DecodingResult::F64(_) => "f64",
        DecodingResult::I8(_) => "i8",
        DecodingResult::I16(_) => "i16",
        DecodingResult::I32(_) => "i32",
        DecodingResult::I64(_) => "i64",
    }
}

// ---------------------------------------------------------------------------
// Metadata layout
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Axis {
    Z,
    C,
    T,
    Y,
    X,
}

impl Axis {
    fn from_char(c: char) -> Option<Self> {
        match c.to_ascii_uppercase() {
            'Z' => Some(Axis::Z),
            'C' => Some(Axis::C),
            'T' => Some(Axis::T),
            'Y' => Some(Axis::Y),
            'X' => Some(Axis::X),
            _ => None,
        }
    }
}

/// Declared stack layout.  `axes` is ordered slowest-to-fastest and always ends
/// in `Y, X` so that one TIFF page is one spatial plane.
struct Layout {
    z: usize,
    c: usize,
    width: usize,
    height: usize,
    /// Non-spatial axes in slowest-to-fastest order with their sizes.
    plane_axes: Vec<(Axis, usize)>,
}

impl Layout {
    fn from_axes(
        path: &Path,
        axes: &[Axis],
        sizes: &[usize],
        page_count: usize,
        width: usize,
        height: usize,
    ) -> Result<Self> {
        if axes.len() != sizes.len() {
            bail!(
                "TIFF metadata in {} has {} axes but {} sizes",
                path.display(),
                axes.len(),
                sizes.len()
            );
        }
        let n = axes.len();
        if n < 2 || axes[n - 2] != Axis::Y || axes[n - 1] != Axis::X {
            bail!(
                "unsupported TIFF axis order {:?} in {}: pages must be Y-X planes",
                axes,
                path.display()
            );
        }
        let mut seen = Vec::new();
        let mut z = 1usize;
        let mut c = 1usize;
        let mut t = 1usize;
        let mut y = 1usize;
        let mut x = 1usize;
        for (&axis, &size) in axes.iter().zip(sizes) {
            if size == 0 {
                bail!(
                    "TIFF metadata in {} declares a zero-length axis",
                    path.display()
                );
            }
            if seen.contains(&axis) {
                bail!(
                    "TIFF metadata in {} repeats axis {:?}",
                    path.display(),
                    axis
                );
            }
            seen.push(axis);
            match axis {
                Axis::Z => z = size,
                Axis::C => c = size,
                Axis::T => t = size,
                Axis::Y => y = size,
                Axis::X => x = size,
            }
        }
        if t != 1 {
            bail!(
                "unsupported TIFF metadata in {}: time series (T={}) cannot be loaded as one volume",
                path.display(),
                t
            );
        }
        if y != height || x != width {
            bail!(
                "TIFF metadata in {} declares YX={}x{}, but pages are {}x{}",
                path.display(),
                y,
                x,
                height,
                width
            );
        }
        let planes = z
            .checked_mul(c)
            .and_then(|v| v.checked_mul(t))
            .ok_or_else(|| anyhow::anyhow!("TIFF plane count overflows for {}", path.display()))?;
        if planes != page_count {
            bail!(
                "TIFF metadata in {} declares {}x{}x{} ({} planes), but the file contains {} pages",
                path.display(),
                z,
                c,
                t,
                planes,
                page_count
            );
        }
        let plane_axes = axes[..n - 2]
            .iter()
            .copied()
            .zip(sizes[..n - 2].iter().copied())
            .collect();
        Ok(Self {
            z,
            c,
            width,
            height,
            plane_axes,
        })
    }

    fn page_to_zc(&self, mut page: usize) -> (usize, usize) {
        let mut z = 0usize;
        let mut c = 0usize;
        for &(axis, size) in self.plane_axes.iter().rev() {
            let coordinate = page % size;
            page /= size;
            match axis {
                Axis::Z => z = coordinate,
                Axis::C => c = coordinate,
                Axis::T => {}
                Axis::Y | Axis::X => unreachable!(),
            }
        }
        (z, c)
    }
}

fn parse_metadata(path: &Path, raw: &RawStack) -> Result<Option<Layout>> {
    let Some(description) = raw.description.as_deref() else {
        return Ok(None);
    };
    let text = description.trim();
    if text.is_empty() {
        return Ok(None);
    }
    let page_count = raw.planes.len();

    if text.starts_with('{') {
        let value: serde_json::Value = serde_json::from_str(text)
            .with_context(|| format!("invalid JSON TIFF metadata in {}", path.display()))?;
        return json_layout(path, &value, page_count, raw.width, raw.height).map(Some);
    }
    if text.contains("<OME") || text.starts_with("<?xml") {
        return ome_layout(path, text, page_count, raw.width, raw.height).map(Some);
    }
    if text.contains("ImageJ=") {
        return imagej_layout(path, text, page_count, raw.width, raw.height).map(Some);
    }
    // Unrelated free-form description: treat the pages as a plain Z stack.
    Ok(None)
}

fn json_layout(
    path: &Path,
    value: &serde_json::Value,
    page_count: usize,
    width: usize,
    height: usize,
) -> Result<Layout> {
    let shape = value
        .get("shape")
        .and_then(|v| v.as_array())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "unsupported TIFF metadata in {}: JSON description has no shape array",
                path.display()
            )
        })?;
    let sizes: Vec<usize> = shape
        .iter()
        .map(|v| {
            v.as_u64()
                .map(|n| n as usize)
                .ok_or_else(|| anyhow::anyhow!("invalid dimension in TIFF metadata shape"))
        })
        .collect::<Result<_>>()?;

    let axes: Vec<Axis> = if let Some(axes) = value.get("axes").and_then(|v| v.as_str()) {
        if axes.chars().count() != sizes.len() {
            bail!(
                "TIFF metadata in {} has axes {:?} but shape {:?}",
                path.display(),
                axes,
                sizes
            );
        }
        axes.chars()
            .map(|c| {
                Axis::from_char(c).ok_or_else(|| {
                    anyhow::anyhow!(
                        "unsupported TIFF metadata in {}: axis {:?} in {:?}",
                        path.display(),
                        c,
                        axes
                    )
                })
            })
            .collect::<Result<_>>()?
    } else {
        match sizes.len() {
            2 => vec![Axis::Y, Axis::X],
            3 => vec![Axis::Z, Axis::Y, Axis::X],
            4 => vec![Axis::Z, Axis::C, Axis::Y, Axis::X],
            n => bail!(
                "unsupported TIFF metadata in {}: cannot infer axes for a {}-dimensional shape {:?}",
                path.display(),
                n,
                sizes
            ),
        }
    };

    Layout::from_axes(path, &axes, &sizes, page_count, width, height)
}

fn ome_layout(
    path: &Path,
    text: &str,
    page_count: usize,
    width: usize,
    height: usize,
) -> Result<Layout> {
    let pixels = text
        .find("<Pixels")
        .map(|start| &text[start..])
        .and_then(|tail| tail.find('>').map(|end| &tail[..=end]))
        .ok_or_else(|| {
            anyhow::anyhow!(
                "unsupported OME-TIFF in {}: missing <Pixels> metadata",
                path.display()
            )
        })?;
    let dimension_order = xml_attr(pixels, "DimensionOrder").ok_or_else(|| {
        anyhow::anyhow!(
            "unsupported OME-TIFF in {}: missing Pixels DimensionOrder",
            path.display()
        )
    })?;
    let size_x = xml_attr_usize(pixels, "SizeX")?;
    let size_y = xml_attr_usize(pixels, "SizeY")?;
    let size_c = xml_attr_usize(pixels, "SizeC")?;
    let size_z = xml_attr_usize(pixels, "SizeZ")?;
    let size_t = xml_attr_usize(pixels, "SizeT").unwrap_or(1);
    if size_x != width || size_y != height {
        bail!(
            "OME-TIFF metadata in {} declares YX={}x{}, but pages are {}x{}",
            path.display(),
            size_y,
            size_x,
            height,
            width
        );
    }
    let sizes: Vec<usize> = dimension_order
        .chars()
        .rev()
        .map(|c| match c.to_ascii_uppercase() {
            'X' => Ok(size_x),
            'Y' => Ok(size_y),
            'C' => Ok(size_c),
            'Z' => Ok(size_z),
            'T' => Ok(size_t),
            other => bail!(
                "unsupported OME-TIFF dimension {:?} in DimensionOrder {:?} ({})",
                other,
                dimension_order,
                path.display()
            ),
        })
        .collect::<Result<_>>()?;
    // OME dimensions are ordered fastest-to-slowest; TIFF pages follow the
    // slowest-to-fastest (C order) convention used by tifffile.
    let axes: Vec<Axis> = dimension_order
        .chars()
        .rev()
        .map(|c| Axis::from_char(c).expect("validated above"))
        .collect();
    Layout::from_axes(path, &axes, &sizes, page_count, width, height)
}

fn imagej_layout(
    path: &Path,
    text: &str,
    page_count: usize,
    width: usize,
    height: usize,
) -> Result<Layout> {
    let mut fields = HashMap::new();
    for line in text.lines() {
        if let Some((key, value)) = line.split_once('=') {
            fields.insert(key.trim(), value.trim());
        }
    }
    let images = fields.get("images").and_then(|s| s.parse::<usize>().ok());
    let channels = fields.get("channels").and_then(|s| s.parse::<usize>().ok());
    let slices = fields.get("slices").and_then(|s| s.parse::<usize>().ok());
    let frames = fields
        .get("frames")
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(1);

    let (c, z) = match (channels, slices, images) {
        (Some(c), Some(z), _) => (c, z),
        (Some(c), None, Some(images)) if c > 0 => (c, images / c),
        (None, Some(z), _) => (1, z),
        (None, None, Some(images)) => (1, images),
        _ => {
            bail!(
                "unsupported ImageJ TIFF metadata in {}: no channel/slice counts",
                path.display()
            )
        }
    };
    if frames != 1 {
        bail!(
            "unsupported ImageJ TIFF in {}: time series with frames={} cannot be loaded as one volume",
            path.display(),
            frames
        );
    }
    let axes = if let Some(axes) = fields.get("axes") {
        let axes = axes.to_string();
        if axes.chars().count() != 4 {
            bail!(
                "unsupported ImageJ TIFF axes {:?} in {} (expected four axes including Z,C,Y,X)",
                axes,
                path.display()
            );
        }
        axes.chars()
            .map(|ch| {
                Axis::from_char(ch).ok_or_else(|| {
                    anyhow::anyhow!(
                        "unsupported ImageJ TIFF axis {:?} in {}",
                        ch,
                        path.display()
                    )
                })
            })
            .collect::<Result<Vec<_>>>()?
    } else {
        vec![Axis::Z, Axis::C, Axis::Y, Axis::X]
    };
    // ImageJ writes pages with X varying fastest, so the axes given in its
    // metadata are already in slowest-to-fastest order.
    let sizes: Vec<usize> = axes
        .iter()
        .map(|axis| match axis {
            Axis::Z => z,
            Axis::C => c,
            Axis::Y => height,
            Axis::X => width,
            Axis::T => 1,
        })
        .collect();
    Layout::from_axes(path, &axes, &sizes, page_count, width, height)
}

fn xml_attr<'a>(tag: &'a str, name: &str) -> Option<&'a str> {
    let needle = format!("{name}=\"");
    let start = tag.find(&needle)? + needle.len();
    let rest = &tag[start..];
    let end = rest.find('"')?;
    Some(&rest[..end])
}

fn xml_attr_usize(tag: &str, name: &str) -> Result<usize> {
    let value = xml_attr(tag, name)
        .ok_or_else(|| anyhow::anyhow!("missing OME Pixels attribute {name}"))?;
    value
        .parse::<usize>()
        .with_context(|| format!("invalid OME Pixels attribute {name}={value:?}"))
}

// ---------------------------------------------------------------------------
// Public loading
// ---------------------------------------------------------------------------

pub fn load_tiff(path: &Path, dual_channel: bool) -> Result<ImageStack> {
    let raw = read_raw_stack(path)?;
    let layout = parse_metadata(path, &raw)?;

    let Some(layout) = layout else {
        if dual_channel {
            bail!(
                "cannot determine 2 channels in {}: no channel metadata (ambiguous single-channel stack with {} pages); expected ImageJ, OME or tifffile shape metadata",
                path.display(),
                raw.planes.len()
            );
        }
        // No metadata: every page is one Z slice of a single-channel stack.
        let shape = [raw.planes.len(), raw.height, raw.width];
        let mut data = Vec::with_capacity(raw.planes[0].len() * shape[0]);
        for plane in &raw.planes {
            data.extend(plane.to_u16()?);
        }
        return Ok(ImageStack {
            membrane: Volume::new(shape, data)?,
            nuclei: None,
        });
    };

    if dual_channel {
        if layout.c != 2 {
            bail!(
                "dual-channel loading requires exactly 2 channels (nuclei + membrane), but metadata in {} reports {} channel(s)",
                path.display(),
                layout.c
            );
        }
    } else if layout.c != 1 {
        bail!(
            "file {} contains {} channels; enable dual-channel loading to read nuclei and membrane",
            path.display(),
            layout.c
        );
    }

    let shape = [layout.z, layout.height, layout.width];
    let plane_len = layout
        .width
        .checked_mul(layout.height)
        .ok_or_else(|| anyhow::anyhow!("TIFF plane dimensions overflow"))?;
    let volume_len = plane_len
        .checked_mul(layout.z)
        .ok_or_else(|| anyhow::anyhow!("TIFF volume dimensions overflow"))?;
    let mut channel_data: Vec<Vec<u16>> = (0..layout.c).map(|_| vec![0u16; volume_len]).collect();

    for (page, raw_plane) in raw.planes.iter().enumerate() {
        let (z, c) = layout.page_to_zc(page);
        if z >= layout.z || c >= layout.c {
            bail!(
                "TIFF page {} maps outside the declared ZC layout in {}",
                page,
                path.display()
            );
        }
        let values = raw_plane.to_u16()?;
        let start = z * plane_len;
        channel_data[c][start..start + plane_len].copy_from_slice(&values);
    }

    if dual_channel {
        let nuclei = Volume::new(shape, channel_data.remove(0))?;
        let membrane = Volume::new(shape, channel_data.remove(0))?;
        Ok(ImageStack {
            membrane,
            nuclei: Some(nuclei),
        })
    } else {
        let membrane = Volume::new(shape, channel_data.remove(0))?;
        Ok(ImageStack {
            membrane,
            nuclei: None,
        })
    }
}

pub fn load_labels(path: &Path) -> Result<Volume<u32>> {
    let raw = read_raw_stack(path)?;
    let shape = [raw.planes.len(), raw.height, raw.width];
    let mut data = Vec::with_capacity(raw.planes.iter().map(|p| p.len()).sum());
    for plane in &raw.planes {
        data.extend(plane.to_u32()?);
    }
    Volume::new(shape, data)
}

// ---------------------------------------------------------------------------
// Export
// ---------------------------------------------------------------------------

pub fn export_labels(
    input_path: &Path,
    labels: &Volume<u32>,
    nuclei: bool,
) -> Result<(PathBuf, PathBuf)> {
    let stem = input_path.file_stem().ok_or_else(|| {
        anyhow::anyhow!(
            "cannot export labels: input path {} has no file stem",
            input_path.display()
        )
    })?;
    let parent = input_path.parent().unwrap_or_else(|| Path::new("."));
    let suffix = if nuclei { "_nuc_lbl" } else { "_lbl" };
    // Append the extension to the whole file name instead of replacing it:
    // `with_extension` would turn "sample.v1" into "sample.tif".
    let mut tif_name = stem.to_os_string();
    tif_name.push(suffix);
    let mut csv_name = tif_name.clone();
    tif_name.push(".tif");
    csv_name.push(".csv");
    let tif_path = parent.join(&tif_name);
    let csv_path = parent.join(&csv_name);

    if same_file(input_path, &tif_path) || same_file(input_path, &csv_path) {
        bail!(
            "refusing to overwrite input file {} with label output",
            input_path.display()
        );
    }

    let write_tif = || -> Result<()> {
        let mut writer = BufWriter::new(File::create(&tif_path)?);
        write_label_tiff(&mut writer, labels)
    };
    write_tif().with_context(|| format!("failed to write label TIFF {}", tif_path.display()))?;
    write_label_csv(&csv_path, labels)
        .with_context(|| format!("failed to write label CSV {}", csv_path.display()))?;
    Ok((tif_path, csv_path))
}

fn same_file(a: &Path, b: &Path) -> bool {
    match (std::fs::canonicalize(a), std::fs::canonicalize(b)) {
        (Ok(a), Ok(b)) => a == b,
        _ => a == b,
    }
}

fn write_label_tiff<W: Write + Seek>(writer: &mut W, labels: &Volume<u32>) -> Result<()> {
    let [depth, height, width] = labels.shape;
    let max_label = labels.data.iter().copied().max().unwrap_or(0);
    let plane_len = width * height;
    {
        let mut encoder = TiffEncoder::new(&mut *writer)?;
        if max_label <= u16::MAX as u32 {
            for z in 0..depth {
                let plane: Vec<u16> = labels.data[z * plane_len..(z + 1) * plane_len]
                    .iter()
                    .map(|&v| v as u16)
                    .collect();
                encoder.write_image::<Gray16>(width as u32, height as u32, &plane)?;
            }
        } else {
            for z in 0..depth {
                encoder.write_image::<Gray32>(
                    width as u32,
                    height as u32,
                    &labels.data[z * plane_len..(z + 1) * plane_len],
                )?;
            }
        }
    }
    // The encoder only finalises the directory when dropped, so flush after
    // that scope or the trailing write errors would be lost.
    writer.flush()?;
    Ok(())
}

fn write_label_csv(path: &Path, labels: &Volume<u32>) -> Result<()> {
    let mut stats: HashMap<u32, (u64, u64, u64, u64)> = HashMap::new();
    let [depth, height, width] = labels.shape;
    for z in 0..depth {
        for y in 0..height {
            for x in 0..width {
                let label = labels.data[(z * height + y) * width + x];
                if label == 0 {
                    continue;
                }
                let acc = stats.entry(label).or_insert((0, 0, 0, 0));
                acc.0 += z as u64;
                acc.1 += y as u64;
                acc.2 += x as u64;
                acc.3 += 1;
            }
        }
    }
    let mut rows: Vec<(u32, (u64, u64, u64, u64))> = stats.into_iter().collect();
    rows.sort_by_key(|(label, _)| *label);

    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    writeln!(writer, "Cell,CZ,CY,CX,Volume (vox)")?;
    for (label, (sz, sy, sx, count)) in rows {
        let n = count as f64;
        writeln!(
            writer,
            "{},{},{},{},{}",
            label,
            csv_float(sz as f64 / n),
            csv_float(sy as f64 / n),
            csv_float(sx as f64 / n),
            csv_float(n)
        )?;
    }
    writer.flush()?;
    Ok(())
}

/// Format like Python's `repr` (and therefore pandas' default CSV output)
/// for floats: shortest representation, `1e-05` instead of `0.00001`, always
/// with a decimal point so integer-valued centroids/volumes read as floats.
fn csv_float(value: f64) -> String {
    // Rust's Debug for floats is the same shortest round-trip representation
    // as Python's repr and switches to an exponent at the same thresholds; it
    // only omits the exponent sign and zero padding ("1e-5" vs "1e-05").
    let text = format!("{value:?}");
    let Some((mantissa, exponent)) = text.split_once('e') else {
        return text;
    };
    let (sign, digits) = match exponent.as_bytes().first() {
        Some(b'+' | b'-') => (&exponent[..1], &exponent[1..]),
        _ => ("+", exponent),
    };
    format!("{mantissa}e{sign}{digits:0>2}")
}

// ---------------------------------------------------------------------------
// Demo data
// ---------------------------------------------------------------------------

pub fn demo_stack() -> ImageStack {
    let shape = [21usize, 96, 96];
    let [depth, height, width] = shape;
    let mut membrane = vec![0u16; depth * height * width];
    let mut nuclei = vec![0u16; depth * height * width];

    // Two membrane-enclosed cells.  The ellipsoid radii are expressed in
    // physical units (the default z ratio is 6.92), so the demo remains
    // detectable with the default DoG parameters.
    let cells = [
        ([10.0f64, 32.0, 32.0], [3.5f64, 17.0, 17.0]),
        ([10.0f64, 64.0, 64.0], [3.5f64, 17.0, 17.0]),
    ];
    for &(center, radii) in &cells {
        for z in 0..depth {
            for y in 0..height {
                for x in 0..width {
                    let dz = (z as f64 - center[0]) / radii[0];
                    let dy = (y as f64 - center[1]) / radii[1];
                    let dx = (x as f64 - center[2]) / radii[2];
                    let d = (dz * dz + dy * dy + dx * dx).sqrt();
                    let i = (z * height + y) * width + x;
                    if (0.80..=1.05).contains(&d) {
                        membrane[i] = 52_000;
                    }
                    let g = (-0.5
                        * ((z as f64 - center[0]).powi(2) / (1.35 * 1.35)
                            + (y as f64 - center[1]).powi(2) / (6.0 * 6.0)
                            + (x as f64 - center[2]).powi(2) / (6.0 * 6.0)))
                        .exp();
                    let value = (g * 58_000.0) as u16;
                    nuclei[i] = nuclei[i].max(value);
                }
            }
        }
    }

    ImageStack {
        membrane: Volume::new(shape, membrane).expect("demo shape is valid"),
        nuclei: Some(Volume::new(shape, nuclei).expect("demo shape is valid")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{self, SeekFrom};

    /// Accepts every write and seek but fails the flush.  A real file cannot
    /// express "only the flush fails", so this is the double that isolates
    /// whether the buffered TIFF writer is flushed and its error reported.
    struct FlushFails(Vec<u8>);

    impl Write for FlushFails {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.0.extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Err(io::Error::other("flush failed"))
        }
    }

    impl Seek for FlushFails {
        fn seek(&mut self, _pos: SeekFrom) -> io::Result<u64> {
            Ok(0)
        }
    }

    #[test]
    fn tiff_flush_error_is_reported() {
        let labels = Volume::new([1, 2, 2], vec![0, 4, 4, 0]).unwrap();
        let mut writer = FlushFails(Vec::new());
        let err = write_label_tiff(&mut writer, &labels).unwrap_err();
        assert!(
            err.to_string().contains("flush failed"),
            "unexpected error: {err}"
        );
    }
}
