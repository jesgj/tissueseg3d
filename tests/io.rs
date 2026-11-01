use std::path::{Path, PathBuf};

use tissueseg3d::io::{demo_stack, export_labels, load_labels, load_tiff};
use tissueseg3d::processing::{detect_seeds, segment_cells, SeedParams};
use tissueseg3d::volume::Volume;

fn fixture(name: &str) -> PathBuf {
    [env!("CARGO_MANIFEST_DIR"), "tests", "fixtures", name]
        .iter()
        .collect()
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("tissueseg3d-io-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn loads_plain_single_channel_zyx_tiff() {
    let stack = load_tiff(&fixture("single_zyx.tif"), false).unwrap();
    assert!(stack.nuclei.is_none());
    assert_eq!(stack.membrane.shape, [4, 6, 7]);
    assert_eq!(stack.membrane.data[0], 0);
    assert_eq!(stack.membrane.data[(6 + 2) * 7 + 3], 18349);
}

#[test]
fn loads_imagej_zcyx_dual_channel_tiff() {
    let stack = load_tiff(&fixture("dual_imagej.tif"), true).unwrap();
    assert_eq!(stack.membrane.shape, [3, 4, 5]);
    let nuc = stack.nuclei.unwrap();
    // channel 0 is nuclei, channel 1 is membrane; C is the fastest page axis.
    assert_eq!(nuc.data[1], 1);
    assert_eq!(nuc.data[(2 * 4 + 1) * 5 + 2], 21 + 2000);
    assert_eq!(stack.membrane.data[1], 1 + 500);
    assert_eq!(stack.membrane.data[(2 * 4 + 1) * 5 + 2], 21 + 2000 + 500);
}

#[test]
fn loads_ome_zcyx_dual_channel_tiff() {
    let stack = load_tiff(&fixture("dual_ome.tif"), true).unwrap();
    assert_eq!(stack.membrane.shape, [3, 4, 5]);
    let nuc = stack.nuclei.unwrap();
    assert_eq!(nuc.data[(2 * 4 + 1) * 5 + 2], 21 + 2000);
    assert_eq!(stack.membrane.data[(2 * 4 + 1) * 5 + 2], 21 + 2500);
}

#[test]
fn loads_tifffile_json_shape_zcyx_dual_channel_tiff() {
    let stack = load_tiff(&fixture("dual_json.tif"), true).unwrap();
    assert_eq!(stack.membrane.shape, [3, 4, 5]);
    let nuc = stack.nuclei.unwrap();
    assert_eq!(nuc.data[(2 * 4 + 1) * 5 + 2], 21 + 2000);
    assert_eq!(stack.membrane.data[(2 * 4 + 1) * 5 + 2], 21 + 2500);
}

#[test]
fn rejects_dual_channel_request_for_single_channel_file() {
    let err = load_tiff(&fixture("single_zyx.tif"), true).unwrap_err();
    assert!(
        err.to_string().contains("channel") || err.to_string().contains("2 channels"),
        "unexpected error: {err}"
    );
}

#[test]
fn rejects_single_channel_request_for_two_channel_file() {
    let err = load_tiff(&fixture("dual_imagej.tif"), false).unwrap_err();
    assert!(
        err.to_string().contains("channel"),
        "unexpected error: {err}"
    );
}

#[test]
fn rejects_ambiguous_dual_channel_pages_without_metadata() {
    let err = load_tiff(&fixture("dual_nometa.tif"), true).unwrap_err();
    assert!(
        err.to_string().contains("metadata") || err.to_string().contains("ambiguous"),
        "unexpected error: {err}"
    );
}

#[test]
fn rejects_three_channel_dual_request() {
    let err = load_tiff(&fixture("dual_three_channels.tif"), true).unwrap_err();
    assert!(err.to_string().contains("2"), "unexpected error: {err}");
}

#[test]
fn rejects_float_tiff_instead_of_silently_casting() {
    let err = load_tiff(&fixture("float_image.tif"), false).unwrap_err();
    assert!(
        err.to_string().contains("float") || err.to_string().contains("unsupported"),
        "unexpected error: {err}"
    );
}

#[test]
fn loads_16bit_labels_as_u32() {
    let labels = load_labels(&fixture("labels_u16.tif")).unwrap();
    assert_eq!(labels.shape, [3, 4, 5]);
    assert_eq!(labels.data[0], 2);
    assert_eq!(labels.data[(4 + 1) * 5 + 1], 5);
    assert_eq!(labels.data[(2 * 4 + 3) * 5 + 4], 9);
}

#[test]
fn loads_32bit_labels_without_truncation() {
    let labels = load_labels(&fixture("labels_u32.tif")).unwrap();
    assert_eq!(labels.shape, [2, 3, 4]);
    assert_eq!(labels.data[0], 70000);
    assert_eq!(labels.data[(3 + 2) * 4 + 3], 70001);
}

#[test]
fn exports_labels_csv_and_safe_tiff_paths() {
    let dir = scratch("export");
    let input = dir.join("sample.tif");
    std::fs::write(&input, b"original-source-bytes").unwrap();
    let mut data = vec![0u32; 2 * 3 * 4];
    for &(z, y, x) in &[(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1)] {
        data[(z * 3 + y) * 4 + x] = 1;
    }
    for &(z, y, x) in &[(1, 1, 2), (1, 1, 3), (1, 2, 2), (1, 2, 3)] {
        data[(z * 3 + y) * 4 + x] = 3;
    }
    let labels = Volume::new([2, 3, 4], data).unwrap();
    let (tif, csv) = export_labels(&input, &labels, false).unwrap();
    assert_eq!(tif, dir.join("sample_lbl.tif"));
    assert_eq!(csv, dir.join("sample_lbl.csv"));

    assert_eq!(std::fs::read(&input).unwrap(), b"original-source-bytes");
    assert_eq!(load_labels(&tif).unwrap(), labels);
    let text = std::fs::read_to_string(&csv).unwrap();
    assert_eq!(
        text,
        "Cell,CZ,CY,CX,Volume (vox)\n1,0.0,0.5,0.5,4.0\n3,1.0,1.5,2.5,4.0\n"
    );
}

#[test]
fn export_uses_file_stem_for_tiff_extension_and_nucleus_names() {
    let dir = scratch("extension");
    let input = dir.join("stack.tiff");
    std::fs::write(&input, b"source").unwrap();
    let labels = Volume::new([1, 2, 2], vec![0, 4, 4, 0]).unwrap();
    let (tif, csv) = export_labels(&input, &labels, true).unwrap();
    assert_eq!(tif, dir.join("stack_nuc_lbl.tif"));
    assert_eq!(csv, dir.join("stack_nuc_lbl.csv"));
    assert_eq!(load_labels(&tif).unwrap(), labels);
}

#[test]
fn export_uses_gray32_instead_of_truncating_large_labels() {
    let dir = scratch("gray32");
    let input = dir.join("large.tif");
    std::fs::write(&input, b"source").unwrap();
    let labels = Volume::new([1, 2, 2], vec![0, 70000, 70001, 4]).unwrap();
    let (tif, _) = export_labels(&input, &labels, false).unwrap();
    assert_eq!(load_labels(&tif).unwrap(), labels);
}

#[test]
fn demo_stack_is_two_channel_and_detectable() {
    let stack = demo_stack();
    let nuc = stack.nuclei.as_ref().expect("demo must include nuclei");
    assert_eq!(stack.membrane.shape, nuc.shape);
    assert!(stack.membrane.shape.iter().all(|&d| d > 1));
    assert!(stack.membrane.data.iter().any(|&v| v != 0));
    assert!(nuc.data.iter().any(|&v| v != 0));

    let seeds = detect_seeds(&stack, &SeedParams::default()).expect("demo seed detection");
    assert!(
        !seeds.is_empty(),
        "demo stack should contain detectable nuclei"
    );
    for &[z, y, x] in &seeds {
        assert!(z < stack.membrane.shape[0]);
        assert!(y < stack.membrane.shape[1]);
        assert!(x < stack.membrane.shape[2]);
    }
    let cells = segment_cells(&stack.membrane, &seeds, &Default::default()).unwrap();
    assert!(
        cells.data.iter().any(|&v| v != 0),
        "demo cells should survive default filters"
    );
}

#[test]
fn missing_file_reports_path_context() {
    let err = load_tiff(Path::new("/definitely/not/here.tif"), false).unwrap_err();
    let text = format!("{err:#}");
    assert!(
        text.contains("here.tif"),
        "error should mention file path: {text}"
    );
}

#[test]
fn export_preserves_dotted_file_stem_bytes() {
    let dir = scratch("dotted");
    let input = dir.join("sample.v1.tif");
    std::fs::write(&input, b"source").unwrap();
    let labels = Volume::new([1, 2, 2], vec![0, 4, 4, 0]).unwrap();
    let (tif, csv) = export_labels(&input, &labels, false).unwrap();
    assert_eq!(
        tif.file_name().unwrap().as_encoded_bytes(),
        b"sample.v1_lbl.tif"
    );
    assert_eq!(
        csv.file_name().unwrap().as_encoded_bytes(),
        b"sample.v1_lbl.csv"
    );
    assert_eq!(tif.parent(), Some(dir.as_path()));
    assert_eq!(std::fs::read(&input).unwrap(), b"source");
    assert_eq!(load_labels(&tif).unwrap(), labels);
}

#[test]
fn export_preserves_dotted_file_stem_bytes_for_nuclei() {
    let dir = scratch("dotted-nuclei");
    let input = dir.join("sample.v1.tif");
    std::fs::write(&input, b"source").unwrap();
    let labels = Volume::new([1, 2, 2], vec![0, 4, 4, 0]).unwrap();
    let (tif, csv) = export_labels(&input, &labels, true).unwrap();
    assert_eq!(
        tif.file_name().unwrap().as_encoded_bytes(),
        b"sample.v1_nuc_lbl.tif"
    );
    assert_eq!(
        csv.file_name().unwrap().as_encoded_bytes(),
        b"sample.v1_nuc_lbl.csv"
    );
}

#[test]
fn csv_formats_small_floats_like_python_repr() {
    let dir = scratch("exponent");
    let input = dir.join("exponent.tif");
    std::fs::write(&input, b"source").unwrap();
    // Label 7 covers 100000 voxels: all but one of the z=0 plane plus the
    // voxel (z=1, y=0, x=0).  Python repr(1 / 100000.0) == "1e-05", and
    // pandas writes exactly the same CZ/CY/CX values for this volume.
    let shape = [2usize, 400, 250];
    let plane = shape[1] * shape[2];
    let mut data = vec![0u32; 2 * plane];
    data[..plane].fill(7);
    data[plane - 1] = 0;
    data[plane] = 7;
    let labels = Volume::new(shape, data).unwrap();
    let (_, csv) = export_labels(&input, &labels, false).unwrap();
    let text = std::fs::read_to_string(&csv).unwrap();
    assert_eq!(
        text,
        "Cell,CZ,CY,CX,Volume (vox)\n7,1e-05,199.49601,124.49751,100000.0\n"
    );
}

#[cfg(unix)]
#[test]
fn reports_tiff_flush_failure_instead_of_silently_dropping_it() {
    // Writing to /dev/full keeps succeeding into the BufWriter until it is
    // flushed, so the failure is only visible if the writer is flushed.
    if !Path::new("/dev/full").exists() {
        return;
    }
    let dir = scratch("flush");
    let input = dir.join("sample.tif");
    std::fs::write(&input, b"source").unwrap();
    std::os::unix::fs::symlink("/dev/full", dir.join("sample_lbl.tif")).unwrap();
    let labels = Volume::new([1, 2, 2], vec![0, 4, 4, 0]).unwrap();
    let err = export_labels(&input, &labels, false).unwrap_err();
    let text = format!("{err:#}");
    assert!(
        text.contains("sample_lbl.tif"),
        "error should name the TIFF path: {text}"
    );
}
