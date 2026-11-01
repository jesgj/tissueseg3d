#![cfg(feature = "gui")]
#![allow(dead_code)] // Test-imported GUI modules are only partially exercised.

#[path = "../src/app.rs"]
mod app;
#[path = "../src/view.rs"]
mod view;

use std::path::PathBuf;
use std::sync::Arc;

use app::{seed_centroids, validate_seeds, WorkflowState};
use tissueseg3d::volume::{ImageStack, Volume};

fn stack(shape: [usize; 3]) -> ImageStack {
    let n = shape[0] * shape[1] * shape[2];
    ImageStack {
        membrane: Volume::new(shape, vec![7u16; n]).unwrap(),
        nuclei: Some(Volume::new(shape, vec![3u16; n]).unwrap()),
    }
}

fn vol(shape: [usize; 3], data: Vec<u32>) -> Volume<u32> {
    Volume::new(shape, data).unwrap()
}

fn loaded_state() -> WorkflowState {
    let mut s = WorkflowState::new();
    s.apply_load(Ok((PathBuf::from("/tmp/sample.tif"), stack([1, 2, 3]))))
        .unwrap();
    s.set_seeds(vec![[0, 0, 0], [0, 1, 2]]).unwrap();
    s
}

#[test]
fn successful_load_resets_seeds_and_downstream_results() {
    let mut s = loaded_state();
    s.apply_cells(Ok(vol([1, 2, 3], vec![1, 1, 2, 0, 2, 2])))
        .unwrap();

    s.apply_load(Ok((PathBuf::from("/tmp/new.tif"), stack([2, 2, 2]))))
        .unwrap();

    assert_eq!(s.input_path, Some(PathBuf::from("/tmp/new.tif")));
    assert_eq!(s.image.as_ref().unwrap().membrane.shape, [2, 2, 2]);
    assert!(s.seeds.is_empty());
    assert!(s.cells.is_none());
    assert!(s.nuclei.is_none());
    assert_eq!(s.cell_count, 0);
    assert_eq!(s.nucleus_count, 0);
}

#[test]
fn failed_load_preserves_current_working_data() {
    let mut s = loaded_state();
    s.apply_cells(Ok(vol([1, 2, 3], vec![1, 1, 2, 0, 2, 2])))
        .unwrap();
    let image_before = s.image.clone().unwrap();
    let cells_before = s.cells.clone().unwrap();
    let seeds_before = s.seeds.clone();

    let err = s
        .apply_load(Err("cannot read TIFF".to_owned()))
        .unwrap_err();

    assert!(err.contains("cannot read TIFF"));
    assert_eq!(s.input_path, Some(PathBuf::from("/tmp/sample.tif")));
    assert!(Arc::ptr_eq(&image_before, s.image.as_ref().unwrap()));
    assert!(Arc::ptr_eq(&cells_before, s.cells.as_ref().unwrap()));
    assert_eq!(s.seeds, seeds_before);
}

#[test]
fn changing_seeds_invalidates_cells_and_nuclei() {
    let mut s = loaded_state();
    s.apply_cells(Ok(vol([1, 2, 3], vec![1, 1, 2, 0, 2, 2])))
        .unwrap();
    s.apply_nuclei(Ok(vol([1, 2, 3], vec![1, 0, 2, 0, 0, 2])))
        .unwrap();

    assert!(s.add_seed([0, 1, 0]));
    assert!(s.cells.is_none());
    assert!(s.nuclei.is_none());
    assert_eq!(s.cell_count, 0);
    assert_eq!(s.nucleus_count, 0);
    assert_eq!(s.seeds.len(), 3);
}

#[test]
fn removing_nearest_seed_invalidates_downstream_results() {
    let mut s = loaded_state();
    s.apply_cells(Ok(vol([1, 2, 3], vec![1, 1, 2, 0, 2, 2])))
        .unwrap();

    assert_eq!(s.remove_nearest_seed([0, 0, 1]), Some([0, 0, 0]));
    assert_eq!(s.seeds, vec![[0, 1, 2]]);
    assert!(s.cells.is_none());
}

#[test]
fn invalid_or_out_of_bounds_seeds_are_rejected_without_changes() {
    let mut s = loaded_state();
    s.set_seeds(vec![[0, 0, 0], [0, 1, 2]]).unwrap();
    assert!(!s.add_seed([1, 0, 0]));
    assert!(!s.add_seed([0, 2, 0]));
    assert_eq!(s.seeds, vec![[0, 0, 0], [0, 1, 2]]);

    assert!(validate_seeds([2, 3, 4], &[[1, 2, 3]]).is_ok());
    assert!(validate_seeds([2, 3, 4], &[[2, 0, 0]]).is_err());
}

#[test]
fn cell_segmentation_invalidates_nuclei_but_not_cells() {
    let mut s = loaded_state();
    s.apply_cells(Ok(vol([1, 2, 3], vec![1, 1, 0, 0, 0, 0])))
        .unwrap();
    s.apply_nuclei(Ok(vol([1, 2, 3], vec![1, 0, 2, 0, 0, 2])))
        .unwrap();

    s.apply_cells(Ok(vol([1, 2, 3], vec![1, 1, 2, 0, 2, 2])))
        .unwrap();

    assert!(s.nuclei.is_none());
    assert_eq!(s.cell_count, 2);
    assert_eq!(s.cells.as_ref().unwrap().data, vec![1, 1, 2, 0, 2, 2]);
}

#[test]
fn zratio_change_invalidates_results_and_keeps_coordinates() {
    let mut s = loaded_state();
    s.apply_cells(Ok(vol([1, 2, 3], vec![1, 1, 2, 0, 2, 2])))
        .unwrap();
    s.apply_nuclei(Ok(vol([1, 2, 3], vec![1, 0, 2, 0, 0, 2])))
        .unwrap();

    s.set_z_ratio(1.5);

    assert_eq!(s.z_ratio, 1.5);
    assert_eq!(s.seeds, vec![[0, 0, 0], [0, 1, 2]]);
    assert!(s.cells.is_none());
    assert!(s.nuclei.is_none());
}

#[test]
fn removing_a_label_clears_it_from_both_masks() {
    let mut s = loaded_state();
    s.apply_cells(Ok(vol([1, 2, 3], vec![1, 1, 2, 0, 2, 2])))
        .unwrap();
    s.apply_nuclei(Ok(vol([1, 2, 3], vec![1, 0, 2, 0, 2, 0])))
        .unwrap();
    s.selected_label = 2;

    s.remove_label(2).unwrap();

    assert_eq!(s.cells.as_ref().unwrap().data, vec![1, 1, 0, 0, 0, 0]);
    assert_eq!(s.nuclei.as_ref().unwrap().data, vec![1, 0, 0, 0, 0, 0]);
    assert_eq!(s.cell_count, 1);
    assert_eq!(s.nucleus_count, 1);
    assert_eq!(s.selected_label, 0);
}

#[test]
fn merging_labels_renumbers_both_masks_and_selects_kept_label() {
    let mut s = loaded_state();
    s.apply_cells(Ok(vol([1, 2, 3], vec![1, 1, 2, 0, 2, 2])))
        .unwrap();
    s.apply_nuclei(Ok(vol([1, 2, 3], vec![1, 0, 2, 0, 2, 0])))
        .unwrap();

    s.merge_labels(1, 2).unwrap();

    assert_eq!(s.cells.as_ref().unwrap().data, vec![1, 1, 1, 0, 1, 1]);
    assert_eq!(s.nuclei.as_ref().unwrap().data, vec![1, 0, 1, 0, 1, 0]);
    assert_eq!(s.cell_count, 1);
    assert_eq!(s.nucleus_count, 1);
    assert_eq!(s.selected_label, 1);
}

#[test]
fn empty_or_identical_merge_is_rejected() {
    let mut s = loaded_state();
    s.apply_cells(Ok(vol([1, 2, 3], vec![1, 1, 2, 0, 2, 2])))
        .unwrap();
    assert!(s.merge_labels(1, 1).is_err());
    assert!(s.merge_labels(1, 0).is_err());
    assert!(s.merge_labels(9, 2).is_err()); // keep label must exist
}

#[test]
fn seed_labels_of_different_shape_are_rejected_and_preserve_seeds() {
    let mut s = loaded_state();
    let before = s.seeds.clone();

    // Every voxel is labelled 1, so the converted centroid [0, 1, 2] would be
    // inside the loaded [1, 2, 3] image even though the mask is [1, 4, 6].
    let labels = vol([1, 4, 6], vec![1u32; 24]);
    let converted = seed_centroids(&labels, s.shape().unwrap());

    let err = s.apply_seed_load(converted).unwrap_err();
    assert!(err.contains("shape"), "unexpected error: {err}");
    assert_eq!(s.seeds, before);
}

#[test]
fn seed_labels_of_matching_shape_are_converted_from_centroids() {
    let labels = vol([1, 2, 3], vec![0, 0, 0, 2, 0, 0]);
    assert_eq!(seed_centroids(&labels, [1, 2, 3]).unwrap(), vec![[0, 1, 0]]);
}

#[test]
fn nuclei_cannot_be_applied_without_cells() {
    let mut s = loaded_state();
    let err = s
        .apply_nuclei(Ok(vol([1, 2, 3], vec![1, 0, 2, 0, 0, 2])))
        .unwrap_err();
    assert!(err.contains("cell"));
    assert!(s.nuclei.is_none());
}
