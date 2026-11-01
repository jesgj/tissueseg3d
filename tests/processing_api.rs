use tissueseg3d::processing::{
    label_centroids, merge_labels, remove_label, segment_cells, segment_nuclei, CellParams,
    NucleusParams, SeedParams,
};
use tissueseg3d::volume::Volume;

#[test]
fn parameter_defaults_match_python_widgets() {
    let s = SeedParams::default();
    assert_eq!(
        (
            s.min_sigma,
            s.max_sigma,
            s.threshold,
            s.merge_distance,
            s.membrane_delta,
            s.z_ratio
        ),
        (4.0, 12.0, 1.0, 100.0, 25.0, 6.92)
    );
    let c = CellParams::default();
    assert_eq!(
        (c.gaussian_sigma, c.median_size, c.min_volume, c.max_volume),
        (0.5, 5, 2000, 100000)
    );
    let n = NucleusParams::default();
    assert_eq!(
        (n.gaussian_sigma, n.threshold, n.median_size, n.z_ratio),
        (1.5, 500, 5, 6.92)
    );
}

#[test]
fn empty_seed_list_produces_background_only_not_panic() {
    let membrane = Volume::filled([3, 5, 5], 0u16).unwrap();
    let out = segment_cells(&membrane, &[], &CellParams::default()).unwrap();
    assert_eq!(out.shape, membrane.shape);
    assert!(
        out.data.iter().all(|&v| v == 0),
        "background flood leaked into cells: {out:?}"
    );
}

#[test]
fn out_of_bounds_seed_is_rejected() {
    let membrane = Volume::filled([3, 5, 5], 0u16).unwrap();
    let err = segment_cells(&membrane, &[[3, 1, 1]], &CellParams::default()).unwrap_err();
    assert!(
        err.to_string().contains("outside"),
        "unexpected error: {err}"
    );
}

#[test]
fn inverted_cell_volume_range_is_rejected() {
    let membrane = Volume::filled([3, 5, 5], 0u16).unwrap();
    let params = CellParams {
        min_volume: 10,
        max_volume: 5,
        ..CellParams::default()
    };
    let err = segment_cells(&membrane, &[[1, 2, 2]], &params).unwrap_err();
    assert!(
        err.to_string().contains("min volume"),
        "unexpected error: {err}"
    );
}

#[test]
fn nucleus_shape_mismatch_is_rejected() {
    let nuclei = Volume::filled([3, 5, 5], 0u16).unwrap();
    let cells = Volume::filled([3, 5, 6], 0u32).unwrap();
    let err = segment_nuclei(&nuclei, &cells, &NucleusParams::default()).unwrap_err();
    assert!(err.to_string().contains("match"), "unexpected error: {err}");
}

#[test]
fn label_centroids_are_ordered_by_label_and_truncated_like_python() {
    // hand-derived: label 1 has z=(0+1+1+0)/4=0.5 -> 0, y=1, x=(1+1+2+2)/4=1.5 -> 1;
    // label 2 is the two voxels at x=0,1 in z=0,y=0 -> (0,0,0.5)->(0,0,0);
    // label 5 is a single voxel at (1,2,2).
    let shape = [2, 3, 3];
    let mut data = vec![0u32; 18];
    let mut put = |z: usize, y: usize, x: usize, v: u32| data[(z * 3 + y) * 3 + x] = v;
    put(0, 1, 1, 1);
    put(1, 1, 1, 1);
    put(1, 1, 2, 1);
    put(0, 1, 2, 1);
    put(0, 0, 0, 2);
    put(0, 0, 1, 2);
    put(1, 2, 2, 5);
    let labels = Volume::new(shape, data).unwrap();
    assert_eq!(
        label_centroids(&labels),
        vec![[0, 1, 1], [0, 0, 0], [1, 2, 2]]
    );
}

#[test]
fn remove_label_clears_cell_and_nucleus_masks() {
    let mut cells = Volume::new([1, 2, 3], vec![1, 2, 2, 3, 3, 3]).unwrap();
    let mut nuc = Volume::new([1, 2, 3], vec![0, 2, 0, 3, 0, 3]).unwrap();
    remove_label(&mut cells, Some(&mut nuc), 2);
    assert_eq!(cells.data, vec![1, 0, 0, 3, 3, 3]);
    assert_eq!(nuc.data, vec![0, 0, 0, 3, 0, 3]);
}

#[test]
fn merge_labels_reassigns_both_masks() {
    let mut cells = Volume::new([1, 2, 3], vec![1, 2, 2, 3, 3, 3]).unwrap();
    let mut nuc = Volume::new([1, 2, 3], vec![0, 2, 0, 3, 0, 3]).unwrap();
    merge_labels(&mut cells, Some(&mut nuc), 5, 3);
    assert_eq!(cells.data, vec![1, 2, 2, 5, 5, 5]);
    assert_eq!(nuc.data, vec![0, 2, 0, 5, 0, 5]);
}

#[test]
fn merge_into_background_is_ignored() {
    let mut cells = Volume::new([1, 1, 3], vec![0, 2, 2]).unwrap();
    merge_labels(&mut cells, None, 0, 2);
    assert_eq!(cells.data, vec![0, 2, 2]);
}
