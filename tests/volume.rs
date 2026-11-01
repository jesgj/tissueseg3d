use tissueseg3d::volume::Volume;

#[test]
fn new_rejects_zero_dimension() {
    let err = Volume::<u16>::new([0, 4, 4], vec![]).unwrap_err();
    assert!(err.to_string().contains("zero"), "unexpected error: {err}");
}

#[test]
fn new_rejects_data_length_mismatch() {
    let err = Volume::<u16>::new([2, 2, 2], vec![1, 2, 3]).unwrap_err();
    assert!(
        err.to_string().contains("length"),
        "unexpected error: {err}"
    );
}

#[test]
fn new_rejects_shape_product_overflow() {
    let err = Volume::<u16>::new([usize::MAX, 2, 1], vec![]).unwrap_err();
    assert!(
        err.to_string().contains("overflow") || err.to_string().contains("large"),
        "unexpected error: {err}"
    );
}

#[test]
fn idx_uses_z_y_x_row_major_order() {
    let v = Volume::new([2, 3, 4], (0u16..24).collect()).unwrap();
    assert_eq!(v.idx(0, 0, 0), 0);
    assert_eq!(v.idx(0, 1, 0), 4);
    assert_eq!(v.idx(1, 0, 0), 12);
    assert_eq!(v.idx(1, 2, 3), 23);
}
