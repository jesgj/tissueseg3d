#![cfg(feature = "gui")]
#![allow(dead_code)] // Test-imported GUI modules are only partially exercised.

#[path = "../src/view.rs"]
mod view;

use eframe::egui::{pos2, vec2, Rect};
use tissueseg3d::volume::Volume;
use view::{pan_for_zoom, project, read_slice, Plane};

#[test]
fn xy_projection_maps_hand_checked_pixels() {
    let rect = Rect::from_min_size(pos2(0.0, 0.0), vec2(500.0, 500.0));
    let p = project(rect, [4, 3, 5], Plane::Xy, 6.92, 1.0, vec2(0.0, 0.0)).unwrap();

    // 5x3 XY slice fitted into 500x500: 100 points per pixel, centred vertically.
    assert_eq!(p.pixel.x, 100.0);
    assert_eq!(p.pixel.y, 100.0);
    assert_eq!(p.image_rect.min, pos2(0.0, 100.0));
    assert_eq!(p.image_rect.max, pos2(500.0, 400.0));

    // Screen x is data x, screen y is data y.
    assert_eq!(p.data_at(pos2(150.0, 250.0)), Some([1, 1]));
    assert_eq!(p.data_at(pos2(499.0, 399.0)), Some([4, 2]));
    assert_eq!(p.data_at(pos2(150.0, 450.0)), None);
}

#[test]
fn xz_projection_uses_zratio_vertical_spacing() {
    let rect = Rect::from_min_size(pos2(0.0, 0.0), vec2(500.0, 500.0));
    let p = project(rect, [4, 3, 5], Plane::Xz, 2.0, 1.0, vec2(0.0, 0.0)).unwrap();

    // Physical slice is 5 wide and 4*2=8 tall. Fit is 500/8=62.5 points
    // per XY voxel; one Z step is therefore 125 points tall.
    assert_eq!(p.pixel.x, 62.5);
    assert_eq!(p.pixel.y, 125.0);
    assert_eq!(p.image_rect.width(), 312.5);
    assert_eq!(p.image_rect.height(), 500.0);

    // Horizontal screen axis is X, vertical screen axis is Z.
    // The 312.5-point-wide image is horizontally centred (left edge 93.75).
    assert_eq!(p.data_at(pos2(100.0, 10.0)), Some([0, 0]));
    assert_eq!(p.data_at(pos2(100.0, 130.0)), Some([0, 1]));
    assert_eq!(p.data_at(pos2(400.0, 380.0)), Some([4, 3]));
    assert_eq!(p.data_at(pos2(400.0, 499.0)), Some([4, 3]));
}

#[test]
fn yz_projection_maps_y_horizontally_and_z_vertically() {
    let rect = Rect::from_min_size(pos2(0.0, 0.0), vec2(500.0, 500.0));
    let p = project(rect, [4, 3, 5], Plane::Yz, 2.0, 1.0, vec2(0.0, 0.0)).unwrap();

    // Physical slice is 3 wide and 8 tall, so fit is 500/8=62.5.
    assert_eq!(p.pixel.x, 62.5);
    assert_eq!(p.pixel.y, 125.0);
    // 187.5 points wide, left edge 156.25, so u=0 spans 156.25..218.75.
    assert_eq!(p.data_at(pos2(160.0, 130.0)), Some([0, 1]));
    assert_eq!(p.data_at(pos2(330.0, 490.0)), Some([2, 3]));
}

#[test]
fn zoom_pan_keeps_pointer_over_same_data_point() {
    let rect = Rect::from_min_size(pos2(0.0, 0.0), vec2(500.0, 500.0));
    let pointer = pos2(100.0, 250.0);
    let pan = pan_for_zoom(
        rect,
        [4, 3, 5],
        Plane::Xy,
        1.0,
        1.0,
        vec2(0.0, 0.0),
        2.0,
        pointer,
    );

    // The 500x300 image doubles to 1000x600. Keeping screen (100,250)
    // over the same data pixel gives a centre of (400,250), pan (150,0).
    assert_eq!(pan, vec2(150.0, 0.0));
    let p = project(rect, [4, 3, 5], Plane::Xy, 1.0, 2.0, pan).unwrap();
    assert_eq!(p.data_at(pointer), Some([1, 1]));
}

#[test]
fn slice_reads_are_hand_checked_for_all_three_planes() {
    // shape [z=2, y=3, x=4], value = 100z + 10y + x.
    let mut data = Vec::new();
    for z in 0..2 {
        for y in 0..3 {
            for x in 0..4 {
                data.push((100 * z + 10 * y + x) as u16);
            }
        }
    }
    let vol = Volume::new([2, 3, 4], data).unwrap();

    // XY at z=1: horizontal x, vertical y.
    assert_eq!(read_slice(&vol, Plane::Xy, 1, 3, 2), 123);
    // XZ at y=2: horizontal x, vertical z.
    assert_eq!(read_slice(&vol, Plane::Xz, 2, 3, 1), 123);
    // YZ at x=2: horizontal y, vertical z.
    assert_eq!(read_slice(&vol, Plane::Yz, 2, 2, 1), 122);

    assert_eq!(Plane::Xy.voxel(1, 3, 2), [1, 2, 3]);
    assert_eq!(Plane::Xz.voxel(2, 3, 1), [1, 2, 3]);
    assert_eq!(Plane::Yz.voxel(2, 2, 1), [1, 2, 2]);
}
