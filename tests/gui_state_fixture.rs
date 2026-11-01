#![cfg(feature = "gui")]
#![allow(dead_code)] // Test-imported GUI modules are only partially exercised.

// Small end-to-end state check against the real core: segmentation results can
// be loaded into the workflow state, edited coherently and exported without
// mutating the original true masks while selecting a label.
#[path = "../src/app.rs"]
mod app;
#[path = "../src/view.rs"]
mod view;

use std::sync::Arc;

use app::WorkflowState;
use tissueseg3d::processing::Seed;
use tissueseg3d::volume::{ImageStack, Volume};

#[test]
fn selected_label_editing_does_not_touch_untargeted_voxels() {
    let shape = [1, 2, 3];
    let membrane = Volume::new(shape, vec![1u16; 6]).unwrap();
    let mut s = WorkflowState::new();
    s.apply_load(Ok((
        std::path::PathBuf::from("/tmp/fixture.tif"),
        ImageStack {
            membrane,
            nuclei: None,
        },
    )))
    .unwrap();
    s.set_seeds(vec![Seed::from([0, 0, 0])]).unwrap();
    s.apply_cells(Ok(Volume::new(shape, vec![1, 1, 2, 0, 2, 2]).unwrap()))
        .unwrap();
    let before = Arc::clone(s.cells.as_ref().unwrap());

    s.selected_label = 2;
    s.remove_label(2).unwrap();

    // The original Arc still describes the old mask; editing used copy-on-write.
    assert_eq!(before.data, vec![1, 1, 2, 0, 2, 2]);
    assert_eq!(s.cells.as_ref().unwrap().data, vec![1, 1, 0, 0, 0, 0]);
}
