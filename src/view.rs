//! Orthogonal view geometry, slice extraction and image compositing.
//!
//! This module deliberately keeps the display path separate from the stored
//! label masks: overlays are rasterized into temporary textures and never
//! write into the volumes used by segmentation or export.

use eframe::egui::{
    self, pos2, vec2, Color32, ColorImage, Pos2, Rect, Sense, Stroke, StrokeKind, TextureHandle,
    TextureOptions, Vec2,
};
use tissueseg3d::processing::Seed;
use tissueseg3d::volume::{ImageStack, Volume};

/// The three orthogonal slices shown in the main viewer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Plane {
    Xy,
    Xz,
    Yz,
}

impl Plane {
    pub fn index(self) -> usize {
        match self {
            Self::Xy => 0,
            Self::Xz => 1,
            Self::Yz => 2,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Xy => "XY",
            Self::Xz => "XZ",
            Self::Yz => "YZ",
        }
    }

    /// Axis that is held constant in this view.
    pub fn fixed_axis(self) -> usize {
        match self {
            Self::Xy => 0,
            Self::Xz => 1,
            Self::Yz => 2,
        }
    }

    /// Data axes used for (horizontal, vertical) image coordinates.
    pub fn axes(self) -> (usize, usize) {
        match self {
            Self::Xy => (2, 1),
            Self::Xz => (2, 0),
            Self::Yz => (1, 0),
        }
    }

    /// Turn a 2D view coordinate plus the current fixed slice into a voxel.
    pub fn voxel(self, fixed: usize, horizontal: usize, vertical: usize) -> Seed {
        match self {
            Self::Xy => [fixed, vertical, horizontal],
            Self::Xz => [vertical, fixed, horizontal],
            Self::Yz => [vertical, horizontal, fixed],
        }
    }
}

/// Read one value from an orthogonal slice. Bounds are the caller's
/// responsibility; the viewer only asks for coordinates inside `shape`.
pub fn read_slice<T: Copy>(
    volume: &Volume<T>,
    plane: Plane,
    fixed: usize,
    horizontal: usize,
    vertical: usize,
) -> T {
    let [z, y, x] = plane.voxel(fixed, horizontal, vertical);
    volume.data[volume.idx(z, y, x)]
}

/// Geometry for one displayed orthogonal slice.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Projection {
    pub image_rect: Rect,
    pub pixel: Vec2,
    pub size: [usize; 2],
}

impl Projection {
    /// Map a screen point to a (horizontal, vertical) view pixel.
    pub fn data_at(&self, position: Pos2) -> Option<[usize; 2]> {
        let local = position - self.image_rect.min;
        if local.x < 0.0
            || local.y < 0.0
            || local.x >= self.image_rect.width()
            || local.y >= self.image_rect.height()
        {
            return None;
        }
        let mut u = (local.x / self.pixel.x).floor() as usize;
        let mut v = (local.y / self.pixel.y).floor() as usize;
        u = u.min(self.size[0].saturating_sub(1));
        v = v.min(self.size[1].saturating_sub(1));
        Some([u, v])
    }

    /// Centre of a view pixel in screen coordinates.
    pub fn screen_of(&self, horizontal: usize, vertical: usize) -> Pos2 {
        self.image_rect.min
            + vec2(
                (horizontal as f32 + 0.5) * self.pixel.x,
                (vertical as f32 + 0.5) * self.pixel.y,
            )
    }
}

/// Fit a slice into `rect`, preserving physical aspect ratio with `z_ratio`.
pub fn project(
    rect: Rect,
    shape: [usize; 3],
    plane: Plane,
    z_ratio: f64,
    zoom: f32,
    pan: Vec2,
) -> Option<Projection> {
    if shape.contains(&0) || !z_ratio.is_finite() || z_ratio <= 0.0 {
        return None;
    }
    if !zoom.is_finite() || zoom <= 0.0 || rect.width() <= 0.0 || rect.height() <= 0.0 {
        return None;
    }

    let (horizontal_axis, vertical_axis) = plane.axes();
    let width_voxels = shape[horizontal_axis] as f32;
    let vertical_spacing = if vertical_axis == 0 {
        z_ratio as f32
    } else {
        1.0
    };
    let height_physical = shape[vertical_axis] as f32 * vertical_spacing;

    let fit_scale = (rect.width() / width_voxels).min(rect.height() / height_physical);
    let scale = fit_scale * zoom;
    if !scale.is_finite() || scale <= 0.0 {
        return None;
    }

    let pixel = vec2(scale, scale * vertical_spacing);
    let size = [shape[horizontal_axis], shape[vertical_axis]];
    let image_size = vec2(size[0] as f32 * pixel.x, size[1] as f32 * pixel.y);
    let image_rect = Rect::from_center_size(rect.center() + pan, image_size);

    Some(Projection {
        image_rect,
        pixel,
        size,
    })
}

/// Pan needed to keep the data point under `pointer` fixed while zooming.
// The parameters are scalar view geometry; grouping them would not make this
// calculation clearer.
#[allow(clippy::too_many_arguments)]
pub fn pan_for_zoom(
    rect: Rect,
    shape: [usize; 3],
    plane: Plane,
    z_ratio: f64,
    old_zoom: f32,
    old_pan: Vec2,
    new_zoom: f32,
    pointer: Pos2,
) -> Vec2 {
    let Some(old) = project(rect, shape, plane, z_ratio, old_zoom, old_pan) else {
        return old_pan;
    };
    if !new_zoom.is_finite() || new_zoom <= 0.0 || old_zoom <= 0.0 {
        return old_pan;
    }
    let factor = new_zoom / old_zoom;
    let local = pointer - old.image_rect.min;
    let new_size = old.image_rect.size() * factor;
    let new_min = pointer - local * factor;
    new_min + new_size * 0.5 - rect.center()
}

/// Per-channel display settings. Ranges are inclusive 16-bit intensity bounds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DisplaySettings {
    pub show_membrane: bool,
    pub show_nuclei: bool,
    pub show_cells: bool,
    pub show_nuclei_labels: bool,
    pub membrane_range: [u16; 2],
    pub nuclei_range: [u16; 2],
    pub cell_opacity: u8,
    pub nucleus_opacity: u8,
}

impl Default for DisplaySettings {
    fn default() -> Self {
        Self {
            show_membrane: true,
            show_nuclei: true,
            show_cells: true,
            show_nuclei_labels: true,
            membrane_range: [0, 65_535],
            nuclei_range: [0, 65_535],
            cell_opacity: 95,
            nucleus_opacity: 155,
        }
    }
}

/// Immutable view of the data needed to draw one frame.
#[derive(Clone, Copy)]
pub struct SceneRef<'a> {
    pub image: &'a ImageStack,
    pub cells: Option<&'a Volume<u32>>,
    pub nuclei: Option<&'a Volume<u32>>,
    pub selected_label: u32,
    pub z_ratio: f64,
    pub revision: u64,
}

#[derive(Clone, PartialEq, Eq)]
struct TextureKey {
    plane: Plane,
    fixed: usize,
    show_membrane: bool,
    show_nuclei: bool,
    show_cells: bool,
    show_nuclei_labels: bool,
    membrane_range: [u16; 2],
    nuclei_range: [u16; 2],
    cell_opacity: u8,
    nucleus_opacity: u8,
    selected_label: u32,
    revision: u64,
}

struct CachedTexture {
    key: TextureKey,
    handle: TextureHandle,
}

/// Viewer interaction returned by [`show_view`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ViewAction {
    Primary { voxel: Seed },
    Secondary { voxel: Seed, seed: Option<Seed> },
}

/// Cursor, per-plane zoom/pan and the small texture cache for the three slices.
pub struct ViewerState {
    pub cursor: Seed,
    pub zoom: [f32; 3],
    pub pan: [Vec2; 3],
    textures: [Option<CachedTexture>; 3],
}

impl Default for ViewerState {
    fn default() -> Self {
        Self::new()
    }
}

impl ViewerState {
    pub fn new() -> Self {
        Self {
            cursor: [0, 0, 0],
            zoom: [1.0; 3],
            pan: [Vec2::ZERO; 3],
            textures: std::array::from_fn(|_| None),
        }
    }

    /// Reset to a centred cursor and fit-to-window zoom, dropping old textures.
    pub fn reset_for_shape(&mut self, shape: [usize; 3]) {
        self.cursor = [shape[0] / 2, shape[1] / 2, shape[2] / 2];
        self.zoom = [1.0; 3];
        self.pan = [Vec2::ZERO; 3];
        self.textures = std::array::from_fn(|_| None);
    }

    pub fn clamp_cursor(&mut self, shape: [usize; 3]) {
        for (axis, dim) in shape.iter().enumerate() {
            let last = dim.saturating_sub(1);
            self.cursor[axis] = self.cursor[axis].min(last);
        }
    }

    pub fn cursor_2d(&self, plane: Plane) -> (usize, usize) {
        let (horizontal, vertical) = plane.axes();
        (self.cursor[horizontal], self.cursor[vertical])
    }

    pub fn reset_zoom(&mut self) {
        self.zoom = [1.0; 3];
        self.pan = [Vec2::ZERO; 3];
    }
}

fn normalized(value: u16, range: [u16; 2]) -> f32 {
    let span = (range[1] as f32 - range[0] as f32).max(1.0);
    ((value as f32 - range[0] as f32) / span).clamp(0.0, 1.0)
}

fn label_color(label: u32) -> [u8; 3] {
    let hash = (label as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    [
        70 + ((hash >> 8) & 0x9F) as u8,
        70 + ((hash >> 24) & 0x9F) as u8,
        70 + ((hash >> 40) & 0x9F) as u8,
    ]
}

fn blend(dst: &mut [u8], color: [u8; 3], alpha: u8) {
    let a = alpha as u32;
    let inv = 255 - a;
    for channel in 0..3 {
        dst[channel] = ((color[channel] as u32 * a + dst[channel] as u32 * inv) / 255) as u8;
    }
}

fn build_slice(
    scene: SceneRef<'_>,
    display: &DisplaySettings,
    plane: Plane,
    fixed: usize,
) -> ColorImage {
    let shape = scene.image.membrane.shape;
    let (horizontal_axis, vertical_axis) = plane.axes();
    let width = shape[horizontal_axis];
    let height = shape[vertical_axis];
    let mut rgba = vec![0u8; width * height * 4];

    let cells = scene.cells.filter(|v| v.shape == shape);
    let nuclei = scene.nuclei.filter(|v| v.shape == shape);

    for v in 0..height {
        for u in 0..width {
            let offset = (v * width + u) * 4;
            let mut pixel = [13u8, 14, 17, 255];

            if display.show_membrane {
                let value = read_slice(&scene.image.membrane, plane, fixed, u, v);
                let intensity = (normalized(value, display.membrane_range) * 255.0).round() as u8;
                pixel[1] = pixel[1].saturating_add(intensity);
            }

            if let (true, Some(channel)) = (display.show_nuclei, scene.image.nuclei.as_ref()) {
                let value = read_slice(channel, plane, fixed, u, v);
                let intensity = (normalized(value, display.nuclei_range) * 255.0).round() as u8;
                pixel[0] = pixel[0].saturating_add(intensity);
                pixel[1] = pixel[1].saturating_add(intensity);
                pixel[2] = pixel[2].saturating_add(intensity);
            }

            if display.show_cells {
                if let Some(cells) = cells {
                    let label = read_slice(cells, plane, fixed, u, v);
                    if label != 0 {
                        let alpha = if label == scene.selected_label {
                            255
                        } else {
                            display.cell_opacity
                        };
                        blend(&mut pixel, label_color(label), alpha);
                    }
                }
            }

            if display.show_nuclei_labels {
                if let Some(nuclei) = nuclei {
                    let label = read_slice(nuclei, plane, fixed, u, v);
                    if label != 0 {
                        let alpha = if label == scene.selected_label {
                            255
                        } else {
                            display.nucleus_opacity
                        };
                        blend(&mut pixel, label_color(label), alpha);
                    }
                }
            }

            rgba[offset..offset + 4].copy_from_slice(&pixel);
        }
    }

    ColorImage::from_rgba_unmultiplied([width, height], &rgba)
}

fn nearest_seed(
    projection: &Projection,
    plane: Plane,
    fixed: usize,
    seeds: &[Seed],
    pointer: Pos2,
    radius: f32,
) -> Option<Seed> {
    let (horizontal_axis, vertical_axis) = plane.axes();
    seeds
        .iter()
        .filter(|seed| seed[plane.fixed_axis()] == fixed)
        .filter_map(|seed| {
            if seed[horizontal_axis] >= projection.size[0]
                || seed[vertical_axis] >= projection.size[1]
            {
                return None;
            }
            let screen = projection.screen_of(seed[horizontal_axis], seed[vertical_axis]);
            let distance = screen.distance(pointer);
            (distance <= radius).then_some((distance, *seed))
        })
        .min_by(|a, b| a.0.total_cmp(&b.0))
        .map(|(_, seed)| seed)
}

/// Draw one orthogonal view and return a user action, if any.
#[allow(clippy::too_many_arguments)]
pub fn show_view(
    ui: &mut egui::Ui,
    ctx: &egui::Context,
    scene: SceneRef<'_>,
    display: &DisplaySettings,
    viewer: &mut ViewerState,
    plane: Plane,
    rect: Rect,
    seeds: &[Seed],
) -> Option<ViewAction> {
    if rect.width() < 4.0 || rect.height() < 4.0 {
        return None;
    }
    let shape = scene.image.membrane.shape;
    viewer.clamp_cursor(shape);
    let fixed = viewer.cursor[plane.fixed_axis()];
    let projection = project(
        rect,
        shape,
        plane,
        scene.z_ratio,
        viewer.zoom[plane.index()],
        viewer.pan[plane.index()],
    )?;

    let painter = ui.painter().with_clip_rect(rect);
    painter.rect_filled(rect, 0.0, Color32::from_rgb(14, 15, 18));

    let key = TextureKey {
        plane,
        fixed,
        show_membrane: display.show_membrane,
        show_nuclei: display.show_nuclei,
        show_cells: display.show_cells,
        show_nuclei_labels: display.show_nuclei_labels,
        membrane_range: display.membrane_range,
        nuclei_range: display.nuclei_range,
        cell_opacity: display.cell_opacity,
        nucleus_opacity: display.nucleus_opacity,
        selected_label: scene.selected_label,
        revision: scene.revision,
    };

    let index = plane.index();
    let stale = viewer.textures[index]
        .as_ref()
        .is_none_or(|cached| cached.key != key);
    if stale {
        let image = build_slice(scene, display, plane, fixed);
        let handle = ctx.load_texture(
            format!("tissueseg3d-slice-{index}"),
            image,
            TextureOptions::NEAREST,
        );
        viewer.textures[index] = Some(CachedTexture { key, handle });
    }

    if let Some(cached) = viewer.textures[index].as_ref() {
        painter.image(
            cached.handle.id(),
            projection.image_rect,
            Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)),
            Color32::WHITE,
        );
    }

    // Crosshair at the current cursor.
    let (cursor_u, cursor_v) = viewer.cursor_2d(plane);
    let cursor_screen = projection.screen_of(cursor_u, cursor_v);
    let crosshair = Stroke::new(
        0.75_f32,
        Color32::from_rgba_unmultiplied(170, 215, 255, 150),
    );
    painter.line_segment(
        [
            pos2(rect.left(), cursor_screen.y),
            pos2(rect.right(), cursor_screen.y),
        ],
        crosshair,
    );
    painter.line_segment(
        [
            pos2(cursor_screen.x, rect.top()),
            pos2(cursor_screen.x, rect.bottom()),
        ],
        crosshair,
    );

    // Kept seeds belonging to this slice.
    let (horizontal_axis, vertical_axis) = plane.axes();
    for seed in seeds {
        if seed[plane.fixed_axis()] != fixed
            || seed[horizontal_axis] >= projection.size[0]
            || seed[vertical_axis] >= projection.size[1]
        {
            continue;
        }
        let point = projection.screen_of(seed[horizontal_axis], seed[vertical_axis]);
        if !rect.contains(point) {
            continue;
        }
        painter.circle_filled(point, 4.5, Color32::from_rgb(80, 235, 110));
        painter.circle_stroke(
            point,
            4.5,
            Stroke::new(1.0_f32, Color32::from_rgb(8, 25, 12)),
        );
    }

    painter.rect_stroke(
        rect,
        0.0,
        Stroke::new(1.0_f32, Color32::from_rgb(58, 63, 72)),
        StrokeKind::Inside,
    );

    let axis_name = ["Z", "Y", "X"][plane.fixed_axis()];
    let tag = format!(
        "{}   {}={}/{}",
        plane.label(),
        axis_name,
        fixed + 1,
        shape[plane.fixed_axis()]
    );
    painter.text(
        rect.left_top() + vec2(8.0, 6.0),
        egui::Align2::LEFT_TOP,
        tag,
        egui::FontId::monospace(12.0),
        Color32::from_gray(220),
    );

    let response = ui.interact(
        rect,
        ui.id().with(("ortho-view", plane.index())),
        Sense::click_and_drag(),
    );

    if response.hovered() {
        ctx.set_cursor_icon(egui::CursorIcon::Crosshair);
    }

    // Primary-drag pans the slice.
    if response.dragged() {
        viewer.pan[index] += response.drag_delta();
    }

    // Wheel over a view zooms around the pointer.
    if response.hovered() {
        let scroll = ctx.input(|input| input.smooth_scroll_delta.y);
        if scroll.abs() > 0.01 {
            if let Some(pointer) = ctx.input(|input| input.pointer.hover_pos()) {
                let old_zoom = viewer.zoom[index];
                let new_zoom = (old_zoom * (scroll * 0.0035).exp()).clamp(0.15, 48.0);
                let new_pan = pan_for_zoom(
                    rect,
                    shape,
                    plane,
                    scene.z_ratio,
                    old_zoom,
                    viewer.pan[index],
                    new_zoom,
                    pointer,
                );
                viewer.zoom[index] = new_zoom;
                viewer.pan[index] = new_pan;
            }
        }
    }

    if response.clicked_by(egui::PointerButton::Primary) {
        if let Some(pointer) = response.interact_pointer_pos() {
            if let Some([u, v]) = projection.data_at(pointer) {
                let voxel = plane.voxel(fixed, u, v);
                viewer.cursor = voxel;
                return Some(ViewAction::Primary { voxel });
            }
        }
    }

    if response.clicked_by(egui::PointerButton::Secondary) {
        if let Some(pointer) = response.interact_pointer_pos() {
            if let Some([u, v]) = projection.data_at(pointer) {
                let voxel = plane.voxel(fixed, u, v);
                viewer.cursor = voxel;
                let seed = nearest_seed(&projection, plane, fixed, seeds, pointer, 14.0);
                return Some(ViewAction::Secondary { voxel, seed });
            }
        }
    }

    None
}
