//! Application state, workflow transitions and the desktop UI.

use std::any::Any;
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::sync::Arc;
use std::time::{Duration, Instant};

use eframe::egui::{self, RichText};
use tissueseg3d::io;
use tissueseg3d::processing::{self, CellParams, NucleusParams, Seed, SeedParams};
use tissueseg3d::volume::{ImageStack, Volume};

use crate::view::{show_view, DisplaySettings, Plane, SceneRef, ViewAction, ViewerState};

/// Validate that every seed is inside a stack of `shape` (z, y, x).
pub fn validate_seeds(shape: [usize; 3], seeds: &[Seed]) -> Result<(), String> {
    for (index, seed) in seeds.iter().enumerate() {
        if seed[0] >= shape[0] || seed[1] >= shape[1] || seed[2] >= shape[2] {
            return Err(format!(
                "seed {} {:?} is outside stack shape {:?} (z, y, x)",
                index + 1,
                seed,
                shape
            ));
        }
    }
    Ok(())
}

/// Convert a labelled seed mask into seed coordinates, rejecting a mask whose
/// shape does not match the loaded image.
pub fn seed_centroids(labels: &Volume<u32>, image_shape: [usize; 3]) -> Result<Vec<Seed>, String> {
    if labels.shape != image_shape {
        return Err(format!(
            "seed label shape {:?} does not match image shape {:?} (z, y, x)",
            labels.shape, image_shape
        ));
    }
    Ok(processing::label_centroids(labels))
}

fn count_labels(labels: &Volume<u32>) -> usize {
    labels
        .data
        .iter()
        .copied()
        .filter(|&label| label != 0)
        .collect::<HashSet<_>>()
        .len()
}

fn contains_label(labels: &Volume<u32>, label: u32) -> bool {
    labels.data.contains(&label)
}

fn label_at(labels: &Volume<u32>, voxel: Seed) -> u32 {
    if voxel[0] >= labels.shape[0] || voxel[1] >= labels.shape[1] || voxel[2] >= labels.shape[2] {
        return 0;
    }
    labels.data[labels.idx(voxel[0], voxel[1], voxel[2])]
}

/// All user-visible workflow data. This is deliberately separate from the
/// widget layout so that invalidation and editing rules can be tested.
pub struct WorkflowState {
    pub input_path: Option<PathBuf>,
    pub dual_channel: bool,
    pub z_ratio: f64,
    pub image: Option<Arc<ImageStack>>,
    pub seeds: Vec<Seed>,
    pub cells: Option<Arc<Volume<u32>>>,
    pub nuclei: Option<Arc<Volume<u32>>>,
    pub selected_label: u32,
    pub cell_count: usize,
    pub nucleus_count: usize,
    pub data_revision: u64,
}

impl Default for WorkflowState {
    fn default() -> Self {
        Self::new()
    }
}

impl WorkflowState {
    pub fn new() -> Self {
        Self {
            input_path: None,
            dual_channel: true,
            z_ratio: 6.92,
            image: None,
            seeds: Vec::new(),
            cells: None,
            nuclei: None,
            selected_label: 0,
            cell_count: 0,
            nucleus_count: 0,
            data_revision: 0,
        }
    }

    pub fn shape(&self) -> Option<[usize; 3]> {
        self.image.as_ref().map(|image| image.membrane.shape)
    }

    fn bump_revision(&mut self) {
        self.data_revision = self.data_revision.wrapping_add(1);
    }

    fn invalidate_downstream(&mut self) {
        self.cells = None;
        self.nuclei = None;
        self.cell_count = 0;
        self.nucleus_count = 0;
        self.selected_label = 0;
        self.bump_revision();
    }

    fn ingest_image(&mut self, path: Option<PathBuf>, stack: ImageStack) {
        self.input_path = path;
        self.image = Some(Arc::new(stack));
        self.seeds.clear();
        self.invalidate_downstream();
    }

    /// Apply a completed load. On error the previous working data is retained.
    pub fn apply_load(
        &mut self,
        result: Result<(PathBuf, ImageStack), String>,
    ) -> Result<(), String> {
        let (path, stack) = result?;
        self.ingest_image(Some(path), stack);
        Ok(())
    }

    /// Apply an in-memory demo stack. Demo data has no source path.
    pub fn apply_demo(&mut self, stack: ImageStack) {
        self.ingest_image(None, stack);
    }

    /// Replace kept seeds, validating against the current image shape.
    pub fn set_seeds(&mut self, seeds: Vec<Seed>) -> Result<(), String> {
        let shape = self
            .shape()
            .ok_or_else(|| "No image is loaded".to_owned())?;
        validate_seeds(shape, &seeds)?;
        let mut seen = HashSet::new();
        self.seeds = seeds
            .into_iter()
            .filter(|seed| seen.insert(*seed))
            .collect();
        self.invalidate_downstream();
        Ok(())
    }

    /// Apply detected or label-derived seeds without losing previous data on error.
    pub fn apply_seed_load(&mut self, result: Result<Vec<Seed>, String>) -> Result<(), String> {
        let seeds = result?;
        self.set_seeds(seeds)
    }

    pub fn apply_detected_seeds(
        &mut self,
        result: Result<Vec<Seed>, String>,
    ) -> Result<(), String> {
        self.apply_seed_load(result)
    }

    /// Add one manually placed seed. Duplicates and out-of-bounds points are
    /// rejected without changing downstream data.
    pub fn add_seed(&mut self, seed: Seed) -> bool {
        let Some(shape) = self.shape() else {
            return false;
        };
        if validate_seeds(shape, &[seed]).is_err() || self.seeds.contains(&seed) {
            return false;
        }
        self.seeds.push(seed);
        self.invalidate_downstream();
        true
    }

    /// Remove the kept seed closest to `target`, using physical z spacing.
    pub fn remove_nearest_seed(&mut self, target: Seed) -> Option<Seed> {
        let mut best: Option<(f64, usize)> = None;
        for (index, seed) in self.seeds.iter().enumerate() {
            let dz = (seed[0] as f64 - target[0] as f64) * self.z_ratio;
            let dy = seed[1] as f64 - target[1] as f64;
            let dx = seed[2] as f64 - target[2] as f64;
            let distance = dz * dz + dy * dy + dx * dx;
            if best.is_none_or(|(best_distance, _)| distance < best_distance) {
                best = Some((distance, index));
            }
        }
        let (_, index) = best?;
        let removed = self.seeds.remove(index);
        self.invalidate_downstream();
        Some(removed)
    }

    /// Change the physical z spacing. Results computed with the old spacing are stale.
    pub fn set_z_ratio(&mut self, z_ratio: f64) {
        if !z_ratio.is_finite() || z_ratio <= 0.0 || (self.z_ratio - z_ratio).abs() < f64::EPSILON {
            return;
        }
        self.z_ratio = z_ratio;
        self.invalidate_downstream();
    }

    pub fn apply_cells(&mut self, result: Result<Volume<u32>, String>) -> Result<(), String> {
        let cells = result?;
        if let Some(shape) = self.shape() {
            if cells.shape != shape {
                return Err(format!(
                    "cell mask shape {:?} does not match image shape {:?}",
                    cells.shape, shape
                ));
            }
        }
        self.cell_count = count_labels(&cells);
        self.cells = Some(Arc::new(cells));
        self.nuclei = None;
        self.nucleus_count = 0;
        self.selected_label = 0;
        self.bump_revision();
        Ok(())
    }

    pub fn apply_nuclei(&mut self, result: Result<Volume<u32>, String>) -> Result<(), String> {
        let nuclei = result?;
        if self.cells.is_none() {
            return Err("nucleus labels require cell labels first".to_owned());
        }
        if let Some(shape) = self.shape() {
            if nuclei.shape != shape {
                return Err(format!(
                    "nucleus mask shape {:?} does not match image shape {:?}",
                    nuclei.shape, shape
                ));
            }
        }
        self.nucleus_count = count_labels(&nuclei);
        self.nuclei = Some(Arc::new(nuclei));
        self.bump_revision();
        Ok(())
    }

    /// Remove a label from the cell mask and, when present, the nucleus mask.
    pub fn remove_label(&mut self, label: u32) -> Result<(), String> {
        if label == 0 {
            return Err("label 0 is background and cannot be removed".to_owned());
        }
        let Some(cells) = self.cells.as_mut() else {
            return Err("no cell labels are available".to_owned());
        };
        let nuclei = self.nuclei.as_mut();
        processing::remove_label(Arc::make_mut(cells), nuclei.map(Arc::make_mut), label);
        self.cell_count = count_labels(cells.as_ref());
        if let Some(nuclei) = self.nuclei.as_ref() {
            self.nucleus_count = count_labels(nuclei.as_ref());
        }
        if self.selected_label == label {
            self.selected_label = 0;
        }
        self.bump_revision();
        Ok(())
    }

    /// Merge `remove` into `keep` in both masks and select the kept label.
    pub fn merge_labels(&mut self, keep: u32, remove: u32) -> Result<(), String> {
        if keep == 0 || remove == 0 {
            return Err("merge labels must both be non-zero".to_owned());
        }
        if keep == remove {
            return Err("cannot merge a label into itself".to_owned());
        }
        let Some(cells) = self.cells.as_mut() else {
            return Err("no cell labels are available".to_owned());
        };
        if !contains_label(cells.as_ref(), keep) {
            return Err(format!("label {keep} does not exist in the cell mask"));
        }
        if !contains_label(cells.as_ref(), remove) {
            return Err(format!("label {remove} does not exist in the cell mask"));
        }
        let nuclei = self.nuclei.as_mut();
        processing::merge_labels(
            Arc::make_mut(cells),
            nuclei.map(Arc::make_mut),
            keep,
            remove,
        );
        self.cell_count = count_labels(cells.as_ref());
        if let Some(nuclei) = self.nuclei.as_ref() {
            self.nucleus_count = count_labels(nuclei.as_ref());
        }
        self.selected_label = keep;
        self.bump_revision();
        Ok(())
    }

    pub fn has_nucleus_channel(&self) -> bool {
        self.image
            .as_ref()
            .is_some_and(|image| image.nuclei.is_some())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Startup {
    Empty,
    Demo,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JobKind {
    Load,
    LoadSeeds,
    DetectSeeds,
    SegmentCells,
    SegmentNuclei,
    Export,
    Demo,
}

impl JobKind {
    fn label(self) -> &'static str {
        match self {
            Self::Load => "Loading TIFF",
            Self::LoadSeeds => "Loading seed labels",
            Self::DetectSeeds => "Detecting seeds",
            Self::SegmentCells => "Segmenting cells",
            Self::SegmentNuclei => "Segmenting nuclei",
            Self::Export => "Exporting masks",
            Self::Demo => "Building demo stack",
        }
    }
}

struct LoadedImage {
    path: Option<PathBuf>,
    stack: ImageStack,
    membrane_range: [u16; 2],
    nuclei_range: Option<[u16; 2]>,
}

enum JobPayload {
    Loaded(Result<LoadedImage, String>),
    Seeds(Result<Vec<Seed>, String>),
    Cells(Result<Volume<u32>, String>),
    Nuclei(Result<Volume<u32>, String>),
    Export(Result<Vec<PathBuf>, String>),
    Panic(String),
}

struct JobMessage {
    kind: JobKind,
    elapsed: Duration,
    payload: JobPayload,
}

struct JobState {
    kind: JobKind,
    started: Instant,
}

pub struct App {
    pub state: WorkflowState,
    pub view: ViewerState,
    pub display: DisplaySettings,
    pub seed_params: SeedParams,
    pub cell_params: CellParams,
    pub nucleus_params: NucleusParams,
    pub seed_edit: bool,
    pub merge_keep: u32,
    pub merge_remove: u32,
    pub status: String,
    pub last_error: Option<String>,
    pub last_timing: Option<(String, Duration)>,
    pub output_dir: Option<PathBuf>,
    job: Option<JobState>,
    receiver: Option<Receiver<JobMessage>>,
    show_about: bool,
    startup_demo: bool,
}

impl App {
    pub fn new(cc: &eframe::CreationContext<'_>, startup: Startup) -> Self {
        configure_style(&cc.egui_ctx);
        Self {
            state: WorkflowState::new(),
            view: ViewerState::new(),
            display: DisplaySettings::default(),
            seed_params: SeedParams::default(),
            cell_params: CellParams::default(),
            nucleus_params: NucleusParams::default(),
            seed_edit: true,
            merge_keep: 1,
            merge_remove: 2,
            status: "Open a TIFF stack or the demo to begin".to_owned(),
            last_error: None,
            last_timing: None,
            output_dir: None,
            job: None,
            receiver: None,
            show_about: false,
            startup_demo: startup == Startup::Demo,
        }
    }

    fn busy(&self) -> bool {
        self.job.is_some()
    }

    fn start_job<F>(&mut self, kind: JobKind, ctx: &egui::Context, work: F)
    where
        F: FnOnce() -> JobPayload + Send + 'static,
    {
        if self.busy() {
            return;
        }
        let (sender, receiver) = mpsc::channel();
        let started = Instant::now();
        let repaint_ctx = ctx.clone();
        self.job = Some(JobState { kind, started });
        self.receiver = Some(receiver);
        self.status = format!("{}...", kind.label());
        self.last_error = None;

        std::thread::spawn(move || {
            let payload = std::panic::catch_unwind(std::panic::AssertUnwindSafe(work))
                .unwrap_or_else(|panic| JobPayload::Panic(panic_text(panic)));
            let _ = sender.send(JobMessage {
                kind,
                elapsed: started.elapsed(),
                payload,
            });
            repaint_ctx.request_repaint();
        });
    }

    fn poll_job(&mut self) {
        let mut message = None;
        let mut disconnected = false;
        if let Some(receiver) = self.receiver.as_ref() {
            match receiver.try_recv() {
                Ok(received) => message = Some(received),
                Err(TryRecvError::Empty) => {}
                Err(TryRecvError::Disconnected) => disconnected = true,
            }
        }
        if let Some(message) = message {
            self.receiver = None;
            self.job = None;
            self.apply_job_message(message);
        } else if disconnected {
            let kind = self.job.as_ref().map(|job| job.kind);
            self.receiver = None;
            self.job = None;
            let name = kind.map_or("Worker", JobKind::label);
            self.last_error = Some(format!("{name} stopped without returning a result"));
            self.status = "Worker stopped unexpectedly".to_owned();
        }
    }

    fn apply_job_message(&mut self, message: JobMessage) {
        let JobMessage {
            kind,
            elapsed,
            payload,
        } = message;
        match payload {
            JobPayload::Loaded(result) => match result {
                Ok(loaded) => {
                    let LoadedImage {
                        path,
                        stack,
                        membrane_range,
                        nuclei_range,
                    } = loaded;
                    let shape = stack.membrane.shape;
                    let channels = if stack.nuclei.is_some() { 2 } else { 1 };
                    let label = path
                        .as_ref()
                        .and_then(|path| path.file_name())
                        .map(|name| name.to_string_lossy().into_owned())
                        .unwrap_or_else(|| "demo stack".to_owned());
                    let applied = match path {
                        Some(path) => self.state.apply_load(Ok((path, stack))),
                        None => {
                            self.state.apply_demo(stack);
                            Ok(())
                        }
                    };
                    match applied {
                        Ok(()) => {
                            self.view.reset_for_shape(shape);
                            self.display.membrane_range = membrane_range;
                            if let Some(range) = nuclei_range {
                                self.display.nuclei_range = range;
                            }
                            self.finish_ok(
                                format!(
                                    "Loaded {label}: z={} y={} x={}, {channels} channel{}",
                                    shape[0],
                                    shape[1],
                                    shape[2],
                                    if channels == 1 { "" } else { "s" }
                                ),
                                elapsed,
                            );
                        }
                        Err(error) => self.finish_error(kind, elapsed, error),
                    }
                }
                Err(error) => self.finish_error(kind, elapsed, error),
            },
            JobPayload::Seeds(result) => {
                let applied = if kind == JobKind::LoadSeeds {
                    self.state.apply_seed_load(result)
                } else {
                    self.state.apply_detected_seeds(result)
                };
                match applied {
                    Ok(()) => {
                        let count = self.state.seeds.len();
                        let verb = if kind == JobKind::LoadSeeds {
                            "Loaded"
                        } else {
                            "Detected"
                        };
                        self.finish_ok(format!("{verb} {count} seeds"), elapsed);
                    }
                    Err(error) => self.finish_error(kind, elapsed, error),
                }
            }
            JobPayload::Cells(result) => match self.state.apply_cells(result) {
                Ok(()) => {
                    let count = self.state.cell_count;
                    self.merge_keep = 1;
                    self.merge_remove = 2;
                    self.finish_ok(format!("Segmented {count} cells"), elapsed);
                }
                Err(error) => self.finish_error(kind, elapsed, error),
            },
            JobPayload::Nuclei(result) => match self.state.apply_nuclei(result) {
                Ok(()) => {
                    let count = self.state.nucleus_count;
                    self.finish_ok(format!("Segmented {count} nuclei"), elapsed);
                }
                Err(error) => self.finish_error(kind, elapsed, error),
            },
            JobPayload::Export(result) => match result {
                Ok(paths) => {
                    let text = paths
                        .iter()
                        .map(|path| path.display().to_string())
                        .collect::<Vec<_>>()
                        .join(", ");
                    self.finish_ok(format!("Exported {text}"), elapsed);
                }
                Err(error) => self.finish_error(kind, elapsed, error),
            },
            JobPayload::Panic(text) => {
                self.finish_error(kind, elapsed, format!("worker panicked: {text}"));
            }
        }
    }

    fn finish_ok(&mut self, message: String, elapsed: Duration) {
        self.status = format!("{message}  ({:.2} s)", elapsed.as_secs_f64());
        self.last_timing = Some((message, elapsed));
        self.last_error = None;
    }

    fn finish_error(&mut self, kind: JobKind, elapsed: Duration, error: String) {
        self.status = format!(
            "{} failed after {:.2} s",
            kind.label(),
            elapsed.as_secs_f64()
        );
        self.last_error = Some(format!("{}: {error}", kind.label()));
        self.last_timing = Some((format!("{} (failed)", kind.label()), elapsed));
    }

    fn start_load(&mut self, ctx: &egui::Context, path: PathBuf) {
        if self.busy() {
            return;
        }
        let dual_channel = self.state.dual_channel;
        self.start_job(JobKind::Load, ctx, move || {
            let result = io::load_tiff(&path, dual_channel)
                .map(|stack| {
                    let membrane_range = channel_range(&stack.membrane);
                    let nuclei_range = stack.nuclei.as_ref().map(channel_range);
                    LoadedImage {
                        path: Some(path.clone()),
                        stack,
                        membrane_range,
                        nuclei_range,
                    }
                })
                .map_err(format_error);
            JobPayload::Loaded(result)
        });
    }

    fn start_demo(&mut self, ctx: &egui::Context) {
        if self.busy() {
            return;
        }
        self.start_job(JobKind::Demo, ctx, || {
            let stack = io::demo_stack();
            let membrane_range = channel_range(&stack.membrane);
            let nuclei_range = stack.nuclei.as_ref().map(channel_range);
            JobPayload::Loaded(Ok(LoadedImage {
                path: None,
                stack,
                membrane_range,
                nuclei_range,
            }))
        });
    }

    fn start_load_seeds(&mut self, ctx: &egui::Context, path: PathBuf) {
        if self.busy() {
            return;
        }
        let Some(image_shape) = self.state.shape() else {
            self.last_error = Some("Load an image before loading seed labels".to_owned());
            return;
        };
        self.start_job(JobKind::LoadSeeds, ctx, move || {
            let result = io::load_labels(&path)
                .map_err(format_error)
                .and_then(|labels| seed_centroids(&labels, image_shape));
            JobPayload::Seeds(result)
        });
    }

    fn start_detect_seeds(&mut self, ctx: &egui::Context) {
        if self.busy() {
            return;
        }
        let Some(image) = self.state.image.clone() else {
            self.last_error = Some("Load an image before detecting seeds".to_owned());
            return;
        };
        if image.nuclei.is_none() {
            self.last_error =
                Some("Seed detection needs a nucleus channel; add seeds manually".to_owned());
            return;
        }
        if self.seed_params.min_sigma > self.seed_params.max_sigma {
            self.last_error = Some("Minimum seed sigma must not exceed maximum".to_owned());
            return;
        }
        if self.seed_params.min_sigma <= 0.0
            || !self.seed_params.max_sigma.is_finite()
            || self.seed_params.max_sigma <= 0.0
            || !self.seed_params.threshold.is_finite()
            || self.seed_params.threshold < 0.0
            || !self.seed_params.merge_distance.is_finite()
            || self.seed_params.merge_distance < 0.0
            || !self.seed_params.membrane_delta.is_finite()
            || self.seed_params.membrane_delta < 0.0
        {
            self.last_error = Some("Seed parameters must be finite and non-negative".to_owned());
            return;
        }
        self.seed_params.z_ratio = self.state.z_ratio;
        let params = self.seed_params.clone();
        self.start_job(JobKind::DetectSeeds, ctx, move || {
            let result = processing::detect_seeds(image.as_ref(), &params).map_err(format_error);
            JobPayload::Seeds(result)
        });
    }

    fn start_segment_cells(&mut self, ctx: &egui::Context) {
        if self.busy() {
            return;
        }
        let Some(image) = self.state.image.clone() else {
            self.last_error = Some("Load an image before labeling cells".to_owned());
            return;
        };
        if self.state.seeds.is_empty() {
            self.last_error = Some("Add or detect at least one seed first".to_owned());
            return;
        }
        if self.cell_params.gaussian_sigma < 0.0
            || !self.cell_params.gaussian_sigma.is_finite()
            || self.cell_params.median_size == 0
            || self.cell_params.min_volume > self.cell_params.max_volume
        {
            self.last_error = Some(
                "Check cell parameters: non-negative sigma, median >= 1, min <= max".to_owned(),
            );
            return;
        }
        let seeds = self.state.seeds.clone();
        let params = self.cell_params.clone();
        self.start_job(JobKind::SegmentCells, ctx, move || {
            let result =
                processing::segment_cells(&image.membrane, &seeds, &params).map_err(format_error);
            JobPayload::Cells(result)
        });
    }

    fn start_segment_nuclei(&mut self, ctx: &egui::Context) {
        if self.busy() {
            return;
        }
        let Some(image) = self.state.image.clone() else {
            self.last_error = Some("Load an image before labeling nuclei".to_owned());
            return;
        };
        if image.nuclei.is_none() {
            self.last_error = Some("No nucleus channel is available".to_owned());
            return;
        }
        let Some(cells) = self.state.cells.clone() else {
            self.last_error = Some("Label cells before labeling nuclei".to_owned());
            return;
        };
        if self.nucleus_params.gaussian_sigma < 0.0
            || !self.nucleus_params.gaussian_sigma.is_finite()
            || self.nucleus_params.median_size == 0
        {
            self.last_error =
                Some("Check nucleus parameters: non-negative sigma, median >= 1".to_owned());
            return;
        }
        self.nucleus_params.z_ratio = self.state.z_ratio;
        let params = self.nucleus_params.clone();
        self.start_job(JobKind::SegmentNuclei, ctx, move || {
            let channel = image
                .nuclei
                .as_ref()
                .expect("nucleus channel checked before starting job");
            let result =
                processing::segment_nuclei(channel, cells.as_ref(), &params).map_err(format_error);
            JobPayload::Nuclei(result)
        });
    }

    fn export(&mut self, ctx: &egui::Context) {
        if self.busy() {
            return;
        }
        let Some(cells) = self.state.cells.clone() else {
            self.last_error = Some("Segment cells before exporting".to_owned());
            return;
        };
        let default_dir = self
            .output_dir
            .clone()
            .or_else(|| {
                self.state
                    .input_path
                    .as_ref()
                    .and_then(|path| path.parent().map(PathBuf::from))
            })
            .or_else(|| std::env::current_dir().ok())
            .unwrap_or_else(|| PathBuf::from("."));
        let Some(directory) = rfd::FileDialog::new()
            .set_title("Export masks and measurements to folder")
            .set_directory(&default_dir)
            .pick_folder()
        else {
            self.status = "Export cancelled".to_owned();
            return;
        };
        let base_path = match self
            .state
            .input_path
            .as_ref()
            .and_then(|path| path.file_name())
        {
            Some(name) => directory.join(name),
            None => directory.join("demo.tif"),
        };
        let nuclei = self.state.nuclei.clone();
        self.output_dir = Some(directory);
        self.start_job(JobKind::Export, ctx, move || {
            let result = (|| -> Result<Vec<PathBuf>, String> {
                let mut outputs = Vec::new();
                let (mask, csv) =
                    io::export_labels(&base_path, cells.as_ref(), false).map_err(format_error)?;
                outputs.push(mask);
                outputs.push(csv);
                if let Some(nuclei) = nuclei {
                    let (mask, csv) = io::export_labels(&base_path, nuclei.as_ref(), true)
                        .map_err(format_error)?;
                    outputs.push(mask);
                    outputs.push(csv);
                }
                Ok(outputs)
            })();
            JobPayload::Export(result)
        });
    }

    fn pick_open(&mut self, ctx: &egui::Context) {
        let Some(path) = rfd::FileDialog::new()
            .add_filter("TIFF stack", &["tif", "tiff"])
            .set_title("Open membrane/nucleus TIFF stack")
            .pick_file()
        else {
            return;
        };
        self.start_load(ctx, path);
    }

    fn pick_seed_labels(&mut self, ctx: &egui::Context) {
        let Some(path) = rfd::FileDialog::new()
            .add_filter("Label TIFF", &["tif", "tiff"])
            .set_title("Load nucleus labels as seeds")
            .pick_file()
        else {
            return;
        };
        self.start_load_seeds(ctx, path);
    }

    fn handle_dropped_files(&mut self, ctx: &egui::Context) {
        if self.busy() {
            return;
        }
        let dropped = ctx.input(|input| input.raw.dropped_files.clone());
        let Some(path) = dropped.into_iter().find_map(|file| file.path) else {
            return;
        };
        let is_tiff = path
            .extension()
            .and_then(|extension| extension.to_str())
            .map(|extension| {
                let extension = extension.to_ascii_lowercase();
                extension == "tif" || extension == "tiff"
            })
            .unwrap_or(false);
        if is_tiff {
            self.start_load(ctx, path);
        } else {
            self.last_error = Some(format!("Dropped file is not a TIFF: {}", path.display()));
        }
    }

    fn handle_view_action(&mut self, action: ViewAction) {
        match action {
            ViewAction::Primary { voxel } => {
                self.view.cursor = voxel;
                if self.busy() {
                    return;
                }
                if self.seed_edit {
                    if self.state.add_seed(voxel) {
                        let count = self.state.seeds.len();
                        self.status = format!("Added seed at {voxel:?}; {count} kept seeds");
                        self.last_error = None;
                    }
                } else {
                    let cell_label = self
                        .state
                        .cells
                        .as_ref()
                        .map_or(0, |labels| label_at(labels, voxel));
                    let nucleus_label = self
                        .state
                        .nuclei
                        .as_ref()
                        .map_or(0, |labels| label_at(labels, voxel));
                    let label = if cell_label != 0 {
                        cell_label
                    } else {
                        nucleus_label
                    };
                    self.state.selected_label = label;
                    if label != 0 {
                        self.merge_keep = label;
                    }
                    self.status = if label == 0 {
                        "No label at cursor".to_owned()
                    } else {
                        format!("Selected label {label}")
                    };
                }
            }
            ViewAction::Secondary { voxel, seed } => {
                self.view.cursor = voxel;
                if self.busy() {
                    return;
                }
                if let Some(seed) = seed {
                    if self.state.remove_nearest_seed(seed).is_some() {
                        let count = self.state.seeds.len();
                        self.status = format!("Removed seed; {count} kept seeds");
                        self.last_error = None;
                    }
                }
            }
        }
    }

    fn set_z_ratio(&mut self, z_ratio: f64) {
        self.state.set_z_ratio(z_ratio);
        self.seed_params.z_ratio = self.state.z_ratio;
        self.nucleus_params.z_ratio = self.state.z_ratio;
    }

    fn add_seed_at_cursor(&mut self) {
        if self.busy() {
            return;
        }
        let seed = self.view.cursor;
        if self.state.add_seed(seed) {
            self.status = format!(
                "Added seed at {seed:?}; {} kept seeds",
                self.state.seeds.len()
            );
            self.last_error = None;
        } else {
            self.last_error = Some("Seed is already present or outside the image".to_owned());
        }
    }

    fn remove_seed_at_cursor(&mut self) {
        if self.busy() {
            return;
        }
        let cursor = self.view.cursor;
        if let Some(seed) = self.state.remove_nearest_seed(cursor) {
            self.status = format!(
                "Removed seed at {seed:?}; {} kept seeds",
                self.state.seeds.len()
            );
            self.last_error = None;
        }
    }

    fn remove_selected_label(&mut self) {
        if self.busy() {
            return;
        }
        let label = self.state.selected_label;
        match self.state.remove_label(label) {
            Ok(()) => {
                self.status = format!("Removed label {label}");
                self.last_error = None;
            }
            Err(error) => self.last_error = Some(error),
        }
    }

    fn merge_selected_labels(&mut self) {
        if self.busy() {
            return;
        }
        let keep = self.merge_keep;
        let remove = self.merge_remove;
        match self.state.merge_labels(keep, remove) {
            Ok(()) => {
                self.status = format!("Merged label {remove} into {keep}");
                self.last_error = None;
            }
            Err(error) => self.last_error = Some(error),
        }
    }

    fn top_bar(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::top("top-bar")
            .frame(
                egui::Frame::default()
                    .fill(egui::Color32::from_rgb(29, 31, 37))
                    .inner_margin(egui::Margin::symmetric(12, 7)),
            )
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.heading(RichText::new("TissueSeg3D").strong());
                    ui.label(
                        RichText::new("cell and nucleus segmentation")
                            .small()
                            .weak(),
                    );
                    ui.separator();
                    let busy = self.busy();
                    ui.add_enabled_ui(!busy, |ui| {
                        if ui.button("Open TIFF").clicked() {
                            self.pick_open(ctx);
                        }
                        if ui.button("Demo").clicked() {
                            self.start_demo(ctx);
                        }
                        if ui.button("Export...").clicked() {
                            self.export(ctx);
                        }
                    });
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.button("About").clicked() {
                            self.show_about = true;
                        }
                    });
                });
            });
    }

    fn status_bar(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::bottom("status-bar")
            .frame(
                egui::Frame::default()
                    .fill(egui::Color32::from_rgb(27, 29, 34))
                    .inner_margin(egui::Margin::symmetric(10, 5)),
            )
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    if let Some(job) = self.job.as_ref() {
                        ui.spinner();
                        ui.label(format!(
                            "{}... {:.1} s",
                            job.kind.label(),
                            job.started.elapsed().as_secs_f64()
                        ));
                    } else {
                        ui.label(&self.status);
                    }
                    ui.separator();
                    ui.label(format!("{} seeds", self.state.seeds.len()));
                    ui.label(format!("{} cells", self.state.cell_count));
                    ui.label(format!("{} nuclei", self.state.nucleus_count));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if let Some((operation, elapsed)) = self.last_timing.as_ref() {
                            ui.label(
                                RichText::new(format!(
                                    "{operation}: {:.2} s",
                                    elapsed.as_secs_f64()
                                ))
                                .small()
                                .weak(),
                            );
                        }
                    });
                });
                if let Some(error) = self.last_error.as_ref() {
                    ui.colored_label(egui::Color32::from_rgb(255, 135, 135), error);
                }
            });
    }

    fn side_panel(&mut self, ctx: &egui::Context) {
        egui::SidePanel::left("workflow-panel")
            .resizable(true)
            .default_width(375.0)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        self.load_section(ui, ctx);
                        ui.separator();
                        self.seed_section(ui, ctx);
                        ui.separator();
                        self.cell_section(ui, ctx);
                        ui.separator();
                        self.nucleus_section(ui, ctx);
                        ui.separator();
                        self.edit_section(ui);
                        ui.separator();
                        self.display_section(ui);
                        ui.separator();
                        self.view_section(ui);
                        ui.add_space(8.0);
                    });
            });
    }

    fn load_section(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.heading("1. Stack");
        match (self.state.input_path.as_ref(), self.state.image.as_ref()) {
            (Some(path), Some(image)) => {
                ui.label(RichText::new(path.display().to_string()).small());
                ui.label(
                    RichText::new(format!(
                        "z={} y={} x={} / {} channel{}",
                        image.membrane.shape[0],
                        image.membrane.shape[1],
                        image.membrane.shape[2],
                        if image.nuclei.is_some() { 2 } else { 1 },
                        if image.nuclei.is_some() { "s" } else { "" }
                    ))
                    .small()
                    .weak(),
                );
            }
            (None, Some(image)) => {
                ui.label(RichText::new("Demo stack").small());
                ui.label(
                    RichText::new(format!(
                        "z={} y={} x={} / {} channel{}",
                        image.membrane.shape[0],
                        image.membrane.shape[1],
                        image.membrane.shape[2],
                        if image.nuclei.is_some() { 2 } else { 1 },
                        if image.nuclei.is_some() { "s" } else { "" }
                    ))
                    .small()
                    .weak(),
                );
            }
            _ => {
                ui.label("No stack loaded.");
            }
        }

        ui.horizontal(|ui| {
            let busy = self.busy();
            if ui
                .add_enabled(!busy, egui::Button::new("Open TIFF..."))
                .clicked()
            {
                self.pick_open(ctx);
            }
            if ui
                .add_enabled(!busy, egui::Button::new("Open demo"))
                .clicked()
            {
                self.start_demo(ctx);
            }
        });
        ui.checkbox(
            &mut self.state.dual_channel,
            "Two channels (nucleus + membrane)",
        );
        let mut z_ratio = self.state.z_ratio;
        ui.horizontal(|ui| {
            ui.label("Z ratio");
            let changed = ui
                .add(
                    egui::DragValue::new(&mut z_ratio)
                        .speed(0.05)
                        .range(0.01..=100.0)
                        .fixed_decimals(2),
                )
                .changed();
            if changed {
                self.set_z_ratio(z_ratio);
            }
        });
        ui.label(
            RichText::new("Z ratio is physical Z spacing / XY pixel size.")
                .small()
                .weak(),
        );
        ui.label(
            RichText::new("Tip: drop a .tif/.tiff file onto the window to open it.")
                .small()
                .weak(),
        );
    }

    fn seed_section(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.heading("2. Detect seeds");
        ui.label(
            RichText::new("Requires the nucleus channel. Single-channel stacks use manual seeds.")
                .small()
                .weak(),
        );
        egui::Grid::new("seed-params")
            .num_columns(2)
            .spacing([10.0, 5.0])
            .show(ui, |ui| {
                ui.label("Scale min / max");
                ui.horizontal(|ui| {
                    ui.add(
                        egui::DragValue::new(&mut self.seed_params.min_sigma)
                            .speed(0.1)
                            .range(1.0..=16.0)
                            .fixed_decimals(1),
                    );
                    ui.add(
                        egui::DragValue::new(&mut self.seed_params.max_sigma)
                            .speed(0.1)
                            .range(1.0..=16.0)
                            .fixed_decimals(1),
                    );
                });
                ui.end_row();

                ui.label("Detection threshold");
                ui.add(egui::Slider::new(&mut self.seed_params.threshold, 0.0..=2.0).step_by(0.05));
                ui.end_row();

                ui.label("Merge distance");
                ui.add(
                    egui::DragValue::new(&mut self.seed_params.merge_distance)
                        .speed(1.0)
                        .range(1.0..=250.0)
                        .fixed_decimals(0),
                );
                ui.end_row();

                ui.label("Membrane delta");
                ui.add(
                    egui::DragValue::new(&mut self.seed_params.membrane_delta)
                        .speed(1.0)
                        .range(1.0..=100.0)
                        .fixed_decimals(0),
                );
                ui.end_row();
            });

        let can_detect = !self.busy() && self.state.has_nucleus_channel();
        ui.horizontal(|ui| {
            if ui
                .add_enabled(can_detect, egui::Button::new("Detect seeds"))
                .clicked()
            {
                self.start_detect_seeds(ctx);
            }
            if ui
                .add_enabled(
                    !self.busy() && self.state.image.is_some(),
                    egui::Button::new("From labels..."),
                )
                .clicked()
            {
                self.pick_seed_labels(ctx);
            }
        });
        ui.label(format!("Kept seeds: {}", self.state.seeds.len()));
    }

    fn cell_section(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.heading("3. Label cells");
        egui::Grid::new("cell-params")
            .num_columns(2)
            .spacing([10.0, 5.0])
            .show(ui, |ui| {
                ui.label("Gaussian sigma");
                ui.add(
                    egui::Slider::new(&mut self.cell_params.gaussian_sigma, 0.0..=1.5)
                        .step_by(0.05),
                );
                ui.end_row();

                ui.label("Median size");
                ui.add(
                    egui::DragValue::new(&mut self.cell_params.median_size)
                        .range(1..=9)
                        .speed(0.2),
                );
                ui.end_row();

                ui.label("Min volume");
                ui.add(
                    egui::DragValue::new(&mut self.cell_params.min_volume)
                        .range(1..=5_000)
                        .speed(10.0),
                );
                ui.end_row();

                ui.label("Max volume");
                ui.add(
                    egui::DragValue::new(&mut self.cell_params.max_volume)
                        .range(1..=200_000)
                        .speed(100.0),
                );
                ui.end_row();
            });
        let can_label = !self.busy() && self.state.image.is_some() && !self.state.seeds.is_empty();
        if ui
            .add_enabled(can_label, egui::Button::new("Label cells"))
            .clicked()
        {
            self.start_segment_cells(ctx);
        }
        ui.label(format!("Segmented cells: {}", self.state.cell_count));
    }

    fn nucleus_section(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.heading("4. Label nuclei");
        egui::Grid::new("nucleus-params")
            .num_columns(2)
            .spacing([10.0, 5.0])
            .show(ui, |ui| {
                ui.label("Gaussian sigma");
                ui.add(
                    egui::Slider::new(&mut self.nucleus_params.gaussian_sigma, 0.0..=3.0)
                        .step_by(0.05),
                );
                ui.end_row();

                ui.label("Threshold");
                ui.add(
                    egui::DragValue::new(&mut self.nucleus_params.threshold)
                        .range(0..=9999)
                        .speed(5.0),
                );
                ui.end_row();

                ui.label("Median size");
                ui.add(
                    egui::DragValue::new(&mut self.nucleus_params.median_size)
                        .range(1..=9)
                        .speed(0.2),
                );
                ui.end_row();
            });
        let can_label =
            !self.busy() && self.state.has_nucleus_channel() && self.state.cells.is_some();
        if ui
            .add_enabled(can_label, egui::Button::new("Label nuclei"))
            .clicked()
        {
            self.start_segment_nuclei(ctx);
        }
        ui.label(format!("Segmented nuclei: {}", self.state.nucleus_count));
    }

    fn edit_section(&mut self, ui: &mut egui::Ui) {
        ui.heading("5. Edit");
        ui.checkbox(
            &mut self.seed_edit,
            "Seed editing: left-click adds, right-click removes",
        );
        ui.label(
            RichText::new("Turn seed editing off to select the label under a click.")
                .small()
                .weak(),
        );
        let busy = self.busy();
        ui.horizontal(|ui| {
            if ui
                .add_enabled(!busy, egui::Button::new("Add seed at cursor"))
                .clicked()
            {
                self.add_seed_at_cursor();
            }
            if ui
                .add_enabled(!busy, egui::Button::new("Remove nearest seed"))
                .clicked()
            {
                self.remove_seed_at_cursor();
            }
        });

        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.label("Selected label");
            let selected_changed = ui
                .add_enabled(
                    !busy,
                    egui::DragValue::new(&mut self.state.selected_label)
                        .range(0..=1_000_000)
                        .speed(1.0),
                )
                .changed();
            if selected_changed && self.state.selected_label != 0 {
                self.merge_keep = self.state.selected_label;
            }
            if ui
                .add_enabled(
                    !busy && self.state.cells.is_some(),
                    egui::Button::new("Remove label"),
                )
                .clicked()
            {
                self.remove_selected_label();
            }
        });
        ui.horizontal(|ui| {
            ui.label("Merge");
            ui.add_enabled(
                !busy,
                egui::DragValue::new(&mut self.merge_remove).range(1..=1_000_000),
            );
            ui.label("into");
            ui.add_enabled(
                !busy,
                egui::DragValue::new(&mut self.merge_keep).range(1..=1_000_000),
            );
            if ui
                .add_enabled(
                    !busy && self.state.cells.is_some(),
                    egui::Button::new("Merge"),
                )
                .clicked()
            {
                self.merge_selected_labels();
            }
        });
    }

    fn display_section(&mut self, ui: &mut egui::Ui) {
        ui.heading("Display");
        ui.horizontal_wrapped(|ui| {
            ui.checkbox(&mut self.display.show_membrane, "Membrane");
            ui.checkbox(&mut self.display.show_nuclei, "Nuclei channel");
            ui.checkbox(&mut self.display.show_cells, "Cell labels");
            ui.checkbox(&mut self.display.show_nuclei_labels, "Nucleus labels");
        });
        egui::Grid::new("contrast")
            .num_columns(3)
            .spacing([8.0, 4.0])
            .show(ui, |ui| {
                ui.label("");
                ui.label(RichText::new("black").small());
                ui.label(RichText::new("white").small());
                ui.end_row();

                ui.label("Membrane");
                ui.add(
                    egui::DragValue::new(&mut self.display.membrane_range[0])
                        .range(0..=65_534)
                        .speed(100.0),
                );
                ui.add(
                    egui::DragValue::new(&mut self.display.membrane_range[1])
                        .range(1..=65_535)
                        .speed(100.0),
                );
                ui.end_row();

                ui.label("Nuclei");
                ui.add(
                    egui::DragValue::new(&mut self.display.nuclei_range[0])
                        .range(0..=65_534)
                        .speed(100.0),
                );
                ui.add(
                    egui::DragValue::new(&mut self.display.nuclei_range[1])
                        .range(1..=65_535)
                        .speed(100.0),
                );
                ui.end_row();
            });
        if self.display.membrane_range[0] >= self.display.membrane_range[1] {
            self.display.membrane_range[1] = self.display.membrane_range[0].saturating_add(1);
        }
        if self.display.nuclei_range[0] >= self.display.nuclei_range[1] {
            self.display.nuclei_range[1] = self.display.nuclei_range[0].saturating_add(1);
        }
        ui.horizontal(|ui| {
            ui.label("Cell opacity");
            ui.add(egui::Slider::new(&mut self.display.cell_opacity, 0..=255));
        });
        ui.horizontal(|ui| {
            ui.label("Nucleus opacity");
            ui.add(egui::Slider::new(
                &mut self.display.nucleus_opacity,
                0..=255,
            ));
        });
    }

    fn view_section(&mut self, ui: &mut egui::Ui) {
        ui.heading("View");
        if let Some(shape) = self.state.shape() {
            self.view.clamp_cursor(shape);
            ui.add(
                egui::Slider::new(&mut self.view.cursor[0], 0..=shape[0].saturating_sub(1))
                    .text("Z slice"),
            );
            ui.add(
                egui::Slider::new(&mut self.view.cursor[1], 0..=shape[1].saturating_sub(1))
                    .text("Y slice"),
            );
            ui.add(
                egui::Slider::new(&mut self.view.cursor[2], 0..=shape[2].saturating_sub(1))
                    .text("X slice"),
            );
            ui.label(
                RichText::new(format!(
                    "Cursor: z={} y={} x={}",
                    self.view.cursor[0], self.view.cursor[1], self.view.cursor[2]
                ))
                .small(),
            );
        } else {
            ui.label("No stack loaded.");
        }
        if ui.button("Reset zoom / pan").clicked() {
            self.view.reset_zoom();
        }
        ui.label(
            RichText::new("Drag to pan, scroll over a view to zoom.")
                .small()
                .weak(),
        );
    }

    fn central_viewer(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default()
            .frame(
                egui::Frame::NONE
                    .fill(egui::Color32::from_rgb(13, 14, 17))
                    .inner_margin(egui::Margin::same(4)),
            )
            .show(ctx, |ui| {
                let Some(image) = self.state.image.clone() else {
                    ui.centered_and_justified(|ui| {
                        ui.label(
                            RichText::new("Open a TIFF stack or the demo to begin")
                                .size(18.0)
                                .weak(),
                        );
                    });
                    return;
                };
                let shape = image.membrane.shape;
                self.view.clamp_cursor(shape);

                let available = ui.available_rect_before_wrap();
                if available.width() < 40.0 || available.height() < 40.0 {
                    return;
                }
                let gap = 6.0;
                let top_height = ((available.height() - gap) * 0.58).max(60.0);
                let bottom_top = available.top() + top_height + gap;
                let bottom_height = (available.bottom() - bottom_top).max(40.0);
                let half_width = ((available.width() - gap) * 0.5).max(40.0);

                let xy_rect = egui::Rect::from_min_size(
                    available.min,
                    egui::vec2(available.width(), top_height),
                );
                let xz_rect = egui::Rect::from_min_size(
                    egui::pos2(available.left(), bottom_top),
                    egui::vec2(half_width, bottom_height),
                );
                let yz_rect = egui::Rect::from_min_size(
                    egui::pos2(available.left() + half_width + gap, bottom_top),
                    egui::vec2(half_width, bottom_height),
                );

                let scene = SceneRef {
                    image: image.as_ref(),
                    cells: self.state.cells.as_deref(),
                    nuclei: self.state.nuclei.as_deref(),
                    selected_label: self.state.selected_label,
                    z_ratio: self.state.z_ratio,
                    revision: self.state.data_revision,
                };
                let mut actions = Vec::new();
                if let Some(action) = show_view(
                    ui,
                    ctx,
                    scene,
                    &self.display,
                    &mut self.view,
                    Plane::Xy,
                    xy_rect,
                    &self.state.seeds,
                ) {
                    actions.push(action);
                }
                if let Some(action) = show_view(
                    ui,
                    ctx,
                    scene,
                    &self.display,
                    &mut self.view,
                    Plane::Xz,
                    xz_rect,
                    &self.state.seeds,
                ) {
                    actions.push(action);
                }
                if let Some(action) = show_view(
                    ui,
                    ctx,
                    scene,
                    &self.display,
                    &mut self.view,
                    Plane::Yz,
                    yz_rect,
                    &self.state.seeds,
                ) {
                    actions.push(action);
                }
                for action in actions {
                    self.handle_view_action(action);
                }
            });
    }

    fn about_window(&mut self, ctx: &egui::Context) {
        if !self.show_about {
            return;
        }
        egui::Window::new("About TissueSeg3D")
            .open(&mut self.show_about)
            .resizable(false)
            .collapsible(false)
            .show(ctx, |ui| {
                ui.label(format!("Version {}", env!("CARGO_PKG_VERSION")));
                ui.label("Native cell and nucleus segmentation for membrane/nucleus TIFF stacks.");
                ui.add_space(4.0);
                ui.label(
                    RichText::new(
                        "Display overlays are temporary textures; segmentation and export use the stored masks.",
                    )
                    .small()
                    .weak(),
                );
            });
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll_job();
        if self.startup_demo {
            self.startup_demo = false;
            self.start_demo(ctx);
        }
        if self.busy() {
            ctx.request_repaint_after(Duration::from_millis(100));
        }
        self.handle_dropped_files(ctx);
        self.top_bar(ctx);
        self.status_bar(ctx);
        self.side_panel(ctx);
        self.central_viewer(ctx);
        self.about_window(ctx);
    }
}

fn configure_style(ctx: &egui::Context) {
    ctx.set_visuals(egui::Visuals::dark());
    let mut style = (*ctx.style()).clone();
    style.spacing.item_spacing = egui::vec2(8.0, 6.0);
    style.spacing.button_padding = egui::vec2(8.0, 4.0);
    ctx.set_style(style);
}

fn channel_range(volume: &Volume<u16>) -> [u16; 2] {
    let mut low = u16::MAX;
    let mut high = 0u16;
    for &value in &volume.data {
        low = low.min(value);
        high = high.max(value);
    }
    if low == u16::MAX && high == 0 {
        return [0, 65_535];
    }
    if high <= low {
        high = low.saturating_add(1);
    }
    [low, high]
}

fn format_error(error: impl std::fmt::Display) -> String {
    format!("{error:#}")
}

fn panic_text(panic: Box<dyn Any + Send>) -> String {
    if let Some(message) = panic.downcast_ref::<&str>() {
        (*message).to_owned()
    } else if let Some(message) = panic.downcast_ref::<String>() {
        message.clone()
    } else {
        "unknown panic payload".to_owned()
    }
}
