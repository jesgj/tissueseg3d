//! TissueSeg3D native entry point: GUI and headless batch mode.

mod app;
mod view;

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{bail, Context};
use eframe::egui;

use app::{validate_seeds, App, Startup};
use tissueseg3d::io;
use tissueseg3d::processing::{self, CellParams, NucleusParams, Seed, SeedParams};

const HELP: &str = "\
TissueSeg3D - native cell and nucleus segmentation for TIFF stacks

Usage:
  tissueseg3d [--demo]
  tissueseg3d --batch INPUT --output-dir DIR [OPTIONS]
  tissueseg3d --batch-demo --output-dir DIR [OPTIONS]
  tissueseg3d --help | --version

GUI:
  --demo                       Open the built-in synthetic stack in the GUI
  (no arguments)               Open the empty GUI

Batch (no window):
  --batch PATH                 Input .tif/.tiff stack
  --batch-demo                 Use the built-in synthetic stack
  --output-dir DIR             Destination directory (required in batch mode)
  --single-channel             Input has no nucleus channel
  --skip-nuclei                Segment and export cells only
  --z-ratio N                  Physical Z spacing / XY pixel size (default 6.92)
  --seeds FILE                 JSON array [[z,y,x], ...] instead of detection

Seed detection parameters:
  --seed-min-sigma N           Default 4
  --seed-max-sigma N           Default 12
  --seed-threshold N           Default 1
  --merge-distance N           Default 100
  --membrane-delta N           Default 25

Cell parameters:
  --cell-sigma N               Default 0.5
  --cell-median N              Default 5
  --min-volume N               Default 2000
  --max-volume N               Default 100000

Nucleus parameters:
  --nucleus-sigma N            Default 1.5
  --nucleus-threshold N        Default 500
  --nucleus-median N           Default 5

Outputs use the input file stem in the output directory, for example
sample_lbl.tif, sample_lbl.csv, sample_nuc_lbl.tif and sample_nuc_lbl.csv.
The input file is never overwritten.
";

#[derive(Debug, Clone)]
enum Command {
    Help,
    Version,
    Gui { demo: bool },
    Batch(BatchOptions),
}

#[derive(Debug, Clone)]
struct BatchOptions {
    input: Option<PathBuf>,
    demo: bool,
    output_dir: PathBuf,
    dual_channel: bool,
    seeds_file: Option<PathBuf>,
    seed_params: SeedParams,
    cell_params: CellParams,
    nucleus_params: NucleusParams,
    skip_nuclei: bool,
}

fn next_value(iter: &mut std::vec::IntoIter<String>, flag: &str) -> Result<String, String> {
    iter.next()
        .ok_or_else(|| format!("{flag} requires a value\n\n{HELP}"))
}

fn parse_f64(iter: &mut std::vec::IntoIter<String>, flag: &str) -> Result<f64, String> {
    let raw = next_value(iter, flag)?;
    let value: f64 = raw
        .parse()
        .map_err(|_| format!("invalid value '{raw}' for {flag}"))?;
    if !value.is_finite() {
        return Err(format!("{flag} must be finite"));
    }
    Ok(value)
}

fn parse_usize(iter: &mut std::vec::IntoIter<String>, flag: &str) -> Result<usize, String> {
    let raw = next_value(iter, flag)?;
    raw.parse()
        .map_err(|_| format!("invalid non-negative integer '{raw}' for {flag}"))
}

fn parse_u16(iter: &mut std::vec::IntoIter<String>, flag: &str) -> Result<u16, String> {
    let raw = next_value(iter, flag)?;
    raw.parse()
        .map_err(|_| format!("invalid 16-bit integer '{raw}' for {flag}"))
}

/// Parse command-line arguments (excluding the executable name).
fn parse_args<I, S>(args: I) -> Result<Command, String>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let mut iter = args
        .into_iter()
        .map(Into::into)
        .collect::<Vec<_>>()
        .into_iter();
    let mut demo_gui = false;
    let mut batch_input: Option<PathBuf> = None;
    let mut batch_demo = false;
    let mut output_dir: Option<PathBuf> = None;
    let mut dual_channel = true;
    let mut z_ratio = 6.92;
    let mut seeds_file = None;
    let mut skip_nuclei = false;
    let mut seed_params = SeedParams::default();
    let mut cell_params = CellParams::default();
    let mut nucleus_params = NucleusParams::default();
    let mut used_batch_option = false;

    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--help" | "-h" => return Ok(Command::Help),
            "--version" | "-V" => return Ok(Command::Version),
            "--demo" => demo_gui = true,
            "--batch" => {
                if batch_input.is_some() || batch_demo {
                    return Err("--batch can only be given once".to_owned());
                }
                batch_input = Some(PathBuf::from(next_value(&mut iter, "--batch")?));
            }
            "--batch-demo" => {
                if batch_input.is_some() || batch_demo {
                    return Err("--batch/--batch-demo can only be given once".to_owned());
                }
                batch_demo = true;
            }
            "--output-dir" => {
                used_batch_option = true;
                output_dir = Some(PathBuf::from(next_value(&mut iter, "--output-dir")?));
            }
            "--single-channel" => {
                used_batch_option = true;
                dual_channel = false;
            }
            "--skip-nuclei" => {
                used_batch_option = true;
                skip_nuclei = true;
            }
            "--z-ratio" => {
                used_batch_option = true;
                let value = parse_f64(&mut iter, "--z-ratio")?;
                if value <= 0.0 {
                    return Err("--z-ratio must be greater than zero".to_owned());
                }
                z_ratio = value;
            }
            "--seeds" => {
                used_batch_option = true;
                seeds_file = Some(PathBuf::from(next_value(&mut iter, "--seeds")?));
            }
            "--seed-min-sigma" => {
                used_batch_option = true;
                seed_params.min_sigma = parse_f64(&mut iter, "--seed-min-sigma")?;
            }
            "--seed-max-sigma" => {
                used_batch_option = true;
                seed_params.max_sigma = parse_f64(&mut iter, "--seed-max-sigma")?;
            }
            "--seed-threshold" => {
                used_batch_option = true;
                seed_params.threshold = parse_f64(&mut iter, "--seed-threshold")?;
            }
            "--merge-distance" => {
                used_batch_option = true;
                seed_params.merge_distance = parse_f64(&mut iter, "--merge-distance")?;
            }
            "--membrane-delta" => {
                used_batch_option = true;
                seed_params.membrane_delta = parse_f64(&mut iter, "--membrane-delta")?;
            }
            "--cell-sigma" => {
                used_batch_option = true;
                cell_params.gaussian_sigma = parse_f64(&mut iter, "--cell-sigma")?;
            }
            "--cell-median" => {
                used_batch_option = true;
                cell_params.median_size = parse_usize(&mut iter, "--cell-median")?;
            }
            "--min-volume" => {
                used_batch_option = true;
                cell_params.min_volume = parse_usize(&mut iter, "--min-volume")?;
            }
            "--max-volume" => {
                used_batch_option = true;
                cell_params.max_volume = parse_usize(&mut iter, "--max-volume")?;
            }
            "--nucleus-sigma" => {
                used_batch_option = true;
                nucleus_params.gaussian_sigma = parse_f64(&mut iter, "--nucleus-sigma")?;
            }
            "--nucleus-threshold" => {
                used_batch_option = true;
                nucleus_params.threshold = parse_u16(&mut iter, "--nucleus-threshold")?;
            }
            "--nucleus-median" => {
                used_batch_option = true;
                nucleus_params.median_size = parse_usize(&mut iter, "--nucleus-median")?;
            }
            other if other.starts_with('-') => {
                return Err(format!("unknown argument '{other}'\n\n{HELP}"));
            }
            other => {
                return Err(format!("unexpected argument '{other}'\n\n{HELP}"));
            }
        }
    }

    seed_params.z_ratio = z_ratio;
    nucleus_params.z_ratio = z_ratio;

    if seed_params.min_sigma <= 0.0 || seed_params.max_sigma <= 0.0 {
        return Err("seed sigmas must be greater than zero".to_owned());
    }
    if seed_params.min_sigma > seed_params.max_sigma {
        return Err("--seed-min-sigma must not exceed --seed-max-sigma".to_owned());
    }
    if seed_params.threshold < 0.0
        || seed_params.merge_distance < 0.0
        || seed_params.membrane_delta < 0.0
    {
        return Err(
            "seed threshold, merge distance and membrane delta must be non-negative".to_owned(),
        );
    }
    if cell_params.gaussian_sigma < 0.0 {
        return Err("--cell-sigma must be non-negative".to_owned());
    }
    if cell_params.median_size == 0 || nucleus_params.median_size == 0 {
        return Err("median sizes must be at least 1".to_owned());
    }
    if cell_params.min_volume == 0 {
        return Err("--min-volume must be at least 1".to_owned());
    }
    if cell_params.min_volume > cell_params.max_volume {
        return Err("--min-volume must not exceed --max-volume".to_owned());
    }
    if nucleus_params.gaussian_sigma < 0.0 {
        return Err("--nucleus-sigma must be non-negative".to_owned());
    }

    if batch_input.is_some() || batch_demo {
        if demo_gui {
            return Err("--demo opens the GUI and cannot be combined with batch mode".to_owned());
        }
        let output_dir = output_dir.ok_or_else(|| {
            "--output-dir DIR is required with --batch or --batch-demo".to_owned()
        })?;
        Ok(Command::Batch(BatchOptions {
            input: batch_input,
            demo: batch_demo,
            output_dir,
            dual_channel,
            seeds_file,
            seed_params,
            cell_params,
            nucleus_params,
            skip_nuclei,
        }))
    } else {
        if used_batch_option {
            return Err("batch options require --batch or --batch-demo".to_owned());
        }
        Ok(Command::Gui { demo: demo_gui })
    }
}

fn parse_seeds_json(text: &str) -> Result<Vec<Seed>, String> {
    serde_json::from_str::<Vec<Seed>>(text)
        .map_err(|error| format!("invalid seeds JSON (expected [[z,y,x], ...]): {error}"))
}

fn run_batch(options: BatchOptions) -> anyhow::Result<()> {
    let batch_started = Instant::now();
    let stack = if options.demo {
        io::demo_stack()
    } else {
        let input = options
            .input
            .as_ref()
            .context("internal error: batch input is missing")?;
        io::load_tiff(input, options.dual_channel)
            .with_context(|| format!("failed to load {}", input.display()))?
    };

    let seeds = if let Some(seeds_file) = options.seeds_file.as_ref() {
        let text = std::fs::read_to_string(seeds_file)
            .with_context(|| format!("failed to read seeds {}", seeds_file.display()))?;
        let seeds = parse_seeds_json(&text)
            .map_err(|error| anyhow::anyhow!("{}: {error}", seeds_file.display()))?;
        validate_seeds(stack.membrane.shape, &seeds)
            .map_err(|error| anyhow::anyhow!("{}: {error}", seeds_file.display()))?;
        seeds
    } else {
        if stack.nuclei.is_none() {
            bail!(
                "automatic seed detection needs a nucleus channel; pass --seeds FILE with --single-channel"
            );
        }
        let detection_started = Instant::now();
        let seeds = processing::detect_seeds(&stack, &options.seed_params)
            .context("seed detection failed")?;
        println!(
            "detected {} seeds in {:.2} s",
            seeds.len(),
            detection_started.elapsed().as_secs_f64()
        );
        seeds
    };
    if seeds.is_empty() {
        bail!("no seeds available for cell segmentation");
    }

    let cells_started = Instant::now();
    let cells = processing::segment_cells(&stack.membrane, &seeds, &options.cell_params)
        .context("cell segmentation failed")?;
    println!(
        "segmented {} cells in {:.2} s",
        processing::label_centroids(&cells).len(),
        cells_started.elapsed().as_secs_f64()
    );

    let nuclei = if options.skip_nuclei {
        None
    } else if let Some(channel) = stack.nuclei.as_ref() {
        let nuclei_started = Instant::now();
        let nuclei = processing::segment_nuclei(channel, &cells, &options.nucleus_params)
            .context("nucleus segmentation failed")?;
        println!(
            "segmented {} nuclei in {:.2} s",
            processing::label_centroids(&nuclei).len(),
            nuclei_started.elapsed().as_secs_f64()
        );
        Some(nuclei)
    } else {
        println!("no nucleus channel; nucleus segmentation skipped");
        None
    };

    std::fs::create_dir_all(&options.output_dir).with_context(|| {
        format!(
            "failed to create output directory {}",
            options.output_dir.display()
        )
    })?;
    let base = if options.demo {
        options.output_dir.join("demo.tif")
    } else {
        let input = options
            .input
            .as_ref()
            .context("internal error: batch input is missing")?;
        let name = input.file_name().context("input path has no file name")?;
        options.output_dir.join(name)
    };

    let (cell_mask, cell_csv) =
        io::export_labels(&base, &cells, false).context("failed to export cell labels")?;
    println!("wrote {}", cell_mask.display());
    println!("wrote {}", cell_csv.display());
    if let Some(nuclei) = nuclei.as_ref() {
        let (nucleus_mask, nucleus_csv) =
            io::export_labels(&base, nuclei, true).context("failed to export nucleus labels")?;
        println!("wrote {}", nucleus_mask.display());
        println!("wrote {}", nucleus_csv.display());
    }
    println!(
        "batch finished in {:.2} s",
        batch_started.elapsed().as_secs_f64()
    );
    Ok(())
}

fn launch_gui(demo: bool) -> anyhow::Result<()> {
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("TissueSeg3D")
            .with_inner_size([1440.0, 900.0])
            .with_min_inner_size([960.0, 640.0]),
        ..Default::default()
    };
    let startup = if demo { Startup::Demo } else { Startup::Empty };
    eframe::run_native(
        "TissueSeg3D",
        native_options,
        Box::new(move |cc| Ok(Box::new(App::new(cc, startup)))),
    )
    .map_err(|error| anyhow::anyhow!("failed to start GUI: {error}"))
}

fn run() -> anyhow::Result<()> {
    match parse_args(std::env::args().skip(1)) {
        Ok(Command::Help) => {
            print!("{HELP}");
            Ok(())
        }
        Ok(Command::Version) => {
            println!("tissueseg3d {}", env!("CARGO_PKG_VERSION"));
            Ok(())
        }
        Ok(Command::Gui { demo }) => launch_gui(demo),
        Ok(Command::Batch(options)) => run_batch(options),
        Err(error) => bail!("{error}"),
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("tissueseg3d: {error:#}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn batch(args: &[&str]) -> BatchOptions {
        match parse_args(args.iter().copied()).expect("arguments should parse") {
            Command::Batch(options) => options,
            other => panic!("expected batch command, got {other:?}"),
        }
    }

    #[test]
    fn help_and_version_are_recognised() {
        assert!(matches!(parse_args(["--help"]).unwrap(), Command::Help));
        assert!(matches!(parse_args(["-h"]).unwrap(), Command::Help));
        assert!(matches!(
            parse_args(["--version"]).unwrap(),
            Command::Version
        ));
        assert!(matches!(parse_args(["-V"]).unwrap(), Command::Version));
    }

    #[test]
    fn no_arguments_opens_empty_gui() {
        match parse_args(std::iter::empty::<&str>()).unwrap() {
            Command::Gui { demo } => assert!(!demo),
            other => panic!("unexpected command {other:?}"),
        }
    }

    #[test]
    fn demo_opens_demo_gui() {
        match parse_args(["--demo"]).unwrap() {
            Command::Gui { demo } => assert!(demo),
            other => panic!("unexpected command {other:?}"),
        }
    }

    #[test]
    fn batch_defaults_match_python_widgets() {
        let options = batch(&["--batch", "input.tif", "--output-dir", "results"]);
        assert_eq!(options.input, Some(PathBuf::from("input.tif")));
        assert_eq!(options.output_dir, PathBuf::from("results"));
        assert!(options.dual_channel);
        assert!(!options.demo);
        assert!(!options.skip_nuclei);
        assert_eq!(options.seed_params.z_ratio, 6.92);
        assert_eq!(options.nucleus_params.z_ratio, 6.92);
        assert_eq!(options.seed_params.min_sigma, 4.0);
        assert_eq!(options.seed_params.max_sigma, 12.0);
        assert_eq!(options.seed_params.threshold, 1.0);
        assert_eq!(options.seed_params.merge_distance, 100.0);
        assert_eq!(options.seed_params.membrane_delta, 25.0);
        assert_eq!(options.cell_params.gaussian_sigma, 0.5);
        assert_eq!(options.cell_params.median_size, 5);
        assert_eq!(options.cell_params.min_volume, 2000);
        assert_eq!(options.cell_params.max_volume, 100_000);
        assert_eq!(options.nucleus_params.gaussian_sigma, 1.5);
        assert_eq!(options.nucleus_params.threshold, 500);
        assert_eq!(options.nucleus_params.median_size, 5);
    }

    #[test]
    fn batch_parses_every_parameter_flag() {
        let options = batch(&[
            "--batch",
            "input.tiff",
            "--output-dir",
            "out",
            "--single-channel",
            "--z-ratio",
            "2.5",
            "--seeds",
            "seeds.json",
            "--seed-min-sigma",
            "3",
            "--seed-max-sigma",
            "10",
            "--seed-threshold",
            "0.75",
            "--merge-distance",
            "80",
            "--membrane-delta",
            "20",
            "--cell-sigma",
            "0.8",
            "--cell-median",
            "7",
            "--min-volume",
            "100",
            "--max-volume",
            "9000",
            "--nucleus-sigma",
            "2.0",
            "--nucleus-threshold",
            "600",
            "--nucleus-median",
            "6",
            "--skip-nuclei",
        ]);
        assert!(!options.dual_channel);
        assert_eq!(options.seed_params.z_ratio, 2.5);
        assert_eq!(options.nucleus_params.z_ratio, 2.5);
        assert_eq!(options.seeds_file, Some(PathBuf::from("seeds.json")));
        assert_eq!(options.seed_params.min_sigma, 3.0);
        assert_eq!(options.seed_params.max_sigma, 10.0);
        assert_eq!(options.seed_params.threshold, 0.75);
        assert_eq!(options.seed_params.merge_distance, 80.0);
        assert_eq!(options.seed_params.membrane_delta, 20.0);
        assert_eq!(options.cell_params.gaussian_sigma, 0.8);
        assert_eq!(options.cell_params.median_size, 7);
        assert_eq!(options.cell_params.min_volume, 100);
        assert_eq!(options.cell_params.max_volume, 9000);
        assert_eq!(options.nucleus_params.gaussian_sigma, 2.0);
        assert_eq!(options.nucleus_params.threshold, 600);
        assert_eq!(options.nucleus_params.median_size, 6);
        assert!(options.skip_nuclei);
    }

    #[test]
    fn batch_demo_needs_no_input() {
        let options = batch(&["--batch-demo", "--output-dir", "demo-out"]);
        assert!(options.demo);
        assert!(options.input.is_none());
        assert_eq!(options.output_dir, PathBuf::from("demo-out"));
    }

    #[test]
    fn unknown_and_positional_arguments_are_rejected() {
        assert!(parse_args(["--frobnicate"])
            .unwrap_err()
            .contains("unknown"));
        assert!(parse_args(["input.tif"])
            .unwrap_err()
            .contains("unexpected"));
    }

    #[test]
    fn missing_values_are_rejected() {
        for args in [
            vec!["--batch"],
            vec!["--output-dir"],
            vec!["--batch", "in.tif", "--output-dir", "out", "--z-ratio"],
            vec!["--batch", "in.tif", "--output-dir", "out", "--seeds"],
        ] {
            assert!(parse_args(args).is_err());
        }
    }

    #[test]
    fn batch_requires_an_output_directory() {
        let error = parse_args(["--batch", "input.tif"]).unwrap_err();
        assert!(error.contains("output-dir"));
    }

    #[test]
    fn batch_options_without_batch_are_rejected_instead_of_ignored() {
        for args in [
            vec!["--z-ratio", "2"],
            vec!["--single-channel"],
            vec!["--skip-nuclei"],
            vec!["--output-dir", "out"],
            vec!["--cell-sigma", "0.5"],
        ] {
            let error = parse_args(args).unwrap_err();
            assert!(error.contains("require --batch"), "{error}");
        }
    }

    #[test]
    fn demo_and_batch_are_mutually_exclusive() {
        assert!(parse_args(["--demo", "--batch", "in.tif", "--output-dir", "out"]).is_err());
        assert!(parse_args(["--batch-demo", "--demo", "--output-dir", "out"]).is_err());
        assert!(parse_args(["--batch", "in.tif", "--batch-demo", "--output-dir", "out"]).is_err());
    }

    #[test]
    fn invalid_numbers_and_ranges_are_rejected() {
        assert!(
            parse_args(["--batch", "in.tif", "--output-dir", "out", "--z-ratio", "0"]).is_err()
        );
        assert!(parse_args([
            "--batch",
            "in.tif",
            "--output-dir",
            "out",
            "--z-ratio",
            "NaN"
        ])
        .is_err());
        assert!(parse_args([
            "--batch",
            "in.tif",
            "--output-dir",
            "out",
            "--seed-min-sigma",
            "10",
            "--seed-max-sigma",
            "4"
        ])
        .is_err());
        assert!(parse_args([
            "--batch",
            "in.tif",
            "--output-dir",
            "out",
            "--min-volume",
            "100",
            "--max-volume",
            "10"
        ])
        .is_err());
        assert!(parse_args([
            "--batch",
            "in.tif",
            "--output-dir",
            "out",
            "--cell-median",
            "0"
        ])
        .is_err());
        assert!(parse_args([
            "--batch",
            "in.tif",
            "--output-dir",
            "out",
            "--nucleus-threshold",
            "not-a-number"
        ])
        .is_err());
    }

    #[test]
    fn seeds_json_requires_an_array_of_zyx_coordinates() {
        assert_eq!(
            parse_seeds_json("[[1, 2, 3], [4, 5, 6]]").unwrap(),
            vec![[1, 2, 3], [4, 5, 6]]
        );
        assert!(parse_seeds_json("{\"z\": 1}").is_err());
        assert!(parse_seeds_json("[[1, 2], [3, 4, 5]]").is_err());
        assert!(parse_seeds_json("[[-1, 2, 3]]").is_err());
    }
}
