# SimVeRi Code

Reference implementation for generating the SimVeRi dataset, assembling the public release package, and reproducing the paper-side technical validation utilities.

This repository is organized around the same three stages used in the dataset paper:

- co-simulation and raw data collection;
- cleaning, benchmark construction, and supplementary resource export; and
- technical validation for ReID, trajectory reconstruction, and air-ground characterization.

## Repository Layout

```text
SimVeRi-code/
  README.md
  LICENSE
  requirements.txt
  environment.yml
  .gitignore
  config.yaml
  camera_topology.json
  traffic_gen_v2.py
  run_simveri.py
  simveri_collector.py
  clean_and_split.py
  generate_simveri_release.py
  export_twins_extras.py
  analyze_camera_coverage.py
  sumo/
  sumo_integration/
  simveri_validation/
```

## What Each Part Does

- `traffic_gen_v2.py`: generates SUMO traffic actors, route files, and the vehicle manifest used during collection.
- `run_simveri.py`: launches synchronized CARLA-SUMO co-simulation and records raw vehicle crops plus metadata.
- `simveri_collector.py`, `bbox_utils.py`, `occlusion.py`, `config_loader.py`: capture-time helpers for image extraction and metadata export.
- `clean_and_split.py`: filters raw captures using explicit geometric and visibility rules and produces cleaned intermediate outputs.
- `generate_simveri_release.py`: assembles the benchmark-ready release bundle, including the main ground-ground benchmark and air-ground protocol assets.
- `export_twins_extras.py`: exports the Twins subset as a separate supplement under `extras/twins/`.
- `simveri_validation/`: paper-side validation scripts for baseline ReID, air-ground analysis, and trajectory-reconstruction validation.

## External Prerequisites

Some dependencies are intentionally not bundled in `requirements.txt` because they are platform- or CUDA-specific:

1. CARLA 0.9.13, including a Python API build that matches your local Python interpreter.
2. SUMO with `SUMO_HOME` configured so that `traci` and `sumolib` are importable.
3. Optional but recommended for baseline training and evaluation: a compatible PyTorch installation and a local FastReID checkout.

## Environment Setup

Create a basic Python environment for the release-processing scripts:

```bash
conda env create -f environment.yml
conda activate simveri
pip install -r requirements.txt
```

Then configure the external runtimes:

```bash
set SUMO_HOME=C:\path\to\sumo
```

Before running collection, confirm that:

- `camera_topology.json` matches the intended public-release camera deployment;
- `config.yaml` points to the local topology file and uses sensible collection thresholds; and
- the CARLA Python API version matches the interpreter used to run `run_simveri.py`.

## End-to-End Pipeline

### 1. Generate Traffic Definitions

```bash
python traffic_gen_v2.py
```

This writes or refreshes the SUMO route and vehicle-manifest files used by the co-simulation run.

### 2. Run Co-Simulation and Collect Raw Outputs

```bash
python run_simveri.py sumo/simveri.sumocfg --output-dir outputs/raw_collect
```

`run_simveri.py` drives synchronized CARLA-SUMO execution and writes raw image crops plus per-capture metadata.

### 3. Clean Raw Captures

```bash
python clean_and_split.py --input-dir outputs/raw_collect --output-dir outputs/cleaned
```

This step removes invalid captures, limits near-duplicate samples, and creates the cleaned intermediate package required by the release generator.

### 4. Assemble the Main Release Bundle

```bash
python generate_simveri_release.py ^
  --input-dir outputs/cleaned ^
  --output-dir outputs/release ^
  --topology-file camera_topology.json ^
  --ag-protocol-scope both
```

This script builds:

- the primary ground-ground benchmark;
- metadata and statistics files;
- `ag_protocol/`; and
- `ag_protocol_full/`.

### 5. Export the Twins Supplement

```bash
python export_twins_extras.py --clean-dir outputs/cleaned --release-dir outputs/release
```

The Twins subset is kept separate on purpose so standard ReID benchmarking is not conflated with controlled identical-appearance analysis.

## Technical Validation

The `simveri_validation/` directory is part of the public release and should be kept. In particular, `simveri_validation/src/tech_validation_tr/` is the shared helper layer for the trajectory-reconstruction validation scripts reported in the paper.

Typical validation workflows are:

- baseline ReID:

```bash
python simveri_validation/scripts/train_baseline.py --dataset-root <release_root> --output-dir <model_dir>
python simveri_validation/scripts/extract_features.py --data-root <release_root> --model-path <model_dir>\model_final.pth --output-dir <features_dir>
python simveri_validation/scripts/evaluate_baseline.py --features-dir <features_dir> --data-root <release_root> --output-dir <results_dir>
```

- air-ground characterization:

```bash
python simveri_validation/scripts/extract_ag_tracklet_features.py --ag-root <release_root>\ag_protocol --output-dir <features_dir>
python simveri_validation/scripts/evaluate_ag_protocol.py --ag-root <release_root>\ag_protocol --features-dir <features_dir> --output-dir <results_dir>
```

- trajectory-reconstruction validation:

```bash
python simveri_validation/scripts/tv_tr_build_tracklets.py --release-root <release_root> --out-dir <tv_dir>
python simveri_validation/scripts/tv_tr_build_candidates.py ...
python simveri_validation/scripts/tv_tr_fit_coview_pairs.py ...
python simveri_validation/scripts/tv_tr_fit_global_speed_prior.py ...
python simveri_validation/scripts/tv_tr_evaluate.py ...
```

The later trajectory-validation steps are intentionally script-based rather than hidden behind a single wrapper so intermediate artifacts remain inspectable.

## Notes for Public Users

- Several validation scripts still carry experiment-time default paths from local development. In public use, prefer passing explicit CLI arguments instead of relying on the embedded defaults.
- `sumo_integration/` contains adapted CARLA-SUMO co-simulation support code; preserve the original notices in those files.
- Not every script in `simveri_validation/scripts/` is required to regenerate the dataset. Some are included to document paper figures, transfer experiments, and sanity checks for full transparency.

## License

The code in this repository is released under the MIT License. See `LICENSE` for details. Third-party software such as CARLA, SUMO, and FastReID remains subject to its own licensing terms.
