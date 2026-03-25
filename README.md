# SimVeRi Code

Reference implementation for generating the SimVeRi dataset, assembling the public release package, and reproducing the paper-side technical validation utilities.

## Official Project Title

SimVeRi: a synthetic dataset for vehicle re-identification with spatiotemporal annotations and air-ground viewpoints

## Release Metadata

- Code repository: <https://github.com/Zengzhi-Zhang/SimVeRi>
- Code archive DOI: <https://doi.org/10.5281/zenodo.19207143>
- Dataset DOI: <https://doi.org/10.5281/zenodo.19207203>
- Code license: `MIT License`
- Dataset license: `CC BY 4.0`
- Corresponding author: Gang Ren (<rengang@seu.edu.cn>)

## Overview

This repository provides the code used to construct the SimVeRi dataset and to reproduce the technical-validation workflows reported for the release. The repository covers three connected stages:

- co-simulation and raw data collection;
- cleaning, benchmark construction, and supplementary resource export; and
- technical validation for ReID, trajectory reconstruction, and air-ground characterization.

The released dataset is distributed separately through Zenodo. This repository contains the software needed to regenerate or inspect the corresponding processing and validation steps.

## Repository Layout

```text
SimVeRi/
  README.md
  LICENSE
  CITATION.cff
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

## Main Components

- `traffic_gen_v2.py`: generates SUMO traffic actors, route files, and the vehicle manifest used during collection.
- `run_simveri.py`: launches synchronized CARLA-SUMO co-simulation and records raw vehicle crops plus metadata.
- `simveri_collector.py`, `bbox_utils.py`, `occlusion.py`, `config_loader.py`: capture-time helpers for image extraction and metadata export.
- `clean_and_split.py`: filters raw captures using explicit geometric and visibility rules and produces cleaned intermediate outputs.
- `generate_simveri_release.py`: assembles the benchmark-ready release bundle, including the main ground-ground benchmark and air-ground protocol assets.
- `export_twins_extras.py`: exports the Twins subset as a separate supplement under `extras/twins/`.
- `simveri_validation/`: validation scripts for baseline ReID, air-ground analysis, and trajectory-reconstruction experiments.

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

## Citation

If you use this codebase, please cite the code archive:

- Zhang, Z., Cao, Q., Deng, Y., Zhang, M., Wu, C. and Ren, G. SimVeRi code. Zenodo. <https://doi.org/10.5281/zenodo.19207143>

If you use the released dataset, please also cite:

- Zhang, Z., Cao, Q., Deng, Y., Zhang, M., Wu, C. and Ren, G. SimVeRi: a synthetic dataset for vehicle re-identification with spatiotemporal annotations and air-ground viewpoints. Zenodo. <https://doi.org/10.5281/zenodo.19207203>

A machine-readable citation record for the software release is provided in `CITATION.cff`.

## Authors

All authors are affiliated with the Jiangsu Province Collaborative Innovation Center of Modern Urban Traffic Technologies, School of Transportation, Southeast University.

1. Zengzhi Zhang (<zhangzengzhi512@seu.edu.cn>)
2. Qi Cao (<cao_qi@seu.edu.cn>)
3. Yue Deng (<dengyue@seu.edu.cn>)
4. Mengyao Zhang (<zmyao0829@126.com>)
5. Changjian Wu (<changjian12@126.com>)
6. Gang Ren (<rengang@seu.edu.cn>, corresponding author)

## Funding

- National Natural Science Foundation of China Key Project: Key Technologies and Software Research for Ultra-Large-Scale Multi-Modal Transportation System Simulation (`52432010`)
- National Natural Science Foundation of China Project: Vehicle Travel Path Identification and Traffic Congestion Tracing Method Based on AVI Data (`52202399`)
- National Natural Science Foundation of China General Project: Principles of Road Traffic Network Resilience Monitoring and Emergency Control Methods Under Non-Normal Events (`52372314`)

## Notes for Public Users

- `sumo_integration/` contains adapted CARLA-SUMO co-simulation support code; preserve the original notices in those files.
- Not every script in `simveri_validation/scripts/` is required to regenerate the dataset. Some are included to document paper figures, transfer experiments, and sanity checks for transparency and reproducibility.
- The dataset release itself is distributed separately through Zenodo and is not stored in full inside this repository.

## License

The code in this repository is released under the MIT License. See `LICENSE` for details. Third-party software such as CARLA, SUMO, and FastReID remains subject to its own licensing terms.
