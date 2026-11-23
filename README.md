# Active learning for SOC surrogate modeling

MATLAB code and simulation datasets accompanying the manuscript:

**Active Learning for Data-Efficient Surrogate Modeling of Electrochemical Devices**

## Requirements

- MATLAB R2025b
- Deep Learning Toolbox
- Parallel Computing Toolbox
- Supported NVIDIA GPU

## Files

- `global_output_statistics_loop.m` — global-output training and active learning.
- `spatial_output_statistics_loop.m` — spatial-output training and active learning.
- `normalizeTargets.m`, `normalizeSpatialTargets.m` — output normalization.
- `selectHybridSims.m` — hybrid acquisition.
- `Data/` — candidate datasets, independent test datasets, normalization statistics, and random seeds.

## Usage

Set the repository root as the MATLAB current folder and run either training script.

The scripts repeat the main experiments using five random seeds. Results are saved in the automatically created `Results/` folder.
