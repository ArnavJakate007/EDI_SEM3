# sattsr — Cross-Sensor Temporal Super Resolution for Geostationary Thermal Imagery

Raises the effective cadence of geostationary thermal-IR imagery from 30 min to 15 min
and 7.5 min using a learned optical-flow frame-interpolation model, and applies it to
INSAT-3DR/3DS.

## Install

    py -3.13 -m venv .venv
    .\.venv\Scripts\Activate.ps1     # PowerShell
    pip install -e ".[dev,download]"

## Pipeline

    sattsr prepare  --config configs/goes19.yaml
    sattsr train    --config configs/goes19.yaml
    sattsr evaluate --config configs/goes19.yaml --checkpoint checkpoints/goes19/best.pt
    sattsr finetune --config configs/insat.yaml  --checkpoint checkpoints/goes19/best.pt
    sattsr infer    --config configs/insat.yaml  --checkpoint checkpoints/insat/best.pt \
                    --input data/raw/insat --out runs/insat-demo --factor 2
    sattsr serve    --runs-dir runs

Synthetic frames are always flagged `synthetic=1` in the output NetCDF. They are
decision-support augmentation, never a substitute for real observations.
