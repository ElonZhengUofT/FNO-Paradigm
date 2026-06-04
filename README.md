# FNO-Paradigm

A minimal, reproducible PyTorch testbed for 1D Burgers operator learning with Fourier Neural Operators.

## Goal

Compare two models for learning the map:

\[
u_0(x) \mapsto u(x,T),\quad u_t + u u_x = \nu u_{xx},\; x\in[0,1],\; \text{periodic}
\]

Default: \(\nu=0.1\), \(T=1.0\).

## Models

### 1) Baseline FNO

- Input channels: \([u_0(x), x]\)
- Output: \(u(x,T)\)
- Formulation:

\[
u_{pred} = \mathrm{FNO}_\theta([u_0, x])
\]

### 2) Heat-ansatz residual FNO

Construct a physics ansatz from the initial condition:

\[
u_{heat}(x,T)=\exp(\nu T\partial_{xx})u_0(x)
\]

In Fourier space:

\[
\hat{u}_{heat}(k,T)=\exp(-\nu k^2 T)\hat{u}_0(k)
\]

Then learn the residual:

\[
\nu_{pred}=u_{heat}+\mathrm{FNO}_\theta([u_0,u_{heat},x])
\]

Why this may help: the heat ansatz captures dissipative smoothing, while FNO focuses on nonlinear convective correction.

This repo is a small testbed for the broader idea: **physics ansatz + neural operator correction**.

## Repository layout

```
.
├── README.md
├── requirements.txt
├── configs/
│   └── burgers_default.yaml
├── data/
│   └── .gitkeep
├── scripts/
│   ├── train_baseline_fno_burgers.py
│   ├── train_ansatz_residual_fno_burgers.py
│   └── compare_burgers.py
├── src/
│   ├── data/
│   │   ├── burgers_solver.py
│   │   ├── burgers_dataset.py
│   │   └── mat_reader.py
│   ├── models/
│   │   ├── fno1d.py
│   │   └── ansatz_fno.py
│   ├── ansatz/
│   │   └── heat_ansatz.py
│   ├── train/
│   │   ├── trainer.py
│   │   └── losses.py
│   └── utils/
│       ├── seed.py
│       ├── plotting.py
│       └── checkpoint.py
├── experiments/
│   └── README.md
└── results/
    └── .gitkeep
```

## Data

Two modes:

1. **Synthetic data** (default): pseudo-spectral RK4 Burgers solver generates `u0`, `uT`, `u_heat`, `x` and saves `.npz`.
2. **MAT data**: if `--data-path` ends with `.mat`, loader attempts to read a provided Burgers dataset and computes `u_heat`.

## Install

```bash
pip install -r requirements.txt
```

## Run

Run both baseline and residual:

```bash
python scripts/compare_burgers.py --experiment both
```

Run baseline only:

```bash
python scripts/train_baseline_fno_burgers.py --experiment baseline
```

Run residual only:

```bash
python scripts/train_ansatz_residual_fno_burgers.py --experiment residual
```

Common flags:

- `--experiment baseline|residual|both`
- `--n-grid 256`
- `--num-samples 1200`
- `--epochs 500`
- `--batch-size 20`
- `--modes 16`
- `--width 64`
- `--layers 4`
- `--nu 0.1`
- `--T 1.0`
- `--lr 0.001`
- `--data-path data/burgers_synthetic.npz`
- `--save-dir results/run_name`

## Outputs

Each run saves under `results/<run_name>/<experiment>/`:

- `config.json`
- `metrics.json`
- `model.pt`
- `sample_*.png`
- `summary.md`

Training prints per-epoch:

- train MSE
- test MSE
- test relative L2
