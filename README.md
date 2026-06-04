# FNO-Paradigm

A minimal, reproducible PyTorch testbed for 1D Burgers operator learning with:

1. **Baseline FNO**
2. **Ansatz-conditioned residual FNO** using a heat-equation prior

## Problem setup

We study periodic Burgers on \(x\in[0,1]\):

\[
 u_t + u u_x = \nu u_{xx},\qquad \nu=0.1,\ T=1.0.
\]

Goal: learn the operator \(u_0(x)\mapsto u(x,T)\).

## Models

### Baseline FNO

\[
 u_{\text{pred}} = \mathrm{FNO}_{\theta}([u_0, x]).
\]

### Heat ansatz + residual FNO

Heat ansatz:

\[
 u_{\text{heat}} = \exp(\nu T\partial_{xx})u_0,
\]

or in Fourier space:

\[
 \hat u_{\text{heat}}(k,T)=\exp(-\nu k^2 T)\hat u_0(k).
\]

Residual model:

\[
 u_{\text{pred}} = u_{\text{heat}} + \mathrm{FNO}_{\theta}([u_0, u_{\text{heat}}, x]).
\]

This can improve sample efficiency because the heat ansatz captures dissipative smoothing, while FNO focuses on nonlinear convective correction.

## Repo layout

```text
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

## Data modes

- **Synthetic**: pseudo-spectral RK4 Burgers solver; saves `u0, uT, u_heat, x` to `.npz`.
- **Optional `.mat` loading**: if a Burgers `.mat` file is provided, loader reads it and computes `u_heat`.

## Install

```bash
pip install -r requirements.txt
```

## Run

Main comparison script:

```bash
python scripts/compare_burgers.py \
  --experiment both \
  --n-grid 256 \
  --num-samples 1200 \
  --epochs 500 \
  --batch-size 20 \
  --modes 16 \
  --width 64 \
  --layers 4 \
  --nu 0.1 \
  --T 1.0 \
  --lr 0.001 \
  --data-path data/burgers_synthetic.npz \
  --save-dir results/run_name
```

Single-model wrappers:

```bash
python scripts/train_baseline_fno_burgers.py --save-dir results/baseline_run
python scripts/train_ansatz_residual_fno_burgers.py --save-dir results/residual_run
```

## Outputs

For each experiment, saved under `results/<run>/<experiment>/`:

- `config.json`
- `metrics.json`
- `model.pt`
- `comparison_*.png`
- `summary.md`

Also `comparison_metrics.json` at run root.

## Metrics

- train MSE
- test MSE
- test relative L2: \(\|u_{pred}-u_{true}\|_2 / \|u_{true}\|_2\)

This repository is intended as a compact testbed for the broader idea:
**physics ansatz + neural operator correction**.
