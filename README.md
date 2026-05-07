# AdamFLIP

Code for **AdamFLIP**: an Adam-style first/second-moment optimizer used with a feedback (PI-control) constrained optimization update, with PINN experiments on the 2D heat equation.

> Status: research code (scripts). If you use this repository in academic work, please cite the associated paper/preprint (see **Citation**).

---

## Repository structure (key files)

- `2D_heat/forward/AdamFL-PINN.py`  
  Main training + evaluation script for **AdamFL-PINN** on the 2D heat equation.  
  Produces training curves and solution comparison plots, and saves `.npz`/`.mat` artifacts.

- `2D_heat/forward/ablation_study_beta.py`  
  NeurIPS-style **ablation study** script for Adam moment parameters \((\beta_1, \beta_2)\) across multiple random seeds.  
  Produces publication-quality plots (mean ± std) and saves aggregated statistics.

---

## Environment setup

### Option A: pip (recommended)

```bash
python -m venv .venv
source .venv/bin/activate  # (Linux/macOS)
# .venv\Scripts\activate   # (Windows PowerShell)

pip install --upgrade pip
pip install torch numpy scipy matplotlib tqdm
```

### Option B: conda

```bash
conda create -n adamflip python=3.10 -y
conda activate adamflip
pip install torch numpy scipy matplotlib tqdm
```

**Notes**
- GPU is optional. Scripts will use CUDA if available.
- For reproducibility, scripts set fixed seeds and enforce deterministic CuDNN behavior when possible.

---

## Running the main experiment (AdamFL-PINN)

From repo root:

```bash
python 2D_heat/forward/AdamFL-PINN.py
```

### Expected outputs

The script will create folders (if missing) and write:
- `./training_data/AdamFL_PINN_loss_history.npz`
- `./figures/adamfl_comparison_t0.5.png`
- `./figures/adamfl_comparison_t1.0.png`
- `./data/AdamFL_solution_t0.5.mat`
- `./data/AdamFL_solution_t1.0.mat`

### What it does (high level)

- Solves the 2D heat equation using a PINN:

  \[
  u_t - \alpha (u_{xx} + u_{yy}) = 0,\quad (x,y,t)\in[0,1]^2\times[0,1]
  \]

  with initial condition \(u(x,y,0)=\sin(\pi x)\sin(\pi y)\) and zero Dirichlet boundaries.

- Trains a fully-connected network taking \((x,y,t)\mapsto u\).
- Optimizes a constrained formulation: PDE residual as objective, IC/BC losses as constraints (combined).
- Includes an Adam warm-start, then runs the AdamFL-style constrained update.

---

## Running the ablation study (Adam betas)

From repo root:

```bash
python 2D_heat/forward/ablation_study_beta.py
```

### Expected outputs

- Figure:
  - `./figures/ablation_study_beta_bc_loss.png`
- Aggregated stats per configuration:
  - `./training_data/ablation_beta_<config>_history.npz`

### What it measures

For each \((\beta_1,\beta_2)\) setting and each seed, the script logs (every 100 iterations):
- physics loss (PDE residual objective)
- IC loss and BC loss (constraint components)
- KKT gap proxy

It then plots **boundary condition loss vs iteration** using mean ± std across seeds, with selective shaded error bands.

---

## Configuration knobs

You can edit these constants at the top of each script:

Common:
- `NUM_ITERATIONS` (training iterations)
- `NUM_POINTS_PDE_MASTER`, `NUM_POINTS_PDE`, `NUM_POINTS_BC`, `NUM_POINTS_IC` (sampling)
- `loss_weights_g` (weights for IC/BC/PDE)
- `Kp`, `Ki` (PI-control gains)
- `eta` (step size)

Ablation:
- `NUM_SEEDS`
- `BETA_CONFIGS`

---

## Reproducibility

- Seeds are set for `torch`, `numpy`, and Python `random`.
- If CUDA is available, CUDA seeds are set and CuDNN is configured for determinism (may reduce performance).

---

## Citation

If you use this code, please cite:

- **AdamFLIP** (paper / preprint): _TBD_

Add your BibTeX here, e.g.

```bibtex
@article{adamflip2026,
  title   = {AdamFLIP: ...},
  author  = {...},
  journal = {...},
  year    = {2026}
}
```

---

## License

Specify a license (e.g., MIT, Apache-2.0) by adding a `LICENSE` file. If omitted, the default is “all rights reserved” under copyright law.

---

