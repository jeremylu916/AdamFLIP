"""
Inverse pendulum ODE: recover unknown squared frequency omega0^2 = g/L from sparse
noisy observations of theta(t), jointly with a PINN for theta(t).

Compares:
  - Standard Adam (weighted sum: data + IC + physics residual)
 AdamFL: objective = data loss; constraints = IC loss + residual loss (KKT-style step + Adam moments).

Run from anywhere:
  python ODE/pendulum/inverse_run_compare.py
"""
from __future__ import annotations

import os
import time
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.io import savemat
from scipy.linalg import solve
from tqdm import trange

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE_DIR, "figures")
DATA_DIR = os.path.join(BASE_DIR, "data")
LOG_DIR = os.path.join(BASE_DIR, "training_logs")

# ============================================================
# Reproducibility + device
# ============================================================
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ============================================================
# Problem config
# ============================================================
@dataclass(frozen=True)
class InversePendulumConfig:
    g: float = 9.81
    L: float = 1.0  # used only to form omega0^2_true = g/L for data generation
    t0: float = 0.0
    t1: float = 10.0
    theta0: float = 1.0
    omega0_ic: float = 0.0  # initial angular velocity dtheta/dt at t=0

    n_coll: int = 2000
    n_eval: int = 400
    n_obs: int = 120
    obs_noise_std: float = 0.02  # relative noise scale * std(theta_obs)

    # Wrong initial guess for omega0^2 (true is g/L)
    omega0_sq_init: float = 4.0

    # Adam loss weights
    w_data: float = 10.0
    w_ic: float = 100.0
    w_res: float = 1.0


CFG_INV = InversePendulumConfig()


# ============================================================
# Reference (ground truth parameter + trajectory)
# ============================================================
def omega0_sq_true(cfg: InversePendulumConfig) -> float:
    return cfg.g / cfg.L


def solve_pendulum_reference(cfg: InversePendulumConfig):
    w02 = omega0_sq_true(cfg)

    def rhs(t, y):
        theta, omega = y
        return [omega, -w02 * np.sin(theta)]

    t_eval = np.linspace(cfg.t0, cfg.t1, cfg.n_eval)
    sol = solve_ivp(
        rhs,
        (cfg.t0, cfg.t1),
        y0=[cfg.theta0, cfg.omega0_ic],
        t_eval=t_eval,
        rtol=1e-10,
        atol=1e-12,
        method="DOP853",
    )
    return sol.t.astype(np.float32), sol.y[0].astype(np.float32), sol.y[1].astype(np.float32)


def make_observations(
    t_ref: np.ndarray,
    theta_ref: np.ndarray,
    cfg: InversePendulumConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Sparse observation times and noisy theta (for reproducibility)."""
    rng = np.random.default_rng(SEED + 7)
    idx = np.sort(rng.choice(len(t_ref), size=min(cfg.n_obs, len(t_ref)), replace=False))
    t_obs = t_ref[idx]
    theta_obs = theta_ref[idx].copy()
    noise = cfg.obs_noise_std * np.std(theta_ref) * rng.standard_normal(theta_obs.shape)
    theta_noisy = theta_obs + noise.astype(np.float32)
    return t_obs.astype(np.float32), theta_noisy


# ============================================================
# MLP (same Fourier embedding style as run_compare.py)
# ============================================================
class MLP(nn.Module):
    def __init__(self, hidden: int = 128, layers: int = 4, t_max: float = 10.0):
        super().__init__()
        self.t_max = t_max
        self.register_buffer("freqs", torch.arange(1, 6, dtype=torch.float32) * (2 * np.pi / (t_max / 2)))
        net_in = 1 + 2 * len(self.freqs)
        net = [nn.Linear(net_in, hidden), nn.Tanh()]
        for _ in range(layers - 2):
            net.extend([nn.Linear(hidden, hidden), nn.Tanh()])
        net.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*net)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, t):
        t_norm = 2.0 * (t / self.t_max) - 1.0
        args = t_norm * self.freqs
        sin_feat = torch.sin(args)
        cos_feat = torch.cos(args)
        x = torch.cat([t_norm, sin_feat, cos_feat], dim=-1)
        return self.net(x)


def _softplus_inv_pos(y: float) -> torch.Tensor:
    y_t = torch.tensor(float(y), dtype=torch.float32)
    y_t = torch.clamp(y_t, min=1e-6)
    return torch.log(torch.expm1(y_t))


class InversePendulumPINN(nn.Module):
    """PINN for theta(t) with learnable omega0^2 > 0 (parameter of the ODE)."""

    def __init__(self, cfg: InversePendulumConfig):
        super().__init__()
        self.cfg = cfg
        self.net = MLP(t_max=cfg.t1)
        self.raw_omega0_sq = nn.Parameter(_softplus_inv_pos(cfg.omega0_sq_init))

    def omega0_sq(self) -> torch.Tensor:
        return F.softplus(self.raw_omega0_sq) + 1e-8

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(t)


def ic_loss(model: InversePendulumPINN, cfg: InversePendulumConfig) -> torch.Tensor:
    mse = nn.MSELoss()
    t0 = torch.tensor([[cfg.t0]], dtype=torch.float32, device=device, requires_grad=True)
    theta0_p = model(t0)
    th_t = torch.autograd.grad(theta0_p, t0, torch.ones_like(theta0_p), create_graph=True)[0]
    return mse(theta0_p, torch.tensor([[cfg.theta0]], dtype=torch.float32, device=device)) + mse(
        th_t, torch.tensor([[cfg.omega0_ic]], dtype=torch.float32, device=device)
    )


def residual_loss(model: InversePendulumPINN, t_coll: torch.Tensor, cfg: InversePendulumConfig) -> torch.Tensor:
    mse = nn.MSELoss()
    w02 = model.omega0_sq()
    t = t_coll.clone().detach().requires_grad_(True)
    theta = model(t)
    th_t = torch.autograd.grad(theta, t, torch.ones_like(theta), create_graph=True)[0]
    th_tt = torch.autograd.grad(th_t, t, torch.ones_like(th_t), create_graph=True)[0]
    res = th_tt + w02 * torch.sin(theta)
    return mse(res, torch.zeros_like(res))


def data_loss(model: InversePendulumPINN, t_obs: torch.Tensor, theta_obs: torch.Tensor) -> torch.Tensor:
    return nn.MSELoss()(model(t_obs), theta_obs)


def inverse_losses(
    model: InversePendulumPINN,
    t_coll: torch.Tensor,
    t_obs: torch.Tensor,
    theta_obs: torch.Tensor,
    cfg: InversePendulumConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        data_loss(model, t_obs, theta_obs),
        ic_loss(model, cfg),
        residual_loss(model, t_coll, cfg),
    )


# ============================================================
# Adam baseline
# ============================================================
def train_adam_inverse(
    cfg: InversePendulumConfig,
    t_obs: torch.Tensor,
    theta_obs: torch.Tensor,
    iters: int = 20000,
    lr: float = 1e-3,
) -> tuple[InversePendulumPINN, dict]:
    model = InversePendulumPINN(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.9995)

    log_path = os.path.join(LOG_DIR, "ADAM_inverse_pendulum_loss.txt")
    os.makedirs(LOG_DIR, exist_ok=True)
    hist = {"data": [], "ic": [], "res": [], "omega": [], "time_s": None}

    t_start = time.time()
    pbar = trange(iters, desc="Adam (inverse pendulum)", unit="iter")
    with open(log_path, "w") as f:
        f.write("iter\tloss_data\tloss_ic\tloss_res\tomega0_sq\ttotal\n")
        for it in pbar:
            t_coll = (cfg.t0 + (cfg.t1 - cfg.t0) * torch.rand(cfg.n_coll, 1, device=device)).float()
            ld, li, lr_ = inverse_losses(model, t_coll, t_obs, theta_obs, cfg)
            loss = cfg.w_data * ld + cfg.w_ic * li + cfg.w_res * lr_
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()

            w_hat = model.omega0_sq().item()
            if (it + 1) % 50 == 0 or it == 0:
                lt = float(loss.item())
                hist["data"].append(float(ld.item()))
                hist["ic"].append(float(li.item()))
                hist["res"].append(float(lr_.item()))
                hist["omega"].append(w_hat)
                f.write(
                    f"{it+1}\t{ld.item():.8e}\t{li.item():.8e}\t{lr_.item():.8e}\t{w_hat:.8e}\t{lt:.8e}\n"
                )
                pbar.set_postfix(
                    {"data": f"{ld.item():.2e}", "res": f"{lr_.item():.2e}", "w2": f"{w_hat:.3f}"}
                )

    hist["time_s"] = time.time() - t_start
    return model, hist


# ============================================================
# AdamFL (data objective; IC + physics constraints)
# ============================================================
def get_flat_params(model: nn.Module) -> np.ndarray:
    return np.concatenate([p.detach().cpu().numpy().ravel() for p in model.parameters()])


def set_flat_params(model: nn.Module, flat_params: np.ndarray) -> None:
    flat_params = flat_params.astype(np.float32)
    pointer = 0
    for p in model.parameters():
        n = p.numel()
        vals = torch.from_numpy(flat_params[pointer : pointer + n]).view_as(p).to(device)
        p.data.copy_(vals)
        pointer += n


def get_flat_grads(model: nn.Module) -> np.ndarray:
    grads = []
    for p in model.parameters():
        if p.grad is None:
            grads.append(np.zeros(p.numel(), dtype=np.float32))
        else:
            grads.append(p.grad.detach().cpu().numpy().ravel())
    return np.concatenate(grads)


def get_grads_from_loss(loss: torch.Tensor, params: list, allow_unused: bool = False) -> np.ndarray:
    gs = torch.autograd.grad(
        loss, params, grad_outputs=torch.ones_like(loss), retain_graph=False, allow_unused=allow_unused
    )
    out = []
    for g, p in zip(gs, params):
        if g is not None:
            out.append(g.detach().cpu().numpy().ravel())
        else:
            out.append(np.zeros(p.numel(), dtype=np.float32))
    return np.concatenate(out)


class AdamFLInversePendulum:
    def __init__(self, cfg: InversePendulumConfig, t_obs: torch.Tensor, theta_obs: torch.Tensor):
        self.cfg = cfg
        self.model = InversePendulumPINN(cfg).to(device)
        self.t_obs = t_obs
        self.theta_obs = theta_obs

    def f_sofl(self, x: np.ndarray) -> float:
        set_flat_params(self.model, x)
        return data_loss(self.model, self.t_obs, self.theta_obs).item()

    def df_sofl(self, x: np.ndarray) -> np.ndarray:
        set_flat_params(self.model, x)
        self.model.zero_grad()
        loss_d = data_loss(self.model, self.t_obs, self.theta_obs)
        params = list(self.model.parameters())
        return get_grads_from_loss(loss_d, params, allow_unused=True)

    def h_and_jacobian(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """One shared interior collocation draw so h and J_h refer to the same constraints."""
        set_flat_params(self.model, x)
        t_coll = (self.cfg.t0 + (self.cfg.t1 - self.cfg.t0) * torch.rand(self.cfg.n_coll, 1, device=device)).float()
        params = list(self.model.parameters())

        self.model.zero_grad()
        li = ic_loss(self.model, self.cfg)
        ic_val = float(li.detach().item())
        g_ic = get_grads_from_loss(li, params, allow_unused=True)

        self.model.zero_grad()
        lr_ = residual_loss(self.model, t_coll, self.cfg)
        res_val = float(lr_.detach().item())
        g_res = get_grads_from_loss(lr_, params, allow_unused=True)

        h_x = np.array([ic_val, res_val], dtype=np.float32)
        J_h = np.vstack([g_ic, g_res])
        return h_x, J_h


def adamfl_optimize_inverse(
    solver: AdamFLInversePendulum,
    x_start: np.ndarray,
    Kp: np.ndarray,
    Ki: np.ndarray | None,
    eta: float,
    max_iter: int,
    tol: float,
) -> tuple[np.ndarray, dict]:
    x = np.array(x_start, dtype=np.float32)
    if Ki is None:
        Ki = np.zeros_like(Kp, dtype=np.float32)
    integral = np.zeros(Kp.shape[0], dtype=np.float32)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    t = 0

    log_path = os.path.join(LOG_DIR, "AdamFL_inverse_pendulum_loss.txt")
    hist = {"data": [], "ic": [], "res": [], "kkt": [], "omega": [], "time_s": None}
    t_start = time.time()

    pbar = trange(max_iter, desc="AdamFL (inverse pendulum)", unit="iter")
    with open(log_path, "w") as f:
        f.write("iter\tloss_data\tloss_ic\tloss_res\tkkt_gap\tomega0_sq\n")
        for it in pbar:
            grad_f = solver.df_sofl(x)
            h_x, J_h = solver.h_and_jacobian(x)

            integral = integral + h_x
            i_control = Ki @ integral
            p_control = -Kp @ h_x
            JhJht = J_h @ J_h.T
            rhs = p_control + i_control + J_h @ grad_f
            I = np.eye(JhJht.shape[0], dtype=np.float32)
            lam = -solve(JhJht + 1e-6 * I, rhs, assume_a="pos")
            kkt = grad_f + (J_h.T @ lam).ravel()

            gn = float(np.linalg.norm(kkt))
            if gn > 10.0:
                kkt = kkt * (10.0 / (gn + 1e-12))

            t += 1
            m = beta1 * m + (1.0 - beta1) * kkt
            v = beta2 * v + (1.0 - beta2) * (kkt**2)
            m_hat = m / (1.0 - beta1**t)
            v_hat = v / (1.0 - beta2**t)
            step = eta * m_hat / (np.sqrt(v_hat) + eps)
            x_new = x - step
            step_size = float(np.linalg.norm(step))

            set_flat_params(solver.model, x_new)
            w_hat = float(solver.model.omega0_sq().item())
            ld = float(data_loss(solver.model, solver.t_obs, solver.theta_obs).item())

            if (it + 1) % 50 == 0 or it == 0:
                kkt_gap = float(max(np.linalg.norm(kkt), np.max(np.abs(h_x))))
                hist["data"].append(ld)
                hist["ic"].append(float(abs(h_x[0])))
                hist["res"].append(float(abs(h_x[1])))
                hist["kkt"].append(kkt_gap)
                hist["omega"].append(w_hat)
                f.write(
                    f"{it+1}\t{ld:.8e}\t{abs(h_x[0]):.8e}\t{abs(h_x[1]):.8e}\t{kkt_gap:.8e}\t{w_hat:.8e}\n"
                )
                pbar.set_postfix({"data": f"{ld:.2e}", "ic": f'{abs(h_x[0]):.2e}', "w2": f"{w_hat:.3f}"})

            if step_size < tol:
                x = x_new
                break
            if np.isnan(x_new).any():
                break
            x = x_new

    hist["time_s"] = time.time() - t_start
    return x, hist


def train_adamfl_inverse(
    cfg: InversePendulumConfig,
    t_obs: torch.Tensor,
    theta_obs: torch.Tensor,
    warmup_iters: int = 3000,
    adam_lr: float = 1e-3,
    adamfl_iters: int = 20000,
) -> tuple[InversePendulumPINN, dict]:
    solver = AdamFLInversePendulum(cfg, t_obs, theta_obs)
    opt = torch.optim.Adam(solver.model.parameters(), lr=adam_lr)

    pbar = trange(warmup_iters, desc="Adam warm-start (inverse)", unit="iter")
    for _ in pbar:
        t_coll = (cfg.t0 + (cfg.t1 - cfg.t0) * torch.rand(cfg.n_coll, 1, device=device)).float()
        ld, li, lr_ = inverse_losses(solver.model, t_coll, t_obs, theta_obs, cfg)
        loss = cfg.w_data * ld + cfg.w_ic * li + cfg.w_res * lr_
        opt.zero_grad()
        loss.backward()
        opt.step()
        pbar.set_postfix({"data": f"{ld.item():.2e}", "w2": f'{solver.model.omega0_sq().item():.3f}'})

    x0 = get_flat_params(solver.model)
    Kp = np.array([[500.0, 0.0], [0.0, 500.0]], dtype=np.float32)
    Ki = np.zeros((2, 2), dtype=np.float32)
    x_final, hist = adamfl_optimize_inverse(solver, x0, Kp, Ki, eta=1e-4, max_iter=adamfl_iters, tol=1e-8)
    set_flat_params(solver.model, x_final)
    return solver.model, hist


# ============================================================
# Evaluation
# ============================================================
@torch.no_grad()
def eval_theta(model: InversePendulumPINN, t_np: np.ndarray) -> np.ndarray:
    model.eval()
    t = torch.tensor(t_np.reshape(-1, 1), dtype=torch.float32, device=device)
    return model(t).detach().cpu().numpy().reshape(-1)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    cfg = CFG_INV
    w_true = omega0_sq_true(cfg)

    t_ref, theta_ref, _ = solve_pendulum_reference(cfg)
    t_obs_np, theta_obs_np = make_observations(t_ref, theta_ref, cfg)

    t_obs = torch.tensor(t_obs_np.reshape(-1, 1), dtype=torch.float32, device=device)
    theta_obs = torch.tensor(theta_obs_np.reshape(-1, 1), dtype=torch.float32, device=device)

    print(f"\nTrue omega0^2 = g/L = {w_true:.6f}")
    print(f"Initial guess omega0^2 = {cfg.omega0_sq_init:.6f}")
    print(f"Observations: n_obs = {len(t_obs_np)}")

    print("\n=== Train: Adam (inverse) ===")
    model_adam, hist_adam = train_adam_inverse(cfg, t_obs, theta_obs, iters=20000, lr=1e-3)

    print("\n=== Train: AdamFL (inverse) ===")
    model_af, hist_af = train_adamfl_inverse(cfg, t_obs, theta_obs, warmup_iters=3000, adam_lr=1e-3, adamfl_iters=20000)

    theta_adam = eval_theta(model_adam, t_ref)
    theta_af = eval_theta(model_af, t_ref)

    w_adam = model_adam.omega0_sq().item()
    w_af = model_af.omega0_sq().item()

    rel_theta_adam = np.linalg.norm(theta_adam - theta_ref) / (np.linalg.norm(theta_ref) + 1e-12)
    rel_theta_af = np.linalg.norm(theta_af - theta_ref) / (np.linalg.norm(theta_ref) + 1e-12)
    rel_w_adam = abs(w_adam - w_true) / (w_true + 1e-12)
    rel_w_af = abs(w_af - w_true) / (w_true + 1e-12)

    print("\n=== Summary ===")
    print(f"Adam:    time {hist_adam['time_s']:.2f}s | rel L2(theta)={rel_theta_adam:.3e} | omega0^2={w_adam:.6f} | rel err(param)={rel_w_adam:.3e}")
    print(f"AdamFL:  time {hist_af['time_s']:.2f}s | rel L2(theta)={rel_theta_af:.3e} | omega0^2={w_af:.6f} | rel err(param)={rel_w_af:.3e}")

    savemat(
        os.path.join(DATA_DIR, "pendulum_inverse_compare.mat"),
        {
            "t_ref": t_ref,
            "theta_ref": theta_ref,
            "t_obs": t_obs_np,
            "theta_obs": theta_obs_np,
            "theta_adam": theta_adam,
            "theta_adamfl": theta_af,
            "omega0_sq_true": w_true,
            "omega0_sq_adam": w_adam,
            "omega0_sq_adamfl": w_af,
            "rel_err_theta_adam": rel_theta_adam,
            "rel_err_theta_adamfl": rel_theta_af,
            "rel_err_param_adam": rel_w_adam,
            "rel_err_param_adamfl": rel_w_af,
        },
    )

    # Theta plot
    plt.figure(figsize=(10, 4))
    plt.plot(t_ref, theta_ref, "k-", lw=2, label="Reference (true omega0^2)")
    plt.scatter(t_obs_np, theta_obs_np, c="gray", s=12, alpha=0.7, label="Observations")
    plt.plot(t_ref, theta_adam, lw=1.8, label=f"Adam: rel L2(theta)={rel_theta_adam:.2e}, ω² err={rel_w_adam:.2e}")
    plt.plot(t_ref, theta_af, lw=1.8, label=f"AdamFL: rel L2(theta)={rel_theta_af:.2e}, ω² err={rel_w_af:.2e}")
    plt.xlabel("t")
    plt.ylabel("theta(t)")
    plt.title("Inverse pendulum: recover omega0^2 = g/L")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "pendulum_inverse_theta_compare.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # Learned parameter trajectory (logged every 50 iters)
    plt.figure(figsize=(10, 4))
    plt.axhline(w_true, color="k", ls="--", label=f"true omega0^2 = {w_true:.4f}")
    plt.plot(hist_adam["omega"], label="Adam estimate (every 50 iters)")
    plt.plot(hist_af["omega"], label="AdamFL estimate (every 50 iters)")
    plt.xlabel("log step index (x50)")
    plt.ylabel("omega0^2")
    plt.title("Learned physical parameter vs iteration log")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "pendulum_inverse_omega_trace.png"), dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(hist_adam["data"], label="Adam: data loss")
    plt.plot(hist_adam["res"], label="Adam: res loss")
    plt.plot(hist_af["data"], label="AdamFL: data loss (eval)")
    plt.plot(hist_af["res"], label="AdamFL: |PL constraint|")
    plt.yscale("log")
    plt.xlabel("log step (x50)")
    plt.ylabel("loss")
    plt.title("Inverse training curves")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "pendulum_inverse_loss_compare.png"), dpi=200, bbox_inches="tight")
    plt.close()

    print("\nSaved:")
    print(f" - {FIG_DIR}/pendulum_inverse_theta_compare.png")
    print(f" - {FIG_DIR}/pendulum_inverse_omega_trace.png")
    print(f" - {FIG_DIR}/pendulum_inverse_loss_compare.png")
    print(f" - {DATA_DIR}/pendulum_inverse_compare.mat")


if __name__ == "__main__":
    main()
