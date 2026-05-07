import os
import time
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
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
class PendulumConfig:
    g: float = 9.81
    L: float = 1.0
    t0: float = 0.0
    t1: float = 10.0
    theta0: float = 1.0  # rad
    omega0: float = 0.0  # rad/s

    n_coll: int = 2000
    n_eval: int = 400


CFG = PendulumConfig()


# ============================================================
# Reference solution (solve_ivp)
# ============================================================
def solve_pendulum_reference(cfg: PendulumConfig):
    w02 = cfg.g / cfg.L

    def rhs(t, y):
        theta, omega = y
        return [omega, -w02 * np.sin(theta)]

    t_eval = np.linspace(cfg.t0, cfg.t1, cfg.n_eval)
    sol = solve_ivp(
        rhs,
        (cfg.t0, cfg.t1),
        y0=[cfg.theta0, cfg.omega0],
        t_eval=t_eval,
        rtol=1e-10,
        atol=1e-12,
        method="DOP853",
    )
    return sol.t.astype(np.float32), sol.y[0].astype(np.float32), sol.y[1].astype(np.float32)


# ============================================================
# PINN model (UPDATED: Fourier Features & Time Scaling)
# ============================================================
class MLP(nn.Module):
    def __init__(self, hidden=128, layers=4, t_max=10.0):
        super().__init__()
        self.t_max = t_max
        
        # Define frequencies for the Fourier embedding (1 to 5 Hz)
        # This acts as a hint to the network about the oscillatory nature of the problem
        self.register_buffer("freqs", torch.arange(1, 6, dtype=torch.float32) * (2 * np.pi / (t_max/2)))
        
        # Input dim: 1 (for t_norm) + 2 * len(freqs) (for sin and cos pairs)
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
        # 1. Normalize time to [-1, 1] to prevent Tanh saturation
        t_norm = 2.0 * (t / self.t_max) - 1.0
        
        # 2. Compute Fourier features
        args = t_norm * self.freqs
        sin_features = torch.sin(args)
        cos_features = torch.cos(args)
        
        # 3. Concatenate [t_norm, sin(w*t), cos(w*t)]
        x = torch.cat([t_norm, sin_features, cos_features], dim=-1)
        return self.net(x)


def pendulum_losses(model: nn.Module, t_coll: torch.Tensor, cfg: PendulumConfig):
    w02 = cfg.g / cfg.L
    mse = nn.MSELoss()

    # Collocation residual
    t = t_coll.clone().detach().requires_grad_(True)
    theta = model(t)
    theta_t = torch.autograd.grad(theta, t, torch.ones_like(theta), create_graph=True)[0]
    theta_tt = torch.autograd.grad(theta_t, t, torch.ones_like(theta_t), create_graph=True)[0]
    res = theta_tt + w02 * torch.sin(theta)
    loss_res = mse(res, torch.zeros_like(res))

    # Initial conditions (t=0): theta(0)=theta0, theta'(0)=omega0
    t0 = torch.tensor([[cfg.t0]], dtype=torch.float32, device=device, requires_grad=True)
    theta0_pred = model(t0)
    theta0_t = torch.autograd.grad(theta0_pred, t0, torch.ones_like(theta0_pred), create_graph=True)[0]
    loss_ic = mse(theta0_pred, torch.tensor([[cfg.theta0]], dtype=torch.float32, device=device)) + mse(
        theta0_t, torch.tensor([[cfg.omega0]], dtype=torch.float32, device=device)
    )

    return loss_ic, loss_res


# ============================================================
# Adam baseline training (UPDATED: IC Weighting & Scheduler)
# ============================================================
def train_adam(cfg: PendulumConfig, iters: int = 20000, lr: float = 1e-3):
    model = MLP(t_max=cfg.t1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Scheduler to decay learning rate smoothly
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.95)

    t_coll = (cfg.t0 + (cfg.t1 - cfg.t0) * torch.rand(cfg.n_coll, 1, device=device)).float()

    log_path = os.path.join(LOG_DIR, "ADAM_pendulum_loss.txt")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    hist = {"ic": [], "res": [], "total": [], "time_s": None}

    t_start = time.time()
    pbar = trange(iters, desc="Adam (pendulum)", unit="iter")
    with open(log_path, "w") as f:
        f.write("iter\tloss_ic\tloss_res\ttotal\n")
        for it in pbar:
            loss_ic, loss_res = pendulum_losses(model, t_coll, cfg)
            
            # Weight the Initial Condition heavily to prevent trivial collapse
            loss = 100.0 * loss_ic + loss_res
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            # Step scheduler every 1000 iterations
            if (it + 1) % 1000 == 0:
                scheduler.step()

            if (it + 1) % 50 == 0 or it == 0:
                li = float(loss_ic.item())
                lr_ = float(loss_res.item())
                lt = float(loss.item())
                hist["ic"].append(li)
                hist["res"].append(lr_)
                hist["total"].append(lt)
                f.write(f"{it+1}\t{li:.8e}\t{lr_:.8e}\t{lt:.8e}\n")
                pbar.set_postfix({"ic": f"{li:.2e}", "res": f"{lr_:.2e}", "tot": f"{lt:.2e}"})

    hist["time_s"] = time.time() - t_start
    return model, hist


# ============================================================
# AdamFL (KKT-style) utilities 
# ============================================================
def get_flat_params(model):
    return np.concatenate([p.detach().cpu().numpy().ravel() for p in model.parameters()])


def set_flat_params(model, flat_params):
    flat_params = flat_params.astype(np.float32)
    pointer = 0
    for p in model.parameters():
        n = p.numel()
        vals = torch.from_numpy(flat_params[pointer : pointer + n]).view_as(p).to(device)
        p.data.copy_(vals)
        pointer += n


def get_flat_grads(model):
    grads = []
    for p in model.parameters():
        if p.grad is None:
            grads.append(np.zeros(p.numel(), dtype=np.float32))
        else:
            grads.append(p.grad.detach().cpu().numpy().ravel())
    return np.concatenate(grads)


class AdamFLPendulum:
    def __init__(self, cfg: PendulumConfig):
        self.cfg = cfg
        # UPDATED: Pass t_max to MLP
        self.model = MLP(t_max=cfg.t1).to(device)
        self.t_coll = (cfg.t0 + (cfg.t1 - cfg.t0) * torch.rand(cfg.n_coll, 1, device=device)).float()

    # objective (kept zero like Navier-Stokes script)
    def f(self, x):
        return 0.0

    def df(self, x):
        return np.zeros_like(x)

    # constraints: [ic_loss, residual_loss]
    def h(self, x):
        set_flat_params(self.model, x)
        loss_ic, loss_res = pendulum_losses(self.model, self.t_coll, self.cfg)
        return np.array([loss_ic.item(), loss_res.item()], dtype=np.float32)

    def dh(self, x):
        set_flat_params(self.model, x)

        self.model.zero_grad()
        loss_ic, _ = pendulum_losses(self.model, self.t_coll, self.cfg)
        loss_ic.backward(retain_graph=True)
        grad_ic = get_flat_grads(self.model)

        self.model.zero_grad()
        _, loss_res = pendulum_losses(self.model, self.t_coll, self.cfg)
        loss_res.backward()
        grad_res = get_flat_grads(self.model)

        return np.vstack([grad_ic, grad_res])


def adamfl_optimize(
    solver: AdamFLPendulum,
    x_start: np.ndarray,
    Kp: np.ndarray,
    Ki: np.ndarray | None = None,
    eta: float = 5e-4,
    max_iter: int = 8000,
    tol: float = 1e-7,
):
    x = np.array(x_start, dtype=np.float32)
    if Ki is None:
        Ki = np.zeros_like(Kp, dtype=np.float32)

    integral = np.zeros(Kp.shape[0], dtype=np.float32)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    t = 0

    log_path = os.path.join(LOG_DIR, "AdamFL_pendulum_loss.txt")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    hist = {"ic": [], "res": [], "kkt": [], "time_s": None}
    t_start = time.time()

    pbar = trange(max_iter, desc="AdamFL (pendulum)", unit="iter")
    with open(log_path, "w") as f:
        f.write("iter\tloss_ic\tloss_res\tkkt_gap\tlambda_ic\tlambda_res\n")
        for it in pbar:
            grad_f = solver.df(x)
            J_h = solver.dh(x)
            h_x = solver.h(x)

            JhJht = J_h @ J_h.T
            p_control = -Kp @ h_x
            integral = integral + h_x
            i_control = Ki @ integral
            rhs = p_control + i_control + J_h @ grad_f

            I = np.eye(JhJht.shape[0], dtype=np.float32)
            lambda_vec = -solve(JhJht + 1e-6 * I, rhs, assume_a="pos")

            kkt_grad = grad_f + (J_h.T @ lambda_vec).flatten()
            grad_norm = np.linalg.norm(kkt_grad)
            clip_value = 10.0
            if grad_norm > clip_value:
                kkt_grad = kkt_grad * (clip_value / (grad_norm + 1e-12))

            t += 1
            m = beta1 * m + (1.0 - beta1) * kkt_grad
            v = beta2 * v + (1.0 - beta2) * (kkt_grad**2)
            m_hat = m / (1.0 - beta1**t)
            v_hat = v / (1.0 - beta2**t)

            step_vec = eta * m_hat / (np.sqrt(v_hat) + eps)
            x_new = x - step_vec
            step_size = np.linalg.norm(step_vec)

            if (it + 1) % 50 == 0 or it == 0:
                kkt_gap = float(max(np.linalg.norm(kkt_grad), np.max(np.abs(h_x))))
                ic_val = float(abs(h_x[0]))
                res_val = float(abs(h_x[1]))
                hist["ic"].append(ic_val)
                hist["res"].append(res_val)
                hist["kkt"].append(kkt_gap)
                f.write(
                    f"{it+1}\t{ic_val:.8e}\t{res_val:.8e}\t{kkt_gap:.8e}\t{lambda_vec[0]:.8e}\t{lambda_vec[1]:.8e}\n"
                )
                pbar.set_postfix(
                    {"ic": f"{ic_val:.2e}", "res": f"{res_val:.2e}", "kkt": f"{kkt_gap:.2e}"}
                )

            if step_size < tol:
                x = x_new
                break
            if np.isnan(x_new).any():
                break
            x = x_new

    hist["time_s"] = time.time() - t_start
    return x, hist


def train_adamfl(
    cfg: PendulumConfig,
    warmup_iters: int = 2000,
    adam_lr: float = 1e-3,
    adamfl_iters: int = 20000,
):
    solver = AdamFLPendulum(cfg)

    # Adam warm-start 
    opt = torch.optim.Adam(solver.model.parameters(), lr=adam_lr)
    pbar = trange(warmup_iters, desc="Adam warm-start (pendulum)", unit="iter")
    for _ in pbar:
        loss_ic, loss_res = pendulum_losses(solver.model, solver.t_coll, cfg)
        # Weight IC during warmup as well
        loss = 100.0 * loss_ic + loss_res
        opt.zero_grad()
        loss.backward()
        opt.step()
        pbar.set_postfix({"ic": f"{loss_ic.item():.2e}", "res": f"{loss_res.item():.2e}"})

    x0 = get_flat_params(solver.model)
    Kp = np.array([[500.0, 0.0], [0.0, 500.0]], dtype=np.float32)
    Ki = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    x_final, hist = adamfl_optimize(solver, x0, Kp=Kp, Ki=Ki, eta=5e-4, max_iter=adamfl_iters, tol=1e-7)
    set_flat_params(solver.model, x_final)
    return solver.model, hist


# ============================================================
# Evaluation + plotting
# ============================================================
@torch.no_grad()
def eval_theta(model: nn.Module, t_eval_np: np.ndarray):
    model.eval()
    t = torch.tensor(t_eval_np.reshape(-1, 1), dtype=torch.float32, device=device)
    theta = model(t).detach().cpu().numpy().reshape(-1)
    return theta


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    t_ref, theta_ref, omega_ref = solve_pendulum_reference(CFG)

    print("\n=== Train: Adam ===")
    model_adam, hist_adam = train_adam(CFG, iters=20000, lr=1e-3)

    print("\n=== Train: AdamFL (warm-start + AdamFL) ===")
    model_adamfl, hist_adamfl = train_adamfl(CFG, warmup_iters=2000, adam_lr=1e-3)

    theta_adam = eval_theta(model_adam, t_ref)
    theta_adamfl = eval_theta(model_adamfl, t_ref)

    # Relative L2 errors
    rel_err_adam = np.linalg.norm(theta_adam - theta_ref) / (np.linalg.norm(theta_ref) + 1e-12)
    rel_err_adamfl = np.linalg.norm(theta_adamfl - theta_ref) / (np.linalg.norm(theta_ref) + 1e-12)

    print("\n=== Summary ===")
    print(f"Adam   time: {hist_adam['time_s']:.2f}s | rel L2(theta): {rel_err_adam:.3e}")
    print(f"AdamFL time: {hist_adamfl['time_s']:.2f}s | rel L2(theta): {rel_err_adamfl:.3e}")

    # Save .mat for convenience
    savemat(
        os.path.join(DATA_DIR, "pendulum_compare.mat"),
        {
            "t": t_ref,
            "theta_ref": theta_ref,
            "omega_ref": omega_ref,
            "theta_adam": theta_adam,
            "theta_adamfl": theta_adamfl,
            "rel_err_adam": rel_err_adam,
            "rel_err_adamfl": rel_err_adamfl,
        },
    )

    # Plot theta(t)
    plt.figure(figsize=(10, 4))
    plt.plot(t_ref, theta_ref, "k-", lw=2, label="Reference (solve_ivp)")
    plt.plot(t_ref, theta_adam, lw=1.8, label=f"PINN + Adam (rel L2={rel_err_adam:.2e})")
    plt.plot(t_ref, theta_adamfl, lw=1.8, label=f"PINN + AdamFL (rel L2={rel_err_adamfl:.2e})")
    plt.xlabel("t")
    plt.ylabel("theta(t)")
    plt.title("Nonlinear pendulum: solution comparison")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "pendulum_theta_compare.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # Plot losses (both logged every 50 iters)
    plt.figure(figsize=(10, 4))
    plt.plot(hist_adam["ic"], label="Adam: IC loss")
    plt.plot(hist_adam["res"], label="Adam: residual loss")
    plt.plot(hist_adamfl["ic"], label="AdamFL: IC loss")
    plt.plot(hist_adamfl["res"], label="AdamFL: residual loss")
    plt.yscale("log")
    plt.xlabel("log step (x50 iters)")
    plt.ylabel("loss")
    plt.title("Training losses (logged every 50 iterations)")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "pendulum_loss_compare.png"), dpi=200, bbox_inches="tight")
    plt.close()

    print("\nSaved:")
    print(" - ODE/pendulum/figures/pendulum_theta_compare.png")
    print(" - ODE/pendulum/figures/pendulum_loss_compare.png")
    print(" - ODE/pendulum/data/pendulum_compare.mat")


if __name__ == "__main__":
    main()