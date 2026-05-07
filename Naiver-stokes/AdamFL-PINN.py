import os
import argparse
import random as rm
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import torch
import torch.nn as nn
from scipy.linalg import solve
from tqdm import trange


# ============================================================
# 0) Reproducibility + device
# ============================================================
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
rm.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Optional deterministic mode (slower but reproducible)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ============================================================
# 1) Taylor-Green vortex setup (2D incompressible Navier-Stokes)
# ============================================================
# Domain: x,y in [0, 2pi], t in [0, T_MAX]
T_MAX = 1.0
nu = 0.01  # viscosity

# Training sample sizes (aligned with FL-PINN/AL-PINN)
N_COLL = 10000
N_IC = 1000
N_BC = 1000

Kp = np.array([[800.0, 0.0], [0.0, 800.0]], dtype=np.float32)
Ki = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
ETA = 5e-4
MAX_ITER = 15000
TOL = 1e-7
BETA_MERIT = 1.0
LAMBDA_EMA = 0.9
INT_CLIP = 1e3
JH_REG = 1e-6
MAX_BACKTRACK = 5


# ============================================================
# 2) Exact solution (for IC/periodic BC and evaluation)
# ============================================================
def taylor_green_u(x, y, t):
    return -torch.cos(x) * torch.sin(y) * torch.exp(-2.0 * nu * t)


def taylor_green_v(x, y, t):
    return torch.sin(x) * torch.cos(y) * torch.exp(-2.0 * nu * t)


def taylor_green_p(x, y, t):
    return -0.25 * (torch.cos(2.0 * x) + torch.cos(2.0 * y)) * torch.exp(-4.0 * nu * t)


# ============================================================
# 3) Data generation
# ============================================================
def sample_training_data():
    # Collocation points for PDE residual
    x_coll = 2.0 * np.pi * np.random.rand(N_COLL, 1)
    y_coll = 2.0 * np.pi * np.random.rand(N_COLL, 1)
    t_coll = T_MAX * np.random.rand(N_COLL, 1)
    X_coll = np.hstack([x_coll, y_coll, t_coll]).astype(np.float32)

    # Initial condition points (t = 0)
    x_ic = 2.0 * np.pi * np.random.rand(N_IC, 1)
    y_ic = 2.0 * np.pi * np.random.rand(N_IC, 1)
    t_ic = np.zeros((N_IC, 1), dtype=np.float32)
    X_ic = np.hstack([x_ic, y_ic, t_ic]).astype(np.float32)

    # Periodic boundary points on x-boundaries: x=0 and x=2pi
    y_bx = 2.0 * np.pi * np.random.rand(N_BC, 1)
    t_bx = T_MAX * np.random.rand(N_BC, 1)
    X_bx_l = np.hstack([np.zeros((N_BC, 1)), y_bx, t_bx]).astype(np.float32)
    X_bx_r = np.hstack([2.0 * np.pi * np.ones((N_BC, 1)), y_bx, t_bx]).astype(np.float32)

    # Periodic boundary points on y-boundaries: y=0 and y=2pi
    x_by = 2.0 * np.pi * np.random.rand(N_BC, 1)
    t_by = T_MAX * np.random.rand(N_BC, 1)
    X_by_b = np.hstack([x_by, np.zeros((N_BC, 1)), t_by]).astype(np.float32)
    X_by_t = np.hstack([x_by, 2.0 * np.pi * np.ones((N_BC, 1)), t_by]).astype(np.float32)

    return X_coll, X_ic, X_bx_l, X_bx_r, X_by_b, X_by_t


# ============================================================
# 4) PINN model
# ============================================================
class PINN(nn.Module):
    def __init__(self, layers=(3, 64, 64, 64, 64, 3)):
        super().__init__()
        net = []
        for i in range(len(layers) - 2):
            net.append(nn.Linear(layers[i], layers[i + 1]))
            net.append(nn.Tanh())
        net.append(nn.Linear(layers[-2], layers[-1]))
        self.net = nn.Sequential(*net)

        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, xyt):
        return self.net(xyt)


class PhysicsInformedNS:
    def __init__(self, X_coll, X_ic, X_bx_l, X_bx_r, X_by_b, X_by_t):
        self.model = PINN().to(device)
        self.mse = nn.MSELoss()

        self.X_coll = torch.tensor(X_coll, dtype=torch.float32, device=device, requires_grad=True)
        self.X_ic = torch.tensor(X_ic, dtype=torch.float32, device=device)

        self.X_bx_l = torch.tensor(X_bx_l, dtype=torch.float32, device=device)
        self.X_bx_r = torch.tensor(X_bx_r, dtype=torch.float32, device=device)
        self.X_by_b = torch.tensor(X_by_b, dtype=torch.float32, device=device)
        self.X_by_t = torch.tensor(X_by_t, dtype=torch.float32, device=device)

    def predict_uvp(self, X):
        out = self.model(X)
        u = out[:, 0:1]
        v = out[:, 1:2]
        p = out[:, 2:3]
        return u, v, p

    def residual_loss(self):
        X = self.X_coll.clone().detach().requires_grad_(True)
        u, v, p = self.predict_uvp(X)

        grads_u = torch.autograd.grad(u, X, torch.ones_like(u), create_graph=True)[0]
        u_x, u_y, u_t = grads_u[:, 0:1], grads_u[:, 1:2], grads_u[:, 2:3]

        grads_v = torch.autograd.grad(v, X, torch.ones_like(v), create_graph=True)[0]
        v_x, v_y, v_t = grads_v[:, 0:1], grads_v[:, 1:2], grads_v[:, 2:3]

        grads_p = torch.autograd.grad(p, X, torch.ones_like(p), create_graph=True)[0]
        p_x, p_y = grads_p[:, 0:1], grads_p[:, 1:2]

        u_xx = torch.autograd.grad(u_x, X, torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
        u_yy = torch.autograd.grad(u_y, X, torch.ones_like(u_y), create_graph=True)[0][:, 1:2]

        v_xx = torch.autograd.grad(v_x, X, torch.ones_like(v_x), create_graph=True)[0][:, 0:1]
        v_yy = torch.autograd.grad(v_y, X, torch.ones_like(v_y), create_graph=True)[0][:, 1:2]

        continuity = u_x + v_y
        mom_x = u_t + u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
        mom_y = v_t + u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)

        zero = torch.zeros_like(continuity)
        loss_cont = self.mse(continuity, zero)
        loss_momx = self.mse(mom_x, zero)
        loss_momy = self.mse(mom_y, zero)
        return loss_cont + loss_momx + loss_momy

    def initial_loss(self):
        x = self.X_ic[:, 0:1]
        y = self.X_ic[:, 1:2]
        t = self.X_ic[:, 2:3]
        u_true = taylor_green_u(x, y, t)
        v_true = taylor_green_v(x, y, t)

        u_pred, v_pred, _ = self.predict_uvp(self.X_ic)
        return self.mse(u_pred, u_true) + self.mse(v_pred, v_true)

    def periodic_boundary_loss(self):
        u_l, v_l, p_l = self.predict_uvp(self.X_bx_l)
        u_r, v_r, p_r = self.predict_uvp(self.X_bx_r)

        u_b, v_b, p_b = self.predict_uvp(self.X_by_b)
        u_t, v_t, p_t = self.predict_uvp(self.X_by_t)

        loss_x = self.mse(u_l, u_r) + self.mse(v_l, v_r) + self.mse(p_l, p_r)
        loss_y = self.mse(u_b, u_t) + self.mse(v_b, v_t) + self.mse(p_b, p_t)
        return loss_x + loss_y

    def ibc_loss(self):
        return self.initial_loss() + self.periodic_boundary_loss()


# ============================================================
# 5) AdamFL utilities (same pattern as AdaFL-PINN notebook)
# ============================================================
def get_flat_params(model):
    return np.concatenate([p.detach().cpu().numpy().ravel() for p in model.parameters()])


def set_flat_params(model, flat_params):
    flat_params = flat_params.astype(np.float32)
    pointer = 0
    for p in model.parameters():
        n = p.numel()
        vals = torch.from_numpy(flat_params[pointer:pointer + n]).view_as(p).to(device)
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


pinn_ns = None  # global for wrappers


def f(x):
    # Objective: physics loss (aligned with FL-PINN split)
    set_flat_params(pinn_ns.model, x)
    return pinn_ns.residual_loss().item()


def df(x):
    # Gradient of physics loss
    set_flat_params(pinn_ns.model, x)
    pinn_ns.model.zero_grad()
    loss = pinn_ns.residual_loss()
    loss.backward()
    return get_flat_grads(pinn_ns.model)


def h(x):
    set_flat_params(pinn_ns.model, x)
    loss_ic = pinn_ns.initial_loss().item()
    loss_bc = pinn_ns.periodic_boundary_loss().item()
    return np.array([loss_ic, loss_bc], dtype=np.float32)


def dh(x):
    set_flat_params(pinn_ns.model, x)

    pinn_ns.model.zero_grad()
    loss_ic = pinn_ns.initial_loss()
    loss_ic.backward(retain_graph=True)
    grad_ic = get_flat_grads(pinn_ns.model)

    pinn_ns.model.zero_grad()
    loss_bc = pinn_ns.periodic_boundary_loss()
    loss_bc.backward()
    grad_bc = get_flat_grads(pinn_ns.model)

    return np.vstack([grad_ic, grad_bc])


def merit_value(f_val, h_val, beta_merit):
    return float(f_val + beta_merit * np.dot(h_val, h_val))


def AdamFL_PINN(
    f,
    h,
    df,
    dh,
    x_start,
    Kp,
    Ki=None,
    eta=1e-3,
    max_iter=5000,
    tol=1e-7,
    beta_merit=1.0,
    lambda_ema=0.9,
    int_clip=1e3,
    jh_reg=1e-6,
    max_backtrack=5,
):
    x = np.array(x_start, dtype=np.float32)
    history = []
    lambda_ic_hist = []
    lambda_bc_hist = []

    if Ki is None:
        Ki = np.zeros_like(Kp, dtype=np.float32)

    integral = np.zeros(Kp.shape[0], dtype=np.float32)

    beta1, beta2, eps = 0.9, 0.999, 1e-8
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    t = 0
    eta_curr = float(eta)
    lambda_prev = np.zeros(Kp.shape[0], dtype=np.float32)

    pbar = trange(max_iter, desc="AdamFL-PINN Training", unit="iter")
    for it in pbar:
        grad_f = df(x)
        J_h = dh(x)
        h_x = h(x)

        JhJht = J_h @ J_h.T
        p_control = -Kp @ h_x
        integral = integral + h_x
        i_control = Ki @ integral
        rhs = p_control + i_control + J_h @ grad_f

        I = np.eye(JhJht.shape[0], dtype=np.float32)
        try:
            lambda_raw = -solve(JhJht + jh_reg * I, rhs, assume_a="sym")
        except Exception:
            lambda_raw = -np.linalg.lstsq(JhJht + jh_reg * I, rhs, rcond=None)[0]

        # Smooth dual updates to avoid oscillation in constrained PINN training.
        lambda_vec = lambda_ema * lambda_prev + (1.0 - lambda_ema) * lambda_raw
        lambda_prev = lambda_vec.copy()

        lambda_ic = lambda_vec[0]
        lambda_bc = lambda_vec[1]
        lambda_ic_hist.append(lambda_ic)
        lambda_bc_hist.append(lambda_bc)

        kkt_grad = grad_f + (J_h.T @ lambda_vec).flatten()

        grad_norm = np.linalg.norm(kkt_grad)
        clip_value = 10.0
        if grad_norm > clip_value:
            kkt_grad = kkt_grad * (clip_value / (grad_norm + 1e-12))

        t += 1
        m = beta1 * m + (1.0 - beta1) * kkt_grad
        v = beta2 * v + (1.0 - beta2) * (kkt_grad ** 2)
        m_hat = m / (1.0 - beta1 ** t)
        v_hat = v / (1.0 - beta2 ** t)

        step_base = m_hat / (np.sqrt(v_hat) + eps)
        f_old = f(x)
        merit_old = merit_value(f_old, h_x, beta_merit)

        accepted = False
        step_vec = eta_curr * step_base
        x_new = x - step_vec
        for bt in range(max_backtrack + 1):
            h_new = h(x_new)
            f_new = f(x_new)
            merit_new = merit_value(f_new, h_new, beta_merit)

            # Accept if merit decreases or constraints are clearly improved.
            if merit_new <= merit_old or np.linalg.norm(h_new) <= 0.99 * np.linalg.norm(h_x):
                accepted = True
                break

            step_vec = 0.5 * step_vec
            x_new = x - step_vec

        # Adapt global step size from acceptance behavior.
        if accepted:
            eta_curr = min(1e-2, eta_curr * 1.02)
        else:
            eta_curr = max(1e-6, eta_curr * 0.5)

        step_size = np.linalg.norm(step_vec)

        if (it + 1) % 100 == 0:
            kkt_gap = max(np.linalg.norm(kkt_grad), np.max(np.abs(h_x)))
            ic_val = float(abs(h_x[0]))
            bc_val = float(abs(h_x[1]))
            pbar.set_postfix(
                {
                    "ic_loss": f"{ic_val:.2e}",
                    "bc_loss": f"{bc_val:.2e}",
                    "lambda_ic": f"{lambda_ic:.2e}",
                    "lambda_bc": f"{lambda_bc:.2e}",
                    "eta": f"{eta_curr:.2e}",
                }
            )

        # Save history every iteration for detailed diagnostics.
        kkt_gap = max(np.linalg.norm(kkt_grad), np.max(np.abs(h_x)))
        ic_val = float(abs(h_x[0]))
        bc_val = float(abs(h_x[1]))
        history.append((f(x), kkt_gap, ic_val, bc_val))

        if step_size < tol:
            print(f"\nConvergence at iter {it}: step_size={step_size:.2e} < tol={tol:.2e}")
            x = x_new
            break

        if np.isnan(x_new).any():
            print("\nNaN detected in parameters. Stopping.")
            break

        x = x_new
        integral = np.clip(integral, -int_clip, int_clip)

    return x, history, lambda_ic_hist, lambda_bc_hist


# ============================================================
# 6) Train (Adam warm-start + AdamFL)
# ============================================================
def train(args):
    global pinn_ns

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    fig_dir = os.path.join(base_dir, "figures")
    train_dir = os.path.join(base_dir, "training_data")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)

    X_coll, X_ic, X_bx_l, X_bx_r, X_by_b, X_by_t = sample_training_data()
    pinn_ns = PhysicsInformedNS(X_coll, X_ic, X_bx_l, X_bx_r, X_by_b, X_by_t)

    # AdamFL stage
    print(f"\n--- AdamFL stage ({args.max_iter} iters) ---")
    x0 = get_flat_params(pinn_ns.model)

    t0 = time.time()
    x_final, history, lambda_ic_hist, lambda_bc_hist = AdamFL_PINN(
        f=f,
        h=h,
        df=df,
        dh=dh,
        x_start=x0,
        Kp=Kp,
        Ki=Ki,
        eta=args.eta,
        max_iter=args.max_iter,
        tol=args.tol,
        beta_merit=args.beta_merit,
        lambda_ema=args.lambda_ema,
        int_clip=args.int_clip,
        jh_reg=args.jh_reg,
        max_backtrack=args.max_backtrack,
    )
    t1 = time.time()

    set_flat_params(pinn_ns.model, x_final)

    final_ic, final_bc = h(x_final)
    final_res = f(x_final)
    print("\n--- Optimization Finished ---")
    print(f"Final Initial Loss:  {final_ic:.6e}")
    print(f"Final Boundary Loss: {final_bc:.6e}")
    print(f"Final Physics Loss:  {final_res:.6e}")
    print(f"Training time: {(t1 - t0):.2f} s")

    # Save model + history
    torch.save(pinn_ns.model.state_dict(), os.path.join(data_dir, "AdamFL_PINN_navier_stokes.pth"))

    if len(history) > 0:
        _, kkt_gap, ic_hist, bc_hist = zip(*history)
        np.savez(
            os.path.join(train_dir, "AdamFL_PINN_history.npz"),
            kkt_gap=np.array(kkt_gap),
            initial_loss=np.array(ic_hist),
            boundary_loss=np.array(bc_hist),
            lambda_ic=np.array(lambda_ic_hist),
            lambda_bc=np.array(lambda_bc_hist),
        )

    return pinn_ns, history


# ============================================================
# 7) Evaluation + plotting
# ============================================================
def evaluate_and_plot(pinn_solver, n_grid=80, t_eval=0.5):
    model = pinn_solver.model
    model.eval()

    x = np.linspace(0.0, 2.0 * np.pi, n_grid)
    y = np.linspace(0.0, 2.0 * np.pi, n_grid)
    X, Y = np.meshgrid(x, y)
    T = np.full_like(X, t_eval)

    X_eval = np.stack([X.ravel(), Y.ravel(), T.ravel()], axis=1)
    X_eval_t = torch.tensor(X_eval, dtype=torch.float32, device=device)

    with torch.no_grad():
        uvp_pred = model(X_eval_t).detach().cpu().numpy()

    U_pred = uvp_pred[:, 0].reshape(n_grid, n_grid)
    V_pred = uvp_pred[:, 1].reshape(n_grid, n_grid)
    P_pred = uvp_pred[:, 2].reshape(n_grid, n_grid)

    Xt = torch.tensor(X, dtype=torch.float32)
    Yt = torch.tensor(Y, dtype=torch.float32)
    Tt = torch.tensor(T, dtype=torch.float32)

    U_true = taylor_green_u(Xt, Yt, Tt).numpy()
    V_true = taylor_green_v(Xt, Yt, Tt).numpy()
    P_true = taylor_green_p(Xt, Yt, Tt).numpy()

    err_u = np.linalg.norm(U_pred - U_true) / (np.linalg.norm(U_true) + 1e-12)
    err_v = np.linalg.norm(V_pred - V_true) / (np.linalg.norm(V_true) + 1e-12)
    err_p = np.linalg.norm(P_pred - P_true) / (np.linalg.norm(P_true) + 1e-12)

    print("\n--- Evaluation at t =", t_eval, "---")
    print(f"Relative L2 error u: {err_u:.6e}")
    print(f"Relative L2 error v: {err_v:.6e}")
    print(f"Relative L2 error p: {err_p:.6e}")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    fig_dir = os.path.join(base_dir, "figures")

    scipy.io.savemat(
        os.path.join(data_dir, "AdamFL_PINN_navier_stokes_eval.mat"),
        {
            "x": x,
            "y": y,
            "t_eval": t_eval,
            "U_pred": U_pred,
            "V_pred": V_pred,
            "P_pred": P_pred,
            "U_true": U_true,
            "V_true": V_true,
            "P_true": P_true,
            "err_u": err_u,
            "err_v": err_v,
            "err_p": err_p,
        },
    )

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle(f"Taylor-Green Vortex (AdamFL-PINN), t={t_eval}")

    axs[0, 0].quiver(X, Y, U_true, V_true, scale=20)
    axs[0, 0].set_title("Exact velocity")

    axs[0, 1].quiver(X, Y, U_pred, V_pred, scale=20)
    axs[0, 1].set_title("Predicted velocity")

    im_u = axs[1, 0].imshow(np.abs(U_pred - U_true), extent=[0, 2 * np.pi, 0, 2 * np.pi], origin="lower")
    axs[1, 0].set_title("|u_pred - u_true|")
    fig.colorbar(im_u, ax=axs[1, 0])

    im_v = axs[1, 1].imshow(np.abs(V_pred - V_true), extent=[0, 2 * np.pi, 0, 2 * np.pi], origin="lower")
    axs[1, 1].set_title("|v_pred - v_true|")
    fig.colorbar(im_v, ax=axs[1, 1])

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "AdamFL_PINN_velocity_comparison.png"), dpi=200, bbox_inches="tight")
    plt.show()


def plot_training_history(history):
    if len(history) == 0:
        print("No AdamFL history to plot.")
        return

    phys, kkt, ic_hist, bc_hist = zip(*history)

    plt.figure(figsize=(16, 4))

    plt.subplot(1, 4, 1)
    plt.plot(kkt)
    plt.yscale("log")
    plt.title("KKT gap")
    plt.xlabel("iters")

    plt.subplot(1, 4, 2)
    plt.plot(ic_hist)
    plt.yscale("log")
    plt.title("Initial loss")
    plt.xlabel("iters")

    plt.subplot(1, 4, 3)
    plt.plot(bc_hist)
    plt.yscale("log")
    plt.title("Boundary loss")
    plt.xlabel("iters")

    plt.subplot(1, 4, 4)
    plt.plot(phys)
    plt.yscale("log")
    plt.title("Physics objective")
    plt.xlabel("iters")

    plt.tight_layout()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    fig_dir = os.path.join(base_dir, "figures")
    plt.savefig(os.path.join(fig_dir, "AdamFL_PINN_training_history.png"), dpi=200, bbox_inches="tight")
    plt.show()


def evaluate_l2_over_time(model, n_grid=80, n_time=101):
    times = np.linspace(0.0, 1.0, n_time)
    err_u = []
    err_v = []
    err_p = []
    for t_eval in times:
        x = np.linspace(0.0, 2.0 * np.pi, n_grid)
        y = np.linspace(0.0, 2.0 * np.pi, n_grid)
        X, Y = np.meshgrid(x, y)
        T = np.full_like(X, t_eval)

        X_eval = np.stack([X.ravel(), Y.ravel(), T.ravel()], axis=1)
        X_eval_t = torch.tensor(X_eval, dtype=torch.float32, device=device)

        with torch.no_grad():
            uvp_pred = model(X_eval_t).detach().cpu().numpy()

        U_pred = uvp_pred[:, 0].reshape(n_grid, n_grid)
        V_pred = uvp_pred[:, 1].reshape(n_grid, n_grid)
        P_pred = uvp_pred[:, 2].reshape(n_grid, n_grid)

        Xt = torch.tensor(X, dtype=torch.float32)
        Yt = torch.tensor(Y, dtype=torch.float32)
        Tt = torch.tensor(T, dtype=torch.float32)

        U_true = taylor_green_u(Xt, Yt, Tt).numpy()
        V_true = taylor_green_v(Xt, Yt, Tt).numpy()
        P_true = taylor_green_p(Xt, Yt, Tt).numpy()

        err_u.append(np.linalg.norm(U_pred - U_true) / (np.linalg.norm(U_true) + 1e-12))
        err_v.append(np.linalg.norm(V_pred - V_true) / (np.linalg.norm(V_true) + 1e-12))
        err_p.append(np.linalg.norm(P_pred - P_true) / (np.linalg.norm(P_true) + 1e-12))

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    scipy.io.savemat(
        os.path.join(data_dir, "AdamFL_PINN_time_l2.mat"),
        {
            "time_slices": times,
            "err_u": np.array(err_u),
            "err_v": np.array(err_v),
            "err_p": np.array(err_p),
        },
    )
    print(
        f"Mean relative L2 over time slices: u={np.mean(err_u):.6e}, "
        f"v={np.mean(err_v):.6e}, p={np.mean(err_p):.6e}"
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_iter", type=int, default=MAX_ITER)
    parser.add_argument("--eta", type=float, default=ETA)
    parser.add_argument("--tol", type=float, default=TOL)
    parser.add_argument("--beta_merit", type=float, default=BETA_MERIT)
    parser.add_argument("--lambda_ema", type=float, default=LAMBDA_EMA)
    parser.add_argument("--int_clip", type=float, default=INT_CLIP)
    parser.add_argument("--jh_reg", type=float, default=JH_REG)
    parser.add_argument("--max_backtrack", type=int, default=MAX_BACKTRACK)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    solver, train_history = train(args)
    plot_training_history(train_history)
    evaluate_and_plot(solver, n_grid=80, t_eval=0.5)
    evaluate_l2_over_time(solver.model, n_grid=80, n_time=101)
