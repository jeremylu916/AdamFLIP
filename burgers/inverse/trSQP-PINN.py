import os
import time
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import scipy.io
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import trange

# --- Reproducibility ---
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Optional deterministic behavior
if torch.backends.cudnn.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Problem setup ---
N_f = 10000
N_u = 2500
N_ic = 256
N_bc = 200

TRUE_LAMBDA1 = 1.0
TRUE_LAMBDA2 = 0.01 / np.pi


def safe_softplus_inv(x):
    return np.log(np.exp(x) - 1.0 + 1e-12)


class PhysicsInformedNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 1),
        ).to(device)

        self.lambda1_raw = nn.Parameter(torch.tensor([1.0], dtype=torch.float32, device=device))
        self.lambda2_raw = nn.Parameter(torch.tensor([safe_softplus_inv(TRUE_LAMBDA2)], dtype=torch.float32, device=device))
        self.loss = nn.MSELoss()

    @property
    def lambda1(self):
        return self.lambda1_raw

    @property
    def lambda2(self):
        # Keep viscosity positive and stable.
        return F.softplus(self.lambda2_raw) + 1e-8

    def net_u(self, x, t):
        return self.net(torch.hstack((x, t)))

    def net_f(self, x, t):
        xg = x.clone().detach().requires_grad_(True)
        tg = t.clone().detach().requires_grad_(True)

        u = self.net_u(xg, tg)
        u_t = torch.autograd.grad(u, tg, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
        u_x = torch.autograd.grad(u, xg, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, xg, torch.ones_like(u_x), create_graph=True, retain_graph=True)[0]

        f = u_t + self.lambda1 * u * u_x - self.lambda2 * u_xx
        return f


def get_flat_params(model):
    return np.concatenate([p.detach().cpu().numpy().flatten() for p in model.parameters()])


def set_flat_params(model, flat_params):
    flat_params = flat_params.astype(np.float32)
    pointer = 0
    for p in model.parameters():
        n = p.numel()
        p.data = torch.from_numpy(flat_params[pointer:pointer + n]).view_as(p).to(device)
        pointer += n


def get_grads_from_loss(loss, params):
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    flat = []
    for g, p in zip(grads, params):
        if g is None:
            flat.append(np.zeros(p.numel(), dtype=np.float32))
        else:
            flat.append(g.detach().cpu().numpy().ravel())
    return np.concatenate(flat)


def load_burgers_data():
    data = scipy.io.loadmat("burgers_shock.mat")
    t_vec = data["t"].flatten()[:, None]
    x_vec = data["x"].flatten()[:, None]
    u_exact = np.real(data["usol"])  # shape: [Nx, Nt]

    X_grid, T_grid = np.meshgrid(x_vec.flatten(), t_vec.flatten(), indexing="ij")
    X_star = np.hstack((X_grid.reshape(-1, 1), T_grid.reshape(-1, 1)))
    u_star = u_exact.reshape(-1, 1)

    return x_vec, t_vec, X_star, u_star, u_exact


def build_training_sets(x_vec, t_vec, X_star, u_star):
    n_total = X_star.shape[0]
    idx_obs = np.random.choice(n_total, size=min(N_u, n_total), replace=False)

    X_u = torch.tensor(X_star[idx_obs], dtype=torch.float32, device=device)
    u_u = torch.tensor(u_star[idx_obs], dtype=torch.float32, device=device)

    x_min, x_max = float(np.min(x_vec)), float(np.max(x_vec))
    t_min, t_max = float(np.min(t_vec)), float(np.max(t_vec))

    X_f = np.hstack([
        np.random.uniform(x_min, x_max, (N_f, 1)),
        np.random.uniform(t_min, t_max, (N_f, 1)),
    ]).astype(np.float32)
    X_f = torch.tensor(X_f, dtype=torch.float32, device=device)

    x_ic = torch.linspace(x_min, x_max, N_ic, device=device).view(-1, 1)
    t_ic = torch.zeros_like(x_ic)
    u_ic = -torch.sin(math.pi * x_ic)

    t_bc = torch.linspace(t_min, t_max, N_bc, device=device).view(-1, 1)
    x_l = torch.full_like(t_bc, x_min)
    x_r = torch.full_like(t_bc, x_max)
    u_bc = torch.zeros_like(t_bc)

    return {
        "X_u": X_u,
        "u_u": u_u,
        "X_f": X_f,
        "x_ic": x_ic,
        "t_ic": t_ic,
        "u_ic": u_ic,
        "x_l": x_l,
        "x_r": x_r,
        "t_bc": t_bc,
        "u_bc": u_bc,
        "x_min": x_min,
        "x_max": x_max,
        "t_min": t_min,
        "t_max": t_max,
    }


def objective_obs_loss(model, ds):
    pred = model.net_u(ds["X_u"][:, 0:1], ds["X_u"][:, 1:2])
    return model.loss(pred, ds["u_u"])


def constraint_physics_loss(model, ds):
    f_pred = model.net_f(ds["X_f"][:, 0:1], ds["X_f"][:, 1:2])
    return model.loss(f_pred, torch.zeros_like(f_pred))


def constraint_icbc_loss(model, ds):
    u_ic_pred = model.net_u(ds["x_ic"], ds["t_ic"])
    loss_ic = model.loss(u_ic_pred, ds["u_ic"])

    u_l = model.net_u(ds["x_l"], ds["t_bc"])
    u_r = model.net_u(ds["x_r"], ds["t_bc"])
    loss_bc = model.loss(u_l, ds["u_bc"]) + model.loss(u_r, ds["u_bc"])

    return loss_ic + loss_bc


def compute_lambda_kkt(grad_f, J, reg=1e-8):
    """Solve lambda = argmin ||grad_f + J^T lambda||^2."""
    JJt = J @ J.T
    rhs = -(J @ grad_f)
    try:
        return np.linalg.solve(JJt + reg * np.eye(JJt.shape[0], dtype=np.float32), rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(JJt + reg * np.eye(JJt.shape[0], dtype=np.float32), rhs, rcond=None)[0]


def solve_tr_subproblem(h, grad_L, J, H, Delta, nu, reg=1e-6):
    """Approximate Step 2 with normal (feasibility) + tangential (optimality) steps."""
    n = grad_L.shape[0]
    m = J.shape[0]
    I_m = np.eye(m, dtype=np.float32)

    JJt = J @ J.T
    # Normal step: reduce linearized constraint violation within nu*Delta.
    try:
        d_n = -J.T @ np.linalg.solve(JJt + reg * I_m, h)
    except np.linalg.LinAlgError:
        d_n = -J.T @ np.linalg.lstsq(JJt + reg * I_m, h, rcond=None)[0]

    norm_dn = np.linalg.norm(d_n)
    max_dn = nu * Delta
    if norm_dn > max_dn:
        d_n = d_n * (max_dn / (norm_dn + 1e-16))

    # Tangential step: projected Newton-like step in approximate null(J).
    try:
        d_obj = -np.linalg.solve(H + reg * np.eye(n, dtype=np.float32), grad_L)
    except np.linalg.LinAlgError:
        d_obj = -np.linalg.lstsq(H + reg * np.eye(n, dtype=np.float32), grad_L, rcond=None)[0]

    # Projection without forming NxN projector.
    Jd = J @ d_obj
    try:
        corr = J.T @ np.linalg.solve(JJt + reg * I_m, Jd)
    except np.linalg.LinAlgError:
        corr = J.T @ np.linalg.lstsq(JJt + reg * I_m, Jd, rcond=None)[0]
    d_t = d_obj - corr

    remain = np.sqrt(max(Delta * Delta - np.dot(d_n, d_n), 0.0))
    norm_dt = np.linalg.norm(d_t)
    if norm_dt > remain:
        d_t = d_t * (remain / (norm_dt + 1e-16))

    return d_n + d_t


def pretrain_feasibility(model, ds, max_iter=20000, g_tol=1e-9, f_tol=1e-9):
    """Step 0: theta_init <- argmin ||c(theta)||^2 using L-BFGS with A.1-like stop."""
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=50,
        tolerance_grad=1e-12,
        tolerance_change=1e-12,
        history_size=100,
        line_search_fn="strong_wolfe",
    )

    theta_prev = get_flat_params(model)
    outer_steps = max(1, max_iter // 50)

    for l in range(outer_steps):
        def closure():
            optimizer.zero_grad()
            c_loss = constraint_physics_loss(model, ds) + constraint_icbc_loss(model, ds)
            c_loss.backward()
            return c_loss

        optimizer.step(closure)

        c_loss = constraint_physics_loss(model, ds) + constraint_icbc_loss(model, ds)
        grads = torch.autograd.grad(c_loss, list(model.parameters()), retain_graph=False, allow_unused=True)
        grad_inf = 0.0
        for g in grads:
            if g is not None:
                grad_inf = max(grad_inf, float(g.detach().abs().max().item()))

        theta_new = get_flat_params(model)
        step_norm = float(np.linalg.norm(theta_new - theta_prev))
        theta_prev = theta_new

        if l % 10 == 0:
            print(f"[Pretrain] iter={l:4d} | c={float(c_loss.item()):.3e} | grad_inf={grad_inf:.3e} | step={step_norm:.3e}")

        if grad_inf <= g_tol or step_norm <= f_tol:
            print(f"[Pretrain] stop at iter={l}, c={float(c_loss.item()):.3e}, grad_inf={grad_inf:.3e}, step={step_norm:.3e}")
            break


def trSQP_train(model, ds, max_iter=1000, delta=0.2, nu=0.8, eta_low=1e-8, eta_upp=0.3, rho_scale=2.0, g_tol=1e-9, f_tol=1e-9):
    params = list(model.parameters())
    theta = get_flat_params(model)

    n = theta.shape[0]
    H = np.eye(n, dtype=np.float32)

    mu_prev = 1.0
    Delta = 1.0
    min_stop_iter = min(1000, max_iter)

    # Step 2 init in Algorithm 3: lambda0 = argmin ||grad l0 + J0^T lambda||^2
    set_flat_params(model, theta)
    loss_f0 = objective_obs_loss(model, ds)
    loss_h10 = constraint_physics_loss(model, ds)
    loss_h20 = constraint_icbc_loss(model, ds)
    grad_f0 = get_grads_from_loss(loss_f0, params)
    J10 = get_grads_from_loss(loss_h10, params)
    J20 = get_grads_from_loss(loss_h20, params)
    J0 = np.vstack([J10, J20])
    lambda_k = compute_lambda_kkt(grad_f0, J0)

    pbar = trange(max_iter, desc="trSQP Training", unit="iter")
    for k in pbar:
        set_flat_params(model, theta)

        # f = observation loss, h = [physics, IC+BC]
        loss_f = objective_obs_loss(model, ds)
        loss_h1 = constraint_physics_loss(model, ds)
        loss_h2 = constraint_icbc_loss(model, ds)

        f_val = loss_f.item()
        h = np.array([loss_h1.item(), loss_h2.item()], dtype=np.float32)

        grad_f = get_grads_from_loss(loss_f, params)
        J1 = get_grads_from_loss(loss_h1, params)
        J2 = get_grads_from_loss(loss_h2, params)
        J = np.vstack([J1, J2])

        grad_L = grad_f + J.T @ lambda_k

        # Step 2: approximate trust-region subproblem using nu split.
        d_k = solve_tr_subproblem(h, grad_L, J, H, Delta, nu)

        denom = 0.7 * (np.linalg.norm(h) - np.linalg.norm(h + J @ d_k))
        if denom == 0:
            mu_k = mu_prev
        else:
            mu_k = max(mu_prev, (np.dot(grad_L, d_k) + 0.5 * d_k @ (H @ d_k)) / denom)

        # Merit model and merit reduction ratio in Step 3.
        q0 = f_val + mu_k * np.linalg.norm(h)
        qd = f_val + np.dot(grad_f, d_k) + 0.5 * d_k @ (H @ d_k) + mu_k * np.linalg.norm(h + J @ d_k)
        Pred_k = q0 - qd

        theta_trial = theta + d_k
        set_flat_params(model, theta_trial)
        f_trial = objective_obs_loss(model, ds).item()
        h_trial = np.array([
            constraint_physics_loss(model, ds).item(),
            constraint_icbc_loss(model, ds).item(),
        ], dtype=np.float32)

        phi_k = f_val + mu_k * np.linalg.norm(h)
        phi_trial = f_trial + mu_k * np.linalg.norm(h_trial)
        Ared_k = phi_k - phi_trial

        eta_k = -np.inf if Pred_k <= 0 else Ared_k / Pred_k

        accepted = eta_k >= eta_low
        if accepted:
            theta_new = theta + d_k
            if eta_k >= eta_upp:
                Delta = rho_scale * Delta

            set_flat_params(model, theta_new)
            loss_f_new = objective_obs_loss(model, ds)
            loss_h1_new = constraint_physics_loss(model, ds)
            loss_h2_new = constraint_icbc_loss(model, ds)

            grad_f_new = get_grads_from_loss(loss_f_new, params)
            J1_new = get_grads_from_loss(loss_h1_new, params)
            J2_new = get_grads_from_loss(loss_h2_new, params)
            J_new = np.vstack([J1_new, J2_new])

            lambda_new = compute_lambda_kkt(grad_f_new, J_new)

            grad_L_new = grad_f_new + J_new.T @ lambda_new
            s = theta_new - theta
            y = grad_L_new - grad_L
            sHs = s @ (H @ s)
            if s @ y >= 0.2 * sHs:
                y_bar = y
            else:
                theta_scale = (0.8 * sHs) / (sHs - s @ y + 1e-16)
                y_bar = theta_scale * y + (1 - theta_scale) * (H @ s)
            rho_b = 1.0 / (s @ y_bar + 1e-16)
            I = np.eye(n, dtype=np.float32)
            H = (I - rho_b * np.outer(s, y_bar)) @ H @ (I - rho_b * np.outer(y_bar, s)) + rho_b * np.outer(y_bar, y_bar)

            theta = theta_new
            mu_prev = mu_k
            lambda_k = lambda_new
        else:
            Delta = Delta / rho_scale

        if (k + 1) >= min_stop_iter and eta_k < eta_low and (
            np.linalg.norm(d_k) <= f_tol or np.max(np.abs(grad_L)) <= g_tol
        ):
            print(f"Converged at iter {k}")
            break

        if k % 10 == 0:
            pbar.set_postfix({
                "f_obs": f"{f_val:.2e}",
                "h_phy": f"{h[0]:.2e}",
                "h_icbc": f"{h[1]:.2e}",
                "lam1": f"{model.lambda1.item():.3f}",
                "lam2": f"{model.lambda2.item():.4e}",
                "Delta": f"{Delta:.2e}",
                "eta": f"{eta_k:.2e}",
            })

    set_flat_params(model, theta)
    return model


def save_solution_and_error(model, x_vec, t_vec, X_star, u_star_grid):
    with torch.no_grad():
        x_star_t = torch.tensor(X_star[:, 0:1], dtype=torch.float32, device=device)
        t_star_t = torch.tensor(X_star[:, 1:2], dtype=torch.float32, device=device)
        u_pred = model.net_u(x_star_t, t_star_t).detach().cpu().numpy().reshape(u_star_grid.shape)

    err_grid = np.abs(u_pred - u_star_grid)

    os.makedirs("./data", exist_ok=True)
    scipy.io.savemat("./data/trSQP_PINN_solution.mat", {
        "solution": u_pred,
        "x": x_vec.flatten(),
        "t": t_vec.flatten(),
        "u_true": u_star_grid,
    })
    scipy.io.savemat("./data/trSQP_PINN_error.mat", {
        "error": err_grid,
        "x": x_vec.flatten(),
        "t": t_vec.flatten(),
        "u_true": u_star_grid,
    })

    os.makedirs("./figures/solution", exist_ok=True)

    plt.rcParams["font.size"] = "15"
    fig = plt.figure(figsize=(5, 6))
    ax = fig.add_subplot(111)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$x$")
    ax.set_title("trSQP-PINN Solution")

    img = ax.imshow(
        u_pred,
        interpolation="nearest",
        cmap="rainbow",
        extent=[t_vec.min(), t_vec.max(), x_vec.min(), x_vec.max()],
        origin="lower",
        aspect="auto",
    )
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.10)
    cbar = fig.colorbar(img, cax=cax)
    cbar.ax.tick_params(labelsize=10)
    plt.savefig("./figures/solution/trSQP_PINN_solution.png", dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

    fig_err = plt.figure(figsize=(5, 6))
    ax_err = fig_err.add_subplot(111)
    ax_err.set_xlabel(r"$t$")
    ax_err.set_ylabel(r"$x$")
    ax_err.set_title("trSQP-PINN Error")

    img_err = ax_err.imshow(
        err_grid,
        interpolation="nearest",
        cmap="bwr",
        extent=[t_vec.min(), t_vec.max(), x_vec.min(), x_vec.max()],
        origin="lower",
        aspect="auto",
    )
    divider_err = make_axes_locatable(ax_err)
    cax_err = divider_err.append_axes("right", size="5%", pad=0.10)
    cbar_err = fig_err.colorbar(img_err, cax=cax_err)
    cbar_err.ax.tick_params(labelsize=10)
    plt.savefig("./figures/solution/trSQP_PINN_error.png", dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig_err)


if __name__ == "__main__":
    os.makedirs("./figures", exist_ok=True)
    os.makedirs("./data", exist_ok=True)

    x_vec, t_vec, X_star, u_star, u_star_grid = load_burgers_data()
    ds = build_training_sets(x_vec, t_vec, X_star, u_star)

    model = PhysicsInformedNN().to(device)

    pretrain_feasibility(model, ds, max_iter=1000, g_tol=1e-9, f_tol=1e-9)

    start = time.time()
    trained = trSQP_train(
        model,
        ds,
        max_iter=1000,
        delta=0.2,
        nu=0.8,
        eta_low=1e-8,
        eta_upp=0.3,
        rho_scale=2.0,
        g_tol=1e-9,
        f_tol=1e-9,
    )
    elapsed = time.time() - start
    print(f"Training finished in {elapsed:.1f}s")

    with torch.no_grad():
        x_star_t = torch.tensor(X_star[:, 0:1], dtype=torch.float32, device=device)
        t_star_t = torch.tensor(X_star[:, 1:2], dtype=torch.float32, device=device)
        u_pred_all = trained.net_u(x_star_t, t_star_t)
        u_true_all = torch.tensor(u_star, dtype=torch.float32, device=device)

    loss_obs = objective_obs_loss(trained, ds).item()
    loss_phy = constraint_physics_loss(trained, ds).item()
    loss_icbc = constraint_icbc_loss(trained, ds).item()
    mse_full = nn.MSELoss()(u_pred_all, u_true_all).item()
    l2_rel = (torch.linalg.norm(u_pred_all - u_true_all) / (torch.linalg.norm(u_true_all) + 1e-16)).item()

    print("Final metrics:")
    print(f"Observation loss: {loss_obs:.6e}")
    print(f"Physics loss: {loss_phy:.6e}")
    print(f"IC+BC loss: {loss_icbc:.6e}")
    print(f"MSE (full grid): {mse_full:.6e}")
    print(f"L2 relative (full grid): {l2_rel:.6e}")
    print(f"Estimated lambda1: {trained.lambda1.item():.6f} (true {TRUE_LAMBDA1:.6f})")
    print(f"Estimated lambda2: {trained.lambda2.item():.6e} (true {TRUE_LAMBDA2:.6e})")

    save_solution_and_error(trained, x_vec, t_vec, X_star, u_star_grid)
