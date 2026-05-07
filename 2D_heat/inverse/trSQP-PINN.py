#!/usr/bin/env python3
"""trSQP-PINN for 2D heat inverse problem.

Uses the same trust-region SQP core as 2D_heat/forward/trSQP-PINN.py while
keeping the inverse heat problem setup (observation objective + physics/ICBC
constraints).
"""

import argparse
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.io import savemat
from tqdm import trange


parser = argparse.ArgumentParser()
parser.add_argument("--EPOCH", type=int, default=100)
parser.add_argument("--ordinal", type=int, default=0)
parser.add_argument("--N_u", type=int, default=4000)
parser.add_argument("--N_f", type=int, default=8000)
parser.add_argument("--N_ic", type=int, default=1000)
parser.add_argument("--N_bc", type=int, default=1000)
parser.add_argument("--alpha", type=float, default=0.1)
parser.add_argument("--noise", type=float, default=0.0)
parser.add_argument("--pretrain_max_iter", type=int, default=100)
parser.add_argument("--delta", type=float, default=0.2)
parser.add_argument("--nu", type=float, default=0.8)
parser.add_argument("--eta_low", type=float, default=1e-8)
parser.add_argument("--eta_upp", type=float, default=0.3)
parser.add_argument("--rho_scale", type=float, default=2.0)
parser.add_argument("--g_tol", type=float, default=1e-9)
parser.add_argument("--f_tol", type=float, default=1e-9)
args = parser.parse_args()


seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
if torch.backends.cudnn.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for this script. No GPU detected.")

device = torch.device(f"cuda:{args.ordinal}")
print(f"Using device: {device} ({torch.cuda.get_device_name(device)})")


class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)


def exact_solution(x, y, t, alpha):
    return torch.exp(-2 * torch.pi ** 2 * alpha * t) * torch.sin(torch.pi * x) * torch.sin(torch.pi * y)


def initial_condition(x, y):
    return torch.sin(torch.pi * x) * torch.sin(torch.pi * y)


def pde_residual(model, x, y, t, alpha):
    inp = torch.cat([x, y, t], dim=1)
    u = model(inp)
    grads = torch.autograd.grad(u, [x, y, t], grad_outputs=torch.ones_like(u), create_graph=True)
    u_x, u_y, u_t = grads
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
    return u_t - alpha * (u_xx + u_yy)


def build_training_sets(n_u, n_f, n_ic, n_bc, alpha, noise=0.0):
    x_u = torch.rand(n_u, 1, device=device)
    y_u = torch.rand(n_u, 1, device=device)
    t_u = torch.rand(n_u, 1, device=device)
    u_u = exact_solution(x_u, y_u, t_u, alpha)
    if noise > 0:
        u_u = u_u + noise * torch.std(u_u) * torch.randn_like(u_u)

    x_f = torch.rand(n_f, 1, device=device)
    y_f = torch.rand(n_f, 1, device=device)
    t_f = torch.rand(n_f, 1, device=device)

    x_ic = torch.rand(n_ic, 1, device=device)
    y_ic = torch.rand(n_ic, 1, device=device)
    t_ic = torch.zeros(n_ic, 1, device=device)
    u_ic = initial_condition(x_ic, y_ic)

    n_side = max(1, n_bc // 2)
    y_bx = torch.rand(n_side, 1, device=device)
    t_bx = torch.rand(n_side, 1, device=device)
    x_b0 = torch.zeros_like(y_bx)
    x_b1 = torch.ones_like(y_bx)

    x_by = torch.rand(n_side, 1, device=device)
    t_by = torch.rand(n_side, 1, device=device)
    y_b0 = torch.zeros_like(x_by)
    y_b1 = torch.ones_like(x_by)

    x_bc = torch.cat([x_b0, x_b1, x_by, x_by], dim=0)
    y_bc = torch.cat([y_bx, y_bx, y_b0, y_b1], dim=0)
    t_bc = torch.cat([t_bx, t_bx, t_by, t_by], dim=0)
    u_bc = torch.zeros_like(x_bc)

    return {
        "X_u": torch.cat([x_u, y_u, t_u], dim=1),
        "u_u": u_u,
        "x_f": x_f,
        "y_f": y_f,
        "t_f": t_f,
        "x_ic": x_ic,
        "y_ic": y_ic,
        "t_ic": t_ic,
        "u_ic": u_ic,
        "x_bc": x_bc,
        "y_bc": y_bc,
        "t_bc": t_bc,
        "u_bc": u_bc,
    }


def objective_obs_loss(model, ds):
    pred = model(ds["X_u"])
    return nn.MSELoss()(pred, ds["u_u"])


def constraint_physics_loss(model, ds, alpha):
    x = ds["x_f"].clone().detach().requires_grad_(True)
    y = ds["y_f"].clone().detach().requires_grad_(True)
    t = ds["t_f"].clone().detach().requires_grad_(True)
    r = pde_residual(model, x, y, t, alpha)
    return nn.MSELoss()(r, torch.zeros_like(r))


def constraint_icbc_loss(model, ds):
    u_ic_pred = model(torch.cat([ds["x_ic"], ds["y_ic"], ds["t_ic"]], dim=1))
    loss_ic = nn.MSELoss()(u_ic_pred, ds["u_ic"])

    u_bc_pred = model(torch.cat([ds["x_bc"], ds["y_bc"], ds["t_bc"]], dim=1))
    loss_bc = nn.MSELoss()(u_bc_pred, ds["u_bc"])
    return loss_ic + loss_bc


def get_flat_params(model):
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()]).to(device)


def set_flat_params(model, flat_params):
    flat_params = flat_params.to(device=device, dtype=torch.float32)
    pointer = 0
    with torch.no_grad():
        for p in model.parameters():
            n = p.numel()
            p.copy_(flat_params[pointer:pointer + n].view_as(p))
            pointer += n


def get_grads_from_loss(loss, params):
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    flat = []
    for g, p in zip(grads, params):
        if g is None:
            flat.append(torch.zeros(p.numel(), device=device, dtype=torch.float32))
        else:
            flat.append(g.detach().reshape(-1).to(device=device, dtype=torch.float32))
    return torch.cat(flat)


def compute_lambda_kkt(grad_f, J, reg=1e-8):
    jj_t = J @ J.T
    rhs = -(J @ grad_f)
    eye = torch.eye(jj_t.shape[0], device=device, dtype=torch.float32)
    try:
        return torch.linalg.solve(jj_t + reg * eye, rhs)
    except RuntimeError:
        return torch.linalg.lstsq(jj_t + reg * eye, rhs.unsqueeze(1)).solution.squeeze(1)


def solve_tr_subproblem(h, grad_l, J, H, delta, nu, reg=1e-6):
    n = grad_l.shape[0]
    m = J.shape[0]

    eye_m = torch.eye(m, device=device, dtype=torch.float32)
    jj_t = J @ J.T

    try:
        d_n = -J.T @ torch.linalg.solve(jj_t + reg * eye_m, h)
    except RuntimeError:
        d_n = -J.T @ torch.linalg.lstsq(jj_t + reg * eye_m, h.unsqueeze(1)).solution.squeeze(1)

    norm_dn = torch.linalg.norm(d_n)
    max_dn = nu * delta
    if float(norm_dn.item()) > max_dn:
        d_n = d_n * (max_dn / (norm_dn + 1e-16))

    try:
        d_obj = -torch.linalg.solve(H + reg * torch.eye(n, device=device, dtype=torch.float32), grad_l)
    except RuntimeError:
        d_obj = -torch.linalg.lstsq(
            H + reg * torch.eye(n, device=device, dtype=torch.float32),
            grad_l.unsqueeze(1),
        ).solution.squeeze(1)

    j_d = J @ d_obj
    try:
        corr = J.T @ torch.linalg.solve(jj_t + reg * eye_m, j_d)
    except RuntimeError:
        corr = J.T @ torch.linalg.lstsq(jj_t + reg * eye_m, j_d.unsqueeze(1)).solution.squeeze(1)
    d_t = d_obj - corr

    remain = torch.sqrt(torch.clamp(torch.tensor(delta * delta, device=device) - torch.dot(d_n, d_n), min=0.0))
    norm_dt = torch.linalg.norm(d_t)
    if float(norm_dt.item()) > float(remain.item()):
        d_t = d_t * (remain / (norm_dt + 1e-16))

    return d_n + d_t


def pretrain_feasibility(model, ds, alpha, max_iter=2000, g_tol=1e-9, f_tol=1e-9):
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
            c_loss = constraint_physics_loss(model, ds, alpha) + constraint_icbc_loss(model, ds)
            c_loss.backward()
            return c_loss

        optimizer.step(closure)

        c_loss = constraint_physics_loss(model, ds, alpha) + constraint_icbc_loss(model, ds)
        grads = torch.autograd.grad(c_loss, list(model.parameters()), retain_graph=False, allow_unused=True)
        grad_inf = 0.0
        for g in grads:
            if g is not None:
                grad_inf = max(grad_inf, float(g.detach().abs().max().item()))

        theta_new = get_flat_params(model)
        step_norm = float(torch.linalg.norm(theta_new - theta_prev).item())
        theta_prev = theta_new

        if l % 10 == 0:
            print(
                f"[Pretrain] iter={l:4d} | c={float(c_loss.item()):.3e} "
                f"| grad_inf={grad_inf:.3e} | step={step_norm:.3e}"
            )

        if grad_inf <= g_tol or step_norm <= f_tol:
            print(
                f"[Pretrain] stop at iter={l}, c={float(c_loss.item()):.3e}, "
                f"grad_inf={grad_inf:.3e}, step={step_norm:.3e}"
            )
            break


def trSQP_train(
    model,
    ds,
    alpha,
    max_iter=20000,
    delta=0.2,
    nu=0.8,
    eta_low=1e-8,
    eta_upp=0.3,
    rho_scale=2.0,
    g_tol=1e-9,
    f_tol=1e-9,
):
    params = list(model.parameters())
    theta = get_flat_params(model)
    n = theta.shape[0]
    H = torch.eye(n, device=device, dtype=torch.float32)

    mu_prev = 1.0
    Delta = float(delta)
    min_stop_iter = min(1000, max_iter)

    set_flat_params(model, theta)
    loss_f0 = objective_obs_loss(model, ds)
    loss_h10 = constraint_physics_loss(model, ds, alpha)
    loss_h20 = constraint_icbc_loss(model, ds)
    grad_f0 = get_grads_from_loss(loss_f0, params)
    J10 = get_grads_from_loss(loss_h10, params)
    J20 = get_grads_from_loss(loss_h20, params)
    J0 = torch.vstack([J10, J20])
    lambda_k = compute_lambda_kkt(grad_f0, J0)

    os.makedirs("./training_logs", exist_ok=True)
    logp = "./training_logs/trSQP-PINN_loss.txt"

    with open(logp, "w", encoding="utf-8") as flog:
        flog.write("iter,loss_obs,loss_phy,loss_icbc,Delta,eta,lambda0,lambda1\n")
        pbar = trange(max_iter, desc="trSQP-PINN", unit="iter")

        for k in pbar:
            set_flat_params(model, theta)

            loss_f = objective_obs_loss(model, ds)
            loss_h1 = constraint_physics_loss(model, ds, alpha)
            loss_h2 = constraint_icbc_loss(model, ds)

            f_val = float(loss_f.item())
            h = torch.tensor([float(loss_h1.item()), float(loss_h2.item())], device=device, dtype=torch.float32)

            grad_f = get_grads_from_loss(loss_f, params)
            J1 = get_grads_from_loss(loss_h1, params)
            J2 = get_grads_from_loss(loss_h2, params)
            J = torch.vstack([J1, J2])

            grad_l = grad_f + J.T @ lambda_k
            d_k = solve_tr_subproblem(h, grad_l, J, H, Delta, nu)

            denom = 0.7 * (torch.linalg.norm(h) - torch.linalg.norm(h + J @ d_k))
            if abs(float(denom.item())) < 1e-16:
                mu_k = mu_prev
            else:
                quad = torch.dot(grad_l, d_k) + 0.5 * torch.dot(d_k, H @ d_k)
                mu_k = max(mu_prev, float((quad / denom).item()))

            q0 = f_val + mu_k * float(torch.linalg.norm(h).item())
            qd = (
                f_val
                + float(torch.dot(grad_f, d_k).item())
                + 0.5 * float(torch.dot(d_k, H @ d_k).item())
                + mu_k * float(torch.linalg.norm(h + J @ d_k).item())
            )
            pred_k = q0 - qd

            theta_trial = theta + d_k
            set_flat_params(model, theta_trial)
            f_trial = float(objective_obs_loss(model, ds).item())
            h_trial = torch.tensor([
                float(constraint_physics_loss(model, ds, alpha).item()),
                float(constraint_icbc_loss(model, ds).item()),
            ], device=device, dtype=torch.float32)

            phi_k = f_val + mu_k * float(torch.linalg.norm(h).item())
            phi_trial = f_trial + mu_k * float(torch.linalg.norm(h_trial).item())
            ared_k = phi_k - phi_trial

            eta_k = -np.inf if pred_k <= 0 else ared_k / pred_k
            accepted = eta_k >= eta_low

            if accepted:
                theta_new = theta + d_k
                if eta_k >= eta_upp:
                    Delta = rho_scale * Delta

                set_flat_params(model, theta_new)
                loss_f_new = objective_obs_loss(model, ds)
                loss_h1_new = constraint_physics_loss(model, ds, alpha)
                loss_h2_new = constraint_icbc_loss(model, ds)

                grad_f_new = get_grads_from_loss(loss_f_new, params)
                J1_new = get_grads_from_loss(loss_h1_new, params)
                J2_new = get_grads_from_loss(loss_h2_new, params)
                J_new = torch.vstack([J1_new, J2_new])

                lambda_new = compute_lambda_kkt(grad_f_new, J_new)
                grad_l_new = grad_f_new + J_new.T @ lambda_new

                s = theta_new - theta
                y = grad_l_new - grad_l
                sHs = torch.dot(s, H @ s)

                if torch.dot(s, y) >= 0.2 * sHs:
                    y_bar = y
                else:
                    theta_scale = (0.8 * sHs) / (sHs - torch.dot(s, y) + 1e-16)
                    y_bar = theta_scale * y + (1.0 - theta_scale) * (H @ s)

                rho_b = 1.0 / (torch.dot(s, y_bar) + 1e-16)
                I = torch.eye(n, device=device, dtype=torch.float32)
                H = (
                    (I - rho_b * torch.outer(s, y_bar))
                    @ H
                    @ (I - rho_b * torch.outer(y_bar, s))
                    + rho_b * torch.outer(y_bar, y_bar)
                )

                theta = theta_new
                mu_prev = mu_k
                lambda_k = lambda_new
            else:
                Delta = Delta / rho_scale

            if (k + 1) >= min_stop_iter and eta_k < eta_low and (
                float(torch.linalg.norm(d_k).item()) <= f_tol
                or float(torch.max(torch.abs(grad_l)).item()) <= g_tol
            ):
                print(f"Converged at iter {k}")
                break

            if k % 10 == 0:
                pbar.set_postfix({
                    "f_obs": f"{f_val:.2e}",
                    "h_phy": f"{float(h[0].item()):.2e}",
                    "h_icbc": f"{float(h[1].item()):.2e}",
                    "Delta": f"{Delta:.2e}",
                    "eta": f"{eta_k:.2e}",
                })
                flog.write(
                    f"{k},{f_val:.6e},{float(h[0].item()):.6e},{float(h[1].item()):.6e},"
                    f"{Delta:.6e},{eta_k:.6e},{float(lambda_k[0].item()):.6e},{float(lambda_k[1].item()):.6e}\n"
                )

    set_flat_params(model, theta)
    return model


def evaluate_and_save(model, alpha):
    os.makedirs("./training_data", exist_ok=True)

    x = torch.linspace(0, 1, 100, device=device)
    y = torch.linspace(0, 1, 100, device=device)
    t = torch.linspace(0, 1, 100, device=device)
    X, Y, T = torch.meshgrid(x, y, t, indexing="ij")
    inp = torch.stack([X.flatten(), Y.flatten(), T.flatten()], dim=1)

    with torch.no_grad():
        U_pred = model(inp).reshape(X.shape)
    U_exact = exact_solution(X, Y, T, alpha)

    x_full = X.reshape(-1, 1).clone().detach().requires_grad_(True)
    y_full = Y.reshape(-1, 1).clone().detach().requires_grad_(True)
    t_full = T.reshape(-1, 1).clone().detach().requires_grad_(True)
    r_full = pde_residual(model, x_full, y_full, t_full, alpha)
    physics_loss = torch.mean(r_full ** 2).item()

    init_mask = torch.isclose(T, torch.tensor(0.0, device=device))
    initial_loss = torch.mean((U_pred[init_mask] - U_exact[init_mask]) ** 2).item()

    bc_mask = (
        torch.isclose(X, torch.tensor(0.0, device=device))
        | torch.isclose(X, torch.tensor(1.0, device=device))
        | torch.isclose(Y, torch.tensor(0.0, device=device))
        | torch.isclose(Y, torch.tensor(1.0, device=device))
    )
    boundary_loss = torch.mean((U_pred[bc_mask] - U_exact[bc_mask]) ** 2).item()

    diff = (U_pred - U_exact).reshape(-1)
    mse = torch.mean(diff ** 2).item()
    rel_l2 = (torch.linalg.norm(diff) / (torch.linalg.norm(U_exact.reshape(-1)) + 1e-16)).item()

    savemat(
        "./training_data/trSQP_PINN_2D_metrics.mat",
        {
            "mse": mse,
            "rel_l2": rel_l2,
            "physics_loss": physics_loss,
            "initial_loss": initial_loss,
            "boundary_loss": boundary_loss,
        },
    )

    torch.save(
        {k: v.clone().cpu() for k, v in model.state_dict().items()},
        "./training_data/trSQP_PINN_2D_model.pth",
    )

    print("Saved model and metrics")
    print(f"Physics Loss: {physics_loss:.6e}, Initial Loss: {initial_loss:.6e}, Boundary Loss: {boundary_loss:.6e}")
    print(f"MSE: {mse:.6e}, Relative L2: {rel_l2:.6e}")


def plot_solution(model, alpha, t_value=0.5):
    model.eval()
    x = torch.linspace(0, 1, 100, device=device)
    y = torch.linspace(0, 1, 100, device=device)
    X, Y = torch.meshgrid(x, y, indexing="ij")
    T = torch.full(X.shape, t_value, device=device)
    inp = torch.stack([X.flatten(), Y.flatten(), T.flatten()], dim=1)

    with torch.no_grad():
        U_pred = model(inp).reshape(X.shape).cpu().numpy()

    U_exact = exact_solution(X, Y, T, alpha).cpu().numpy()
    err = np.abs(U_pred - U_exact)

    X_np = X.cpu().numpy()
    Y_np = Y.cpu().numpy()

    os.makedirs("./data", exist_ok=True)
    savemat(
        f"./data/trSQP_solution_t{t_value}.mat",
        {
            "X": X_np,
            "Y": Y_np,
            "U_pred": U_pred,
            "U_exact": U_exact,
            "Error": err,
        },
    )

    os.makedirs("./figures", exist_ok=True)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    vmin = min(U_pred.min(), U_exact.min())
    vmax = max(U_pred.max(), U_exact.max())

    ax1.pcolormesh(X_np, Y_np, U_pred, cmap="jet", vmin=vmin, vmax=vmax, shading="auto")
    ax1.set_title("Prediction")

    ax2.pcolormesh(X_np, Y_np, U_exact, cmap="jet", vmin=vmin, vmax=vmax, shading="auto")
    ax2.set_title("Exact")

    ax3.pcolormesh(X_np, Y_np, err, cmap="Reds", shading="auto")
    ax3.set_title("Abs Error")

    fig.savefig(f"./figures/trSQP_PINN_2D_t{t_value}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    ds = build_training_sets(args.N_u, args.N_f, args.N_ic, args.N_bc, args.alpha, noise=args.noise)
    model = PINN().to(device)

    pretrain_feasibility(
        model,
        ds,
        args.alpha,
        max_iter=args.pretrain_max_iter,
        g_tol=args.g_tol,
        f_tol=args.f_tol,
    )

    start = time.time()
    model = trSQP_train(
        model,
        ds,
        args.alpha,
        max_iter=args.EPOCH,
        delta=args.delta,
        nu=args.nu,
        eta_low=args.eta_low,
        eta_upp=args.eta_upp,
        rho_scale=args.rho_scale,
        g_tol=args.g_tol,
        f_tol=args.f_tol,
    )
    print(f"Training finished in {time.time() - start:.1f}s")

    evaluate_and_save(model, args.alpha)
    plot_solution(model, args.alpha, t_value=0.5)
    plot_solution(model, args.alpha, t_value=1.0)
