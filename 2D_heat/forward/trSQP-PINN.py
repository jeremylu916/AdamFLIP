#!/usr/bin/env python3
"""trSQP-PINN for 2D heat forward problem.

This version keeps the same network structure and random data sampling as
2D_heat/forward/Adam_PINN.py, and replaces optimization with a trust-region SQP
routine adapted from the provided algorithm.
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
parser.add_argument("--pretrain", type=int, default=100)
parser.add_argument("--ordinal", type=int, default=0)
parser.add_argument("--delta", type=float, default=0.2)
parser.add_argument("--nu", type=float, default=0.8)
parser.add_argument("--eta_low", type=float, default=1e-8)
parser.add_argument("--eta_upp", type=float, default=0.3)
parser.add_argument("--rho_scale", type=float, default=2.0)
parser.add_argument("--g_tol", type=float, default=1e-9)
parser.add_argument("--f_tol", type=float, default=1e-9)
args = parser.parse_args()


# --- Reproducibility ---
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
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


# Keep architecture exactly aligned with Adam_PINN.py
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)


def initial_condition(x, y):
    return torch.sin(torch.pi * x) * torch.sin(torch.pi * y)


def pde(x, y, t, model):
    input_data = torch.cat([x, y, t], dim=1)
    u = model(input_data)

    grads = torch.autograd.grad(u, [x, y, t], grad_outputs=torch.ones_like(u), create_graph=True)
    u_x, u_y, u_t = grads[0], grads[1], grads[2]

    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]

    alpha = 0.1
    return u_t - alpha * (u_xx + u_yy)


# Keep data sampling exactly aligned with Adam_PINN.py
def generate_training_data(num_points_pde, num_points_bc, num_points_ic):
    x_pde = torch.rand(num_points_pde, 1, requires_grad=True, device=device)
    y_pde = torch.rand(num_points_pde, 1, requires_grad=True, device=device)
    t_pde = torch.rand(num_points_pde, 1, requires_grad=True, device=device)

    x_ic = torch.rand(num_points_ic, 1, device=device)
    y_ic = torch.rand(num_points_ic, 1, device=device)
    t_ic = torch.zeros(num_points_ic, 1, device=device)
    u_ic_exact = initial_condition(x_ic, y_ic)

    y_bc_x = torch.rand(num_points_bc // 2, 1, device=device)
    t_bc_x = torch.rand(num_points_bc // 2, 1, device=device)
    x_bc_0, x_bc_1 = torch.zeros_like(y_bc_x), torch.ones_like(y_bc_x)

    x_bc_y = torch.rand(num_points_bc // 2, 1, device=device)
    t_bc_y = torch.rand(num_points_bc // 2, 1, device=device)
    y_bc_0, y_bc_1 = torch.zeros_like(x_bc_y), torch.ones_like(x_bc_y)

    x_bc = torch.cat([x_bc_0, x_bc_1, x_bc_y, x_bc_y], dim=0)
    y_bc = torch.cat([y_bc_x, y_bc_x, y_bc_0, y_bc_1], dim=0)
    t_bc = torch.cat([t_bc_x, t_bc_x, t_bc_y, t_bc_y], dim=0)
    u_bc_exact = torch.zeros_like(x_bc)

    return (
        x_pde, y_pde, t_pde,
        x_ic, y_ic, t_ic, u_ic_exact,
        x_bc, y_bc, t_bc, u_bc_exact,
    )


def get_flat_params(model):
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()]).to(device)


def set_flat_params(model, flat_params):
    flat_params = flat_params.to(device=device, dtype=torch.float32)
    pointer = 0
    with torch.no_grad():
        for p in model.parameters():
            n = p.numel()
            block = flat_params[pointer:pointer + n].view_as(p)
            p.copy_(block)
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


def objective_obs_loss(model, batch, mse_loss):
    # 2D heat forward setup has no observation set; use zero objective and enforce
    # PDE/IC/BC through constraints as in a pure feasibility-driven SQP stage.
    return sum((p.sum() * 0.0) for p in model.parameters())


def constraint_physics_loss(model, batch, mse_loss):
    x_pde, y_pde, t_pde = batch[0], batch[1], batch[2]
    residual = pde(x_pde, y_pde, t_pde, model)
    return mse_loss(residual, torch.zeros_like(residual))


def constraint_ic_loss(model, batch, mse_loss):
    x_ic, y_ic, t_ic, u_ic_exact = batch[3], batch[4], batch[5], batch[6]
    u_pred_ic = model(torch.cat([x_ic, y_ic, t_ic], dim=1))
    return mse_loss(u_pred_ic, u_ic_exact)


def constraint_bc_loss(model, batch, mse_loss):
    x_bc, y_bc, t_bc, u_bc_exact = batch[7], batch[8], batch[9], batch[10]
    u_pred_bc = model(torch.cat([x_bc, y_bc, t_bc], dim=1))
    return mse_loss(u_pred_bc, u_bc_exact)


def compute_lambda_kkt(grad_f, J, reg=1e-8):
    """Solve lambda = argmin ||grad_f + J^T lambda||^2."""
    jj_t = J @ J.T
    rhs = -(J @ grad_f)
    eye = torch.eye(jj_t.shape[0], device=device, dtype=torch.float32)
    try:
        return torch.linalg.solve(jj_t + reg * eye, rhs)
    except RuntimeError:
        return torch.linalg.lstsq(jj_t + reg * eye, rhs.unsqueeze(1)).solution.squeeze(1)


def solve_tr_subproblem(h, grad_l, J, H, delta, nu, reg=1e-6):
    """Approximate trust-region step with normal + tangential decomposition."""
    n = grad_l.shape[0]
    m = J.shape[0]

    eye_m = torch.eye(m, device=device, dtype=torch.float32)
    jj_t = J @ J.T

    # Normal step: reduce linearized constraint violation
    try:
        d_n = -J.T @ torch.linalg.solve(jj_t + reg * eye_m, h)
    except RuntimeError:
        d_n = -J.T @ torch.linalg.lstsq(jj_t + reg * eye_m, h.unsqueeze(1)).solution.squeeze(1)

    norm_dn = torch.linalg.norm(d_n)
    max_dn = nu * delta
    if float(norm_dn.item()) > max_dn:
        d_n = d_n * (max_dn / (norm_dn + 1e-16))

    # Tangential step: projected Newton-like step
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


def pretrain_feasibility(model, num_points_pde, num_points_bc, num_points_ic, max_iter=1000, g_tol=1e-9, f_tol=1e-9):
    """Step 0: minimize squared constraint violations to get feasible initialization."""
    mse_loss = nn.MSELoss()
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
        batch = generate_training_data(num_points_pde, num_points_bc, num_points_ic)

        def closure():
            optimizer.zero_grad()
            c_loss = (
                constraint_physics_loss(model, batch, mse_loss)
                + constraint_ic_loss(model, batch, mse_loss)
                + constraint_bc_loss(model, batch, mse_loss)
            )
            c_loss.backward()
            return c_loss

        optimizer.step(closure)

        c_loss = (
            constraint_physics_loss(model, batch, mse_loss)
            + constraint_ic_loss(model, batch, mse_loss)
            + constraint_bc_loss(model, batch, mse_loss)
        )

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
    num_points_pde,
    num_points_bc,
    num_points_ic,
    max_iter=1000,
    delta=0.2,
    nu=0.8,
    eta_low=1e-8,
    eta_upp=0.3,
    rho_scale=2.0,
    g_tol=1e-9,
    f_tol=1e-9,
):
    mse_loss = nn.MSELoss()
    params = list(model.parameters())
    theta = get_flat_params(model)

    n = theta.shape[0]
    H = torch.eye(n, device=device, dtype=torch.float32)

    mu_prev = 1.0
    Delta = float(delta)

    # Algorithm init: lambda0 = argmin ||grad f0 + J0^T lambda||^2
    batch0 = generate_training_data(num_points_pde, num_points_bc, num_points_ic)
    set_flat_params(model, theta)
    loss_f0 = objective_obs_loss(model, batch0, mse_loss)
    loss_h10 = constraint_physics_loss(model, batch0, mse_loss)
    loss_h20 = constraint_ic_loss(model, batch0, mse_loss)
    loss_h30 = constraint_bc_loss(model, batch0, mse_loss)

    grad_f0 = get_grads_from_loss(loss_f0, params)
    J10 = get_grads_from_loss(loss_h10, params)
    J20 = get_grads_from_loss(loss_h20, params)
    J30 = get_grads_from_loss(loss_h30, params)
    J0 = torch.vstack([J10, J20, J30])
    lambda_k = compute_lambda_kkt(grad_f0, J0)

    os.makedirs("./training_logs", exist_ok=True)
    loss_file_path = "./training_logs/trSQP-PINN_training_loss.txt"

    start_time = time.time()
    with open(loss_file_path, "w", encoding="utf-8") as f:
        f.write("Epoch,Loss_PDE,Loss_IC,Loss_BC,Merit,Delta,eta\n")

        pbar = trange(max_iter, desc="trSQP Training", unit="iter")
        for k in pbar:
            batch = generate_training_data(num_points_pde, num_points_bc, num_points_ic)
            set_flat_params(model, theta)

            loss_f = objective_obs_loss(model, batch, mse_loss)
            loss_h1 = constraint_physics_loss(model, batch, mse_loss)
            loss_h2 = constraint_ic_loss(model, batch, mse_loss)
            loss_h3 = constraint_bc_loss(model, batch, mse_loss)

            f_val = float(loss_f.item())
            h = torch.tensor([float(loss_h1.item()), float(loss_h2.item()), float(loss_h3.item())], device=device, dtype=torch.float32)

            grad_f = get_grads_from_loss(loss_f, params)
            J1 = get_grads_from_loss(loss_h1, params)
            J2 = get_grads_from_loss(loss_h2, params)
            J3 = get_grads_from_loss(loss_h3, params)
            J = torch.vstack([J1, J2, J3])

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
            f_trial = float(objective_obs_loss(model, batch, mse_loss).item())
            h_trial = torch.tensor([
                float(constraint_physics_loss(model, batch, mse_loss).item()),
                float(constraint_ic_loss(model, batch, mse_loss).item()),
                float(constraint_bc_loss(model, batch, mse_loss).item()),
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

                loss_f_new = objective_obs_loss(model, batch, mse_loss)
                loss_h1_new = constraint_physics_loss(model, batch, mse_loss)
                loss_h2_new = constraint_ic_loss(model, batch, mse_loss)
                loss_h3_new = constraint_bc_loss(model, batch, mse_loss)

                grad_f_new = get_grads_from_loss(loss_f_new, params)
                J1_new = get_grads_from_loss(loss_h1_new, params)
                J2_new = get_grads_from_loss(loss_h2_new, params)
                J3_new = get_grads_from_loss(loss_h3_new, params)
                J_new = torch.vstack([J1_new, J2_new, J3_new])

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

            grad_l_inf = float(torch.max(torch.abs(grad_l)).item())
            if (float(torch.linalg.norm(d_k).item()) <= f_tol or grad_l_inf <= g_tol) and (k + 1) >= min(1000, max_iter):
                print(f"Converged at iter {k}")
                break

            if k % 10 == 0:
                pbar.set_postfix({
                    "h_phy": f"{float(h[0].item()):.2e}",
                    "h_ic": f"{float(h[1].item()):.2e}",
                    "h_bc": f"{float(h[2].item()):.2e}",
                    "Delta": f"{Delta:.2e}",
                    "eta": f"{eta_k:.2e}",
                })
                f.write(
                    f"{k},{float(h[0].item()):.6e},{float(h[1].item()):.6e},{float(h[2].item()):.6e},"
                    f"{phi_k:.6e},{Delta:.6e},{eta_k:.6e}\n"
                )

    elapsed = time.time() - start_time
    print(f"Training finished in {elapsed:.1f}s")

    set_flat_params(model, theta)
    return model


def plot_solution_comparison(model, t_value):
    model.eval()
    alpha = 0.1

    x = torch.linspace(0, 1, 100, device=device)
    y = torch.linspace(0, 1, 100, device=device)
    X, Y = torch.meshgrid(x, y, indexing="ij")
    T = torch.full(X.shape, t_value, device=device)

    input_tensor = torch.stack([X.flatten(), Y.flatten(), T.flatten()], dim=1)
    with torch.no_grad():
        U_pred = model(input_tensor).reshape(X.shape)

    exp_term = np.exp(-2 * np.pi ** 2 * alpha * t_value)
    U_exact = exp_term * torch.sin(np.pi * X) * torch.sin(np.pi * Y)
    Error = torch.abs(U_pred - U_exact)

    X_np = X.cpu().numpy()
    Y_np = Y.cpu().numpy()
    U_pred_np = U_pred.cpu().numpy()
    U_exact_np = U_exact.cpu().numpy()
    Error_np = Error.cpu().numpy()

    os.makedirs("./mat_files", exist_ok=True)
    savemat(
        f"./mat_files/trSQP_solution_comparison_t{t_value}.mat",
        {
            "X": X_np,
            "Y": Y_np,
            "U_pred": U_pred_np,
            "U_exact": U_exact_np,
            "Error": Error_np,
        },
    )

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"trSQP-PINN Comparison at t = {t_value}", fontsize=16)

    v_min = min(U_pred_np.min(), U_exact_np.min())
    v_max = max(U_pred_np.max(), U_exact_np.max())

    c1 = ax1.pcolormesh(X_np, Y_np, U_pred_np, cmap="jet", vmin=v_min, vmax=v_max, shading="auto")
    fig.colorbar(c1, ax=ax1)
    ax1.set_title("Prediction")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.axis("square")

    c2 = ax2.pcolormesh(X_np, Y_np, U_exact_np, cmap="jet", vmin=v_min, vmax=v_max, shading="auto")
    fig.colorbar(c2, ax=ax2)
    ax2.set_title("Exact Solution")
    ax2.set_xlabel("x")
    ax2.axis("square")

    c3 = ax3.pcolormesh(X_np, Y_np, Error_np, cmap="Reds", shading="auto")
    fig.colorbar(c3, ax=ax3)
    ax3.set_title("Absolute Error")
    ax3.set_xlabel("x")
    ax3.axis("square")

    plt.subplots_adjust(wspace=0.1, hspace=0.1, top=0.88)
    os.makedirs("./figures", exist_ok=True)
    fig.savefig(f"./figures/trSQP_pinn_comparison_t{t_value}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def calculate_l2_relative_error(model):
    model.eval()
    alpha = 0.1
    x = torch.linspace(0, 1, 100, device=device)
    y = torch.linspace(0, 1, 100, device=device)
    t = torch.linspace(0, 1, 100, device=device)
    X, Y, T = torch.meshgrid(x, y, t, indexing="ij")
    input_tensor = torch.stack([X.flatten(), Y.flatten(), T.flatten()], dim=1)

    with torch.no_grad():
        U_pred = model(input_tensor).reshape(X.shape)

    exp_term = torch.exp(-2 * torch.pi ** 2 * alpha * T)
    U_exact = exp_term * torch.sin(torch.pi * X) * torch.sin(torch.pi * Y)

    mse_error = torch.mean((U_pred - U_exact) ** 2)
    l2_error = torch.linalg.norm(U_pred - U_exact) / (torch.linalg.norm(U_exact) + 1e-16)

    print("\n--- Final L2 Relative Error ---")
    print(f"L2 Relative Error: {l2_error.item():.6e}")
    print(f"Mean Squared Error (MSE): {mse_error.item():.6e}")


def calculate_final_losses(model, num_points_pde, num_points_bc, num_points_ic):
    print("\n--- Calculating Final Losses on a New Batch ---")
    model.eval()
    mse_loss = nn.MSELoss()

    batch = generate_training_data(num_points_pde, num_points_bc, num_points_ic)
    with torch.no_grad():
        loss_ic = constraint_ic_loss(model, batch, mse_loss)
        loss_bc = constraint_bc_loss(model, batch, mse_loss)

    # PDE loss requires gradients through input coordinates.
    loss_pde = constraint_physics_loss(model, batch, mse_loss)
    total = loss_ic + loss_bc + loss_pde

    print(f"Final Initial Condition Loss: {loss_ic.item():.6e}")
    print(f"Final Boundary Condition Loss: {loss_bc.item():.6e}")
    print(f"Final Physics (PDE) Loss:      {loss_pde.item():.6e}")
    print(f"Final Total Loss:              {total.item():.6e}")


if __name__ == "__main__":
    NUM_POINTS_PDE = 2000
    NUM_POINTS_BC = 500
    NUM_POINTS_IC = 500

    model = PINN().to(device)

    pretrain_feasibility(
        model,
        num_points_pde=NUM_POINTS_PDE,
        num_points_bc=NUM_POINTS_BC,
        num_points_ic=NUM_POINTS_IC,
        max_iter=args.pretrain,
        g_tol=args.g_tol,
        f_tol=args.f_tol,
    )

    trained = trSQP_train(
        model,
        num_points_pde=NUM_POINTS_PDE,
        num_points_bc=NUM_POINTS_BC,
        num_points_ic=NUM_POINTS_IC,
        max_iter=args.EPOCH,
        delta=args.delta,
        nu=args.nu,
        eta_low=args.eta_low,
        eta_upp=args.eta_upp,
        rho_scale=args.rho_scale,
        g_tol=args.g_tol,
        f_tol=args.f_tol,
    )

    os.makedirs("./training_data", exist_ok=True)
    torch.save(trained.state_dict(), "./training_data/trSQP_PINN_2D_model.pth")

    plot_solution_comparison(trained, t_value=0.5)
    plot_solution_comparison(trained, t_value=1.0)
    calculate_l2_relative_error(trained)
    calculate_final_losses(trained, NUM_POINTS_PDE, NUM_POINTS_BC, NUM_POINTS_IC)
