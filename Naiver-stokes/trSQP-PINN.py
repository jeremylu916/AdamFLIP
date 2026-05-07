#!/usr/bin/env python3
"""trSQP-PINN for 2D incompressible Navier-Stokes (Taylor-Green vortex).

Uses the same trust-region SQP core as 2D_heat/forward/trSQP-PINN.py while
keeping Navier-Stokes PINN architecture, data generation, and evaluation flow.
"""

import argparse
import os
import random as rm
import time

import numpy as np
import scipy.io
import torch
import torch.nn as nn
from tqdm import trange

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
rm.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
if torch.backends.cudnn.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


T_MAX = 1.0
nu = 0.01

N_COLL = 10000
N_IC = 1000
N_BC = 1000


def taylor_green_u(x, y, t):
    return -torch.cos(x) * torch.sin(y) * torch.exp(-2.0 * nu * t)


def taylor_green_v(x, y, t):
    return torch.sin(x) * torch.cos(y) * torch.exp(-2.0 * nu * t)


def taylor_green_p(x, y, t):
    return -0.25 * (torch.cos(2.0 * x) + torch.cos(2.0 * y)) * torch.exp(-4.0 * nu * t)


def sample_training_data():
    x_coll = 2.0 * np.pi * np.random.rand(N_COLL, 1)
    y_coll = 2.0 * np.pi * np.random.rand(N_COLL, 1)
    t_coll = T_MAX * np.random.rand(N_COLL, 1)
    X_coll = np.hstack([x_coll, y_coll, t_coll]).astype(np.float32)

    x_ic = 2.0 * np.pi * np.random.rand(N_IC, 1)
    y_ic = 2.0 * np.pi * np.random.rand(N_IC, 1)
    t_ic = np.zeros((N_IC, 1), dtype=np.float32)
    X_ic = np.hstack([x_ic, y_ic, t_ic]).astype(np.float32)

    y_bx = 2.0 * np.pi * np.random.rand(N_BC, 1)
    t_bx = T_MAX * np.random.rand(N_BC, 1)
    X_bx_l = np.hstack([np.zeros((N_BC, 1)), y_bx, t_bx]).astype(np.float32)
    X_bx_r = np.hstack([2.0 * np.pi * np.ones((N_BC, 1)), y_bx, t_bx]).astype(np.float32)

    x_by = 2.0 * np.pi * np.random.rand(N_BC, 1)
    t_by = T_MAX * np.random.rand(N_BC, 1)
    X_by_b = np.hstack([x_by, np.zeros((N_BC, 1)), t_by]).astype(np.float32)
    X_by_t = np.hstack([x_by, 2.0 * np.pi * np.ones((N_BC, 1)), t_by]).astype(np.float32)

    return X_coll, X_ic, X_bx_l, X_bx_r, X_by_b, X_by_t


class PINN(nn.Module):
    def __init__(self, layers=(3, 96, 96, 96, 96, 3)):
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
        # Map domain [0, 2pi]x[0, 2pi]x[0, 1] to [-1, 1]^3 for tanh stability.
        x = 2.0 * (xyt[:, 0:1] / (2.0 * np.pi)) - 1.0
        y = 2.0 * (xyt[:, 1:2] / (2.0 * np.pi)) - 1.0
        t = 2.0 * (xyt[:, 2:3] / T_MAX) - 1.0
        xyt_scaled = torch.cat([x, y, t], dim=1)
        return self.net(xyt_scaled)


class PhysicsInformedNS:
    def __init__(self, X_coll, X_ic, X_bx_l, X_bx_r, X_by_b, X_by_t, device):
        self.device = device
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
        return out[:, 0:1], out[:, 1:2], out[:, 2:3]

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
        return self.mse(continuity, zero) + self.mse(mom_x, zero) + self.mse(mom_y, zero)

    def initial_loss(self):
        x = self.X_ic[:, 0:1]
        y = self.X_ic[:, 1:2]
        t = self.X_ic[:, 2:3]
        u_true = taylor_green_u(x, y, t)
        v_true = taylor_green_v(x, y, t)

        u_pred, v_pred, _ = self.predict_uvp(self.X_ic)
        return self.mse(u_pred, u_true) + self.mse(v_pred, v_true)

    def periodic_boundary_loss(self):
        X_bx_l = self.X_bx_l.clone().detach().requires_grad_(True)
        X_bx_r = self.X_bx_r.clone().detach().requires_grad_(True)
        X_by_b = self.X_by_b.clone().detach().requires_grad_(True)
        X_by_t = self.X_by_t.clone().detach().requires_grad_(True)

        u_l, v_l, p_l = self.predict_uvp(X_bx_l)
        u_r, v_r, p_r = self.predict_uvp(X_bx_r)

        u_b, v_b, p_b = self.predict_uvp(X_by_b)
        u_t, v_t, p_t = self.predict_uvp(X_by_t)

        # Enforce periodicity on both values and normal derivatives.
        du_l = torch.autograd.grad(u_l, X_bx_l, torch.ones_like(u_l), create_graph=True)[0][:, 0:1]
        du_r = torch.autograd.grad(u_r, X_bx_r, torch.ones_like(u_r), create_graph=True)[0][:, 0:1]
        dv_l = torch.autograd.grad(v_l, X_bx_l, torch.ones_like(v_l), create_graph=True)[0][:, 0:1]
        dv_r = torch.autograd.grad(v_r, X_bx_r, torch.ones_like(v_r), create_graph=True)[0][:, 0:1]
        dp_l = torch.autograd.grad(p_l, X_bx_l, torch.ones_like(p_l), create_graph=True)[0][:, 0:1]
        dp_r = torch.autograd.grad(p_r, X_bx_r, torch.ones_like(p_r), create_graph=True)[0][:, 0:1]

        du_b = torch.autograd.grad(u_b, X_by_b, torch.ones_like(u_b), create_graph=True)[0][:, 1:2]
        du_t = torch.autograd.grad(u_t, X_by_t, torch.ones_like(u_t), create_graph=True)[0][:, 1:2]
        dv_b = torch.autograd.grad(v_b, X_by_b, torch.ones_like(v_b), create_graph=True)[0][:, 1:2]
        dv_t = torch.autograd.grad(v_t, X_by_t, torch.ones_like(v_t), create_graph=True)[0][:, 1:2]
        dp_b = torch.autograd.grad(p_b, X_by_b, torch.ones_like(p_b), create_graph=True)[0][:, 1:2]
        dp_t = torch.autograd.grad(p_t, X_by_t, torch.ones_like(p_t), create_graph=True)[0][:, 1:2]

        loss_x = self.mse(u_l, u_r) + self.mse(v_l, v_r) + self.mse(p_l, p_r)
        loss_y = self.mse(u_b, u_t) + self.mse(v_b, v_t) + self.mse(p_b, p_t)
        loss_dx = self.mse(du_l, du_r) + self.mse(dv_l, dv_r) + self.mse(dp_l, dp_r)
        loss_dy = self.mse(du_b, du_t) + self.mse(dv_b, dv_t) + self.mse(dp_b, dp_t)
        return loss_x + loss_y + 0.5 * (loss_dx + loss_dy)

    def pressure_gauge_loss(self):
        # Pressure is defined up to a constant; fixing its mean improves p error.
        _, _, p = self.predict_uvp(self.X_coll)
        return torch.mean(p) ** 2


def get_flat_params(model, device):
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()]).to(device)


def set_flat_params(model, flat_params, device):
    flat_params = flat_params.to(device=device, dtype=torch.float32)
    pointer = 0
    with torch.no_grad():
        for p in model.parameters():
            n = p.numel()
            p.copy_(flat_params[pointer:pointer + n].view_as(p))
            pointer += n


def get_grads_from_loss(loss, params, device):
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    flat = []
    for g, p in zip(grads, params):
        if g is None:
            flat.append(torch.zeros(p.numel(), device=device, dtype=torch.float32))
        else:
            flat.append(g.detach().reshape(-1).to(device=device, dtype=torch.float32))
    return torch.cat(flat)


def compute_lambda_kkt(grad_f, J, device, reg=1e-8):
    jj_t = J @ J.T
    rhs = -(J @ grad_f)
    eye = torch.eye(jj_t.shape[0], device=device, dtype=torch.float32)
    try:
        return torch.linalg.solve(jj_t + reg * eye, rhs)
    except RuntimeError:
        return torch.linalg.lstsq(jj_t + reg * eye, rhs.unsqueeze(1)).solution.squeeze(1)


def solve_tr_subproblem_light(h, grad_l, J, h_diag, Delta, nu_split, device, reg=1e-6):
    m = J.shape[0]

    eye_m = torch.eye(m, device=device, dtype=torch.float32)
    jj_t = J @ J.T

    try:
        d_n = -J.T @ torch.linalg.solve(jj_t + reg * eye_m, h)
    except RuntimeError:
        d_n = -J.T @ torch.linalg.lstsq(jj_t + reg * eye_m, h.unsqueeze(1)).solution.squeeze(1)

    norm_dn = torch.linalg.norm(d_n)
    max_dn = nu_split * Delta
    if float(norm_dn.item()) > max_dn:
        d_n = d_n * (max_dn / (norm_dn + 1e-16))

    # Lightweight tangential step: diagonal quasi-Newton direction.
    d_obj = -grad_l / (h_diag + reg)

    Jd = J @ d_obj
    try:
        corr = J.T @ torch.linalg.solve(jj_t + reg * eye_m, Jd)
    except RuntimeError:
        corr = J.T @ torch.linalg.lstsq(jj_t + reg * eye_m, Jd.unsqueeze(1)).solution.squeeze(1)
    d_t = d_obj - corr

    remain = torch.sqrt(torch.clamp(torch.tensor(Delta * Delta, device=device) - torch.dot(d_n, d_n), min=0.0))
    norm_dt = torch.linalg.norm(d_t)
    if float(norm_dt.item()) > float(remain.item()):
        d_t = d_t * (remain / (norm_dt + 1e-16))

    return d_n + d_t


def objective_obs_loss(pinn_ns):
    return pinn_ns.residual_loss() + 0.1 * pinn_ns.pressure_gauge_loss()


def constraint_ic_loss(pinn_ns):
    return pinn_ns.initial_loss()


def constraint_bc_loss(pinn_ns):
    return pinn_ns.periodic_boundary_loss()


def pretrain_feasibility(pinn_ns, max_iter=100, g_tol=1e-9, f_tol=1e-9):
    model = pinn_ns.model
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=50,
        tolerance_grad=1e-12,
        tolerance_change=1e-12,
        history_size=100,
        line_search_fn="strong_wolfe",
    )

    theta_prev = get_flat_params(model, pinn_ns.device)
    outer_steps = max(1, max_iter // 50)

    for l in range(outer_steps):
        def closure():
            optimizer.zero_grad()
            c_loss = constraint_ic_loss(pinn_ns) + constraint_bc_loss(pinn_ns)
            c_loss.backward()
            return c_loss

        optimizer.step(closure)

        c_loss = constraint_ic_loss(pinn_ns) + constraint_bc_loss(pinn_ns)
        grads = torch.autograd.grad(c_loss, list(model.parameters()), retain_graph=False, allow_unused=True)
        grad_inf = 0.0
        for g in grads:
            if g is not None:
                grad_inf = max(grad_inf, float(g.detach().abs().max().item()))

        theta_new = get_flat_params(model, pinn_ns.device)
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


def tr_sqp_pinn(pinn_ns, args):
    model = pinn_ns.model
    device = pinn_ns.device
    params = list(model.parameters())

    theta = get_flat_params(model, device)
    n = theta.shape[0]
    h_diag = torch.ones(n, device=device, dtype=torch.float32)

    mu_prev = 1.0
    Delta = float(args.delta)
    min_stop_iter = min(1000, args.max_iter)

    set_flat_params(model, theta, device)
    loss_f0 = objective_obs_loss(pinn_ns)
    loss_h10 = constraint_ic_loss(pinn_ns)
    loss_h20 = constraint_bc_loss(pinn_ns)
    grad_f0 = get_grads_from_loss(loss_f0, params, device)
    J10 = get_grads_from_loss(loss_h10, params, device)
    J20 = get_grads_from_loss(loss_h20, params, device)
    J0 = torch.vstack([J10, J20])
    lambda_k = compute_lambda_kkt(grad_f0, J0, device)

    history = {
        "f": [],
        "ic": [],
        "bc": [],
        "kkt": [],
        "delta": [],
        "eta": [],
        "lambda_ic": [],
        "lambda_bc": [],
    }

    pbar = trange(1, args.max_iter + 1, desc="trSQP-PINN Training", unit="iter")
    for k in pbar:
        set_flat_params(model, theta, device)

        loss_f = objective_obs_loss(pinn_ns)
        loss_h1 = constraint_ic_loss(pinn_ns)
        loss_h2 = constraint_bc_loss(pinn_ns)

        f_val = float(loss_f.item())
        h = torch.tensor([float(loss_h1.item()), float(loss_h2.item())], device=device, dtype=torch.float32)

        grad_f = get_grads_from_loss(loss_f, params, device)
        J1 = get_grads_from_loss(loss_h1, params, device)
        J2 = get_grads_from_loss(loss_h2, params, device)
        J = torch.vstack([J1, J2])

        grad_l = grad_f + J.T @ lambda_k
        d_k = solve_tr_subproblem_light(h, grad_l, J, h_diag, Delta, args.nu, device)

        denom = 0.7 * (torch.linalg.norm(h) - torch.linalg.norm(h + J @ d_k))
        if abs(float(denom.item())) < 1e-16:
            mu_k = mu_prev
        else:
            quad = torch.dot(grad_l, d_k) + 0.5 * torch.dot(h_diag * d_k, d_k)
            mu_k = max(mu_prev, float((quad / denom).item()))

        q0 = f_val + mu_k * float(torch.linalg.norm(h).item())
        qd = (
            f_val
            + float(torch.dot(grad_f, d_k).item())
                + 0.5 * float(torch.dot(h_diag * d_k, d_k).item())
            + mu_k * float(torch.linalg.norm(h + J @ d_k).item())
        )
        pred_k = q0 - qd

        theta_trial = theta + d_k
        set_flat_params(model, theta_trial, device)
        f_trial = float(objective_obs_loss(pinn_ns).item())
        h_trial = torch.tensor(
            [float(constraint_ic_loss(pinn_ns).item()), float(constraint_bc_loss(pinn_ns).item())],
            device=device,
            dtype=torch.float32,
        )

        phi_k = f_val + mu_k * float(torch.linalg.norm(h).item())
        phi_trial = f_trial + mu_k * float(torch.linalg.norm(h_trial).item())
        ared_k = phi_k - phi_trial

        eta_k = -np.inf if pred_k <= 0 else ared_k / pred_k
        accepted = eta_k >= args.eta_low

        if accepted:
            theta_new = theta + d_k
            if eta_k >= args.eta_upp:
                Delta = min(args.delta_max, args.rho_scale * Delta)

            set_flat_params(model, theta_new, device)
            loss_f_new = objective_obs_loss(pinn_ns)
            loss_h1_new = constraint_ic_loss(pinn_ns)
            loss_h2_new = constraint_bc_loss(pinn_ns)

            grad_f_new = get_grads_from_loss(loss_f_new, params, device)
            J1_new = get_grads_from_loss(loss_h1_new, params, device)
            J2_new = get_grads_from_loss(loss_h2_new, params, device)
            J_new = torch.vstack([J1_new, J2_new])

            lambda_new = compute_lambda_kkt(grad_f_new, J_new, device)
            grad_l_new = grad_f_new + J_new.T @ lambda_new

            s = theta_new - theta
            y = grad_l_new - grad_l
            h_s = h_diag * s
            sHs = torch.dot(s, h_s)

            if torch.dot(s, y) >= 0.2 * sHs:
                y_bar = y
            else:
                theta_scale = (0.8 * sHs) / (sHs - torch.dot(s, y) + 1e-16)
                y_bar = theta_scale * y + (1.0 - theta_scale) * h_s

            # Diagonal damped-BFGS-style update for efficiency and stability.
            s_sq = s * s
            yb_s = y_bar * s
            valid = s_sq > 1e-16
            h_new = h_diag.clone()
            h_new[valid] = h_diag[valid] + yb_s[valid] / (s_sq[valid] + 1e-16) - (h_s[valid] * h_s[valid]) / (sHs + 1e-16)
            h_diag = torch.clamp(h_new, min=args.h_floor, max=args.h_cap)

            theta = theta_new
            mu_prev = mu_k
            lambda_k = lambda_new
            lambda_log = lambda_new
        else:
            Delta = max(args.delta_min, Delta / args.rho_scale)
            lambda_log = lambda_k

        kkt_gap = max(
            float(torch.linalg.norm(grad_l).item()),
            float(torch.max(torch.abs(h)).item()),
        )

        history["f"].append(f_val)
        history["ic"].append(float(h[0].item()))
        history["bc"].append(float(h[1].item()))
        history["kkt"].append(kkt_gap)
        history["delta"].append(float(Delta))
        history["eta"].append(float(eta_k))
        history["lambda_ic"].append(float(lambda_log[0].item()))
        history["lambda_bc"].append(float(lambda_log[1].item()))

        if k % 50 == 0:
            pbar.set_postfix(
                {
                    "f": f"{f_val:.2e}",
                    "ic": f"{float(h[0].item()):.2e}",
                    "bc": f"{float(h[1].item()):.2e}",
                    "kkt": f"{kkt_gap:.2e}",
                    "Delta": f"{Delta:.2e}",
                }
            )

        if (k >= min_stop_iter and eta_k < args.eta_low and (
            float(torch.linalg.norm(d_k).item()) <= args.f_tol
            or float(torch.max(torch.abs(grad_l)).item()) <= args.g_tol
        )) or (kkt_gap < args.tol):
            print(f"Converged at iter {k}, KKT={kkt_gap:.3e}")
            break

    set_flat_params(model, theta, device)
    return theta, history


def evaluate_l2_uvp(model, device, t_eval=0.5, n_grid=80):
    model.eval()

    x = np.linspace(0.0, 2.0 * np.pi, n_grid)
    y = np.linspace(0.0, 2.0 * np.pi, n_grid)
    X, Y = np.meshgrid(x, y)
    T = np.full_like(X, t_eval)

    X_eval = np.stack([X.ravel(), Y.ravel(), T.ravel()], axis=1)
    X_eval_t = torch.tensor(X_eval, dtype=torch.float32, device=device)

    with torch.no_grad():
        pred = model(X_eval_t).cpu().numpy()

    U_pred = pred[:, 0].reshape(n_grid, n_grid)
    V_pred = pred[:, 1].reshape(n_grid, n_grid)
    P_pred = pred[:, 2].reshape(n_grid, n_grid)

    Xt = torch.tensor(X, dtype=torch.float32)
    Yt = torch.tensor(Y, dtype=torch.float32)
    Tt = torch.tensor(T, dtype=torch.float32)

    U_true = taylor_green_u(Xt, Yt, Tt).numpy()
    V_true = taylor_green_v(Xt, Yt, Tt).numpy()
    P_true = taylor_green_p(Xt, Yt, Tt).numpy()

    err_u = np.linalg.norm(U_pred - U_true) / (np.linalg.norm(U_true) + 1e-12)
    err_v = np.linalg.norm(V_pred - V_true) / (np.linalg.norm(V_true) + 1e-12)
    err_p = np.linalg.norm(P_pred - P_true) / (np.linalg.norm(P_true) + 1e-12)

    return float(err_u), float(err_v), float(err_p)


def save_velocity_uv_error_artifacts(model, device, t_val, n_grid, data_dir, fig_dir):
    model.eval()
    x = np.linspace(0.0, 2.0 * np.pi, n_grid)
    y = np.linspace(0.0, 2.0 * np.pi, n_grid)
    X, Y = np.meshgrid(x, y)
    T = np.full_like(X, t_val)

    X_eval = np.stack([X.ravel(), Y.ravel(), T.ravel()], axis=1)
    X_eval_t = torch.tensor(X_eval, dtype=torch.float32, device=device)
    with torch.no_grad():
        pred = model(X_eval_t).cpu().numpy()

    U_pred = pred[:, 0].reshape(n_grid, n_grid)
    V_pred = pred[:, 1].reshape(n_grid, n_grid)
    Xt = torch.tensor(X, dtype=torch.float32)
    Yt = torch.tensor(Y, dtype=torch.float32)
    Tt = torch.tensor(T, dtype=torch.float32)
    U_true = taylor_green_u(Xt, Yt, Tt).numpy()
    V_true = taylor_green_v(Xt, Yt, Tt).numpy()

    diff_u = np.abs(U_pred - U_true)
    diff_v = np.abs(V_pred - V_true)

    vel_pred = np.sqrt(U_pred ** 2 + V_pred ** 2)
    vel_true = np.sqrt(U_true ** 2 + V_true ** 2)
    vel_err = np.abs(vel_pred - vel_true)

    os.makedirs(data_dir, exist_ok=True)
    scipy.io.savemat(
        os.path.join(data_dir, f"trSQP_PINN_velocity_t{t_val:.2f}.mat"),
        {
            "x": x,
            "y": y,
            "t": t_val,
            "velocity_pred": vel_pred,
            "velocity_true": vel_true,
            "velocity_err": vel_err,
            "U_pred": U_pred,
            "V_pred": V_pred,
            "U_true": U_true,
            "V_true": V_true,
        },
    )
    scipy.io.savemat(
        os.path.join(data_dir, f"trSQP_PINN_u_error_t{t_val:.2f}.mat"),
        {"x": x, "y": y, "t": t_val, "u_error": diff_u},
    )
    scipy.io.savemat(
        os.path.join(data_dir, f"trSQP_PINN_v_error_t{t_val:.2f}.mat"),
        {"x": x, "y": y, "t": t_val, "v_error": diff_v},
    )

    if plt is None:
        return

    os.makedirs(fig_dir, exist_ok=True)

    # Velocity figure (pred, true, error)
    fig_v, ax_v = plt.subplots(1, 3, figsize=(14, 4.5))
    vmin = min(vel_pred.min(), vel_true.min())
    vmax = max(vel_pred.max(), vel_true.max())
    im0 = ax_v[0].pcolormesh(X, Y, vel_pred, cmap="jet", vmin=vmin, vmax=vmax, shading="auto")
    ax_v[0].set_title("|V| pred")
    plt.colorbar(im0, ax=ax_v[0])
    im1 = ax_v[1].pcolormesh(X, Y, vel_true, cmap="jet", vmin=vmin, vmax=vmax, shading="auto")
    ax_v[1].set_title("|V| true")
    plt.colorbar(im1, ax=ax_v[1])
    im2 = ax_v[2].pcolormesh(X, Y, vel_err, cmap="Reds", shading="auto")
    ax_v[2].set_title("|V| error")
    plt.colorbar(im2, ax=ax_v[2])
    for ax in ax_v:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    fig_v.suptitle(f"trSQP-PINN Velocity at t={t_val:.2f}")
    fig_v.tight_layout()
    fig_v.savefig(os.path.join(fig_dir, f"trSQP_PINN_velocity_t{t_val:.2f}.png"), dpi=200, bbox_inches="tight")
    plt.close(fig_v)

    # U error figure
    fig_u, ax_u = plt.subplots(figsize=(5.5, 4.5))
    imu = ax_u.pcolormesh(X, Y, diff_u, cmap="Reds", shading="auto")
    ax_u.set_title(f"|U error| at t={t_val:.2f}")
    ax_u.set_xlabel("x")
    ax_u.set_ylabel("y")
    plt.colorbar(imu, ax=ax_u)
    fig_u.tight_layout()
    fig_u.savefig(os.path.join(fig_dir, f"trSQP_PINN_u_error_t{t_val:.2f}.png"), dpi=200, bbox_inches="tight")
    plt.close(fig_u)

    # V error figure
    fig_w, ax_w = plt.subplots(figsize=(5.5, 4.5))
    imw = ax_w.pcolormesh(X, Y, diff_v, cmap="Reds", shading="auto")
    ax_w.set_title(f"|V error| at t={t_val:.2f}")
    ax_w.set_xlabel("x")
    ax_w.set_ylabel("y")
    plt.colorbar(imw, ax=ax_w)
    fig_w.tight_layout()
    fig_w.savefig(os.path.join(fig_dir, f"trSQP_PINN_v_error_t{t_val:.2f}.png"), dpi=200, bbox_inches="tight")
    plt.close(fig_w)


def save_plot_comparison_mat(model, device, t_val=0.5, n_grid=80, fig_dir=None, data_dir=None):
    model.eval()
    x = np.linspace(0.0, 2.0 * np.pi, n_grid)
    y = np.linspace(0.0, 2.0 * np.pi, n_grid)
    X, Y = np.meshgrid(x, y)
    T = np.full_like(X, t_val)

    X_eval = np.stack([X.ravel(), Y.ravel(), T.ravel()], axis=1)
    X_eval_t = torch.tensor(X_eval, dtype=torch.float32, device=device)
    with torch.no_grad():
        pred = model(X_eval_t).cpu().numpy()

    U_pred = pred[:, 0].reshape(n_grid, n_grid)
    V_pred = pred[:, 1].reshape(n_grid, n_grid)
    Xt = torch.tensor(X, dtype=torch.float32)
    Yt = torch.tensor(Y, dtype=torch.float32)
    Tt = torch.tensor(T, dtype=torch.float32)
    U_true = taylor_green_u(Xt, Yt, Tt).numpy()
    V_true = taylor_green_v(Xt, Yt, Tt).numpy()
    diff_u = np.abs(U_pred - U_true)
    diff_v = np.abs(V_pred - V_true)

    if data_dir is None:
        data_dir = "./data"

    os.makedirs(data_dir, exist_ok=True)
    scipy.io.savemat(
        os.path.join(data_dir, f"trSQP_PINN_t{t_val:.2f}.mat"),
        {
            "x": x,
            "y": y,
            "t": t_val,
            "U_pred": U_pred,
            "V_pred": V_pred,
            "U_true": U_true,
            "V_true": V_true,
            "diff_u": diff_u,
            "diff_v": diff_v,
        },
    )

    # Additional AL-style artifacts requested by user.
    if fig_dir is not None:
        save_velocity_uv_error_artifacts(model, device, t_val, n_grid, data_dir, fig_dir)

    if plt is not None and fig_dir is not None:
        os.makedirs(fig_dir, exist_ok=True)
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))

        vmin_u = min(U_pred.min(), U_true.min())
        vmax_u = max(U_pred.max(), U_true.max())
        vmin_v = min(V_pred.min(), V_true.min())
        vmax_v = max(V_pred.max(), V_true.max())

        im00 = axes[0, 0].pcolormesh(X, Y, U_pred, cmap="jet", vmin=vmin_u, vmax=vmax_u, shading="auto")
        axes[0, 0].set_title("U pred")
        plt.colorbar(im00, ax=axes[0, 0])

        im01 = axes[0, 1].pcolormesh(X, Y, U_true, cmap="jet", vmin=vmin_u, vmax=vmax_u, shading="auto")
        axes[0, 1].set_title("U true")
        plt.colorbar(im01, ax=axes[0, 1])

        im02 = axes[0, 2].pcolormesh(X, Y, diff_u, cmap="Reds", shading="auto")
        axes[0, 2].set_title("|U error|")
        plt.colorbar(im02, ax=axes[0, 2])

        im10 = axes[1, 0].pcolormesh(X, Y, V_pred, cmap="jet", vmin=vmin_v, vmax=vmax_v, shading="auto")
        axes[1, 0].set_title("V pred")
        plt.colorbar(im10, ax=axes[1, 0])

        im11 = axes[1, 1].pcolormesh(X, Y, V_true, cmap="jet", vmin=vmin_v, vmax=vmax_v, shading="auto")
        axes[1, 1].set_title("V true")
        plt.colorbar(im11, ax=axes[1, 1])

        im12 = axes[1, 2].pcolormesh(X, Y, diff_v, cmap="Reds", shading="auto")
        axes[1, 2].set_title("|V error|")
        plt.colorbar(im12, ax=axes[1, 2])

        for ax in axes.flat:
            ax.set_xlabel("x")
            ax.set_ylabel("y")

        fig.suptitle(f"trSQP-PINN Navier-Stokes at t={t_val:.2f}")
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"trSQP_PINN_t{t_val:.2f}.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)


def plot_training_history(history, fig_dir="./figures"):
    if plt is None:
        print("matplotlib is not installed; skipping training-history plot.")
        return

    loss_res = np.array(history["f"])
    loss_ic = np.abs(np.array(history["ic"]))
    loss_bc = np.abs(np.array(history["bc"]))
    loss_total = loss_res + loss_ic + loss_bc

    plt.figure(figsize=(16, 4))

    plt.subplot(1, 4, 1)
    plt.plot(loss_total)
    plt.yscale("log")
    plt.title("Total trSQP Loss")

    plt.subplot(1, 4, 2)
    plt.plot(loss_res)
    plt.yscale("log")
    plt.title("Physics Loss")

    plt.subplot(1, 4, 3)
    plt.plot(loss_ic)
    plt.yscale("log")
    plt.title("Initial Loss")

    plt.subplot(1, 4, 4)
    plt.plot(loss_bc)
    plt.yscale("log")
    plt.title("Boundary Loss")

    plt.tight_layout()
    out_path = os.path.join(fig_dir, "trSQP_PINN_training_history.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_iter", type=int, default=None)
    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--pretrain", type=int, default=100)
    parser.add_argument("--tol", type=float, default=1e-7)
    parser.add_argument("--delta", type=float, default=1.0)
    parser.add_argument("--delta_min", type=float, default=1e-6)
    parser.add_argument("--delta_max", type=float, default=100.0)
    parser.add_argument("--eta_low", type=float, default=1e-4)
    parser.add_argument("--eta_upp", type=float, default=0.5)
    parser.add_argument("--rho_scale", type=float, default=2.0)
    parser.add_argument("--nu", type=float, default=0.8)
    parser.add_argument("--g_tol", type=float, default=1e-9)
    parser.add_argument("--f_tol", type=float, default=1e-9)
    parser.add_argument("--h_floor", type=float, default=1e-8)
    parser.add_argument("--h_cap", type=float, default=1e4)
    parser.add_argument("--ordinal", type=int, default=0)
    args = parser.parse_args()
    if args.max_iter is None:
        args.max_iter = args.epoch
    return args


if __name__ == "__main__":
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script. No GPU detected.")

    device = torch.device(f"cuda:{args.ordinal}")
    print(f"Using device: {device} ({torch.cuda.get_device_name(device)})")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    fig_dir = os.path.join(base_dir, "figures")
    train_dir = os.path.join(base_dir, "training_data")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)

    X_coll, X_ic, X_bx_l, X_bx_r, X_by_b, X_by_t = sample_training_data()
    pinn_ns = PhysicsInformedNS(X_coll, X_ic, X_bx_l, X_bx_r, X_by_b, X_by_t, device)

    pretrain_feasibility(
        pinn_ns,
        max_iter=args.pretrain,
        g_tol=args.g_tol,
        f_tol=args.f_tol,
    )

    t0 = time.time()
    _, history = tr_sqp_pinn(pinn_ns, args)
    t1 = time.time()

    final_res = float(objective_obs_loss(pinn_ns).item())
    final_ic = float(constraint_ic_loss(pinn_ns).item())
    final_bc = float(constraint_bc_loss(pinn_ns).item())

    print("\n--- Optimization Finished ---")
    print(f"Final Initial Loss:  {final_ic:.6e}")
    print(f"Final Boundary Loss: {final_bc:.6e}")
    print(f"Final Physics Loss:  {final_res:.6e}")
    print(f"Training time: {(t1 - t0):.2f} s")

    torch.save(pinn_ns.model.state_dict(), os.path.join(data_dir, "trSQP_PINN_navier_stokes.pth"))

    np.savez(
        os.path.join(train_dir, "trSQP_PINN_history.npz"),
        physics=np.array(history["f"]),
        initial=np.array(history["ic"]),
        boundary=np.array(history["bc"]),
        kkt=np.array(history["kkt"]),
        delta=np.array(history["delta"]),
        eta=np.array(history["eta"]),
        lambda_ic=np.array(history["lambda_ic"]),
        lambda_bc=np.array(history["lambda_bc"]),
    )

    plot_training_history(history, fig_dir)
    for t_save in [0.00, 0.50, 1.00]:
        save_plot_comparison_mat(
            pinn_ns.model,
            device,
            t_val=t_save,
            n_grid=80,
            fig_dir=fig_dir,
            data_dir=data_dir,
        )

    errs_u, errs_v, errs_p = [], [], []
    time_slices = np.linspace(0.0, 1.0, 1000)

    for t_slice in time_slices:
        eu, ev, ep = evaluate_l2_uvp(pinn_ns.model, device, t_eval=float(t_slice), n_grid=80)
        errs_u.append(eu)
        errs_v.append(ev)
        errs_p.append(ep)

    scipy.io.savemat(
        os.path.join(data_dir, "trSQP_PINN_navier_stokes_eval.mat"),
        {
            "time_slices": time_slices,
            "err_u": np.array(errs_u),
            "err_v": np.array(errs_v),
            "err_p": np.array(errs_p),
        },
    )

    print("\nMean over time slices:")
    print(f"L2(u) mean = {np.mean(errs_u):.6e}")
    print(f"L2(v) mean = {np.mean(errs_v):.6e}")
    print(f"L2(p) mean = {np.mean(errs_p):.6e}")
