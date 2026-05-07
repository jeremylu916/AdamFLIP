#!/usr/bin/env python3
"""trSQP-PINN for FPDE inverse problem.

Uses a Burgers-style trSQP core:
- objective: observation/data loss
- constraints: physics residual and (IC+BC)
- trust-region step with damped BFGS Hessian updates
"""
import os
import time
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from scipy.io import savemat
from tqdm import trange

parser = argparse.ArgumentParser()
parser.add_argument('--EPOCH', type=int, default=1000)
parser.add_argument('--inner', type=int, default=5)  # kept for CLI compatibility
parser.add_argument('--lr', type=float, default=1e-3)  # kept for CLI compatibility
parser.add_argument('--rho', type=float, default=1e-3)  # kept for CLI compatibility
parser.add_argument('--max_delta', type=float, default=0.1)  # kept for CLI compatibility
parser.add_argument('--beta', type=float, default=1.0)  # kept for CLI compatibility
parser.add_argument('--ordinal', type=int, default=0)
args = parser.parse_args()

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

device = torch.device(f"cuda:{args.ordinal}" if torch.cuda.is_available() else "cpu")
print('Device:', device)

# --- Grid and parameters ---
M = 200
N = 200
h = np.pi / M
tau = 1.0 / N

# Create global input grid (time x space)
t = torch.linspace(0, 1, N, device=device)
x = torch.linspace(0, np.pi, M, device=device)
T, X = torch.meshgrid(t, x, indexing='ij')
x1 = X.flatten()[:, None]
t1 = T.flatten()[:, None]
X_input = torch.cat((x1, t1), dim=1).to(device)


def initial_condition(x_in):
    return torch.sin(x_in) * 0.0


def boundary_condition(t_in):
    return torch.zeros_like(t_in)


def u_true(x_in, t_in):
    return torch.sin(x_in) * (t_in ** 3)


u_truth = u_true(X, T).reshape(-1, 1).to(device)
obs_noise = 0.01 * torch.std(u_truth)
u_truth_noisy = u_truth + torch.randn_like(u_truth) * obs_noise


def f_source(x_in, t_in, alpha_local):
    t_ = t_in.reshape(-1, 1)
    x_ = x_in.reshape(1, -1)
    gamma_4 = torch.lgamma(torch.tensor(4.0, device=device)).exp()
    term = (gamma_4 / torch.lgamma(4 - alpha_local).exp()) * (t_ ** (3 - alpha_local)) * torch.sin(x_) + (t_ ** 3) * torch.sin(x_)
    return term[1:, 1:-1]


def S(m_local):
    s_mat = torch.diag(torch.full((m_local,), -2.0, dtype=torch.float32, device=device))
    off_diag = torch.ones(m_local - 1, dtype=torch.float32, device=device)
    s_mat += torch.diag(off_diag, diagonal=1)
    s_mat += torch.diag(off_diag, diagonal=-1)
    return s_mat


S_M = S(M)

# Cache IC/BC points once to avoid reallocation every iteration
NUM_IC = 200
NUM_BC = 200
x_ic = torch.linspace(0, torch.pi, NUM_IC, device=device).view(-1, 1)
t_ic = torch.zeros_like(x_ic, device=device)
u_ic = initial_condition(x_ic)
t_bc = torch.linspace(0, 1, NUM_BC, device=device).view(-1, 1)
x_bc_0 = torch.zeros_like(t_bc, device=device)
x_bc_pi = torch.full_like(t_bc, torch.pi, device=device)
u_bc = boundary_condition(t_bc)


class MLP(nn.Module):
    def __init__(self, layers):
        super(MLP, self).__init__()
        self.layers = layers
        self.activation = nn.ReLU()
        self.linear = nn.ModuleList([nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)])
        self.alpha = nn.Parameter(torch.tensor(1.0, dtype=torch.float32, device=device))
        for i in range(len(layers) - 1):
            nn.init.xavier_normal_(self.linear[i].weight.data, gain=1.0)
            nn.init.zeros_(self.linear[i].bias.data)

    def forward(self, inp):
        if not torch.is_tensor(inp):
            inp = torch.from_numpy(inp).to(device)
        a = self.activation(self.linear[0](inp))
        for i in range(1, len(self.layers) - 2):
            z = self.linear[i](a)
            a = self.activation(z)
        a = self.linear[-1](a)
        return a


def FPDE(alpha_local, u_hat):
    alpha_1 = 1 - alpha_local
    i_minus_j = torch.arange(1, N, device=device).view(-1, 1) - torch.arange(0, N - 1, device=device).view(1, -1)
    a_mat = (torch.tril(i_minus_j) ** alpha_1) - 2 * (torch.tril(i_minus_j - 1) ** alpha_1) + (torch.tril(i_minus_j - 2).fill_diagonal_(0) ** alpha_1)
    a_mat = a_mat.fill_diagonal_(1)
    b_vec = torch.matmul(a_mat, u_hat[1:, :])
    i_minus_j_1 = torch.arange(1, N, device=device).reshape(-1, 1)
    c_vec = (i_minus_j_1 ** alpha_1) - ((i_minus_j_1 - 1) ** alpha_1)
    b_vec = b_vec - torch.matmul(c_vec, u_hat[0, :].view(1, -1))
    a_scale = tau ** (-alpha_local) / torch.lgamma(2 - alpha_local).exp()
    d_vec = torch.mul(a_scale, b_vec)
    return d_vec[:, 1:-1]


def residual_loss(model):
    u_hat = model(X_input).reshape(N, M)
    u_nn_xx = h ** (-2) * torch.matmul(u_hat, S_M)[1:, 1:-1]
    return nn.MSELoss()(f_source(x, t, model.alpha), FPDE(model.alpha, u_hat) - u_nn_xx)


def initial_loss(model):
    ic_input = torch.cat((x_ic, t_ic), dim=-1)
    u_pred_ic = model(ic_input)
    return nn.MSELoss()(u_pred_ic, u_ic)


def boundary_loss(model):
    bc_x0_input = torch.cat((x_bc_0, t_bc), dim=-1)
    bc_xpi_input = torch.cat((x_bc_pi, t_bc), dim=-1)
    u_pred_bc_0 = model(bc_x0_input)
    u_pred_bc_pi = model(bc_xpi_input)
    return nn.MSELoss()(u_pred_bc_0, u_bc) + nn.MSELoss()(u_pred_bc_pi, u_bc)


def obs_loss(model):
    pred = model(X_input)
    return nn.MSELoss()(pred, u_truth_noisy)


def get_flat_params(model):
    return parameters_to_vector(model.parameters()).detach().clone().to(device)


def set_flat_params(model, flat_params):
    with torch.no_grad():
        vector_to_parameters(flat_params, model.parameters())


def get_grads_from_loss(loss_value, params):
    grads = torch.autograd.grad(loss_value, params, retain_graph=True, allow_unused=True)
    flat = []
    for g, p in zip(grads, params):
        if g is None:
            flat.append(torch.zeros(p.numel(), dtype=torch.float32, device=device))
        else:
            flat.append(g.detach().reshape(-1))
    return torch.cat(flat)


def trSQP_train(model, max_iter=1000):
    params = list(model.parameters())
    theta = get_flat_params(model)
    n = theta.numel()
    # Diagonal quasi-Newton approximation (much cheaper than full dense H)
    h_diag = torch.ones(n, dtype=torch.float32, device=device)
    I_2 = torch.eye(2, dtype=torch.float32, device=device)

    mu_prev = 1.0
    Delta = 1.0
    eta_low = 1e-12
    eta_upp = 0.3
    rho_scale = 2.0
    f_tol = 1e-12
    g_tol = 1e-8
    min_stop_iter = min(1000, max_iter)

    os.makedirs('./training_logs', exist_ok=True)
    os.makedirs('./training_data', exist_ok=True)
    logp = './training_logs/trSQP_FPDE_inverse_loss.txt'

    with open(logp, 'w') as flog:
        flog.write('iter,loss_obs,loss_pde,loss_icbc,Delta,eta,alpha\n')
        pbar = trange(max_iter, desc='trSQP-PINN')

        for k in pbar:
            set_flat_params(model, theta)

            loss_d = obs_loss(model)
            loss_c1 = residual_loss(model)
            loss_ic = initial_loss(model)
            loss_bc = boundary_loss(model)
            loss_c2 = loss_ic + loss_bc

            f_val = loss_d.item()
            h_vec = torch.tensor([loss_c1.item(), loss_c2.item()], dtype=torch.float32, device=device)

            grad_f = get_grads_from_loss(loss_d, params)
            J1 = get_grads_from_loss(loss_c1, params)
            J2 = get_grads_from_loss(loss_c2, params)
            J = torch.stack([J1, J2], dim=0)

            JJt = J @ J.T
            eps_reg = 1e-8
            try:
                lambda_k = -torch.linalg.solve(JJt + eps_reg * I_2, h_vec)
            except RuntimeError:
                lambda_k = -(torch.linalg.pinv(JJt + eps_reg * I_2) @ h_vec)

            grad_L = grad_f + J.T @ lambda_k

            reg = 1e-6
            d_newton = -grad_L / (h_diag + reg)

            norm_d = torch.linalg.norm(d_newton).item()
            if norm_d <= Delta:
                d_k = d_newton
            else:
                d_k = d_newton * (Delta / (norm_d + 1e-16))

            Pred_k = -(torch.dot(grad_L, d_k) + 0.5 * torch.sum(h_diag * d_k * d_k)).item()

            denom = 0.7 * (torch.linalg.norm(h_vec).item() - torch.linalg.norm(h_vec + J @ d_k).item())
            if denom == 0:
                mu_k = mu_prev
            else:
                mu_k = max(mu_prev, (torch.dot(grad_L, d_k) + 0.5 * torch.sum(h_diag * d_k * d_k)).item() / denom)

            theta_trial = theta + d_k
            set_flat_params(model, theta_trial)
            f_trial = obs_loss(model).item()
            Ared_k = f_val - f_trial

            eta_k = -np.inf if Pred_k <= 0 else Ared_k / Pred_k

            if eta_k >= eta_upp:
                theta_new = theta + d_k
                Delta = rho_scale * Delta

                set_flat_params(model, theta_new)
                loss_d_new = obs_loss(model)
                loss_c1_new = residual_loss(model)
                loss_c2_new = initial_loss(model) + boundary_loss(model)

                grad_f_new = get_grads_from_loss(loss_d_new, params)
                J1_new = get_grads_from_loss(loss_c1_new, params)
                J2_new = get_grads_from_loss(loss_c2_new, params)
                J_new = torch.stack([J1_new, J2_new], dim=0)
                h_new = torch.tensor([loss_c1_new.item(), loss_c2_new.item()], dtype=torch.float32, device=device)

                JJt_new = J_new @ J_new.T
                try:
                    lambda_new = -torch.linalg.solve(JJt_new + eps_reg * I_2, h_new)
                except RuntimeError:
                    lambda_new = -(torch.linalg.pinv(JJt_new + eps_reg * I_2) @ h_new)

                grad_L_new = grad_f_new + J_new.T @ lambda_new
                s = theta_new - theta
                y = grad_L_new - grad_L
                sHs = torch.sum(h_diag * s * s)

                if torch.dot(s, y) >= 0.2 * sHs:
                    y_bar = y
                else:
                    theta_scale = (0.8 * sHs) / (sHs - torch.dot(s, y) + 1e-16)
                    y_bar = theta_scale * y + (1 - theta_scale) * (h_diag * s)

                # Diagonal damped BFGS-like update (stable, GPU-friendly)
                secant = y_bar / (s + 1e-12)
                valid = torch.isfinite(secant) & (secant > 1e-8) & (torch.abs(s) > 1e-10)
                secant_clamped = torch.clamp(secant, 1e-8, 1e4)
                h_diag = torch.where(valid, 0.9 * h_diag + 0.1 * secant_clamped, h_diag)

                theta = theta_new
                mu_prev = mu_k
            elif eta_k >= eta_low:
                theta = theta + d_k
                mu_prev = mu_k
            else:
                Delta = Delta / rho_scale

            if (k + 1) >= min_stop_iter and eta_k < eta_low and (torch.linalg.norm(d_k).item() <= f_tol or torch.max(torch.abs(grad_L)).item() <= g_tol):
                print(f'Converged at iter {k}')
                break

            if k % 10 == 0:
                pbar.set_postfix({
                    'f': f'{f_val:.2e}',
                    'c1': f'{h_vec[0].item():.2e}',
                    'c2': f'{h_vec[1].item():.2e}',
                    'Delta': f'{Delta:.2e}',
                    'eta': f'{eta_k:.2e}',
                    'alpha': f'{model.alpha.item():.4f}'
                })
                flog.write(f"{k},{f_val:.6e},{h_vec[0].item():.6e},{h_vec[1].item():.6e},{Delta:.6e},{eta_k:.6e},{model.alpha.item():.6e}\\n")

    set_flat_params(model, theta)
    return model


if __name__ == '__main__':
    layers = [2, 50, 100, 1]
    model = MLP(layers).to(device)

    start = time.time()
    trained = trSQP_train(model, max_iter=args.EPOCH)
    elapsed = time.time() - start
    print(f'Training finished in {elapsed:.1f}s')

    os.makedirs('./training_data', exist_ok=True)
    torch.save({k: v.clone().cpu() for k, v in trained.state_dict().items()}, './training_data/trSQP_FPDE_inverse_model.pth')

    with torch.no_grad():
        pred = trained(X_input)
        u_hat = pred.reshape(N, M)
        u_nn_xx = h ** (-2) * torch.matmul(u_hat, S_M)[1:, 1:-1]
        res = (FPDE(trained.alpha, u_hat) - u_nn_xx).cpu().numpy()
        physics_mse = np.mean(res ** 2)

        init_mse = initial_loss(trained).item()
        bc_mse = boundary_loss(trained).item()
        data_mse = obs_loss(trained).item()
        mse_full = nn.MSELoss()(pred, u_truth).item()
        rel_l2 = (torch.linalg.norm(pred - u_truth) / (torch.linalg.norm(u_truth) + 1e-16)).item()

    print(f'Final physics MSE: {physics_mse:.6e}')
    print(f'Final initial MSE: {init_mse:.6e}')
    print(f'Final boundary MSE: {bc_mse:.6e}')
    print(f'Final data MSE: {data_mse:.6e}')
    print(f'Final full-grid MSE: {mse_full:.6e}')
    print(f'Final relative L2 error: {rel_l2:.6e}')
    print(f'Learned alpha: {trained.alpha.item():.6e}')

    savemat('./training_data/trSQP_FPDE_inverse_metrics.mat', {
        'physics_mse': physics_mse,
        'initial_mse': init_mse,
        'boundary_mse': bc_mse,
        'data_mse': data_mse,
        'full_mse': mse_full,
        'rel_l2': rel_l2,
        'alpha_learned': trained.alpha.item()
    })

    os.makedirs('./data', exist_ok=True)
    with torch.no_grad():
        u_pred_enforced = trained(X_input).detach().cpu().numpy().reshape(N, M)
    u_truth_grid = u_true(X, T).detach().cpu().numpy()
    x_values = x.detach().cpu().numpy()
    t_values = t.detach().cpu().numpy()

    savemat('./data/trSQP-PINN_solution.mat', {
        'u_pred_enforced': u_pred_enforced,
        'x': x_values,
        't': t_values,
        'u_true': u_truth_grid
    })
    savemat('./data/trSQP-PINN_error.mat', {
        'u_pred_enforced': np.abs(u_pred_enforced - u_truth_grid),
        'x': x_values,
        't': t_values,
        'u_true': u_truth_grid
    })
    print('Saved trSQP FPDE inverse model, metrics, solution, and error mats')
