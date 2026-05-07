#!/usr/bin/env python3
"""trSQP-PINN for FPDE forward problem.

Alternating inner-loop network optimization and trust-region updates
for augmented-Lagrangian multipliers enforcing IC/BC.
NN structure and sampling follow the FPDE `Adam-PINN` notebook.
"""
import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.io import savemat

parser = argparse.ArgumentParser()
parser.add_argument('--EPOCH', type=int, default=1000)
parser.add_argument('--inner', type=int, default=5)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--rho', type=float, default=1e-3)
parser.add_argument('--max_delta', type=float, default=0.1)
parser.add_argument('--beta', type=float, default=1.0)
parser.add_argument('--ordinal', type=int, default=0)
args = parser.parse_args()

device = torch.device(f"cuda:{args.ordinal}" if torch.cuda.is_available() else "cpu")
print('Device:', device)

# --- Grid and parameters (match Adam-PINN notebook) ---
M = 200
N = 200
h = np.pi / M
tau = 1.0 / N

# alpha array used in FPDE operator in the notebook
alpha = torch.linspace(0, 0.9, N, device=device).reshape(-1, 1)

# Create global input grid (time x space)
t = torch.linspace(0, 1, N, device=device)
x = torch.linspace(0, np.pi, M, device=device)
T, X = torch.meshgrid(t, x, indexing='ij')
x1 = X.flatten()[:, None]
t1 = T.flatten()[:, None]
X_input = torch.cat((x1, t1), dim=1).to(device)


def initial_condition(x):
    return torch.sin(x) * 0.0


def boundary_condition(t):
    return torch.zeros_like(t)


def u_true(x, t):
    # simple manufactured solution used in the notebook
    # return elementwise sin(x) * t^3 for inputs of matching shape
    return torch.sin(x) * (t ** 3)


def f_source(x, t, alpha_local):
    # match the notebook's source term definition
    t_ = t.reshape(-1, 1)
    x_ = x.reshape(1, -1)
    gamma_4 = torch.lgamma(torch.tensor(4.0, device=device)).exp()
    term = (gamma_4 / torch.lgamma(4 - alpha_local).exp()) * t_ ** (3 - alpha_local) * torch.sin(x_) + t_ ** 3 * torch.sin(x_)
    return term[1:, 1:-1]


def S(M_local):
    S = torch.diag(torch.full((M_local,), -2.0, dtype=torch.float, device=device))
    i, j = np.indices(S.shape)
    S[i == j - 1] = 1.0
    S[i == j + 1] = 1.0
    return S


class MLP(nn.Module):
    def __init__(self, layers):
        super(MLP, self).__init__()
        self.layers = layers
        self.activation = nn.ReLU()
        self.linear = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers) - 1)])
        for i in range(len(layers) - 1):
            nn.init.xavier_normal_(self.linear[i].weight.data, gain=1.0)
            nn.init.zeros_(self.linear[i].bias.data)

    def forward(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).to(device)
        a = self.activation(self.linear[0](x))
        for i in range(1, len(self.layers) - 2):
            z = self.linear[i](a)
            a = self.activation(z)
        a = self.linear[-1](a)
        return a


def FPDE(alpha_local, u_hat):
    # replicated from notebook (kept as-is to preserve behavior)
    alpha_1 = 1 - alpha_local
    i_minus_j = torch.arange(1, N, device=device).view(-1, 1) - torch.arange(0, N - 1, device=device).view(1, -1)
    A = (torch.tril(i_minus_j) ** alpha_1[1:]) - 2 * (torch.tril(i_minus_j - 1) ** alpha_1[1:]) + (torch.tril(i_minus_j - 2).fill_diagonal_(0) ** alpha_1[1:])
    A = A.fill_diagonal_(1)
    B = torch.matmul(A, u_hat[1:, :])
    i_minus_j_1 = torch.arange(1, N, device=device).reshape(-1, 1)
    c = (i_minus_j_1 ** alpha_1[1:]) - ((i_minus_j_1 - 1) ** alpha_1[1:])
    B = B - torch.matmul(c, u_hat[0, :].view(1, -1))
    a = tau ** (-alpha_local) / torch.lgamma(2 - alpha_local).exp()
    d = torch.mul(a[1:], B)
    return d[:, 1:-1]


def residual_loss(model, criterion):
    u_hat = model(X_input)
    u_hat = u_hat.reshape(N, M)
    u_nn_xx = h ** (-2) * torch.matmul(u_hat, S(M))[1:, 1:-1]
    loss = criterion(f_source(x, t, alpha), FPDE(alpha, u_hat) - u_nn_xx)
    return loss


def initial_loss(model, x_ic, t_ic, u_ic, criterion):
    ic_input = torch.cat((x_ic, t_ic), dim=-1)
    u_pred_ic = model(ic_input)
    return criterion(u_pred_ic, u_ic)


def boundary_loss(model, x_bc_0, x_bc_pi, t_bc, u_bc, criterion):
    bc_x0_input = torch.cat((x_bc_0, t_bc), dim=-1)
    bc_xpi_input = torch.cat((x_bc_pi, t_bc), dim=-1)
    u_pred_bc_0 = model(bc_x0_input)
    u_pred_bc_pi = model(bc_xpi_input)
    return criterion(u_pred_bc_0, u_bc) + criterion(u_pred_bc_pi, u_bc)


def train_trsqp_forward(model, epochs, inner_steps, rho, max_delta, beta, lr):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # AL multipliers for IC and BC
    lbd_ic = 0.0
    lbd_bc = 0.0

    os.makedirs('./training_logs', exist_ok=True)
    os.makedirs('./training_data', exist_ok=True)
    logfile = './training_logs/trSQP_FPDE_forward_loss.txt'
    with open(logfile, 'w') as flog:
        flog.write('epoch,loss_res,loss_ic,loss_bc,total,lbd_ic,lbd_bc,val_rel\n')

        # prepare fixed IC/BC samples like in notebook
        num_ic = 200
        num_bc = 200
        x_ic = torch.linspace(0, torch.pi, num_ic, device=device).view(-1, 1)
        t_ic = torch.zeros_like(x_ic, device=device)
        u_ic = initial_condition(x_ic)

        t_bc = torch.linspace(0, 1, num_bc, device=device).view(-1, 1)
        x_bc_0 = torch.zeros_like(t_bc, device=device)
        x_bc_pi = torch.full_like(t_bc, torch.pi, device=device)
        u_bc = boundary_condition(t_bc)

        best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
        val_errs = []
        start = time.time()

        for epoch in range(1, epochs + 1):
            # inner primal updates
            model.train()
            for _ in range(inner_steps):
                optimizer.zero_grad()
                loss_res = residual_loss(model, criterion)
                loss_ic = initial_loss(model, x_ic, t_ic, u_ic, criterion)
                loss_bc = boundary_loss(model, x_bc_0, x_bc_pi, t_bc, u_bc, criterion)

                # mean constraints
                # for IC, use mean error over IC points; for BC, mean of boundary predictions
                with torch.no_grad():
                    ic_input = torch.cat((x_ic, t_ic), dim=-1)
                    u_pred_ic = model(ic_input)
                    mean_ic = (u_pred_ic - u_ic).mean().item()
                    bc_x0_input = torch.cat((x_bc_0, t_bc), dim=-1)
                    bc_xpi_input = torch.cat((x_bc_pi, t_bc), dim=-1)
                    u_pred_bc_0 = model(bc_x0_input)
                    u_pred_bc_pi = model(bc_xpi_input)
                    mean_bc = torch.cat([u_pred_bc_0, u_pred_bc_pi], dim=0).mean().item()

                loss_total = loss_res + beta * loss_ic + beta * loss_bc + lbd_ic * mean_ic + lbd_bc * mean_bc
                loss_total.backward()
                optimizer.step()

            # outer update: trust-region update for multipliers
            # compute fresh means
            model.eval()
            with torch.no_grad():
                ic_input = torch.cat((x_ic, t_ic), dim=-1)
                u_pred_ic = model(ic_input)
                mean_ic = (u_pred_ic - u_ic).mean().item()
                bc_x0_input = torch.cat((x_bc_0, t_bc), dim=-1)
                bc_xpi_input = torch.cat((x_bc_pi, t_bc), dim=-1)
                u_pred_bc_0 = model(bc_x0_input)
                u_pred_bc_pi = model(bc_xpi_input)
                mean_bc = torch.cat([u_pred_bc_0, u_pred_bc_pi], dim=0).mean().item()

            # trust-region scaled increment
            delta_ic = rho * mean_ic
            delta_bc = rho * mean_bc
            # clamp increments
            delta_ic = max(min(delta_ic, max_delta), -max_delta)
            delta_bc = max(min(delta_bc, max_delta), -max_delta)
            lbd_ic += delta_ic
            lbd_bc += delta_bc

            # validation: compare to u_true on a coarse grid
            xg = torch.linspace(0, np.pi, 80, device=device)
            tg = torch.linspace(0, 1, 80, device=device)
            Tg, Xg = torch.meshgrid(tg, xg, indexing='ij')
            inp = torch.stack([Xg.flatten(), Tg.flatten()], dim=1)
            with torch.no_grad():
                pred = model(inp).reshape(Tg.shape)
                U_exact = u_true(Xg, Tg)
                diff = (pred - U_exact).reshape(-1)
                val_rel = torch.linalg.norm(diff, 2).item() / (torch.linalg.norm(U_exact.reshape(-1), 2).item() + 1e-16)

            val_errs.append(val_rel)
            if len(val_errs) == 1 or val_rel < min(val_errs):
                best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}

            if epoch % 50 == 0 or epoch == 1:
                elapsed = time.time() - start
                print(f"Epoch {epoch}/{epochs} | val_rel: {val_rel:.6e} | lbd_ic: {lbd_ic:.6e} | lbd_bc: {lbd_bc:.6e} | time: {elapsed:.1f}s")
                flog.write(f"{epoch},{float(loss_res):.6e},{float(loss_ic):.6e},{float(loss_bc):.6e},{float(loss_total):.6e},{lbd_ic:.6e},{lbd_bc:.6e},{val_rel:.6e}\n")

        print('Training finished')

    torch.save(best_state, './training_data/trSQP_FPDE_forward_model.pth')

    # final evaluation metrics
    # physics residual on coarse grid
    with torch.no_grad():
        u_hat = model(X_input).reshape(N, M)
        u_nn_xx = h ** (-2) * torch.matmul(u_hat, S(M))[1:, 1:-1]
        res = (FPDE(alpha, u_hat) - u_nn_xx).cpu().numpy()
        physics_mse = np.mean(res**2)

    # initial and boundary MSE
    with torch.no_grad():
        u_pred_ic = model(torch.cat((x_ic, t_ic), dim=-1))
        initial_mse = torch.mean((u_pred_ic - u_ic)**2).item()
        u_pred_bc_0 = model(torch.cat((x_bc_0, t_bc), dim=-1))
        u_pred_bc_pi = model(torch.cat((x_bc_pi, t_bc), dim=-1))
        boundary_mse = (torch.mean((u_pred_bc_0 - u_bc)**2) + torch.mean((u_pred_bc_pi - u_bc)**2)).item()

    # data MSE: use samples from u_true
    num_data = 1000
    x_data = torch.rand(num_data, 1, device=device) * np.pi
    t_data = torch.rand(num_data, 1, device=device)
    with torch.no_grad():
        u_pred_data = model(torch.cat((x_data, t_data), dim=-1)).cpu().numpy()
    u_true_data = u_true(x_data, t_data).cpu().numpy()
    data_mse = np.mean((u_pred_data - u_true_data)**2)
    data_rel = np.linalg.norm(u_pred_data - u_true_data) / (np.linalg.norm(u_true_data) + 1e-16)

    # full-grid MSE and relative L2
    with torch.no_grad():
        u_pred_enforced = model(X_input).cpu().numpy().reshape(N, M)
    U_exact_full = u_true(X, T).cpu().numpy()
    full_mse = np.mean((u_pred_enforced - U_exact_full)**2)
    rel_l2 = np.linalg.norm(u_pred_enforced - U_exact_full) / (np.linalg.norm(U_exact_full) + 1e-16)

    print(f"Final physics MSE: {physics_mse:.6e}")
    print(f"Final initial MSE: {initial_mse:.6e}")
    print(f"Final boundary MSE: {boundary_mse:.6e}")
    print(f"Final data MSE: {data_mse:.6e}")
    print(f"Final full-grid MSE: {full_mse:.6e}")
    print(f"Final relative L2 error: {rel_l2:.6e}")
    print(f"Final data relative L2 error: {data_rel:.6e}")

    savemat('./training_data/trSQP_FPDE_forward_metrics.mat', {
        'physics_mse': physics_mse,
        'initial_mse': initial_mse,
        'boundary_mse': boundary_mse,
        'data_mse': data_mse,
        'full_mse': full_mse,
        'rel_l2': rel_l2,
        'data_rel_l2': data_rel,
        'lbd_ic': lbd_ic,
        'lbd_bc': lbd_bc
    })

    os.makedirs('./data', exist_ok=True)
    x_values = x.detach().cpu().numpy()
    t_values = t.detach().cpu().numpy()
    savemat('./data/trSQP-PINN_solution.mat', {
        'u_pred_enforced': u_pred_enforced,
        'x': x_values,
        't': t_values,
        'u_true': U_exact_full
    })
    savemat('./data/trSQP-PINN_error.mat', {
        'u_pred_enforced': np.abs(u_pred_enforced - U_exact_full),
        'x': x_values,
        't': t_values,
        'u_true': U_exact_full
    })
    print('Saved trSQP forward FPDE model and metrics')
    print('Saved solution and error mats to ./data')


if __name__ == '__main__':
    layers = [2, 50, 100, 1]
    model = MLP(layers).to(device)
    train_trsqp_forward(model, args.EPOCH, args.inner, args.rho, args.max_delta, args.beta, args.lr)
