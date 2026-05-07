import os
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.io import savemat

# AL-PINN for FPDE inverse problem
parser = argparse.ArgumentParser()
parser.add_argument('--EPOCH', type=int, default=2000)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--rho', type=float, default=1e-3)
parser.add_argument('--beta', type=float, default=1.0)
parser.add_argument('--ordinal', type=int, default=0)
args = parser.parse_args()

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

device = torch.device(f"cuda:{args.ordinal}" if torch.cuda.is_available() else "cpu")
print('Device:', device)

# --- Grid and parameters (match Adam-PINN.py in FPDE/inverse) ---
M = 100
N = 100
h = np.pi / M
tau = 1 / N

# Create grid
t = torch.linspace(0, 1, N, device=device)
x = torch.linspace(0, np.pi, M, device=device)
T, X = torch.meshgrid(t, x, indexing='ij')
x1 = X.flatten()[:, None]
t1 = T.flatten()[:, None]
X_input = torch.cat((x1, t1), dim=1).to(device)

# Conditions and sampling
def initial_condition(x):
    return torch.sin(x) * 0

def boundary_condition(t):
    return torch.zeros_like(t)

num_ic = 100
num_bc = 100
num_pde = 5000

x_ic = torch.linspace(0, torch.pi, num_ic, device=device).view(-1, 1)
t_ic = torch.zeros_like(x_ic, device=device)
u_ic = initial_condition(x_ic)

t_bc = torch.linspace(0, 1, num_bc, device=device).view(-1, 1)
x_bc_0 = torch.zeros_like(t_bc, device=device)
x_bc_pi = torch.full_like(t_bc, torch.pi, device=device)
u_bc = boundary_condition(t_bc)

x_pde = (torch.rand(num_pde, 1, device=device) * torch.pi)
t_pde = torch.rand(num_pde, 1, device=device)

# True function and noisy observations
def u_true(x, t):
    return torch.sin(x) * t ** 3

u_truth = u_true(X, T).reshape(-1, 1).to(device)
obs_noise = 0.01 * torch.std(u_truth)
u_truth_noisy = u_truth + torch.randn_like(u_truth) * obs_noise


def S(M_local):
    S = torch.diag(torch.full((M_local,), -2, dtype=torch.float, device=device))
    i, j = np.indices(S.shape)
    S[i == j - 1] = 1
    S[i == j + 1] = 1
    return S


class MLP(nn.Module):
    def __init__(self, layers):
        super(MLP, self).__init__()
        self.layers = layers
        self.activation = nn.ReLU()
        self.linear = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers) - 1)])
        self.alpha = nn.Parameter(torch.tensor(1.0, dtype=torch.float32, device=device))
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


def FPDE(alpha, u_hat):
    alpha_1 = 1 - alpha
    i_minus_j = torch.arange(1, N, device=device).view(-1, 1) - torch.arange(0, N - 1, device=device).view(1, -1)
    A = torch.tril(i_minus_j) ** alpha_1 - 2 * torch.tril(i_minus_j - 1) ** alpha_1 + (torch.tril(i_minus_j - 2).fill_diagonal_(0) ** alpha_1)
    A = A.fill_diagonal_(1)
    B = torch.matmul(A, u_hat[1:, :])
    i_minus_j_1 = torch.arange(1, N, device=device).reshape(-1, 1)
    c = (i_minus_j_1 ** alpha_1) - ((i_minus_j_1 - 1) ** alpha_1)
    B = B - torch.matmul(c, u_hat[0, :].view(1, -1))
    a = tau ** (-alpha) / torch.lgamma(2 - alpha).exp()
    d = torch.mul(a, B)
    return d[:, 1:-1]


def residual_loss(model):
    u_hat = model(X_input)
    u_hat = u_hat.reshape(N, M)
    u_nn_xx = h ** (-2) * torch.matmul(u_hat, S(M))[1:, 1:-1]
    loss = nn.MSELoss()(FPDE(model.alpha, u_hat) - u_nn_xx, torch.zeros_like(u_nn_xx))
    return loss


def obs_loss(model):
    predicted = model(X_input)
    return nn.MSELoss()(predicted, u_truth)


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


def train_al(model, epochs, lr, rho, beta):
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # explicit AL multipliers
    lbd_ic = 0.0
    lbd_bc = 0.0

    os.makedirs('./training_logs', exist_ok=True)
    os.makedirs('./training_data', exist_ok=True)
    logfile = './training_logs/AL_FPDE_inverse_loss.txt'
    with open(logfile, 'w') as flog:
        flog.write('epoch,loss_pde,loss_obs,loss_ic,loss_bc,total,lbd_ic,lbd_bc,val_rel\n')

        best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
        val_errs = []
        start = time.time()

        for epoch in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad()
            loss_pde = residual_loss(model)
            loss_obs = obs_loss(model)
            loss_ic = initial_loss(model)
            loss_bc = boundary_loss(model)

            # means for AL
            with torch.no_grad():
                mean_ic = (model(torch.cat((x_ic, t_ic), dim=-1)) - u_ic).mean().item()
                bc_pred = torch.cat([model(torch.cat((x_bc_0, t_bc), dim=-1)), model(torch.cat((x_bc_pi, t_bc), dim=-1))], dim=0)
                mean_bc = bc_pred.mean().item()

            total = loss_pde + loss_obs + beta * loss_ic + beta * loss_bc + lbd_ic * mean_ic + lbd_bc * mean_bc
            total.backward()
            optimizer.step()

            # dual ascent updates (explicit)
            lbd_ic = float(np.clip(lbd_ic + rho * mean_ic, -1e6, 1e6))
            lbd_bc = float(np.clip(lbd_bc + rho * mean_bc, -1e6, 1e6))

            # validation
            model.eval()
            with torch.no_grad():
                pred = model(X_input)
                diff = (pred - u_truth).reshape(-1)
                val_rel = torch.linalg.norm(diff, 2).item() / (torch.linalg.norm(u_truth.reshape(-1), 2).item() + 1e-16)

            val_errs.append(val_rel)
            if len(val_errs) == 1 or val_rel < min(val_errs):
                best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}

            if epoch % 50 == 0 or epoch == 1:
                elapsed = time.time() - start
                print(f"Epoch {epoch}/{epochs} | val_rel: {val_rel:.6e} | lbd_ic: {lbd_ic:.6e} | lbd_bc: {lbd_bc:.6e} | time: {elapsed:.1f}s")
                flog.write(f"{epoch},{float(loss_pde):.6e},{float(loss_obs):.6e},{float(loss_ic):.6e},{float(loss_bc):.6e},{float(total):.6e},{lbd_ic:.6e},{lbd_bc:.6e},{val_rel:.6e}\n")

        print('Training finished')

    torch.save(best_state, './training_data/AL_FPDE_inverse_model.pth')

    # final metrics
    with torch.no_grad():
        pred = model(X_input)
        physics_loss = residual_loss(model).item()
        init_loss = initial_loss(model).item()
        bd_loss = boundary_loss(model).item()
        data_loss = obs_loss(model).item()
        mse_full = nn.MSELoss()(pred, u_truth).item()
        rel = torch.linalg.norm(pred - u_truth) / (torch.linalg.norm(u_truth) + 1e-16)

    print(f"Final physics MSE: {physics_loss:.6e}")
    print(f"Final initial MSE: {init_loss:.6e}")
    print(f"Final boundary MSE: {bd_loss:.6e}")
    print(f"Final data MSE: {data_loss:.6e}")
    print(f"Final full-grid MSE: {mse_full:.6e}")
    print(f"Final relative L2 error: {rel:.6e}")
    print(f"Learned alpha: {model.alpha.item():.6e}")

    savemat('./training_data/AL_FPDE_inverse_metrics.mat', {
        'physics_mse': physics_loss,
        'initial_mse': init_loss,
        'boundary_mse': bd_loss,
        'data_mse': data_loss,
        'full_mse': mse_full,
        'rel_l2': rel.item() if isinstance(rel, torch.Tensor) else float(rel),
        'alpha_learned': model.alpha.item()
    })

    os.makedirs('./data', exist_ok=True)
    with torch.no_grad():
        u_pred_enforced = model(X_input).detach().cpu().numpy().reshape(N, M)
    u_truth_grid = u_true(X, T).detach().cpu().numpy()
    x_values = x.detach().cpu().numpy()
    t_values = t.detach().cpu().numpy()

    savemat('./data/AL-PINN_solution.mat', {
        'u_pred_enforced': u_pred_enforced,
        'x': x_values,
        't': t_values,
        'u_true': u_truth_grid
    })
    savemat('./data/AL-PINN_error.mat', {
        'u_pred_enforced': np.abs(u_pred_enforced - u_truth_grid),
        'x': x_values,
        't': t_values,
        'u_true': u_truth_grid
    })
    print('Saved solution and error mats to ./data')


if __name__ == '__main__':
    layers = [2, 50, 100, 1]
    model = MLP(layers).to(device)
    train_al(model, args.EPOCH, args.lr, args.rho, args.beta)
