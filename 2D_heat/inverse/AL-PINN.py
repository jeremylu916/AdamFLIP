import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import random
import time
from scipy.io import savemat
from tqdm import trange

# Reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

ALPHA_TRUE = 0.1


class PINN(nn.Module):
    def __init__(self, initial_alpha_guess=1.0):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.alpha = nn.Parameter(torch.tensor([initial_alpha_guess], dtype=torch.float32, device=device))

    def forward(self, x):
        return self.net(x)


def initial_condition(x, y):
    return torch.sin(torch.pi * x) * torch.sin(torch.pi * y)


def analytical_solution(x, y, t, alpha):
    return torch.sin(torch.pi * x) * torch.sin(torch.pi * y) * torch.exp(-2 * torch.pi**2 * alpha * t)


def pde(x, y, t, model, learned_alpha):
    input_data = torch.cat([x, y, t], dim=1)
    u = model(input_data)
    grads = torch.autograd.grad(u, [x, y, t], grad_outputs=torch.ones_like(u), create_graph=True)
    u_x, u_y, u_t = grads[0], grads[1], grads[2]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
    return u_t - learned_alpha * (u_xx + u_yy)


def generate_training_data(num_points_pde, num_points_bc, num_points_ic, num_points_data):
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

    # sensor data sampled from true solution
    x_data = torch.rand(num_points_data, 1, device=device)
    y_data = torch.rand(num_points_data, 1, device=device)
    t_data = torch.rand(num_points_data, 1, device=device)
    u_data_exact = analytical_solution(x_data, y_data, t_data, ALPHA_TRUE)

    return (x_pde, y_pde, t_pde,
            x_ic, y_ic, t_ic, u_ic_exact,
            x_bc, y_bc, t_bc, u_bc_exact,
            x_data, y_data, t_data, u_data_exact)


def save_solution_comparison(model, t_value):
    model.eval()
    x = torch.linspace(0, 1, 100, device=device)
    y = torch.linspace(0, 1, 100, device=device)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    T = torch.full(X.shape, t_value, device=device)
    input_tensor = torch.stack([X.flatten(), Y.flatten(), T.flatten()], dim=1)

    with torch.no_grad():
        U_pred = model(input_tensor).reshape(X.shape)

    exp_term = np.exp(-2 * np.pi**2 * ALPHA_TRUE * t_value)
    U_exact = exp_term * torch.sin(np.pi * X.cpu()) * torch.sin(np.pi * Y.cpu())
    U_exact = U_exact.to(device)
    Error = torch.abs(U_pred - U_exact)

    X_np = X.cpu().numpy()
    Y_np = Y.cpu().numpy()
    U_pred_np = U_pred.cpu().numpy()
    U_exact_np = U_exact.cpu().numpy()
    Error_np = Error.cpu().numpy()

    os.makedirs("./data", exist_ok=True)
    savemat(f"./data/AL_solution_t{t_value}.mat", {
        "X": X_np,
        "Y": Y_np,
        "U_pred": U_pred_np,
        "U_exact": U_exact_np,
        "Error": Error_np
    })
    print(f"Successfully saved data to ./data/AL_solution_t{t_value}.mat")


def train_al_inverse(model, num_iterations, num_points_pde, num_points_bc, num_points_ic, num_points_data,
                     beta=1.0, lr=5e-3, rho=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()

    # explicit dual multipliers (scalars)
    lbd_ic = torch.tensor(0.0, device=device)
    lbd_bc = torch.tensor(0.0, device=device)

    os.makedirs('./training_logs', exist_ok=True)
    os.makedirs('./training_data', exist_ok=True)
    logpath = './training_logs/AL_INV_PINN_loss.txt'
    with open(logpath, 'w') as flog:
        flog.write('epoch,loss_pde,loss_ic,loss_bc,loss_data,total,alpha,lbd_ic,lbd_bc\n')

        start = time.time()
        pbar = trange(num_iterations, desc="AL-PINN")
        for it in pbar:
            optimizer.zero_grad()
            (x_pde, y_pde, t_pde,
             x_ic, y_ic, t_ic, u_ic_exact,
             x_bc, y_bc, t_bc, u_bc_exact,
             x_data, y_data, t_data, u_data_exact) = generate_training_data(num_points_pde, num_points_bc, num_points_ic, num_points_data)

            learned_alpha = model.alpha

            u_pred_ic = model(torch.cat([x_ic, y_ic, t_ic], dim=1))
            loss_ic = mse(u_pred_ic, u_ic_exact)

            u_pred_bc = model(torch.cat([x_bc, y_bc, t_bc], dim=1))
            loss_bc = mse(u_pred_bc, u_bc_exact)

            residual = pde(x_pde, y_pde, t_pde, model, learned_alpha)
            loss_pde = mse(residual, torch.zeros_like(residual))

            u_pred_data = model(torch.cat([x_data, y_data, t_data], dim=1))
            loss_data = mse(u_pred_data, u_data_exact)

            mean_ic = (u_pred_ic - u_ic_exact).mean()
            mean_bc = u_pred_bc.mean()

            total = loss_pde + loss_data + beta * loss_ic + (lbd_ic * mean_ic) + beta * loss_bc + (lbd_bc * mean_bc)

            total.backward()
            optimizer.step()

            # explicit dual ascent
            with torch.no_grad():
                lbd_ic = torch.clamp(lbd_ic + rho * mean_ic.detach(), -1e3, 1e3)
                lbd_bc = torch.clamp(lbd_bc + rho * mean_bc.detach(), -1e3, 1e3)

            if it % 100 == 0:
                elapsed = time.time() - start
                print(f"Iter: {it:5d}, Total: {total.item():.4e}, pde: {loss_pde.item():.4e}, data: {loss_data.item():.4e}, ic: {loss_ic.item():.4e}, bc: {loss_bc.item():.4e}, alpha: {learned_alpha.item():.6f}, lbd_ic: {float(lbd_ic):.6e}, lbd_bc: {float(lbd_bc):.6e}, time: {elapsed:.2f}s")
                flog.write(f"{it},{loss_pde.item():.6e},{loss_ic.item():.6e},{loss_bc.item():.6e},{loss_data.item():.6e},{total.item():.6e},{learned_alpha.item():.6e},{float(lbd_ic):.6e},{float(lbd_bc):.6e}\n")

            pbar.set_postfix({
                "tot": f"{total.item():.2e}",
                "pde": f"{loss_pde.item():.2e}",
                "data": f"{loss_data.item():.2e}",
                "alpha": f"{learned_alpha.item():.4f}"
            })

    # save model and metrics
    torch.save({k: v.clone().cpu() for k, v in model.state_dict().items()}, './training_data/AL_INV_PINN_2D_model.pth')

    # evaluation on full grid
    x = torch.linspace(0, 1, 100, device=device)
    y = torch.linspace(0, 1, 100, device=device)
    t = torch.linspace(0, 1, 100, device=device)
    X, Y, T = torch.meshgrid(x, y, t, indexing='ij')
    inp = torch.stack([X.flatten(), Y.flatten(), T.flatten()], dim=1)

    with torch.no_grad():
        U_pred = model(inp).reshape(X.shape)

    # physics residual on full grid
    x_flat = X.reshape(-1,1).to(device).requires_grad_(True)
    y_flat = Y.reshape(-1,1).to(device).requires_grad_(True)
    t_flat = T.reshape(-1,1).to(device).requires_grad_(True)
    f_full = pde(x_flat, y_flat, t_flat, model, model.alpha).detach().cpu().numpy()
    physics_test_loss = np.mean(f_full**2)

    # initial and boundary losses
    alpha_val = model.alpha.item()
    exp_term = np.exp(-2 * np.pi**2 * ALPHA_TRUE * T.cpu().numpy())
    U_exact = exp_term * np.sin(np.pi * X.cpu().numpy()) * np.sin(np.pi * Y.cpu().numpy())
    U_pred_np = U_pred.cpu().numpy()
    # initial (t==0)
    ini_mask = np.isclose(T.cpu().numpy(), 0.0)
    initial_test_loss = np.mean((U_pred_np[ini_mask] - U_exact[ini_mask])**2)
    # boundary mask
    bdry_mask = (np.isclose(X.cpu().numpy(), 0.0) | np.isclose(X.cpu().numpy(), 1.0) | np.isclose(Y.cpu().numpy(), 0.0) | np.isclose(Y.cpu().numpy(), 1.0))
    boundary_test_loss = np.mean((U_pred_np[bdry_mask] - U_exact[bdry_mask])**2)

    mse = np.mean((U_pred_np - U_exact)**2)
    rel = np.linalg.norm(U_pred_np - U_exact) / (np.linalg.norm(U_exact) + 1e-16)

    # data loss on sensor points
    # regenerate sensor grid (same sampling) deterministically
    x_data = torch.rand(num_points_data, 1, device=device)
    y_data = torch.rand(num_points_data, 1, device=device)
    t_data = torch.rand(num_points_data, 1, device=device)
    with torch.no_grad():
        u_pred_data = model(torch.cat([x_data, y_data, t_data], dim=1)).cpu().numpy()
    u_true_data = analytical_solution(x_data, y_data, t_data, ALPHA_TRUE).cpu().numpy()
    data_mse = np.mean((u_pred_data - u_true_data)**2)
    data_rel = np.linalg.norm(u_pred_data - u_true_data) / (np.linalg.norm(u_true_data) + 1e-16)

    savemat('./training_data/AL_INV_PINN_2D_metrics.mat', {
        'physics_test_loss': physics_test_loss,
        'initial_test_loss': initial_test_loss,
        'boundary_test_loss': boundary_test_loss,
        'mse': mse,
        'rel_l2': rel,
        'data_mse': data_mse,
        'data_rel_l2': data_rel,
        'alpha_learned': alpha_val
    })
    print('Saved model and metrics to training_data')
    print(f"Physics Loss (MSE): {physics_test_loss:.6e}")
    print(f"Initial Loss (MSE): {initial_test_loss:.6e}")
    print(f"Boundary Loss (MSE): {boundary_test_loss:.6e}")
    print(f"Data Loss (MSE): {data_mse:.6e}")
    print(f"Final MSE (grid): {mse:.6e}")
    print(f"Final Relative L2 Error: {rel:.6e}")
    print(f"Learned alpha: {alpha_val:.6e}")

    save_solution_comparison(model, t_value=0.5)
    save_solution_comparison(model, t_value=1.0)


if __name__ == '__main__':
    NUM_ITERATIONS = 10000
    NUM_POINTS_PDE = 2000
    NUM_POINTS_BC = 500
    NUM_POINTS_IC = 500
    NUM_POINTS_DATA = 1000

    beta = 0.5
    lr = 5e-3
    rho = 1e-3

    pinn = PINN(initial_alpha_guess=1.0).to(device)
    train_al_inverse(pinn, NUM_ITERATIONS, NUM_POINTS_PDE, NUM_POINTS_BC, NUM_POINTS_IC, NUM_POINTS_DATA, beta=beta, lr=lr, rho=rho)
