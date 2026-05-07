import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import random
import time
from scipy.io import savemat
import matplotlib.pyplot as plt
from tqdm import trange

# --- Reproducibility ---
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


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
            nn.Linear(64, 1)
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


def generate_training_data(num_points_pde, num_points_bc, num_points_ic):
    x_pde = torch.rand(num_points_pde, 1, requires_grad=True, device=device)
    y_pde = torch.rand(num_points_pde, 1, requires_grad=True, device=device)
    t_pde = torch.rand(num_points_pde, 1, requires_grad=True, device=device)

    x_ic = torch.rand(num_points_ic, 1, device=device)
    y_ic = torch.rand(num_points_ic, 1, device=device)
    t_ic = torch.zeros(num_points_ic, 1, device=device)
    u_ic_exact = initial_condition(x_ic, y_ic)

    # boundary points: x=0/1 or y=0/1
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

    return (x_pde, y_pde, t_pde,
            x_ic, y_ic, t_ic, u_ic_exact,
            x_bc, y_bc, t_bc, u_bc_exact)


def train_pinn_al(model, num_iterations, num_points_pde, num_points_bc, num_points_ic, beta=1.0, lr=5e-3, lbd_lr=1e-3):
    # scalar lambdas for IC and BC (augmented-Lagrangian multipliers)
    # use explicit dual ascent updates (do not include lambdas in optimizer)
    lbd_ic = torch.tensor(0.0, device=device)
    lbd_bc = torch.tensor(0.0, device=device)

    # dual step size (rho)
    rho = float(lbd_lr)

    optimizer = optim.Adam(list(model.parameters()), lr=lr)
    mse_loss = nn.MSELoss()

    os.makedirs("./training_logs", exist_ok=True)
    loss_file_path = "./training_logs/AL-PINN_training_loss.txt"
    with open(loss_file_path, "w") as f:
        f.write("Epoch,Loss_IC,Loss_BC,Loss_PDE,Total_Loss,lbd_ic,lbd_bc\n")

        start_time = time.time()
        pbar = trange(num_iterations, desc="AL-PINN Training", unit="iter")
        for iteration in pbar:
            optimizer.zero_grad()

            (x_pde, y_pde, t_pde,
             x_ic, y_ic, t_ic, u_ic_exact,
             x_bc, y_bc, t_bc, u_bc_exact) = generate_training_data(num_points_pde, num_points_bc, num_points_ic)

            # Forward
            u_pred_ic = model(torch.cat([x_ic, y_ic, t_ic], dim=1))
            loss_ic = mse_loss(u_pred_ic, u_ic_exact)

            u_pred_bc = model(torch.cat([x_bc, y_bc, t_bc], dim=1))
            loss_bc = mse_loss(u_pred_bc, u_bc_exact)

            residual = pde(x_pde, y_pde, t_pde, model)
            loss_pde = mse_loss(residual, torch.zeros_like(residual))

            # Augmented-Lagrangian objective
            # use mean terms for the lambda coupling
            mean_ic = (u_pred_ic - u_ic_exact).mean()
            mean_bc = u_pred_bc.mean()
            loss = loss_pde + beta * loss_ic + (lbd_ic * mean_ic) + beta * loss_bc + (lbd_bc * mean_bc)

            loss.backward()
            optimizer.step()

            # explicit dual-ascent update for lambdas (use detached means)
            mean_ic_det = mean_ic.detach()
            mean_bc_det = mean_bc.detach()
            # update lambdas and clamp to keep stable
            with torch.no_grad():
                lbd_ic = torch.clamp(lbd_ic + rho * mean_ic_det, -1e3, 1e3)
                lbd_bc = torch.clamp(lbd_bc + rho * mean_bc_det, -1e3, 1e3)

            if iteration % 100 == 0:
                elapsed = time.time() - start_time
                pbar.set_postfix({
                    'loss': f'{loss.item():.2e}',
                    'pde': f'{loss_pde.item():.2e}',
                    'ic': f'{loss_ic.item():.2e}',
                    'bc': f'{loss_bc.item():.2e}',
                })
                print(f"Iteration: {iteration:5d}, Loss: {loss.item():.4e}, Loss_pde: {loss_pde.item():.4e}, Loss_ic: {loss_ic.item():.4e}, Loss_bc: {loss_bc.item():.4e}, elapsed: {elapsed:.2f}s")
                f.write(f"{iteration},{loss_ic.item():.6e},{loss_bc.item():.6e},{loss_pde.item():.6e},{loss.item():.6e},{float(lbd_ic):.6e},{float(lbd_bc):.6e}\n")

    # return lambdas for reporting
    return lbd_ic.detach().cpu().item(), lbd_bc.detach().cpu().item()


def plot_solution_comparison(model, t_value, save_solution=True, save_error=True):
    model.eval()
    alpha = 0.1

    x = torch.linspace(0, 1, 100, device=device)
    y = torch.linspace(0, 1, 100, device=device)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    T = torch.full(X.shape, t_value, device=device)

    input_tensor = torch.stack([X.flatten(), Y.flatten(), T.flatten()], dim=1)
    with torch.no_grad():
        U_pred = model(input_tensor).reshape(X.shape)

    exp_term = np.exp(-2 * np.pi**2 * alpha * t_value)
    U_exact = exp_term * torch.sin(torch.pi * X) * torch.sin(torch.pi * Y)
    Error = torch.abs(U_pred - U_exact)

    X_np, Y_np = X.cpu().numpy(), Y.cpu().numpy()
    U_pred_np = U_pred.cpu().numpy()
    U_exact_np = U_exact.cpu().numpy()
    Error_np = Error.cpu().numpy()

    os.makedirs("./data", exist_ok=True)
    if save_solution:
        savemat(f"./data/AL_solution_t{t_value}.mat", {
            "X": X_np,
            "Y": Y_np,
            "U_pred": U_pred_np,
            "U_exact": U_exact_np,
        })
        print(f"Saved ./data/AL_solution_t{t_value}.mat")
    if save_error:
        savemat(f"./data/AL_error_t{t_value}.mat", {
            "X": X_np,
            "Y": Y_np,
            "Error": Error_np,
        })
        print(f"Saved ./data/AL_error_t{t_value}.mat")

    os.makedirs("./figures", exist_ok=True)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    v_min = min(U_pred_np.min(), U_exact_np.min())
    v_max = max(U_pred_np.max(), U_exact_np.max())
    c1 = ax1.pcolormesh(X_np, Y_np, U_pred_np, cmap='jet', vmin=v_min, vmax=v_max, shading='auto')
    fig.colorbar(c1, ax=ax1)
    ax1.set_title("Prediction")
    c2 = ax2.pcolormesh(X_np, Y_np, U_exact_np, cmap='jet', vmin=v_min, vmax=v_max, shading='auto')
    fig.colorbar(c2, ax=ax2)
    ax2.set_title("Exact")
    c3 = ax3.pcolormesh(X_np, Y_np, Error_np, cmap='Reds', shading='auto')
    fig.colorbar(c3, ax=ax3)
    ax3.set_title("Absolute Error")
    plt.subplots_adjust(wspace=0.1, hspace=0.1, top=0.88)
    fig.savefig(f"./figures/AL_pinn_comparison_t{t_value}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def calculate_final_losses(model, num_points_pde, num_points_bc, num_points_ic):
    print("\n--- Calculating Final Losses on a New Batch ---")
    model.eval()
    mse_loss = nn.MSELoss()

    (x_pde, y_pde, t_pde,
     x_ic, y_ic, t_ic, u_ic_exact,
     x_bc, y_bc, t_bc, u_bc_exact) = generate_training_data(num_points_pde, num_points_bc, num_points_ic)

    with torch.no_grad():
        u_pred_ic = model(torch.cat([x_ic, y_ic, t_ic], dim=1))
        loss_ic = mse_loss(u_pred_ic, u_ic_exact)
        u_pred_bc = model(torch.cat([x_bc, y_bc, t_bc], dim=1))
        loss_bc = mse_loss(u_pred_bc, u_bc_exact)

    residual = pde(x_pde, y_pde, t_pde, model)
    loss_pde = mse_loss(residual, torch.zeros_like(residual))

    print(f"Final Initial Condition Loss (unweighted): {loss_ic.item():.6e}")
    print(f"Final Boundary Condition Loss (unweighted): {loss_bc.item():.6e}")
    print(f"Final Physics (PDE) Loss (unweighted):   {loss_pde.item():.6e}")

    return loss_ic.item(), loss_bc.item(), loss_pde.item()


if __name__ == "__main__":
    NUM_ITERATIONS = 30000
    NUM_POINTS_PDE = 2000
    NUM_POINTS_BC = 500
    NUM_POINTS_IC = 500

    beta = 10.0
    lr = 5e-3
    lbd_lr = 1e-3

    pinn_model = PINN().to(device)
    l1, l2 = train_pinn_al(pinn_model, NUM_ITERATIONS, NUM_POINTS_PDE, NUM_POINTS_BC, NUM_POINTS_IC, beta=beta, lr=lr, lbd_lr=lbd_lr)

    # plot comparisons
    plot_solution_comparison(pinn_model, t_value=0.5, save_solution=True, save_error=True)
    plot_solution_comparison(pinn_model, t_value=1.0, save_solution=True, save_error=False)
    plot_solution_comparison(pinn_model, t_value=0.1, save_solution=False, save_error=True)

    # final losses and metrics save
    loss_ic, loss_bc, loss_pde = calculate_final_losses(pinn_model, NUM_POINTS_PDE, NUM_POINTS_BC, NUM_POINTS_IC)

    # compute L2 and MSE on full grid
    x = torch.linspace(0, 1, 100, device=device)
    y = torch.linspace(0, 1, 100, device=device)
    t = torch.linspace(0, 1, 100, device=device)
    X, Y, T = torch.meshgrid(x, y, t, indexing='ij')
    input_tensor = torch.stack([X.flatten(), Y.flatten(), T.flatten()], dim=1)
    with torch.no_grad():
        U_pred = pinn_model(input_tensor).reshape(X.shape)

    alpha = 0.1
    exp_term = torch.exp(-2 * torch.pi**2 * alpha * T)
    U_exact = exp_term * torch.sin(torch.pi * X) * torch.sin(torch.pi * Y)
    mse_error = torch.mean((U_pred - U_exact) ** 2).cpu().item()
    l2_error = torch.linalg.norm(U_pred - U_exact).cpu().item() / torch.linalg.norm(U_exact).cpu().item()

    print(f"Final MSE on grid: {mse_error:.6e}")
    print(f"Final Relative L2 Error: {l2_error:.6e}")

    os.makedirs('./data', exist_ok=True)
    try:
        savemat('./data/AL_PINN_metrics.mat', {'initial_loss': loss_ic,
                                               'boundary_loss': loss_bc,
                                               'physics_loss': loss_pde,
                                               'mse': mse_error,
                                               'rel_l2': l2_error,
                                               'lambda_ic': l1,
                                               'lambda_bc': l2})
        print('Saved metrics to ./data/AL_PINN_metrics.mat')
    except Exception:
        pass
