import os
import time
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import scipy.io
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import trange

# --- Reproducibility ---
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Burgers equation dataset / grid ---
N_f = 10000
N_u = 2500
N_ic = 256
N_bc = 256
NU_OVER_PI = 0.01 / np.pi


def load_burgers_data():
    data = scipy.io.loadmat("burgers_shock.mat")
    t_vec = data["t"].flatten()[:, None]
    x_vec = data["x"].flatten()[:, None]
    u_exact = np.real(data["usol"])  # shape [Nx, Nt]

    X_grid, T_grid = np.meshgrid(x_vec.flatten(), t_vec.flatten(), indexing="ij")
    X_star = np.hstack((X_grid.reshape(-1, 1), T_grid.reshape(-1, 1)))
    u_star = u_exact.reshape(-1, 1)
    return x_vec, t_vec, X_star, u_star, u_exact


def u_init(x_in):
    return -torch.sin(math.pi * x_in)


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
    u_ic = u_init(x_ic)

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

class PhysicsInformedNN():
    def __init__(self, ds):
        self.X_u_full = ds["X_u"]
        self.u_full = ds["u_u"]
        self.X_f_full = ds["X_f"]
        self.x_ic = ds["x_ic"]
        self.t_ic = ds["t_ic"]
        self.u_ic = ds["u_ic"]
        self.x_l = ds["x_l"]
        self.x_r = ds["x_r"]
        self.t_bc = ds["t_bc"]
        self.u_bc = ds["u_bc"]

        self.create_net()
        self.loss = nn.MSELoss()

    def create_net(self):
        self.net = nn.Sequential(
            nn.Linear(2, 20), nn.Tanh(), nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(), nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(), nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(), nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(), nn.Linear(20, 1)
        ).to(device)

    def net_u(self, x, t):
        return self.net(torch.hstack((x, t)))

    def net_f(self, x, t):
        x.requires_grad_(True); t.requires_grad_(True)
        u = self.net_u(x, t)
        u_t = torch.autograd.grad(u, t, torch.ones_like(u), True, True)[0]
        u_x = torch.autograd.grad(u, x, torch.ones_like(u), True, True)[0]
        u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), True, True)[0]
        f = u_t + (u * u_x) - NU_OVER_PI * u_xx
        return f

def residual_loss(model):
    f_pred = model.net_f(model.X_f_full[:, 0:1], model.X_f_full[:, 1:2])
    null_f = torch.zeros((model.X_f_full.shape[0], 1), device=device)
    return model.loss(f_pred, null_f)

def boundary_loss(model):
    u_l = model.net_u(model.x_l, model.t_bc)
    u_r = model.net_u(model.x_r, model.t_bc)
    return model.loss(u_l, model.u_bc) + model.loss(u_r, model.u_bc)


def initial_loss(model):
    u_ic_pred = model.net_u(model.x_ic, model.t_ic)
    return model.loss(u_ic_pred, model.u_ic)


def objective_obs_loss(model):
    u_pred = model.net_u(model.X_u_full[:, 0:1], model.X_u_full[:, 1:2])
    return model.loss(u_pred, model.u_full)


def constraint_icbc_loss(model):
    return initial_loss(model) + boundary_loss(model)

def get_flat_params(model):
    return np.concatenate([p.detach().cpu().numpy().flatten() for p in model.parameters()])

def set_flat_params(model, flat_params):
    flat_params = flat_params.astype(np.float32)
    pointer = 0
    for p in model.parameters():
        num_params = p.numel()
        p.data = torch.from_numpy(flat_params[pointer:pointer + num_params]).view_as(p).to(device)
        pointer += num_params

def get_grads_from_loss(loss, params):
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    flat = []
    for g, p in zip(grads, params):
        if g is None:
            flat.append(np.zeros(p.numel(), dtype=np.float32))
        else:
            flat.append(g.detach().cpu().numpy().ravel())
    return np.concatenate(flat)


def pretrain_feasibility(net_module, max_iter=300):
    optimizer = optim.LBFGS(
        net_module.parameters(),
        lr=1.0,
        max_iter=max_iter,
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
        history_size=100,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()
        c_loss = residual_loss(pinn_model) + constraint_icbc_loss(pinn_model)
        c_loss.backward()
        return c_loss

    start = time.time()
    optimizer.step(closure)
    # residual_loss uses autograd w.r.t. inputs, so do not wrap with no_grad.
    c_after = (residual_loss(pinn_model) + constraint_icbc_loss(pinn_model)).item()
    print(f"Pretraining done in {time.time() - start:.1f}s | feasibility: {c_after:.3e}")

# trSQP-style optimizer wrapper (Algorithm 3-like)
def trSQP_train(net_module, max_iter=2000):
    # flattened parameter vector (theta)
    params = list(net_module.parameters())
    theta = get_flat_params(net_module)

    n = theta.shape[0]
    # Initialize quasi-Newton matrix H (identity)
    H = np.eye(n, dtype=np.float32)

    # Algorithm parameters (from paper screenshot)
    mu_prev = 1.0
    Delta = 1.0
    eta_low = 1e-8
    eta_upp = 0.3
    rho_scale = 2.0
    f_tol = 1e-12
    g_tol = 1e-8
    min_stop_iter = min(500, max_iter)

    pbar = trange(max_iter, desc="trSQP Training", unit="iter")
    for k in pbar:
        # set model to current theta
        set_flat_params(net_module, theta)

        # evaluate objective (data loss) and constraints
        loss_d = objective_obs_loss(pinn_model)
        loss_c1 = residual_loss(pinn_model)
        loss_c2 = constraint_icbc_loss(pinn_model)

        f_val = loss_d.item()
        h = np.array([loss_c1.item(), loss_c2.item()], dtype=np.float32)

        # gradients
        grad_f = get_grads_from_loss(loss_d, params)
        J1 = get_grads_from_loss(loss_c1, params)
        J2 = get_grads_from_loss(loss_c2, params)
        J = np.vstack([J1, J2])

        # estimate dual lambda_k
        JJt = J @ J.T
        eps_reg = 1e-8
        try:
            lambda_k = -np.linalg.solve(JJt + eps_reg * np.eye(JJt.shape[0], dtype=np.float32), h)
        except np.linalg.LinAlgError:
            lambda_k = -np.linalg.lstsq(JJt + eps_reg * np.eye(JJt.shape[0], dtype=np.float32), h, rcond=None)[0]

        grad_L = grad_f + J.T @ lambda_k

        # solve approximate TR subproblem via (regularized) Newton step
        reg = 1e-6
        try:
            d_newton = -np.linalg.solve(H + reg * np.eye(n, dtype=np.float32), grad_L)
        except np.linalg.LinAlgError:
            d_newton = -np.linalg.lstsq(H + reg * np.eye(n, dtype=np.float32), grad_L, rcond=None)[0]

        # enforce trust region
        if np.linalg.norm(d_newton) <= Delta:
            d_k = d_newton
        else:
            d_k = d_newton * (Delta / np.linalg.norm(d_newton))

        # Merit model q_mu(d): quadratic objective model + mu * ||h + Jd||
        quad_term = np.dot(grad_f, d_k) + 0.5 * d_k @ (H @ d_k)
        h_lin_trial = h + J @ d_k

        denom = 0.7 * (np.linalg.norm(h) - np.linalg.norm(h + J @ d_k))
        if denom == 0:
            mu_candidate = mu_prev
        else:
            mu_candidate = max(mu_prev, (np.dot(grad_L, d_k) + 0.5 * d_k @ (H @ d_k)) / denom)
        mu_k = mu_candidate

        q0 = f_val + mu_k * np.linalg.norm(h)
        qd = f_val + quad_term + mu_k * np.linalg.norm(h_lin_trial)
        Pred_k = q0 - qd

        # trial step
        theta_trial = theta + d_k
        set_flat_params(net_module, theta_trial)
        f_trial = objective_obs_loss(pinn_model).item()
        h_trial = np.array([
            residual_loss(pinn_model).item(),
            constraint_icbc_loss(pinn_model).item(),
        ], dtype=np.float32)

        phi_k = f_val + mu_k * np.linalg.norm(h)
        phi_trial = f_trial + mu_k * np.linalg.norm(h_trial)
        Ared_k = phi_k - phi_trial

        if Pred_k <= 0:
            eta_k = -np.inf
        else:
            eta_k = Ared_k / Pred_k

        if eta_k >= eta_low:
            theta_new = theta + d_k
            if eta_k >= eta_upp:
                Delta = rho_scale * Delta

            # Recompute at accepted point for quasi-Newton update.
            set_flat_params(net_module, theta_new)
            loss_d_new = objective_obs_loss(pinn_model)
            loss_c1_new = residual_loss(pinn_model)
            loss_c2_new = constraint_icbc_loss(pinn_model)

            grad_f_new = get_grads_from_loss(loss_d_new, params)
            J1_new = get_grads_from_loss(loss_c1_new, params)
            J2_new = get_grads_from_loss(loss_c2_new, params)
            J_new = np.vstack([J1_new, J2_new])
            JJt_new = J_new @ J_new.T
            h_new = np.array([loss_c1_new.item(), loss_c2_new.item()], dtype=np.float32)
            try:
                lambda_new = -np.linalg.solve(
                    JJt_new + eps_reg * np.eye(JJt_new.shape[0], dtype=np.float32),
                    h_new,
                )
            except np.linalg.LinAlgError:
                lambda_new = -np.linalg.lstsq(
                    JJt_new + eps_reg * np.eye(JJt_new.shape[0], dtype=np.float32),
                    h_new,
                    rcond=None,
                )[0]
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
        else:
            Delta = Delta / rho_scale

        # stopping
        if (k + 1) >= min_stop_iter and eta_k < eta_low and (np.linalg.norm(d_k) <= f_tol or np.max(np.abs(grad_L)) <= g_tol):
            print(f"Converged at iter {k}")
            break

        if k % 10 == 0:
            pbar.set_postfix({
                "f_obs": f"{f_val:.2e}",
                "h_phy": f"{h[0]:.2e}",
                "h_icbc": f"{h[1]:.2e}",
                "Delta": f"{Delta:.2e}",
                "eta": f"{eta_k:.2e}",
            })

    set_flat_params(net_module, theta)
    return net_module


if __name__ == '__main__':
    os.makedirs('./figures', exist_ok=True)
    os.makedirs('./data', exist_ok=True)

    x_vec, t_vec, X_star, u_star, u_star_grid = load_burgers_data()
    ds = build_training_sets(x_vec, t_vec, X_star, u_star)

    # Instantiate PINN with observation + physics/IC/BC sets.
    pinn_model = PhysicsInformedNN(ds)

    start = time.time()
    pretrain_feasibility(pinn_model.net, max_iter=300)
    # increase max iterations to allow longer training
    trained = trSQP_train(pinn_model.net, max_iter=1000)
    elapsed = time.time() - start
    print(f"Training finished in {elapsed:.1f}s")

    # save final prediction on grid
    with torch.no_grad():
        x_star_t = torch.tensor(X_star[:, 0:1], dtype=torch.float32, device=device)
        t_star_t = torch.tensor(X_star[:, 1:2], dtype=torch.float32, device=device)
        U = trained(torch.hstack((x_star_t, t_star_t))).detach().cpu().numpy().reshape(u_star_grid.shape)

    plt.figure(figsize=(6, 4))
    plt.imshow(U, extent=[t_vec.min(), t_vec.max(), x_vec.min(), x_vec.max()], aspect='auto', origin='lower')
    plt.colorbar()
    plt.title('trSQP-PINN Burgers: learned u(x,t)')
    figpath = './figures/trSQP-PINN-solution.png'
    plt.savefig(figpath, dpi=200, bbox_inches='tight')
    print(f"Saved figure to {figpath}")

    scipy.io.savemat('./data/trSQP-PINN-solution.mat', {'u_pred': U, 'u_true': u_star_grid, 'x': x_vec, 't': t_vec})
    print('Saved data to ./data/trSQP-PINN-solution.mat')
    
    # Final losses and metrics
    phys_loss = residual_loss(pinn_model).item()
    bnd_loss = boundary_loss(pinn_model).item()

    # initial loss
    init_loss = initial_loss(pinn_model).item()

    # MSE and L2 on full grid
    u_true_full = torch.tensor(u_star, dtype=torch.float32, device=device)
    u_pred_full = trained(torch.hstack((x_star_t, t_star_t)))
    mse_full = nn.MSELoss()(u_pred_full, u_true_full).item()
    l2_rel = (torch.linalg.norm(u_pred_full - u_true_full) / (torch.linalg.norm(u_true_full) + 1e-16)).item()

    print('Final metrics:')
    print(f'Physics loss: {phys_loss:.6e}')
    print(f'Initial loss: {init_loss:.6e}')
    print(f'Boundary loss: {bnd_loss:.6e}')
    print(f'MSE (full grid): {mse_full:.6e}')
    print(f'L2 relative (full grid): {l2_rel:.6e}')

    # === Plotting the solution and error ===
    print("\nPlotting the solution ...")
    os.makedirs("./figures/solution", exist_ok=True)
    x_plot = torch.tensor(x_vec.flatten(), dtype=torch.float32, device=device)
    t_plot = torch.tensor(t_vec.flatten(), dtype=torch.float32, device=device)
    X_grid_plot, T_grid_plot = torch.meshgrid(x_plot, t_plot, indexing='ij')
    xcol_pred = X_grid_plot.reshape(-1, 1)
    tcol_pred = T_grid_plot.reshape(-1, 1)
    with torch.no_grad():
        usol_pred = pinn_model.net_u(xcol_pred, tcol_pred)
    Unp_pred = usol_pred.reshape(x_plot.numel(), t_plot.numel()).cpu().numpy()
    xnp_plot = x_plot.cpu().numpy(); tnp_plot = t_plot.cpu().numpy()

    print("Generating plot...")
    plt.rcParams['font.size'] = '15'
    fig = plt.figure(figsize=(5, 6)); ax = fig.add_subplot(111)
    plt.xlabel(r"$t$"); plt.ylabel(r"$x$"); plt.title("trSQP-PINN Solution")
    img_handle = ax.imshow(Unp_pred, interpolation='nearest', cmap='rainbow',
                           extent=[tnp_plot.min(), tnp_plot.max(), xnp_plot.min(), xnp_plot.max()],
                           origin='lower', aspect='auto', vmin=-1.0, vmax=1.0)
    divider = make_axes_locatable(ax); cax = divider.append_axes("right", size="5%", pad=0.10)
    cbar = fig.colorbar(img_handle, cax=cax); cbar.ax.tick_params(labelsize=10)
    output_filename = "./figures/solution/trSQP_PINN_solution.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight', pad_inches=0.1)
    print(f"Plot saved successfully as {output_filename}")
    plt.show()

    # --- Save solution data to .mat file in ./data ---
    os.makedirs('./data', exist_ok=True)
    try:
        scipy.io.savemat('./data/trSQP_PINN_solution.mat', {'solution': Unp_pred, 'x': xnp_plot, 't': tnp_plot})
        print('Saved solution matrix to ./data/trSQP_PINN_solution.mat')
    except Exception as e:
        print('Warning: failed to save solution .mat file:', e)

    # --- Error Plot ---
    print("Generating error plot...")
    Exact = u_star_grid
    error_grid = np.abs(Exact - Unp_pred)

    plt.rcParams['font.size'] = '15'
    fig_err = plt.figure(figsize=(5, 6))
    ax_err = fig_err.add_subplot(111)
    plt.xlabel(r"$t$")
    plt.ylabel(r"$x$")
    plt.title(r"trSQP-PINN Error")
    img_handle_err = ax_err.imshow(
        error_grid,
        interpolation='nearest',
        cmap='bwr',
        extent=[tnp_plot.min(), tnp_plot.max(), xnp_plot.min(), xnp_plot.max()],
        origin='lower',
        aspect='auto',
        vmin=0.0, vmax=1.0
    )
    divider_err = make_axes_locatable(ax_err)
    cax_err = divider_err.append_axes("right", size="5%", pad=0.10)
    cbar_err = fig_err.colorbar(img_handle_err, cax=cax_err)
    cbar_err.ax.tick_params(labelsize=10)
    output_filename_err = "./figures/solution/trSQP_PINN_error.png"
    plt.savefig(output_filename_err, dpi=300, bbox_inches='tight', pad_inches=0.1)
    print(f"Error plot saved successfully as {output_filename_err}")
    plt.show()

    # --- Save error grid to .mat file in ./data ---
    try:
        scipy.io.savemat('./data/trSQP_PINN_error.mat', {'error': error_grid, 'x': xnp_plot, 't': tnp_plot})
        print('Saved error matrix to ./data/trSQP_PINN_error.mat')
    except Exception as e:
        print('Warning: failed to save error .mat file:', e)
