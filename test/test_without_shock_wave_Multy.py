import torch
import torch.nn as nn
import numpy as np
from tqdm import trange
import random as rm
import scipy.io
from scipy.linalg import solve
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# --- Setup, PINN Class, and Helper functions ---
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"We are using device: {device}")

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
rm.seed(seed)

def save_model(model, path):
    """Saves the model's state dictionary."""
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model, path):
    """Loads the model's state dictionary."""
    model.load_state_dict(torch.load(path))
    model.eval()
    print(f"Model loaded from {path}")

def plot_optimization_history_multi_constraint(*Histories, legends=None, linewidth=1, fontsize=20, ticksize=15, legendsize=15, savename=None):
    plt.figure(figsize=(21, 5))
    
    # --- PLOT 1: KKT Gap ---
    plt.subplot(1, 3, 1)
    for i, history in enumerate(Histories):
        label = legends[i] if legends is not None and i < len(legends) else f'Run {i+1}'
        _, KKT_gaps, _, _ = zip(*history)
        plt.plot(KKT_gaps, label=label, linewidth=linewidth)
    plt.xlabel('Iteration (x100)', fontsize=fontsize)
    plt.ylabel('KKT Gap', fontsize=fontsize)
    plt.yscale('log'); plt.legend(fontsize=legendsize)
    plt.title('KKT Gap', fontsize=fontsize)
    plt.grid(True); plt.xticks(fontsize=ticksize); plt.yticks(fontsize=ticksize)

    # --- PLOT 2: Boundary Constraint ---
    plt.subplot(1, 3, 2)
    for i, history in enumerate(Histories):
        label = legends[i] if legends is not None and i < len(legends) else f'Run {i+1}'
        _, _, h1_violations, _ = zip(*history)
        plt.plot(h1_violations, label=label, linewidth=linewidth)
    plt.xlabel('Iteration (x100)', fontsize=fontsize)
    plt.ylabel('Boundary Violation |h1(x)|', fontsize=fontsize)
    plt.yscale('log'); plt.legend(fontsize=legendsize)
    plt.title('Boundary Constraint', fontsize=fontsize)
    plt.grid(True); plt.xticks(fontsize=ticksize); plt.yticks(fontsize=ticksize)

    # --- PLOT 3: Physics Constraint ---
    plt.subplot(1, 3, 3)
    for i, history in enumerate(Histories):
        label = legends[i] if legends is not None and i < len(legends) else f'Run {i+1}'
        _, _, _, h2_violations = zip(*history)
        plt.plot(h2_violations, label=label, linewidth=linewidth)
    plt.xlabel('Iteration (x100)', fontsize=fontsize)
    plt.ylabel('Physics Violation |h2(x)|', fontsize=fontsize)
    plt.yscale('log'); plt.legend(fontsize=legendsize)
    plt.title('Physics Constraint', fontsize=fontsize)
    plt.grid(True); plt.xticks(fontsize=ticksize); plt.yticks(fontsize=ticksize)

    plt.tight_layout()
    if savename:
        dirpath = os.path.dirname(savename)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        plt.savefig(savename, bbox_inches='tight')
    plt.show()


class PhysicsInformedNN():
    def __init__(self, X_u, u, X_f):
        self.X_u_master = torch.tensor(X_u, dtype=torch.float32).to(device)
        self.u_master = torch.tensor(u, dtype=torch.float32).to(device)
        self.X_f_master = torch.tensor(X_f, dtype=torch.float32).to(device)
        
        self.X_u_full = self.X_u_master
        self.u_full = self.u_master
        self.X_f_full = self.X_f_master
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

    def set_data_for_stage(self, X_u_stage, u_stage, X_f_stage):
        self.X_u_stage = X_u_stage
        self.u_stage = u_stage
        self.X_f_stage = X_f_stage

    def net_u(self, x, t):
        return self.net(torch.hstack((x, t)))

    def net_f(self, x, t):
        x.requires_grad_(True); t.requires_grad_(True)
        u = self.net_u(x, t)
        u_t = torch.autograd.grad(u, t, torch.ones_like(u), True, True)[0]
        u_x = torch.autograd.grad(u, x, torch.ones_like(u), True, True)[0]
        u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), True, True)[0]
        f = u_t + (u * u_x) - (0.01 / np.pi) * u_xx
        return f

def residual_loss(model):
    f_pred = model.net_f(model.X_f_stage[:, 0:1], model.X_f_stage[:, 1:2])
    null_f = torch.zeros((model.X_f_stage.shape[0], 1), device=device)
    return model.loss(f_pred, null_f)

def boundary_loss(model):
    u_pred = model.net_u(model.X_u_stage[:, 0:1], model.X_u_stage[:, 1:2])
    return model.loss(u_pred, model.u_stage)

def get_flat_params(model):
    return np.concatenate([p.detach().cpu().numpy().flatten() for p in model.parameters()])

def set_flat_params(model, flat_params):
    flat_params = flat_params.astype(np.float32)
    pointer = 0
    for p in model.parameters():
        num_params = p.numel()
        p.data = torch.from_numpy(flat_params[pointer:pointer + num_params]).view_as(p).to(device)
        pointer += num_params

pinn_model = None

# --- NEW WRAPPER FUNCTIONS for Multi-Constraint Problem ---
def f(x):
    """Objective function is now zero."""
    return 0

def df(x):
    """Gradient of the objective is a vector of zeros."""
    return np.zeros_like(x)

def h(x):
    """Constraint function now returns a vector of two losses."""
    set_flat_params(pinn_model.net, x)
    loss_h_val = boundary_loss(pinn_model).item()
    loss_f_val = residual_loss(pinn_model).item()
    return np.array([loss_h_val, loss_f_val])

def dh(x):
    """Constraint Jacobian now returns a matrix with two rows of gradients."""
    set_flat_params(pinn_model.net, x)
    
    # Gradient of boundary loss (h1)
    pinn_model.net.zero_grad()
    loss_h = boundary_loss(pinn_model)
    loss_h.backward(retain_graph=True) # Must retain graph for second backward pass
    grad_h = np.concatenate([p.grad.cpu().numpy().flatten() if p.grad is not None else np.zeros(p.numel()) for p in pinn_model.net.parameters()])
    
    # Gradient of physics loss (h2)
    pinn_model.net.zero_grad()
    loss_f = residual_loss(pinn_model)
    loss_f.backward()
    grad_f = np.concatenate([p.grad.cpu().numpy().flatten() if p.grad is not None else np.zeros(p.numel()) for p in pinn_model.net.parameters()])
    
    # Stack gradients into a Jacobian matrix
    return np.vstack([grad_h, grad_f])

# --- Optimizer (Handles multi-constraint case) ---
def SOFL_Adam_PINN(f, h, df, dh, x_start, Kp, Ki=0, eta=0.01, max_iter=1000, tol=1e-6):
    x = np.array(x_start, dtype=np.float32)
    history = []
    Integral = np.zeros(Kp.shape[0]) # Integral is now a vector

    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    m, v, t = np.zeros_like(x), np.zeros_like(x), 0
    
    pbar = trange(max_iter, desc="SOFL_Adam (Multi-Constraint)")
    for iteration in pbar:
        true_grad_f = df(x) # This is now a zero vector
        true_J_h = dh(x)    # This is now a 2xN matrix
        true_h_x = h(x)     # This is now a vector of 2 losses

        JhJht = true_J_h @ true_J_h.T
        Pcontrol = -Kp @ true_h_x
        Integral = Integral + true_h_x
        Icontrol = Ki @ Integral
        rhs = Pcontrol + Icontrol + true_J_h @ true_grad_f
        I = np.eye(np.shape(JhJht)[0])
        lambda_ = -solve(JhJht + 1e-6 * I, rhs, assume_a='pos')

        KKT_grad = true_grad_f + (true_J_h.T @ lambda_).flatten()
        
        clip_value = 10.0
        grad_norm = np.linalg.norm(KKT_grad)
        if grad_norm > clip_value:
            KKT_grad = KKT_grad * (clip_value / grad_norm)

        t += 1
        m = beta1 * m + (1 - beta1) * KKT_grad
        v = beta2 * v + (1 - beta2) * (KKT_grad**2)
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        step_vector = eta * m_hat / (np.sqrt(v_hat) + epsilon)
        x_new = x - step_vector
        step_size = np.linalg.norm(step_vector)
        
        if iteration % 100 == 0:
            true_f_val = f(x)
            KKT_gap = np.max([np.linalg.norm(KKT_grad), np.max(np.abs(true_h_x))])
            pbar.set_postfix({
                'boundary_loss': f'{true_h_x[0]:.2e}',
                'physics_loss': f'{true_h_x[1]:.2e}',
                'KKT_gap': f'{KKT_gap:.2e}',
            })
            history.append((true_f_val, KKT_gap, np.abs(true_h_x[0]), np.abs(true_h_x[1])))

        if step_size < tol:
            print(f"\nConvergence at iter {iteration}: Step size ({step_size:.2e}) < tol ({tol:.2e}).")
            break

        x = x_new
        
        if np.isnan(x).any():
            print("\nError: NaN values detected. Stopping.")
            break
            
    return x, history

# --- Main Execution ---

# 1. Load Data
N_u = 100
N_f = 10000
x_upper = np.ones((N_u // 4, 1), dtype=np.float32); t_upper = np.random.rand(N_u // 4, 1).astype(np.float32)
x_lower = -np.ones((N_u // 4, 1), dtype=np.float32); t_lower = np.random.rand(N_u // 4, 1).astype(np.float32)
t_zero = np.zeros((N_u // 2, 1), dtype=np.float32); x_zero = -1 + 2 * np.random.rand(N_u // 2, 1).astype(np.float32)
X_upper = np.hstack((x_upper, t_upper)); X_lower = np.hstack((x_lower, t_lower)); X_zero = np.hstack((x_zero, t_zero))
X_u_train_np = np.vstack((X_upper, X_lower, X_zero))
u_upper = np.zeros((N_u // 4, 1), dtype=np.float32); u_lower = np.zeros((N_u // 4, 1), dtype=np.float32)
u_zero = -np.sin(np.pi * x_zero).astype(np.float32)
u_train_np = np.vstack((u_upper, u_lower, u_zero))
index = np.arange(N_u); np.random.shuffle(index)
X_u_train_np = X_u_train_np[index, :]; u_train_np = u_train_np[index, :]
X_f_train_np = np.random.uniform([-1, 0], [1, 1], size=(N_f, 2)).astype(np.float32)
X_f_train_np = np.vstack((X_f_train_np, X_u_train_np))

# 2. Initialize PINN model
pinn_model = PhysicsInformedNN(X_u_train_np, u_train_np, X_f_train_np)

# 3. Setup and Run Causal Training Curriculum
x_current = get_flat_params(pinn_model.net)
stages = [(0.33, 10000), (0.66, 10000), (1.0, 10000)]
full_history = []

# --- NEW 2x2 Gain Matrices for the two constraints ---
Kp = np.array([[600.0, 0.0], [0.0, 600.0]])
Ki = np.array([[0.01, 0.0], [0.0, 0.01]])
eta = 1e-4 
tol = 1e-8

for i, (t_max, iters) in enumerate(stages):
    print(f"\n--- Causal Training: Stage {i+1}/{len(stages)} (t <= {t_max}) ---")

    X_f_stage_np = X_f_train_np[X_f_train_np[:, 1] <= t_max]
    
    pinn_model.set_data_for_stage(
        pinn_model.X_u_master,
        pinn_model.u_master,
        torch.tensor(X_f_stage_np, dtype=torch.float32).to(device)
    )

    x_current, history = SOFL_Adam_PINN(
        f, h, df, dh,
        x_current, Kp, Ki=Ki, eta=eta, 
        max_iter=iters, tol=tol
    )
    full_history.extend(history)

x_final = x_current

# 4. Final Evaluation using the full master dataset
set_flat_params(pinn_model.net, x_final)
pinn_model.set_data_for_stage(pinn_model.X_u_master, pinn_model.u_master, pinn_model.X_f_master)
final_losses = h(x_final)

print("\n--- Optimization Finished ---")



save_model(pinn_model.net, "./test_model/trained_Multi-Constraint_pinn_model.pth")

# --- Evaluation ---
# ... (Evaluation code is the same)
print("\n--- Evaluation ---")
data = scipy.io.loadmat('burgers_shock.mat')
t_exact = data['t'].flatten()[:, None]
x_exact = data['x'].flatten()[:, None]
Exact = np.real(data['usol'])
T, X = np.meshgrid(t_exact, x_exact)
X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
u_truth = Exact.flatten()[:, None]

# Define the shock wave region (example: x in [-0.05, 0.05] and t in [0.3, 1])
shock_wave_region = (X.flatten() >= -0.05) & (X.flatten() <= 0.05) & \
                    (T.flatten() >= 0.3) & (T.flatten() <= 1.0)

# Exclude the shock wave region
X_star_no_shock = X_star[~shock_wave_region]
u_truth_no_shock = u_truth[~shock_wave_region]

# Convert to tensors
x_full = torch.from_numpy(X_star_no_shock[:, 0:1]).float().to(device)
t_full = torch.from_numpy(X_star_no_shock[:, 1:2]).float().to(device)

# Evaluate the model
with torch.no_grad():
    u_pred_no_shock = pinn_model.net_u(x_full, t_full).cpu().numpy()

# Calculate the L2 loss and relative error
l2_loss_mse_no_shock = np.mean((u_truth_no_shock - u_pred_no_shock) ** 2)
l2_relative_error_no_shock = np.linalg.norm(u_truth_no_shock - u_pred_no_shock, 2) / np.linalg.norm(u_truth_no_shock, 2)

# Print the results
print(f"Final L2 Loss (MSE) excluding shock wave: {l2_loss_mse_no_shock:.4f}")
print(f"Final L2 Relative Error excluding shock wave: {l2_relative_error_no_shock:.4f}")

def residual_loss(model):
    # Exclude shock wave region
    shock_wave_region = (model.X_f_full[:, 0] >= -0.2) & (model.X_f_full[:, 0] <= 0.2) & \
                        (model.X_f_full[:, 1] >= 0.3) & (model.X_f_full[:, 1] <= 0.7)
    X_f_no_shock = model.X_f_full[~shock_wave_region]

    # Calculate physics loss on the remaining points
    f_pred = model.net_f(X_f_no_shock[:, 0:1], X_f_no_shock[:, 1:2])
    null_f = torch.zeros((X_f_no_shock.shape[0], 1), device=device)
    return model.loss(f_pred, null_f)

def boundary_loss(model):
    # Exclude shock wave region
    shock_wave_region = (model.X_u_full[:, 0] >= -0.2) & (model.X_u_full[:, 0] <= 0.2) & \
                        (model.X_u_full[:, 1] >= 0.3) & (model.X_u_full[:, 1] <= 0.7)
    X_u_no_shock = model.X_u_full[~shock_wave_region]
    u_no_shock = model.u_full[~shock_wave_region]

    # Calculate boundary loss on the remaining points
    u_pred = model.net_u(X_u_no_shock[:, 0:1], X_u_no_shock[:, 1:2])
    return model.loss(u_pred, u_no_shock)


# Final evaluation excluding shock wave
final_physics_loss = residual_loss(pinn_model).item()
final_boundary_loss = boundary_loss(pinn_model).item()

print("\n--- Optimization Finished ---")
print(f"Final Physics Loss (Objective, excluding shock wave): {final_physics_loss:.4f}")
print(f"Final Boundary Loss (Constraint, excluding shock wave): {final_boundary_loss:.4f}")


# --- Plotting the Solution ---
print("\nPlotting the solution ...")
os.makedirs("./figures/solution", exist_ok=True)
x_plot = torch.linspace(-1, 1, 256).to(device); t_plot = torch.linspace(0, 1, 100).to(device)
X_grid_plot, T_grid_plot = torch.meshgrid(x_plot, t_plot, indexing='ij')
xcol_pred = X_grid_plot.reshape(-1, 1); tcol_pred = T_grid_plot.reshape(-1, 1)
with torch.no_grad():
    usol_pred = pinn_model.net_u(xcol_pred, tcol_pred)
Unp_pred = usol_pred.reshape(x_plot.numel(), t_plot.numel()).cpu().numpy()
xnp_plot = x_plot.cpu().numpy(); tnp_plot = t_plot.cpu().numpy()

# --- Solution Plot ---
print("Generating plot...")
plt.rcParams['font.size'] = '15'; fig = plt.figure(figsize=(5, 6)); ax = fig.add_subplot(111)
plt.xlabel(r"$t$"); plt.ylabel(r"$x$"); plt.title(r"Multi-Constraints SOFL-PINN $u(x,t)$")
img_handle = ax.imshow(Unp_pred, interpolation='nearest', cmap='rainbow',
                       extent=[tnp_plot.min(), tnp_plot.max(), xnp_plot.min(), xnp_plot.max()],
                       origin='lower', aspect='auto', vmin=-1.0, vmax=1.0)          
divider = make_axes_locatable(ax); cax = divider.append_axes("right", size="5%", pad=0.10)
cbar = fig.colorbar(img_handle, cax=cax); cbar.ax.tick_params(labelsize=10)
output_filename = "./figures/solution/Multi_Constraints_SOFL_PINN_solution.png"
plt.savefig(output_filename, dpi=300, bbox_inches='tight', pad_inches=0.1)
print(f"Plot saved successfully as {output_filename}")
plt.show()

# --- Error Plot ---
print("Generating error plot...")
error_grid = abs(Exact - Unp_pred)   # absolute error
# Optionally: relative error
# error_grid = (Exact - Unp_pred) / (np.abs(Exact) + 1e-8)

plt.rcParams['font.size'] = '15'
fig_err = plt.figure(figsize=(5, 6))
ax_err = fig_err.add_subplot(111)
plt.xlabel(r"$t$")
plt.ylabel(r"$x$")
plt.title(r"Multi-Constraints SOFL-PINN Error")

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

output_filename_err = "./figures/solution/Multi_Constraints_SOFL_PINN_error.png"
plt.savefig(output_filename_err, dpi=300, bbox_inches='tight', pad_inches=0.1)
print(f"Error plot saved successfully as {output_filename_err}")
plt.show()

# --- Plotting ---
# ... (Solution, Error, and Ground Truth plotting code is the same)
# The history plotting function is updated to handle the new format.
# The call will need to be updated as well if history format changes.
# plot_optimization_history_multi_constraint(full_history, legends=['SOFL-PINN'], linewidth=3, savename='./figures/training_history.png')
plot_optimization_history_multi_constraint(history, legends=['Multi-Constraints SOFL'], linewidth=3, savename='burgers_equation.png')

