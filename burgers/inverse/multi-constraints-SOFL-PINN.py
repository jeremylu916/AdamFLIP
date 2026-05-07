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
import time
from random import uniform

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
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model, path):
    """Loads the model's state dictionary."""
    model.load_state_dict(torch.load(path))
    model.eval()
    print(f"Model loaded from {path}")

def plot_optimization_history_multi_constraint(*Histories, legends=None, linewidth=1, fontsize=20, ticksize=15, legendsize=15, savename=None):
    """Plotting function for the multi-constraint problem."""
    plt.figure(figsize=(21, 5))
    
    # --- PLOT 1: Objective Function ---
    plt.subplot(1, 3, 1)
    for i, history in enumerate(Histories):
        label = legends[i] if legends is not None and i < len(legends) else f'Run {i+1}'
        f_values, _, _, _ = zip(*history)
        plt.plot(f_values, label=label, linewidth=linewidth)
    plt.xlabel('Iteration (x100)', fontsize=fontsize)
    plt.ylabel('$f(x)$ (Observation Loss)', fontsize=fontsize)
    plt.legend(fontsize=legendsize)
    plt.title('Objective Function', fontsize=fontsize)
    plt.grid(True); plt.xticks(fontsize=ticksize); plt.yticks(fontsize=ticksize)
    plt.yscale('log')

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

# --- PINN for Inverse Problem ---
lambda1_init = 2.0
lambda2_init = 0.0
nu_true = 0.01/np.pi
print("True PDE parameters: \u03BB\u2081 = 1.0, \u03BB\u2082 = %.6f" % (nu_true))
print(f"Initial Guess: \u03BB\u2081 = {lambda1_init}, \u03BB\u2082 = {lambda2_init}")


class PhysicsInformedNN():
    def __init__(self, X_u, u, X_f, X_obs, u_obs):
        # Boundary data
        self.x_u = torch.tensor(X_u[:, 0:1], dtype=torch.float32, requires_grad=True).to(device)
        self.t_u = torch.tensor(X_u[:, 1:2], dtype=torch.float32, requires_grad=True).to(device)
        self.u = torch.tensor(u, dtype=torch.float32).to(device)

        # Physics collocation points
        self.x_f = torch.tensor(X_f[:, 0:1], dtype=torch.float32, requires_grad=True).to(device)
        self.t_f = torch.tensor(X_f[:, 1:2], dtype=torch.float32, requires_grad=True).to(device)
        self.null =  torch.zeros((self.x_f.shape[0], 1)).to(device)

        # Observation data
        self.x_obs = torch.tensor(X_obs[:, 0:1], dtype=torch.float32).to(device)
        self.t_obs = torch.tensor(X_obs[:, 1:2], dtype=torch.float32).to(device)
        self.u_obs = torch.tensor(u_obs, dtype=torch.float32).to(device)
        
        self.create_net()
        self.loss = nn.MSELoss()
        self.iter = 0
        
        # --- Learnable PDE parameters ---
        self.lambda1 = torch.tensor([lambda1_init], requires_grad=True).float().to(device)
        self.lambda2 = torch.tensor([lambda2_init], requires_grad=True).float().to(device)
        self.lambda1 = nn.Parameter(self.lambda1)
        self.lambda2 = nn.Parameter(self.lambda2)
        self.net.register_parameter('lambda1', self.lambda1)
        self.net.register_parameter('lambda2', self.lambda2)
        
    def create_net(self):
        self.net = nn.Sequential(
            nn.Linear(2, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 1)
        ).to(device)

    def net_u(self, x, t):
        return self.net( torch.hstack((x, t)) )

    def net_f(self, x, t):
        u = self.net_u(x, t)
        u_t = torch.autograd.grad(u, t, torch.ones_like(u), True, True)[0]
        u_x = torch.autograd.grad(u, x, torch.ones_like(u), True, True)[0]
        u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), True, True)[0]
        f = u_t + self.lambda1 * u * u_x - self.lambda2 * u_xx
        return f

    # --- Loss Functions ---
    def ibc_loss(self):
        u_pred = self.net_u(self.x_u, self.t_u)
        return self.loss(u_pred, self.u)

    def residual_loss(self):
        f_pred = self.net_f(self.x_f, self.t_f)
        return self.loss(f_pred, self.null)

    def observation_loss(self):
        u_pred_obs = self.net_u(self.x_obs, self.t_obs)
        return self.loss(u_pred_obs, self.u_obs)

# --- Data Loading ---
N_u = 100
N_f = 10000

x_upper = np.ones((N_u//4, 1), dtype=float); t_upper = np.random.rand(N_u//4, 1)
x_lower = -np.ones((N_u//4, 1), dtype=float); t_lower = np.random.rand(N_u//4, 1)
t_zero = np.zeros((N_u//2, 1), dtype=float); x_zero = -1 + np.random.rand(N_u//2, 1) * 2
X_upper = np.hstack((x_upper, t_upper)); X_lower = np.hstack((x_lower, t_lower)); X_zero = np.hstack((x_zero, t_zero))
X_u_train = np.vstack((X_upper, X_lower, X_zero))

u_upper = np.zeros((N_u//4, 1), dtype=float); u_lower = np.zeros((N_u//4, 1), dtype=float)
u_zero = -np.sin(np.pi * x_zero)
u_train = np.vstack((u_upper, u_lower, u_zero))

index = np.arange(N_u); np.random.shuffle(index)
X_u_train = X_u_train[index, :]; u_train = u_train[index, :]

X_f_train = np.random.uniform([-1, 0], [1, 1], size=(N_f, 2)).astype(np.float32)
X_f_train = np.vstack((X_f_train, X_u_train))

# --- Load Observation Data ---
data = scipy.io.loadmat('burgers_shock.mat')
t_ = data['t'].flatten()[:, None]
x_ = data['x'].flatten()[:, None]
Exact = np.real(data['usol'])
T, X = np.meshgrid(t_, x_)
X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
u_truth = Exact.flatten()[:, None]

# Create observation data from sparse, noisy samples
N_obs = 2560
idx_obs = np.random.choice(X_star.shape[0], N_obs, replace=False)
X_obs = X_star[idx_obs, :]
u_obs = u_truth[idx_obs, :]
obs_noise = 0.1
u_obs = u_obs + np.random.randn(N_obs, 1) * obs_noise

# --- Helper functions for SOFL Optimizer ---
def get_flat_params(model):
    """Gets all model parameters (weights, biases, and lambdas) as a flat NumPy vector."""
    return np.concatenate([p.detach().cpu().numpy().flatten() for p in model.parameters()])

def get_flat_grads(model):
    """Gets all model gradients as a flat NumPy vector."""
    return np.concatenate([p.grad.cpu().numpy().flatten() if p.grad is not None else np.zeros(p.numel()) for p in model.parameters()])

def set_flat_params(model, flat_params):
    """Sets all model parameters from a flat NumPy vector."""
    flat_params = flat_params.astype(np.float32)
    pointer = 0
    for p in model.parameters():
        num_params = p.numel()
        p.data = torch.from_numpy(flat_params[pointer:pointer + num_params]).view_as(p).to(device)
        pointer += num_params

pinn_model = None # Global model reference

# --- Wrapper Functions for SOFL Optimizer ---
def f(x):
    """Objective function: minimize observation loss."""
    set_flat_params(pinn_model.net, x)
    return pinn_model.observation_loss().item()

def df(x):
    """Gradient of the objective function."""
    set_flat_params(pinn_model.net, x)
    pinn_model.net.zero_grad()
    loss_obs = pinn_model.observation_loss()
    loss_obs.backward()
    return get_flat_grads(pinn_model.net)

def h(x):
    """Constraint function vector: [boundary_loss, physics_loss]."""
    set_flat_params(pinn_model.net, x)
    loss_h_val = pinn_model.ibc_loss().item()
    loss_f_val = pinn_model.residual_loss().item()
    return np.array([loss_h_val, loss_f_val])

def dh(x):
    """Constraint Jacobian: [grad(boundary_loss), grad(physics_loss)]."""
    set_flat_params(pinn_model.net, x)
    
    # Gradient of boundary loss (h1)
    pinn_model.net.zero_grad()
    loss_h = pinn_model.ibc_loss()
    loss_h.backward(retain_graph=True)
    grad_h = get_flat_grads(pinn_model.net)
    
    # Gradient of physics loss (h2)
    pinn_model.net.zero_grad()
    loss_f = pinn_model.residual_loss()
    loss_f.backward()
    grad_f = get_flat_grads(pinn_model.net)
    
    return np.vstack([grad_h, grad_f])

# --- SOFL Optimizer (Handles multi-constraint case) ---
def SOFL_Adam_PINN(f, h, df, dh, x_start, Kp, Ki=0, eta=0.01, max_iter=1000, tol=1e-6):
    x = np.array(x_start, dtype=np.float32)
    history = []
    Integral = np.zeros(Kp.shape[0])

    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    m, v, t = np.zeros_like(x), np.zeros_like(x), 0
    
    pbar = trange(max_iter, desc="SOFL+Adam (Multi-Constraint)")
    for iteration in pbar:
        true_grad_f = df(x) # Gradient of observation loss
        true_J_h = dh(x)    # Gradients of [ibc_loss, residual_loss]
        true_h_x = h(x)     # Values of [ibc_loss, residual_loss]

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
                'obs_loss': f'{true_f_val:.2e}',
                'ibc_loss': f'{true_h_x[0]:.2e}',
                'res_loss': f'{true_h_x[1]:.2e}',
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

# 1. Initialize PINN model
pinn_model = PhysicsInformedNN(X_u_train, u_train, X_f_train, X_obs, u_obs)

# 2. Setup and Run SOFL Optimizer
x_start = get_flat_params(pinn_model.net)
full_history = []

# --- 2x2 Gain Matrices for the two constraints ---
# We give the boundary loss a high priority, and the physics loss a moderate one
Kp = np.array([[500.0, 0.0], [0.0, 500.0]]) 
Ki = np.array([[0.01, 0.0], [0.0, 0.01]])
eta = 1e-4 
tol = 1e-8
max_iter = 20000

x_final, history = SOFL_Adam_PINN(
    f, h, df, dh,
    x_start, Kp, Ki=Ki, eta=eta, 
    max_iter=max_iter, tol=tol
)
full_history.extend(history)
set_flat_params(pinn_model.net, x_final)

# --- Optimization Finished ---
print("\n--- Optimization Finished ---")
final_obs_loss = pinn_model.observation_loss().item()
final_ibc_loss = pinn_model.ibc_loss().item()
final_res_loss = pinn_model.residual_loss().item()
final_lambda1 = pinn_model.lambda1.item()
final_lambda2 = pinn_model.lambda2.item()

print(f"Final Observation Loss (Objective): {final_obs_loss:.6f}")
print(f"Final Boundary Loss (Constraint 1): {final_ibc_loss:.6f}")
print(f"Final Physics Loss (Constraint 2): {final_res_loss:.6f}")
print("--- Discovered PDE Parameters ---")
print(f"Discovered \u03BB\u2081 (True=1.0):   {final_lambda1:.6f}")
print(f"Discovered \u03BB\u2082 (True={nu_true:.6f}): {final_lambda2:.6f}")

save_model(pinn_model.net, "./trained_Inverse_PINN.pth")

# --- Evaluation on Full Test Data ---
print("\n--- Evaluation on Full Test Grid ---")
x_test = torch.tensor(X_star[:, 0:1], dtype=torch.float32).to(device)
t_test = torch.tensor(X_star[:, 1:2], dtype=torch.float32).to(device)
u_truth_tensor = torch.tensor(u_truth, dtype=torch.float32).to(device)

with torch.no_grad():
    u_pred = pinn_model.net_u(x_test, t_test)
    l2_loss_mse = pinn_model.loss(u_pred, u_truth_tensor).item()
    
u_pred_np = u_pred.cpu().numpy()
l2_relative_error = np.linalg.norm(u_truth - u_pred_np, 2) / np.linalg.norm(u_truth, 2)

print(f"Final L2 Loss (MSE): {l2_loss_mse:.6f}")
print(f"Final L2 Relative Error: {l2_relative_error:.6f}")

# --- Plotting ---
plot_optimization_history_multi_constraint(full_history, legends=['SOFL-PINN'], linewidth=3, savename='./figures/inverse_training_history.png')
# ... (Solution plotting code would go here)
'''
plotting 
'''
print("\nPlotting the solution ...")
os.makedirs("./figures/solution", exist_ok=True)
x_plot = torch.linspace(-1, 1, 256).to(device); t_plot = torch.linspace(0, 1, 100).to(device)
X_grid_plot, T_grid_plot = torch.meshgrid(x_plot, t_plot, indexing='ij')
xcol_pred = X_grid_plot.reshape(-1, 1); tcol_pred = T_grid_plot.reshape(-1, 1)
with torch.no_grad():
    usol_pred = pinn_model.net_u(xcol_pred, tcol_pred)
Unp_pred = usol_pred.reshape(x_plot.numel(), t_plot.numel()).cpu().numpy()
xnp_plot = x_plot.cpu().numpy(); tnp_plot = t_plot.cpu().numpy()

print("Generating plot...")
plt.rcParams['font.size'] = '15'; fig = plt.figure(figsize=(5, 6)); ax = fig.add_subplot(111)
plt.xlabel(r"$t$"); plt.ylabel(r"$x$"); plt.title("Multi-Constraint SOFL-PINN Solution")
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
plt.title(r"Multi-Constraint SOFL-PINN Error")

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
