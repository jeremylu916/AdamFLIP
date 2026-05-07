'''
labmda 1, lambda 2
'''
from matplotlib.ticker import ScalarFormatter
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
        self.X_u_full = X_u_stage
        self.u_full = u_stage
        self.X_f_full = X_f_stage

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
    f_pred = model.net_f(model.X_f_full[:, 0:1], model.X_f_full[:, 1:2])
    null_f = torch.zeros((model.X_f_full.shape[0], 1), device=device)
    return model.loss(f_pred, null_f)

def boundary_loss(model):
    u_pred = model.net_u(model.X_u_full[:, 0:1], model.X_u_full[:, 1:2])
    return model.loss(u_pred, model.u_full)


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

def FL_PINN(f, h, df, dh, x_start, Kp, Ki=0, eta=0.01, max_iter=1000, tol=1e-6):
    x = np.array(x_start, dtype=np.float32)
    history = []
    lambda_physics_history = []  # Store lambda_physics values
    lambda_boundary_history = []
    loss_epoch_history = []
    loss_boundary_history = []
    loss_physics_history = []
    loss_total_history = []
    Integral = np.zeros(Kp.shape[0]) # Integral is now a vector

    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    m, v, t = np.zeros_like(x), np.zeros_like(x), 0
    
    pbar = trange(max_iter, desc="FL-PINN Training")
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
        lambda_vector = -solve(JhJht + 1e-6 * I, rhs, assume_a='pos')

        # --- EXPLICIT LAMBDA UNPACKING ---
        lambda_boundary = lambda_vector[0]
        lambda_physics = lambda_vector[1]
        lambda_boundary_history.append(lambda_boundary)
        lambda_physics_history.append(lambda_physics)

        grad_h = true_J_h[0, :]
        grad_f = true_J_h[1, :]

        # The KKT gradient is the weighted sum of the constraint gradients
      
        KKT_grad = true_grad_f + (true_J_h.T @ lambda_vector).flatten()
        #KKT_grad =true_grad_f + (grad_h * lambda_boundary) + (grad_f * lambda_physics)
        
        clip_value = 10.0
        grad_norm = np.linalg.norm(KKT_grad)
        if grad_norm > clip_value:
            KKT_grad = KKT_grad * (clip_value / grad_norm)

        # t += 1
        # m = beta1 * m + (1 - beta1) * KKT_grad
        # v = beta2 * v + (1 - beta2) * (KKT_grad**2)
        # m_hat = m / (1 - beta1**t)
        # v_hat = v / (1 - beta2**t)
        # step_vector = eta * m_hat / (np.sqrt(v_hat) + epsilon)
        # x_new = x - step_vector
        # step_size = np.linalg.norm(step_vector)
        x_new = x - eta * KKT_grad
        
        if (iteration + 1) % 100 == 0:
            true_f_val = f(x)
            boundary_loss = float(np.abs(true_h_x[0]))
            physics_loss = float(np.abs(true_h_x[1]))
            total_loss = boundary_loss + physics_loss
            KKT_gap = np.max([np.linalg.norm(KKT_grad), np.max(np.abs(true_h_x))])
            pbar.set_postfix({
                'boundary_loss': f'{boundary_loss:.2e}',
                'physics_loss': f'{physics_loss:.2e}',
                'lambda_b': f'{lambda_boundary:.2e}',
                'lambda_p': f'{lambda_physics:.2e}',
            })
            history.append((true_f_val, KKT_gap, boundary_loss, physics_loss))
            loss_epoch_history.append(iteration + 1)
            loss_boundary_history.append(boundary_loss)
            loss_physics_history.append(physics_loss)
            loss_total_history.append(total_loss)

        if np.linalg.norm(x_new - x) < tol:
            break

        x = x_new
        
        if np.isnan(x).any():
            print("\nError: NaN values detected. Stopping.")
            break
            
    return x, history, lambda_boundary_history, lambda_physics_history, loss_epoch_history, loss_boundary_history, loss_physics_history, loss_total_history

# 1. Load Data
N_u = 100
N_f = 20000
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
lambda_boundary_full = []
lambda_physics_full = []
# --- 2x2 Gain Matrices for the two constraints ---
Kp = np.array([[500.0, 0.0], [0.0, 500.0]])
Ki = np.array([[0.01, 0.0], [0.0, 0.01]])
eta = 1e-4 
tol = 1e-8
max_iter = 10000

# for i, (t_max, iters) in enumerate(stages):
#     print(f"\n--- Causal Training: Stage {i+1}/{len(stages)} (t <= {t_max}) ---")

#     X_f_stage_np = X_f_train_np[X_f_train_np[:, 1] <= t_max]
    
#     pinn_model.set_data_for_stage(
#         pinn_model.X_u_master,
#         pinn_model.u_master,
#         torch.tensor(X_f_stage_np, dtype=torch.float32).to(device)
#     )

#     x_current, history_stage,lambda_boundary_stage, lambda_physics_stage  = SOFL_Adam_PINN(
#         f, h, df, dh,
#         x_current, Kp, Ki=Ki, eta=eta, 
#         max_iter=iters, tol=tol
#     )
#     full_history.extend(history_stage)
#     lambda_boundary_full.extend(lambda_boundary_stage)
#     lambda_physics_full.extend(lambda_physics_stage)

# # x_final = x_current

adam_pretrain_iters = 2000
adam_lr = 1e-3

print(f"\n--- Adam warm-start ({adam_pretrain_iters} iters) ---")
pinn_model.set_data_for_stage(
    pinn_model.X_u_master,
    pinn_model.u_master,
    pinn_model.X_f_master,
 )
pinn_model.net.train()
optimizer = torch.optim.Adam(pinn_model.net.parameters(), lr=adam_lr)
pbar_pre = trange(adam_pretrain_iters, desc="Adam Warm-start")
for i in pbar_pre:
    optimizer.zero_grad()
    loss = boundary_loss(pinn_model) + residual_loss(pinn_model)
    loss.backward()
    optimizer.step()

print(f"\n--- Training: single stage ({max_iter} iters) ---")
x_current = get_flat_params(pinn_model.net)

x_final, history, lambda_boundary_history, lambda_physics_history, loss_epoch_history, loss_boundary_history, loss_physics_history, loss_total_history = FL_PINN(
    f, h, df, dh,
    x_current, Kp, Ki=Ki, eta=eta,
    max_iter=max_iter, tol=tol
)
full_history.extend(history)
lambda_boundary_full.extend(lambda_boundary_history)
lambda_physics_full.extend(lambda_physics_history)

os.makedirs("./training_data", exist_ok=True)
np.savez(
    "./training_data/FL_PINN_loss_history.npz",
    epochs=np.array(loss_epoch_history),
    boundary_loss=np.array(loss_boundary_history),
    physics_loss=np.array(loss_physics_history),
    total_loss=np.array(loss_total_history),
)
print("Saved loss history to ./training_data/FL_PINN_loss_history.npz")

# 4. Final Evaluation using the full master dataset
set_flat_params(pinn_model.net, x_final)
pinn_model.set_data_for_stage(pinn_model.X_u_master, pinn_model.u_master, pinn_model.X_f_master)
final_losses = h(x_final)

print("\n--- Optimization Finished ---")
# print(f"Final Boundary Loss (Constraint 1): {final_losses[0]:.4f}")
# print(f"Final Physics Loss (Constraint 2): {final_losses[1]:.4f}")
save_model(pinn_model.net, "./trained_FL_PINN_model.pth")

# --- Evaluation ---
# ... (Evaluation code is the same)
data = scipy.io.loadmat('burgers_shock.mat')
t_exact = data['t'].flatten()[:, None]; x_exact = data['x'].flatten()[:, None]
Exact = np.real(data['usol']); T, X = np.meshgrid(t_exact, x_exact)
X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
u_truth = Exact.flatten()[:,None]

load_model(pinn_model.net, "./trained_FL_PINN_model.pth")
x_full = torch.from_numpy(X_star[:, 0:1]).float().to(device)
t_full = torch.from_numpy(X_star[:, 1:2]).float().to(device)
with torch.no_grad():
    u_pred = pinn_model.net_u(x_full, t_full).cpu().numpy()

l2_loss_mse = np.mean((u_truth - u_pred)**2)
print(f"Final L2 Loss (MSE): {l2_loss_mse:.4f}")
error = np.linalg.norm(u_truth - u_pred, 2) / np.linalg.norm(u_truth, 2)
print(f"Final L2 Relative Error: {error:.4f}")

# --- Testing physics loss (residual MSE on full grid) ---
x_full_req = x_full.clone().detach().requires_grad_(True)
t_full_req = t_full.clone().detach().requires_grad_(True)
f_pred_test = pinn_model.net_f(x_full_req, t_full_req).detach().cpu().numpy()
physics_test_loss = np.mean(f_pred_test ** 2)
print(f"Test Physics Loss (MSE of residual): {physics_test_loss:.6f}")

# --- Testing boundary loss (MSE on x=±1 or t=0) ---
x_star = X_star[:, 0]
t_star = X_star[:, 1]
boundary_mask = (np.isclose(x_star, -1.0) | np.isclose(x_star, 1.0) | np.isclose(t_star, 0.0))
u_boundary_pred = u_pred[boundary_mask]
u_boundary_true = u_truth[boundary_mask]
boundary_test_loss = np.mean((u_boundary_pred - u_boundary_true) ** 2)
print(f"Test Boundary Loss (MSE): {boundary_test_loss:.6f}")

# --- Testing initial loss (MSE at t=0) ---
initial_mask = np.isclose(t_star, 0.0)
u_initial_pred = u_pred[initial_mask]
u_initial_true = u_truth[initial_mask]
initial_test_loss = np.mean((u_initial_pred - u_initial_true) ** 2)
print(f"Test Initial Loss (MSE): {initial_test_loss:.6f}")

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
plt.xlabel(r"$t$"); plt.ylabel(r"$x$"); plt.title("FL-PINN Solution")
img_handle = ax.imshow(Unp_pred, interpolation='nearest', cmap='rainbow',
                       extent=[tnp_plot.min(), tnp_plot.max(), xnp_plot.min(), xnp_plot.max()],
                       origin='lower', aspect='auto', vmin=-1.0, vmax=1.0)          
divider = make_axes_locatable(ax); cax = divider.append_axes("right", size="5%", pad=0.10)
cbar = fig.colorbar(img_handle, cax=cax); cbar.ax.tick_params(labelsize=10)
output_filename = "./figures/solution/FL_PINN_solution.png"
plt.savefig(output_filename, dpi=300, bbox_inches='tight', pad_inches=0.1)
print(f"Plot saved successfully as {output_filename}")
plt.show()

# --- Save solution data to .mat file in ./data ---
os.makedirs('./data', exist_ok=True)
try:
    scipy.io.savemat('./data/FL_PINN_solution.mat', {'solution': Unp_pred, 'x': xnp_plot, 't': tnp_plot})
    print('Saved solution matrix to ./data/FL_PINN_solution.mat')
except Exception as e:
    print('Warning: failed to save solution .mat file:', e)

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
plt.title(r"FL-PINN Error")

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

output_filename_err = "./figures/solution/FL_PINN_error.png"
plt.savefig(output_filename_err, dpi=300, bbox_inches='tight', pad_inches=0.1)
print(f"Error plot saved successfully as {output_filename_err}")
plt.show()

# --- Save error grid to .mat file in ./data ---
try:
    scipy.io.savemat('./data/FL_PINN_error.mat', {'error': error_grid, 'x': xnp_plot, 't': tnp_plot})
    print('Saved error matrix to ./data/FL_PINN_error.mat')
except Exception as e:
    print('Warning: failed to save error .mat file:', e)

def plot_optimization_history_multi_constraint(*Histories, legends=None, linewidth=1, fontsize=20, ticksize=15, legendsize=15, savename=None):
    """New plotting function for the multi-constraint problem."""
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
    plt.title('Boundary Loss', fontsize=fontsize)
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
    plt.title('Physics Loss', fontsize=fontsize)
    plt.grid(True); plt.xticks(fontsize=ticksize); plt.yticks(fontsize=ticksize)

    plt.tight_layout()
    if savename:
        dirpath = os.path.dirname(savename)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        plt.savefig(savename, bbox_inches='tight')
    plt.show()

plot_optimization_history_multi_constraint(full_history, legends=['FL-PINN'], linewidth=3, savename='./figures/FL-PINN_training_history.png')