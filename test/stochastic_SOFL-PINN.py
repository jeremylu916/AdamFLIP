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

def plot_optimization_history_2(*Histories, optimal_value=None, legends=None, linewidth=1, fontsize=20, ticksize=15, legendsize=15, savename=None):
    plt.figure(figsize=(21, 5))
    plt.subplot(1, 3, 1)
    if optimal_value is not None:
        plt.axhline(optimal_value, color='g', linestyle='--', label='Optimal $f(x)$', linewidth=linewidth)
    for i, history in enumerate(Histories):
        label = legends[i] if legends is not None and i < len(legends) else f'Run {i+1}'
        f_values, _, _ = zip(*history)
        plt.plot(f_values, label=label, linewidth=linewidth)
    plt.xlabel('Iteration (x100)', fontsize=fontsize)
    plt.ylabel('$f(x)$', fontsize=fontsize)
    plt.legend(fontsize=legendsize)
    plt.title('Function Value', fontsize=fontsize)
    plt.grid(True); plt.xticks(fontsize=ticksize); plt.yticks(fontsize=ticksize)

    plt.subplot(1, 3, 2)
    for i, history in enumerate(Histories):
        label = legends[i] if legends is not None and i < len(legends) else f'Run {i+1}'
        _, KKT_gaps, _ = zip(*history)
        plt.plot(KKT_gaps, label=label, linewidth=linewidth)
    plt.xlabel('Iteration (x100)', fontsize=fontsize)
    plt.ylabel('KKT Gap', fontsize=fontsize)
    plt.yscale('log'); plt.legend(fontsize=legendsize)
    plt.title('KKT Gap', fontsize=fontsize)
    plt.grid(True); plt.xticks(fontsize=ticksize); plt.yticks(fontsize=ticksize)

    plt.subplot(1, 3, 3)
    for i, history in enumerate(Histories):
        label = legends[i] if legends is not None and i < len(legends) else f'Run {i+1}'
        _, _, h_violations = zip(*history)
        plt.plot(h_violations, label=label, linewidth=linewidth)
    plt.xlabel('Iteration (x100)', fontsize=fontsize)
    plt.ylabel('Constraint Violation |h(x)|', fontsize=fontsize)
    plt.yscale('log'); plt.legend(fontsize=legendsize)
    plt.title('Constraint Violation', fontsize=fontsize)
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
        self.X_u_stage = self.X_u_master
        self.u_stage = self.u_master
        self.X_f_stage = self.X_f_master
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
    
    def set_data_batch(self, X_u_batch, u_batch, X_f_batch):
        self.x_u_batch = X_u_batch[:, 0:1]; self.t_u_batch = X_u_batch[:, 1:2]
        self.u_batch = u_batch
        self.x_f_batch = X_f_batch[:, 0:1]; self.t_f_batch = X_f_batch[:, 1:2]
        self.null_batch = torch.zeros((X_f_batch.shape[0], 1), device=device)

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

    def get_losses_and_grads(self, on_stage_data=True):
        self.net.zero_grad()
        if on_stage_data:
            x_u, t_u, u = self.X_u_stage[:,0:1], self.X_u_stage[:,1:2], self.u_stage
            x_f, t_f = self.X_f_stage[:,0:1], self.X_f_stage[:,1:2]
            null = torch.zeros((self.X_f_stage.shape[0], 1), device=device)
        else:
            x_u, t_u, u = self.x_u_batch, self.t_u_batch, self.u_batch
            x_f, t_f = self.x_f_batch, self.t_f_batch
            null = self.null_batch
        
        loss_h = self.loss(self.net_u(x_u, t_u), u)
        loss_h.backward(retain_graph=True)
        grad_h = get_flat_grads(self.net).reshape(1, -1)
        self.net.zero_grad()
        
        loss_f = self.loss(self.net_f(x_f, t_f), null)
        loss_f.backward()
        grad_f = get_flat_grads(self.net)
        self.net.zero_grad()
        
        return loss_f.item(), loss_h.item(), grad_f, grad_h

def get_flat_weights(model):
    return np.concatenate([p.data.cpu().numpy().flatten() for p in model.parameters()])

def get_flat_grads(model):
    return np.concatenate([p.grad.cpu().numpy().flatten() if p.grad is not None else np.zeros(p.numel()) for p in model.parameters()])

def set_flat_params(model, flat_params):
    flat_params = flat_params.astype(np.float32)
    pointer = 0
    for p in model.parameters():
        num_params = p.numel()
        p.data = torch.from_numpy(flat_params[pointer:pointer + num_params]).view_as(p).to(device)
        pointer += num_params

pinn_model = None

def sub_sample(Dx_stage, Dy_stage, batch_size):
    X_f_stage, X_u_stage, u_stage = Dx_stage, Dy_stage[0], Dy_stage[1]
    idx_f = np.random.choice(X_f_stage.shape[0], batch_size, replace=False)
    idx_u = np.random.choice(X_u_stage.shape[0], batch_size, replace=False)
    sub_Dx = X_f_stage[idx_f, :]
    sub_Dy = (X_u_stage[idx_u, :], u_stage[idx_u, :])
    return sub_Dx, sub_Dy

# --- MINI-BATCH OPTIMIZER with Lambda Smoothing and Kp Annealing ---
def SOFL_Adam_Stochastic(Dx_stage, Dy_stage, x_start, Kp_initial, Ki=0, eta=0.01, max_iter=1000, tol=1e-6, batch_size=20, decay_rate=0.9999):
    x = np.array(x_start, dtype=np.float32)
    history = []
    Integral = 0.0
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    m, v, t = np.zeros_like(x), np.zeros_like(x), 0
    beta_lambda = 0.9
    lambda_avg = np.array([[0.0]])
    Kp = Kp_initial

    pbar = trange(max_iter, desc="SOFL+Adam Stochastic")
    for iteration in pbar:
        sub_Dx_np, sub_Dy_np = sub_sample(Dx_stage, Dy_stage, batch_size)
        X_u_batch = torch.from_numpy(sub_Dy_np[0]).to(device)
        u_batch = torch.from_numpy(sub_Dy_np[1]).to(device)
        X_f_batch = torch.from_numpy(sub_Dx_np).to(device)
        pinn_model.set_data_batch(X_u_batch, u_batch, X_f_batch)
        set_flat_params(pinn_model.net, x)

        _, stoch_h_x_val, stoch_grad_f, stoch_J_h = pinn_model.get_losses_and_grads(on_stage_data=False)
        stoch_h_x = np.array([stoch_h_x_val])

        JhJht = stoch_J_h @ stoch_J_h.T
        Pcontrol = -Kp @ stoch_h_x
        Integral = Integral + stoch_h_x
        Icontrol = Ki * Integral
        rhs = Pcontrol + Icontrol + stoch_J_h @ stoch_grad_f
        I = np.eye(np.shape(JhJht)[0])
        lambda_noisy = -solve(JhJht + 1e-6 * I, rhs, assume_a='pos')
        lambda_avg = beta_lambda * lambda_avg + (1 - beta_lambda) * lambda_noisy

        stoch_KKT_grad = (stoch_grad_f + (stoch_J_h.T @ lambda_avg).flatten())
        
        clip_value = 10.0
        grad_norm = np.linalg.norm(stoch_KKT_grad)
        if grad_norm > clip_value:
            stoch_KKT_grad = stoch_KKT_grad * (clip_value / grad_norm)

        t += 1
        m = beta1 * m + (1 - beta1) * stoch_KKT_grad
        v = beta2 * v + (1 - beta2) * (stoch_KKT_grad**2)
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        step_vector = eta * m_hat / (np.sqrt(v_hat) + epsilon)
        x = x - step_vector
        step_size = np.linalg.norm(step_vector)
        
        Kp = Kp * decay_rate

        if iteration % 100 == 0:
            set_flat_params(pinn_model.net, x)
            full_loss_f_val, full_loss_h_val, _, _ = pinn_model.get_losses_and_grads(on_stage_data=True)
            KKT_gap = np.max([grad_norm, np.abs(full_loss_h_val)])
            pbar.set_postfix({
                'physics_loss': f'{full_loss_f_val:.2e}', 
                'boundary_loss': f'{full_loss_h_val:.2e}',
                'Kp': f'{Kp[0,0]:.2e}', 'Step': f'{step_size:.2e}'
            })
            history.append((full_loss_f_val, KKT_gap, full_loss_h_val))

        if step_size < tol:
            print(f"\nConvergence at iter {iteration}: Step size ({step_size:.2e}) < tol ({tol:.2e}).")
            break
            
    return x, history

# --- Main Execution ---

# 1. Load Data
N_u = 100; N_f = 10000
x_upper = np.ones((N_u//4, 1), dtype=np.float32); t_upper = np.random.rand(N_u//4, 1).astype(np.float32)
x_lower = -np.ones((N_u//4, 1), dtype=np.float32); t_lower = np.random.rand(N_u//4, 1).astype(np.float32)
t_zero = np.zeros((N_u//2, 1), dtype=np.float32); x_zero = -1+2*np.random.rand(N_u//2, 1).astype(np.float32)
X_upper = np.hstack((x_upper, t_upper)); X_lower = np.hstack((x_lower, t_lower)); X_zero = np.hstack((x_zero, t_zero))
X_u_train_np = np.vstack((X_upper, X_lower, X_zero))
u_upper = np.zeros((N_u//4, 1), dtype=np.float32); u_lower = np.zeros((N_u//4, 1), dtype=np.float32)
u_zero = -np.sin(np.pi * x_zero).astype(np.float32)
u_train_np = np.vstack((u_upper, u_lower, u_zero))
index = np.arange(N_u); np.random.shuffle(index)
X_u_train_np = X_u_train_np[index, :]; u_train_np = u_train_np[index, :]
X_f_train_np = np.random.uniform([-1, 0], [1, 1], size=(N_f, 2)).astype(np.float32)
X_f_train_np = np.vstack((X_f_train_np, X_u_train_np))

# 2. Initialize PINN model
pinn_model = PhysicsInformedNN(X_u_train_np, u_train_np, X_f_train_np)

# 3. Setup and Run Causal Training with the STOCHASTIC Optimizer
x_current = get_flat_weights(pinn_model.net)
stages = [(0.33, 10000), (0.66, 10000), (1.0, 10000)]
full_history = []

# --- HYPERPARAMETER CORRECTIONS ---
Kp_initial = np.array([[500.0]]) # Lower initial Kp
Ki = 0.01
eta = 1e-4 
tol = 1e-8
batch_size = 64
decay_rate = 0.99995 # Slower decay rate

Kp_current = Kp_initial
for i, (t_max, iters) in enumerate(stages):
    print(f"\n--- Causal Training: Stage {i+1}/{len(stages)} (t <= {t_max}) ---")

    X_f_stage_np = X_f_train_np[X_f_train_np[:, 1] <= t_max]
    Dx_stage = X_f_stage_np
    Dy_stage = (X_u_train_np, u_train_np)

    pinn_model.set_data_for_stage(
        pinn_model.X_u_master,
        pinn_model.u_master,
        torch.tensor(X_f_stage_np, dtype=torch.float32).to(device)
    )

    x_current, history_stage = SOFL_Adam_Stochastic(
        Dx_stage, Dy_stage,
        x_current, Kp_current, Ki=Ki, eta=eta, 
        max_iter=iters, tol=tol, batch_size=batch_size, decay_rate=decay_rate
    )
    full_history.extend(history_stage)
    # Update Kp for the next stage based on the decay during this stage
    Kp_current = Kp_current * (decay_rate ** iters)

x_final = x_current

# 4. Final Evaluation using the full master dataset
set_flat_params(pinn_model.net, x_final)
pinn_model.set_data_for_stage(pinn_model.X_u_master, pinn_model.u_master, pinn_model.X_f_master)
final_loss_f_val, final_loss_h_val, _, _ = pinn_model.get_losses_and_grads(on_stage_data=True)

print("\n--- Optimization Finished ---")
print(f"Final Physics Loss (Objective): {final_loss_f_val:.4f}")
print(f"Final Boundary Loss (Constraint): {final_loss_h_val:.4f}")
save_model(pinn_model.net, "./trained_pinn_model.pth")

# --- Evaluation ---
print("\n--- Evaluation ---")
data = scipy.io.loadmat('burgers_shock.mat')
t_exact = data['t'].flatten()[:, None]; x_exact = data['x'].flatten()[:, None]
Exact = np.real(data['usol']); T, X = np.meshgrid(t_exact, x_exact)
X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
u_truth = Exact.flatten()[:,None]

load_model(pinn_model.net, "./trained_pinn_model.pth")
x_full = torch.from_numpy(X_star[:, 0:1]).float().to(device)
t_full = torch.from_numpy(X_star[:, 1:2]).float().to(device)
with torch.no_grad():
    u_pred = pinn_model.net_u(x_full, t_full).cpu().numpy()

l2_loss_mse = np.mean((u_truth - u_pred)**2)
print(f"Final L2 Loss (MSE): {l2_loss_mse:.4f}")
error = np.linalg.norm(u_truth - u_pred, 2) / np.linalg.norm(u_truth, 2)
print(f"Final L2 Relative Error: {error:.4f}")

# --- Plotting ---
plot_optimization_history_2(full_history, legends=['SOFL-PINN (Stochastic)'], linewidth=3, savename='./figures/training_history.png')

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
plt.xlabel(r"$t$"); plt.ylabel(r"$x$"); plt.title(r"$u(x,t)$")
img_handle = ax.imshow(Unp_pred, interpolation='nearest', cmap='rainbow',
                       extent=[tnp_plot.min(), tnp_plot.max(), xnp_plot.min(), xnp_plot.max()],
                       origin='lower', aspect='auto', vmin=-1.0, vmax=1.0)          
divider = make_axes_locatable(ax); cax = divider.append_axes("right", size="5%", pad=0.10)
cbar = fig.colorbar(img_handle, cax=cax); cbar.ax.tick_params(labelsize=10)
output_filename = "./figures/solution/SOFL-PINN_solution.png"
plt.savefig(output_filename, dpi=300, bbox_inches='tight', pad_inches=0.1)
print(f"Plot saved successfully as {output_filename}")
plt.show()

