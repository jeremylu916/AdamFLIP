import torch
import torch.nn as nn
import numpy as np
import math
import matplotlib.pyplot as plt
import torch.optim as optim
from matplotlib import cm
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
from torch import linalg as LA
import random
import os
import time
from scipy.linalg import solve
from tqdm import trange
from mpl_toolkits.axes_grid1 import make_axes_locatable

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

# --- 1. SET UP DEVICE ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device} 🚀")

# --- 2. Define grid and parameters ---
M = 100
N = 100
spatial_step_h = np.pi / M  # RENAMED from h to avoid conflict with constraint function h
tau = 1 / N

t = torch.linspace(0, 1, N, device=device).requires_grad_(True)
x = torch.linspace(0, np.pi, M, device=device).requires_grad_(True)
alpha = torch.linspace(0, 0.9, N, device=device).reshape(-1, 1).requires_grad_(True)
T, X = torch.meshgrid(t, x, indexing='ij')
x1 = X.flatten()[:, None]
t1 = T.flatten()[:, None]
X_input = torch.cat((x1, t1), dim=1).to(device)

# --- 3. Conditions ---
def initial_condition(x):
    return torch.sin(x) * 0 

def boundary_condition(t):
    return torch.zeros_like(t)

num_ic = 100
num_bc = 100

x_ic = torch.linspace(0, torch.pi, num_ic, device=device).view(-1, 1)
t_ic = torch.zeros_like(x_ic, device=device)
u_ic = initial_condition(x_ic)

t_bc = torch.linspace(0, 1, num_bc, device=device).view(-1, 1)
x_bc_0 = torch.zeros_like(t_bc, device=device)
x_bc_pi = torch.full_like(t_bc, torch.pi, device=device)
u_bc = boundary_condition(t_bc)

# Note: num_pde is not used in this script's loss functions, but we keep it
num_pde = 5000
x_pde = (torch.rand(num_pde, 1, device=device) * torch.pi)
t_pde = torch.rand(num_pde, 1, device=device)

# --- 4. True function and source term ---
def u_true(x, t):
    t = t.reshape(-1, 1)
    x = x.reshape(1, -1)
    return torch.sin(x) * t ** 3

def f_source(x, t, alpha):
    t = t.reshape(-1, 1)
    x = x.reshape(1, -1)
    gamma_4 = torch.lgamma(torch.tensor(4.0, device=device)).exp()
    term = (gamma_4 / torch.lgamma(4 - alpha).exp()) * t ** (3 - alpha) * torch.sin(x) + t ** 3 * torch.sin(x)
    return term[1:, 1:-1]

def S(M):
    S = torch.diag(torch.full((M,), -2, dtype=torch.float, device=device))
    i, j = np.indices(S.shape)
    S[i == j - 1] = 1
    S[i == j + 1] = 1
    return S

# --- 5. MLP Model ---
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

criterion = nn.MSELoss()

# --- 6. FPDE operator ---
def FPDE(alpha, u_hat):
    alpha_1 = 1 - alpha
    i_minus_j = torch.arange(1, N, device=device).view(-1, 1) - torch.arange(0, N - 1, device=device).view(1, -1)
    A = (torch.tril(i_minus_j) ** alpha_1[1:]) - 2 * (torch.tril(i_minus_j - 1) ** alpha_1[1:]) + (torch.tril(i_minus_j - 2).fill_diagonal_(0) ** alpha_1[1:])
    A = A.fill_diagonal_(1)
    B = torch.matmul(A, u_hat[1:, :])
    i_minus_j_1 = torch.arange(1, N, device=device).reshape(-1, 1)
    c = (i_minus_j_1 ** alpha_1[1:]) - ((i_minus_j_1 - 1) ** alpha_1[1:])
    B = B - torch.matmul(c, u_hat[0, :].view(1, -1))
    a = tau ** (-alpha) / torch.lgamma(2 - alpha).exp()
    d = torch.mul(a[1:], B)
    return d[:, 1:-1]

# --- 7. Model and Loss Functions for SOFL ---
layers = [2, 50, 100, 1]
pinn_model = MLP(layers).to(device)

def residual_loss(model):
    u_hat = model(X_input) * X_input[:, 0].view(-1, 1) * (torch.pi - X_input[:, 0]).view(-1, 1)
    u_hat = u_hat.reshape(N, M)
    # --- CORRECTED: Use spatial_step_h ---
    u_nn_xx = spatial_step_h ** (-2) * torch.matmul(u_hat, S(M))[1:, 1:-1]
    loss = criterion(f_source(x, t, alpha), FPDE(alpha, u_hat) - u_nn_xx)
    return loss

def ibc_loss(model):
    ic_input = torch.cat((x_ic, t_ic), dim=-1)
    u_pred_ic = model(ic_input)
    bc_x0_input = torch.cat((x_bc_0, t_bc), dim=-1)
    bc_xpi_input = torch.cat((x_bc_pi, t_bc), dim=-1)
    loss_ic = criterion(u_pred_ic, u_ic)
    u_pred_bc_0 = model(bc_x0_input)
    u_pred_bc_pi = model(bc_xpi_input)
    loss_bc = criterion(u_pred_bc_0, u_bc) + criterion(u_pred_bc_pi, u_bc)
    loss = loss_ic + loss_bc
    return loss

def boundary_loss(model):
    bc_x0_input = torch.cat((x_bc_0, t_bc), dim=-1)
    bc_xpi_input = torch.cat((x_bc_pi, t_bc), dim=-1)
    u_pred_bc_0 = model(bc_x0_input)
    u_pred_bc_pi = model(bc_xpi_input)
    loss_bc = criterion(u_pred_bc_0, u_bc) + criterion(u_pred_bc_pi, u_bc)
    return loss_bc

def initial_loss(model):
    ic_input = torch.cat((x_ic, t_ic), dim=-1)
    u_pred_ic = model(ic_input)
    loss_ic = criterion(u_pred_ic, u_ic)
    return loss_ic

# --- 8. SOFL Helper Functions ---
def get_flat_params(model):
    """Gets all model parameters as a flat NumPy vector."""
    return np.concatenate([p.data.cpu().numpy().flatten() for p in model.parameters()])

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

def get_grads_from_loss(loss, model_params, allow_unused=False):
    """Helper to compute gradients using autograd.grad."""
    grads = torch.autograd.grad(
        loss, model_params, grad_outputs=torch.ones_like(loss),
        retain_graph=True, allow_unused=allow_unused
    )
    flat_grads = []
    for g, p in zip(grads, model_params):
        if g is not None:
            flat_grads.append(g.cpu().numpy().flatten())
        else:
            flat_grads.append(np.zeros(p.numel()))
    return np.concatenate(flat_grads)

# --- 9. SOFL Wrapper Functions ---
def f(x):
    """Objective function is now the physics loss."""
    set_flat_params(pinn_model, x)
    return residual_loss(pinn_model).item()

def df(x):
    """Gradient of the objective function (physics loss)."""
    set_flat_params(pinn_model, x)
    model_params = list(pinn_model.parameters())
    loss_res = residual_loss(pinn_model)
    grad_res = get_grads_from_loss(loss_res, model_params, allow_unused=True)
    return grad_res

def h(x):
    """Constraint function now returns boundary and initial losses."""
    set_flat_params(pinn_model, x)
    loss_bc_val = boundary_loss(pinn_model).item()
    loss_ic_val = initial_loss(pinn_model).item()
    return np.array([loss_bc_val, loss_ic_val])

def dh(x):
    """Constraint Jacobian now returns gradients for boundary and initial losses."""
    set_flat_params(pinn_model, x)
    model_params = list(pinn_model.parameters())
    
    # Gradient of boundary loss (h1)
    loss_bc = boundary_loss(pinn_model)
    grad_bc = get_grads_from_loss(loss_bc, model_params, allow_unused=True)
    
    # Gradient of initial loss (h2)
    loss_ic = initial_loss(pinn_model)
    grad_ic = get_grads_from_loss(loss_ic, model_params, allow_unused=True)
    
    return np.vstack([grad_bc, grad_ic])

# --- 10. SOFL Optimizer ---
def SOFL_Adam_PINN(f, h, df, dh, x_start, Kp, Ki=0, eta=0.01, max_iter=1000, tol=1e-6, weight_decay=0.0):
    # === Logging setup ===
    log_dir = "../training_logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "FL-PINN_training_loss.txt")

    with open(log_path, "w") as log_file:
        log_file.write("Iter\tf_val\tKKT_gap\tbc_loss(c1)\tic_loss(c2)\n")

    os.makedirs("./training_data", exist_ok=True)
    loss_epochs = []
    loss_kkt_history = []
    loss_physics_history = []
    loss_boundary_history = []
    loss_initial_history = []

    # === Initialization ===
    x = np.array(x_start, dtype=np.float32)
    history = []
    Integral = np.zeros(Kp.shape[0])

    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    m, v, t = np.zeros_like(x), np.zeros_like(x), 0
    
    pbar = trange(max_iter, desc="SOFL+Adam")
    for iteration in pbar:
        set_flat_params(pinn_model, x)

        true_grad_f = df(x) # Zero vector
        true_J_h = dh(x)    # 2xN matrix
        true_h_x = h(x)     # 2-element vector

        JhJht = true_J_h @ true_J_h.T
        Pcontrol = -Kp @ true_h_x
        Integral = Integral + true_h_x
        Icontrol = Ki @ Integral
        rhs = Pcontrol + Icontrol + true_J_h @ true_grad_f
        I = np.eye(np.shape(JhJht)[0])
        lambda_ = -solve(JhJht + 1e-6 * I, rhs, assume_a='pos')

        KKT_grad = true_grad_f + (true_J_h.T @ lambda_).flatten()
        
        if weight_decay > 0:
            KKT_grad = KKT_grad + weight_decay * x
            
        clip_value = 10.0
        grad_norm = np.linalg.norm(KKT_grad)
        if grad_norm > clip_value:
            KKT_grad = KKT_grad * (clip_value / grad_norm)

        # === Adam Update ===
        t += 1
        m = beta1 * m + (1 - beta1) * KKT_grad
        v = beta2 * v + (1 - beta2) * (KKT_grad**2)
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        step_vector = eta * m_hat / (np.sqrt(v_hat) + epsilon)
        x_new = x - step_vector
        step_size = np.linalg.norm(step_vector)
        
        # === Logging and progress ===
        if iteration % 50 == 0 or iteration + 1 == max_iter:
            true_f_val = f(x)
            KKT_gap = np.max([np.linalg.norm(KKT_grad), np.max(np.abs(true_h_x))])
            bc_loss, ic_loss_val = np.abs(true_h_x[0]), np.abs(true_h_x[1])
            pbar.set_postfix({
                'bc_loss(c1)': f'{bc_loss:.2e}',
                'ic_loss(c2)': f'{ic_loss_val:.2e}',
                'KKT_gap': f'{KKT_gap:.2e}',
            })

            # Save to history and file
            history.append((true_f_val, KKT_gap, bc_loss, ic_loss_val))
            with open(log_path, "a") as log_file:
                log_file.write(f"{iteration}\t{true_f_val:.6e}\t{KKT_gap:.6e}\t{bc_loss:.6e}\t{ic_loss_val:.6e}\n")

            if iteration % 100 == 0:
                loss_epochs.append(iteration)
                loss_kkt_history.append(KKT_gap)
                loss_physics_history.append(true_f_val)  # physics loss is f_val
                loss_boundary_history.append(bc_loss)
                loss_initial_history.append(ic_loss_val)

        # === Convergence or NaN check ===
        if step_size < tol:
            print(f"\nConvergence at iter {iteration}: Step size ({step_size:.2e}) < tol ({tol:.2e}).")
            break

        if np.isnan(x_new).any():
            print("\nError: NaN values detected. Stopping.")
            break

        x = x_new

    np.savez(
        "./training_data/AdaFL_PINN_loss_history.npz",
        epochs=np.array(loss_epochs),
        kkt_gap=np.array(loss_kkt_history),
        physics_loss=np.array(loss_physics_history),
        boundary_loss=np.array(loss_boundary_history),
        initial_loss=np.array(loss_initial_history),
    )
    print("Saved loss history to ./training_data/AdaFL_PINN_loss_history.npz")

    return x, history


 # --- 12. Main Execution ---
if __name__ == "__main__":
    
    # SOFL Hyperparameters
    SOFL_MAX_ITER = 5000
    SOFL_ETA = 1e-3
    
    # 2x2 Gain Matrices for [physics_loss, ibc_loss]
    Kp = np.array([[500.0, 0.0], [0.0, 500.0]]) # Start with 10x higher weight on physics
    Ki = np.array([[0.0, 0.0], [0.0, 0.0]])  # Start with no integral gain
    SOFL_TOL = 1e-8
    WEIGHT_DECAY = 1e-5 # L2 Regularization

    x_start = get_flat_params(pinn_model)
    
    print("\n--- Starting SOFL+Adam Training for FPDE ---")
    start_time = time.time()
    
    x_final, history = SOFL_Adam_PINN(
        f, h, df, dh, x_start,
        Kp=Kp,
        Ki=Ki,
        eta=SOFL_ETA,
        max_iter=SOFL_MAX_ITER,
        tol=SOFL_TOL,
        weight_decay=WEIGHT_DECAY
    )
    
    elapsed_time = time.time() - start_time
    print(f"Training complete! Total time: {elapsed_time:.2f} seconds")

    set_flat_params(pinn_model, x_final)

    # --- Print Final Losses ---
    
    
    # --- 11. Plotting Function ---
def plot_optimization_history_multi_constraint(*Histories, legends=None, linewidth=1, fontsize=20, ticksize=15, legendsize=15, savename=None):
    """Plotting function for the multi-constraint problem."""
    plt.figure(figsize=(21, 5))
    
    # --- PLOT 1: KKT Gap ---
    plt.subplot(1, 3, 1)
    for i, history in enumerate(Histories):
        label = legends[i] if legends is not None and i < len(legends) else f'Run {i+1}'
        _, KKT_gaps, _, _ = zip(*history)
        plt.plot(KKT_gaps, label=label, linewidth=linewidth)
    plt.xlabel('Iteration (x50)', fontsize=fontsize)
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
    plt.xlabel('Iteration (x50)', fontsize=fontsize)
    plt.ylabel('Boundary Violation |h1(x)|', fontsize=fontsize)
    plt.yscale('log'); plt.legend(fontsize=legendsize)
    plt.title('Boundary Constraint', fontsize=fontsize)
    plt.grid(True); plt.xticks(fontsize=ticksize); plt.yticks(fontsize=ticksize)

    # --- PLOT 3: Initial Constraint ---
    plt.subplot(1, 3, 3)
    for i, history in enumerate(Histories):
        label = legends[i] if legends is not None and i < len(legends) else f'Run {i+1}'
        _, _, _, h2_violations = zip(*history)
        plt.plot(h2_violations, label=label, linewidth=linewidth)
    plt.xlabel('Iteration (x50)', fontsize=fontsize)
    plt.ylabel('Initial Violation |h2(x)|', fontsize=fontsize)
    plt.yscale('log'); plt.legend(fontsize=legendsize)
    plt.title('Initial Constraint', fontsize=fontsize)
    plt.grid(True); plt.xticks(fontsize=ticksize); plt.yticks(fontsize=ticksize)

    plt.tight_layout()
    if savename:
        dirpath = os.path.dirname(savename)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        plt.savefig(savename, bbox_inches='tight')
    plt.show()


# --- Plotting Training History ---
os.makedirs("./figures", exist_ok=True)
plot_optimization_history_multi_constraint(history, legends=['SOFL-PINN'], linewidth=3, savename='./figures/sofl_fpde_history.png')


def u_true_plot(x,t):
    return np.sin(x)*t**3


x_values = np.linspace(0, np.pi, M)
t_values = np.linspace(0, 1, N)
x1, t1 = np.meshgrid(x_values, t_values)

# (assuming u_true_plot is defined elsewhere)
u_truth = u_true_plot(x1, t1)

# Move model output to CPU before converting to numpy
u_hat = pinn_model(X_input) * X_input[:, 0].view(-1, 1) * (torch.pi - X_input[:, 0]).view(-1, 1)
u_plot = u_hat.detach().cpu().numpy()   # <-- FIXED
u_plot = u_plot.reshape(N, M)

# Also move these if they were created on GPU
X_plot = X.detach().cpu().numpy()
T_plot = T.detach().cpu().numpy()

fig = plt.figure(figsize=(12, 6))

ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(x1, t1, u_truth, cmap='viridis', alpha=0.8)
ax1.set_xlabel('$x$')
ax1.set_ylabel('$t$')
ax1.set_zlabel('$u_{\\mathrm{true}}(x, t)$')
ax1.set_title('$u_{\\mathrm{true}}(x, t) = \\sin(x) \\cdot t^3$')

ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(x1, t1, u_plot, cmap='viridis', alpha=0.8)
ax2.set_xlabel('$x$')
ax2.set_ylabel('$t$')
ax2.set_zlabel('$u_{\\mathrm{approx}}(x, t)$')
ax2.set_title('$u_{\\mathrm{approx}}(x, t)$')

plt.tight_layout()
plt.show()


u_truth_tensor = torch.from_numpy(u_truth).view(-1,1).to(device)
u_pred = pinn_model(X_input)

# Calculate MSE loss
print('MSE:', criterion(u_pred, u_truth_tensor).item())

# Calculate L2 relative error
l2_error = torch.linalg.norm(u_pred - u_truth_tensor) / torch.linalg.norm(u_truth_tensor)
print('L2 relative error:', l2_error.item())
# --- Calculate final testing losses ---
def calculate_test_losses(model):
    model.eval()

    # --- Testing Physics Loss (MSE of PDE residual on full grid) ---
    u_hat_test = model(X_input) * X_input[:, 0].view(-1, 1) * (torch.pi - X_input[:, 0]).view(-1, 1)
    u_hat_test = u_hat_test.reshape(N, M)
    u_nn_xx_test = spatial_step_h ** (-2) * torch.matmul(u_hat_test, S(M))[1:, 1:-1]
    residual_test = FPDE(alpha, u_hat_test) - u_nn_xx_test
    physics_test_loss = criterion(residual_test, torch.zeros_like(residual_test)).item()

    # --- Testing Initial Loss (MSE at t=0) ---
    ic_input_test = torch.cat((x_ic, t_ic), dim=-1)
    u_pred_ic_test = model(ic_input_test)
    initial_test_loss = criterion(u_pred_ic_test, u_ic).item()

    # --- Testing Boundary Loss (MSE on x=0 and x=pi) ---
    bc_x0_input_test = torch.cat((x_bc_0, t_bc), dim=-1)
    bc_xpi_input_test = torch.cat((x_bc_pi, t_bc), dim=-1)
    u_pred_bc_0_test = model(bc_x0_input_test)
    u_pred_bc_pi_test = model(bc_xpi_input_test)
    boundary_test_loss = (criterion(u_pred_bc_0_test, u_bc) + criterion(u_pred_bc_pi_test, u_bc)).item()

    print("- Final Test Losses ---")
    print(f"Test Physics Loss (MSE of residual): {physics_test_loss:.6e}")
    print(f"Test Initial Loss (MSE): {initial_test_loss:.6e}")
    print(f"Test Boundary Loss (MSE): {boundary_test_loss:.6e}")

calculate_test_losses(pinn_model)