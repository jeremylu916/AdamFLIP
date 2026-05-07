import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import time
import math
from scipy.io import savemat
from tqdm import trange
from scipy.linalg import solve
from mpl_toolkits.axes_grid1 import make_axes_locatable

# === NeurIPS PUBLICATION-QUALITY ABLATION STUDY ===
# This script runs multiple random seeds and generates publication-ready plots
# with colorblind-friendly styling, proper error bands, and high visual standards.

# --- 1. SET SEED FOR REPRODUCIBILITY ---
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- 2. SET UP DEVICE FOR GPU OR CPU ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device} 🚀")


class PINN_Net(nn.Module):
    """The Neural Network model."""
    def __init__(self):
        super(PINN_Net, self).__init__()
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
        self.net.apply(self.init_weights)
        
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_normal_(m.weight, gain=1.0)
            m.bias.data.fill_(0.001)

    def forward(self, x):
        return self.net(x)

class PhysicsInformedNN():
    """Manages the PINN model, data, and loss calculations."""
    def __init__(self, pde_data, ic_data, bc_data):
        # Store master data
        self.x_pde_master = pde_data[0]
        self.y_pde_master = pde_data[1]
        self.t_pde_master = pde_data[2]
        
        self.x_ic_master, self.y_ic_master, self.t_ic_master, self.u_ic_master = ic_data
        self.x_bc_master, self.y_bc_master, self.t_bc_master, self.u_bc_master = bc_data

        # Initialize stage data (will be overwritten by causal loop)
        self.x_pde_stage = self.x_pde_master
        self.y_pde_stage = self.y_pde_master
        self.t_pde_stage = self.t_pde_master

        self.net = PINN_Net().to(device)
        self.loss = nn.MSELoss()

    def set_data_for_stage(self, t_max):
        """Filters the master PDE data for the current causal stage."""
        indices = self.t_pde_master.squeeze() <= t_max
        self.x_pde_stage = self.x_pde_master[indices]
        self.y_pde_stage = self.y_pde_master[indices]
        self.t_pde_stage = self.t_pde_master[indices]

    def pde_loss(self):
        """Calculates the PDE residual loss (Objective)."""
        x = self.x_pde_stage.detach().requires_grad_(True)
        y = self.y_pde_stage.detach().requires_grad_(True)
        t = self.t_pde_stage.detach().requires_grad_(True)
        residual = pde(x, y, t, self.net)
        loss_pde = self.loss(residual, torch.zeros_like(residual))
        return loss_pde * loss_weights_g['pde']

    def data_loss(self):
        """Calculates the combined IC and BC loss (Constraint)."""
        u_pred_ic = self.net(torch.cat([self.x_ic_master, self.y_ic_master, self.t_ic_master], dim=1))
        loss_ic = self.loss(u_pred_ic, self.u_ic_master)

        u_pred_bc = self.net(torch.cat([self.x_bc_master, self.y_bc_master, self.t_bc_master], dim=1))
        loss_bc = self.loss(u_pred_bc, self.u_bc_master)
        
        return loss_weights_g['ic'] * loss_ic + loss_weights_g['bc'] * loss_bc
    
    def get_ic_bc_losses(self):
        """Calculates IC and BC losses separately for logging."""
        with torch.no_grad():
            u_pred_ic = self.net(torch.cat([self.x_ic_master, self.y_ic_master, self.t_ic_master], dim=1))
            loss_ic = self.loss(u_pred_ic, self.u_ic_master)

            u_pred_bc = self.net(torch.cat([self.x_bc_master, self.y_bc_master, self.t_bc_master], dim=1))
            loss_bc = self.loss(u_pred_bc, self.u_bc_master)
        
        return loss_ic.item() * loss_weights_g['ic'], loss_bc.item() * loss_weights_g['bc']


def initial_condition(x, y):
    """u(x, y, 0) = sin(pi*x) * sin(pi*y)"""
    return torch.sin(torch.pi * x) * torch.sin(torch.pi * y)

def pde(x, y, t, model):
    """Calculates the PDE residual: u_t - alpha * (u_xx + u_yy)"""
    input_data = torch.cat([x, y, t], dim=1)
    u = model(input_data)
    
    u_x, u_y, u_t = torch.autograd.grad(
        u, [x, y, t], grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True
    )

    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
    
    alpha = 0.1 
    return u_t - alpha * (u_xx + u_yy)


def calculate_l2_relative_error(model, num_points=100):
    """Calculates relative L2 error against the analytical 2D heat solution."""
    model.net.eval()
    alpha = 0.1
    x = torch.linspace(0, 1, num_points, device=device)
    y = torch.linspace(0, 1, num_points, device=device)
    t = torch.linspace(0, 1, num_points, device=device)
    X, Y, T = torch.meshgrid(x, y, t, indexing='ij')
    input_tensor = torch.stack([X.flatten(), Y.flatten(), T.flatten()], dim=1)

    with torch.no_grad():
        u_pred = model.net(input_tensor).reshape(X.shape)

    exp_term = torch.exp(-2 * torch.pi**2 * alpha * T)
    u_exact = exp_term * torch.sin(torch.pi * X) * torch.sin(torch.pi * Y)
    l2_error = torch.linalg.norm(u_pred - u_exact) / torch.linalg.norm(u_exact)
    return l2_error.item()


# --- 4. DATA GENERATION ---

def generate_training_data(num_points_pde, num_points_bc, num_points_ic):
    # PDE (collocation) points
    x_pde = torch.rand(num_points_pde, 1, requires_grad=True, device=device)
    y_pde = torch.rand(num_points_pde, 1, requires_grad=True, device=device)
    t_pde = torch.rand(num_points_pde, 1, requires_grad=True, device=device)
    
    # Initial Condition points
    x_ic = torch.rand(num_points_ic, 1, device=device)
    y_ic = torch.rand(num_points_ic, 1, device=device)
    t_ic = torch.zeros(num_points_ic, 1, device=device)
    u_ic_exact = initial_condition(x_ic, y_ic)
    
    # Boundary Condition points
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

    pde_data = (x_pde, y_pde, t_pde)
    ic_data = (x_ic, y_ic, t_ic, u_ic_exact)
    bc_data = (x_bc, y_bc, t_bc, u_bc_exact)
    
    return pde_data, ic_data, bc_data

# --- 5. SOFL OPTIMIZER AND BRIDGE FUNCTIONS ---

pinn_model = None
loss_weights_g = None

def get_flat_params(model):
    """Gets the model's parameters (weights) as a flat NumPy vector."""
    return np.concatenate([p.data.cpu().numpy().flatten() for p in model.parameters()])

def get_flat_grads(model):
    """Gets the model's gradients as a flat NumPy vector."""
    return np.concatenate([p.grad.cpu().numpy().flatten() if p.grad is not None else np.zeros(p.numel()) for p in model.parameters()])

def set_flat_params(model, flat_params):
    """Sets the model's parameters from a flat NumPy vector."""
    flat_params = flat_params.astype(np.float32)
    pointer = 0
    for p in model.parameters():
        num_params = p.numel()
        p.data = torch.from_numpy(flat_params[pointer:pointer + num_params]).view_as(p).to(device)
        pointer += num_params
        
def get_grads_from_loss(loss, model_params):
    """Helper to compute gradients using autograd.grad to avoid graph freeing."""
    grads = torch.autograd.grad(
        loss,
        model_params,
        grad_outputs=torch.ones_like(loss),
        retain_graph=True # Keep the graph alive for the next grad calculation
    )
    return np.concatenate([g.cpu().numpy().flatten() for g in grads])


# --- Wrapper functions for SOFL ---
def f(x):
    """Objective: Minimize PDE residual loss."""
    set_flat_params(pinn_model.net, x)
    return pinn_model.pde_loss().item()

def h(x):
    """Constraint: Data loss (IC + BC) must be zero."""
    set_flat_params(pinn_model.net, x)
    return np.array([pinn_model.data_loss().item()])

def df(x):
    """Gradient of the objective (PDE loss)."""
    set_flat_params(pinn_model.net, x)
    pinn_model.net.zero_grad() # Zero out old gradients
    loss = pinn_model.pde_loss()
    # Retain graph because dh() will be called next
    loss.backward(retain_graph=True) 
    return get_flat_grads(pinn_model.net)

def dh(x):
    """Gradient of the constraint (Data loss)."""
    set_flat_params(pinn_model.net, x)
    pinn_model.net.zero_grad() # Zero out old gradients
    loss = pinn_model.data_loss()
    # Retain graph for the next optimizer step
    loss.backward(retain_graph=True) 
    return get_flat_grads(pinn_model.net).reshape(1, -1)

# --- SOFL Optimizer ---
def AdamFL_PINN(
    f, h, df, dh, x_start, Kp, Ki=0, eta=0.01, max_iter=1000, tol=1e-6,
    beta1=0.9, beta2=0.999, weight_decay=0.0, verbose=False
):
    x = np.array(x_start, dtype=np.float32)
    history = []
    Integral = 0.0

    epsilon = 1e-8
    m, v, t = np.zeros_like(x), np.zeros_like(x), 0
    
    pbar = trange(max_iter, desc="AdaFL Training", disable=not verbose)
    for iteration in pbar:
        global pde_data_g, ic_data_g, bc_data_g
        pde_data_g, ic_data_g, bc_data_g = generate_training_data(
            NUM_POINTS_PDE, NUM_POINTS_BC, NUM_POINTS_IC
        )

        # Set params for this iteration's calculations
        set_flat_params(pinn_model.net, x)

        true_grad_f = df(x)
        true_J_h = dh(x)
        true_h_x = h(x)

        JhJht = true_J_h @ true_J_h.T
        Pcontrol = -Kp @ true_h_x
        Integral = Integral + true_h_x
        Icontrol = Ki * Integral
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

        t += 1
        m = beta1 * m + (1 - beta1) * KKT_grad
        v = beta2 * v + (1 - beta2) * (KKT_grad**2)
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        step_vector = eta * m_hat / (np.sqrt(v_hat) + epsilon)
        x_new = x - step_vector
        step_size = np.linalg.norm(step_vector)
        
        if iteration % 100 == 0 and verbose:
            set_flat_params(pinn_model.net, x) 
            
            true_f_val = f(x) # Physics Loss
            loss_ic_val, loss_bc_val = pinn_model.get_ic_bc_losses()
            true_h_val_combined = loss_ic_val + loss_bc_val
            KKT_gap = np.max([np.linalg.norm(KKT_grad), np.abs(true_h_val_combined)])

            pbar.set_postfix({
                'physics_loss': f'{true_f_val:.2e}', 
                'ic_loss': f'{loss_ic_val:.2e}',
                'bc_loss': f'{loss_bc_val:.2e}',
                'KKT_gap': f'{KKT_gap:.2e}'
            })
        
        # Always record history at checkpoints
        if iteration % 100 == 0:
            set_flat_params(pinn_model.net, x)
            true_f_val = f(x)
            loss_ic_val, loss_bc_val = pinn_model.get_ic_bc_losses()
            true_h_val_combined = loss_ic_val + loss_bc_val
            KKT_gap = np.max([np.linalg.norm(KKT_grad), np.abs(true_h_val_combined)])
            history.append((true_f_val, KKT_gap, loss_ic_val, loss_bc_val))

        if step_size < tol:
            break

        x = x_new
        
        if np.isnan(x).any():
            break
            
    return x, history


# ================================
# ABLATION STUDY: Multiple Adam Beta Configurations with Multiple Runs
# ================================

# Hyperparameters
NUM_ITERATIONS = 2000  # Use fewer iterations for ablation study
NUM_POINTS_PDE_MASTER = 20000
NUM_POINTS_PDE = 2000
NUM_POINTS_BC = 500
NUM_POINTS_IC = 500
NUM_SEEDS = 3  # Multiple random seeds for statistical robustness

# Loss weights
loss_weights_g = {'ic': 10.0, 'bc': 10.0, 'pde': 1.0}

# Fixed PI-control gains while testing Adam moment parameters
Kp_FIXED = 1000.0
Ki_FIXED = 0.01

# Adam beta configurations to test (ablation study)
BETA_CONFIGS = [
    {'name': 'b1_0p0_b2_0p9', 'beta1': 0.0, 'beta2': 0.9},
    {'name': 'b1_0p5_b2_0p99', 'beta1': 0.5, 'beta2': 0.99},
    {'name': 'b1_0p9_b2_0p99', 'beta1': 0.9, 'beta2': 0.99},
    {'name': 'b1_0p9_b2_0p999', 'beta1': 0.9, 'beta2': 0.999},
    {'name': 'b1_0p95_b2_0p999', 'beta1': 0.95, 'beta2': 0.999},
    {'name': 'b1_0p99_b2_0p9999', 'beta1': 0.99, 'beta2': 0.9999},
]

# Dictionary to store results for each beta configuration (indexed by [name][seed_idx])
ablation_results = {cfg['name']: [] for cfg in BETA_CONFIGS}

# Loop over different random seeds THEN different beta configurations
for seed_idx in range(NUM_SEEDS):
    print(f"\n{'='*70}")
    print(f"  SEED {seed_idx + 1}/{NUM_SEEDS}")
    print(f"{'='*70}")
    
    # Set unique random seed for this iteration
    current_seed = seed + seed_idx * 1000
    torch.manual_seed(current_seed)
    np.random.seed(current_seed)
    random.seed(current_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(current_seed)
        torch.cuda.manual_seed_all(current_seed)
    
    # Re-generate master data for each seed
    pde_data_g_master, ic_data_g, bc_data_g = generate_training_data(
        NUM_POINTS_PDE_MASTER, NUM_POINTS_BC, NUM_POINTS_IC
    )

    # Use the same initial network parameters for every beta config within this seed.
    # This makes all curves start from the same boundary-condition loss.
    base_model = PhysicsInformedNN(pde_data_g_master, ic_data_g, bc_data_g)
    x_initial = get_flat_params(base_model.net)
    
    for beta_cfg in BETA_CONFIGS:
        beta_name = beta_cfg['name']
        beta1 = beta_cfg['beta1']
        beta2 = beta_cfg['beta2']
        print(f"  beta1={beta1:.4f}, beta2={beta2:.4f}...", end=" ")
        
        # Re-initialize the model for each beta config, then restore the shared start.
        pinn_model = PhysicsInformedNN(pde_data_g_master, ic_data_g, bc_data_g)
        set_flat_params(pinn_model.net, x_initial)
        
        # Set data stage
        pinn_model.set_data_for_stage(t_max=1.0)
        
        # AdamFL training
        x_current = x_initial.copy()
        
        Kp_matrix = np.array([[Kp_FIXED]])
        eta = 1e-4
        tol = 1e-8
        
        x_final, history = AdamFL_PINN(
            f, h, df, dh,
            x_current, Kp_matrix, Ki=Ki_FIXED, eta=eta,
            max_iter=NUM_ITERATIONS, tol=tol,
            beta1=beta1, beta2=beta2
        )
        
        # Print and store final metrics
        set_flat_params(pinn_model.net, x_final)
        final_physics_loss = pinn_model.pde_loss().item()
        final_bc_loss = pinn_model.get_ic_bc_losses()[1]
        final_l2_error = calculate_l2_relative_error(pinn_model)

        ablation_results[beta_name].append({
            'history': history,
            'x_final': x_final,
            'model': pinn_model,
            'beta1': beta1,
            'beta2': beta2,
            'physics_loss': final_physics_loss,
            'bc_loss': final_bc_loss,
            'l2_relative_error': final_l2_error,
        })

        print(
            f"BC Loss={final_bc_loss:.2e}, "
            f"Physics Loss={final_physics_loss:.2e}, "
            f"L2 Rel Error={final_l2_error:.2e}"
        )


# ================================
# PLOT BC LOSS VS ITERATION (Publication Quality)
# ================================

print("\n" + "="*70)
print("  GENERATING PUBLICATION-QUALITY PLOTS...")
print("="*70)

# Colorblind-friendly palette (tested on colorblind vision simulators)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

# Line styles for B/W printing compatibility
line_styles = ['-', '--', '-.', ':', '-', '--']

# Marker styles for visual distinctness
markers = ['o', 's', '^', 'D', 'v', 'x']

fig, ax = plt.subplots(figsize=(10, 6.5))

# Curves to highlight with error bands (baseline + best/important cases)
shaded_beta_configs = {'b1_0p0_b2_0p9', 'b1_0p9_b2_0p999', 'b1_0p99_b2_0p9999'}

# Extract and plot BC loss for each beta config with mean ± std across seeds
for idx, beta_cfg in enumerate(BETA_CONFIGS):
    beta_name = beta_cfg['name']
    beta1 = beta_cfg['beta1']
    beta2 = beta_cfg['beta2']
    histories = [result['history'] for result in ablation_results[beta_name]]
    n_iters = len(histories[0])
    iterations = np.array([i * 100 for i in range(n_iters)])
    
    # Collect BC loss across all seeds
    bc_loss_matrix = np.array([[h[3] for h in history] for history in histories])
    
    # Compute mean and std
    bc_loss_mean = np.mean(bc_loss_matrix, axis=0)
    bc_loss_std = np.std(bc_loss_matrix, axis=0)
    
    # Plot the mean curve with custom styling
    ax.plot(iterations, bc_loss_mean, 
            label=fr'$\beta_1={beta1:g},\ \beta_2={beta2:g}$',
            linestyle=line_styles[idx % len(line_styles)], 
            marker=markers[idx % len(markers)], 
            color=colors[idx % len(colors)],
            linewidth=3.0,
            markersize=7,
            markevery=max(1, n_iters // 8),  # Avoid marker clutter
            alpha=0.85,
            zorder=10-idx)  # Vary z-order for visual hierarchy
    
    # Add shaded error band (±1 std dev) only for key curves to reduce clutter
    if beta_name in shaded_beta_configs:
        ax.fill_between(iterations, 
                         bc_loss_mean - bc_loss_std, 
                         bc_loss_mean + bc_loss_std,
                         color=colors[idx % len(colors)],
                         alpha=0.2,
                         zorder=1)

# Formatting for publication
ax.set_xlabel('Iteration', fontsize=18, fontweight='bold')
ax.set_ylabel('Boundary Condition Loss', fontsize=18, fontweight='bold')
ax.set_yscale('log')

# Professional legend with frame
ax.legend(fontsize=12, 
          loc='best', 
          framealpha=0.95, 
          edgecolor='black', 
          fancybox=False,
          ncol=2)

# Grid styling
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.7)

# Tick formatting
ax.tick_params(labelsize=14, width=1.5, length=6)
ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=6))

# Make axes spines thicker and darker for publication
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(1.5)

plt.tight_layout()

# Save with high DPI
os.makedirs("./figures", exist_ok=True)
plt.savefig('./figures/ablation_study_beta_bc_loss.png', 
            dpi=300, 
            bbox_inches='tight', 
            facecolor='white',
            edgecolor='none')
print("✓ Saved: ./figures/ablation_study_beta_bc_loss.png")
plt.show()

# ================================
# SAVE ABLATION STUDY DATA
# ================================

os.makedirs("./training_data", exist_ok=True)

# Save aggregated statistics for each beta config across all seeds
for beta_cfg in BETA_CONFIGS:
    beta_name = beta_cfg['name']
    histories = [result['history'] for result in ablation_results[beta_name]]
    iterations = np.array([i * 100 for i in range(len(histories[0]))])
    
    # Extract and compute statistics
    bc_loss_matrix = np.array([[h[3] for h in history] for history in histories])
    physics_loss_matrix = np.array([[h[0] for h in history] for history in histories])
    kkt_gap_matrix = np.array([[h[1] for h in history] for history in histories])
    ic_loss_matrix = np.array([[h[2] for h in history] for history in histories])
    final_bc_losses = np.array([result['bc_loss'] for result in ablation_results[beta_name]])
    final_physics_losses = np.array([result['physics_loss'] for result in ablation_results[beta_name]])
    final_l2_errors = np.array([result['l2_relative_error'] for result in ablation_results[beta_name]])
    
    np.savez(
        f"./training_data/ablation_beta_{beta_name}_history.npz",
        iterations=iterations,
        beta1=beta_cfg['beta1'],
        beta2=beta_cfg['beta2'],
        Kp=Kp_FIXED,
        Ki=Ki_FIXED,
        bc_loss_mean=np.mean(bc_loss_matrix, axis=0),
        bc_loss_std=np.std(bc_loss_matrix, axis=0),
        bc_loss_min=np.min(bc_loss_matrix, axis=0),
        bc_loss_max=np.max(bc_loss_matrix, axis=0),
        physics_loss_mean=np.mean(physics_loss_matrix, axis=0),
        physics_loss_std=np.std(physics_loss_matrix, axis=0),
        kkt_gap_mean=np.mean(kkt_gap_matrix, axis=0),
        kkt_gap_std=np.std(kkt_gap_matrix, axis=0),
        ic_loss_mean=np.mean(ic_loss_matrix, axis=0),
        ic_loss_std=np.std(ic_loss_matrix, axis=0),
        final_bc_loss_mean=np.mean(final_bc_losses),
        final_bc_loss_std=np.std(final_bc_losses),
        final_physics_loss_mean=np.mean(final_physics_losses),
        final_physics_loss_std=np.std(final_physics_losses),
        final_l2_relative_error_mean=np.mean(final_l2_errors),
        final_l2_relative_error_std=np.std(final_l2_errors),
    )

print("✓ Saved statistics to: ./training_data/")

# ================================
# SUMMARY
# ================================

print("\n" + "="*70)
print("  ABLATION STUDY COMPLETE")
print("="*70)
print(f"\nConfiguration:")
print(f"  • Fixed Kp: {Kp_FIXED}")
print(f"  • Fixed Ki: {Ki_FIXED}")
print(f"  • Tested beta configs:")
for beta_cfg in BETA_CONFIGS:
    print(f"    - beta1={beta_cfg['beta1']}, beta2={beta_cfg['beta2']}")
print(f"  • Random seeds: {NUM_SEEDS}")
print(f"  • Training iterations per run: {NUM_ITERATIONS}")
print(f"  • Output: ./figures/ablation_study_beta_bc_loss.png")
print(f"  • Data: ./training_data/ablation_beta_*.npz")
print(f"\nKey observations:")
print(f"  • beta1 controls first-moment smoothing of the KKT-gradient estimate")
print(f"  • beta2 controls second-moment smoothing and step normalization")
print(f"  • Shaded regions represent ±1 standard deviation across {NUM_SEEDS} seeds")
print(f"  • Plot is colorblind-friendly and suitable for B&W printing")

# Calculate and display final metric statistics for each beta config
print(f"\nFinal Metric Statistics:")
for beta_cfg in BETA_CONFIGS:
    beta_name = beta_cfg['name']
    final_bc_losses = np.array([result['bc_loss'] for result in ablation_results[beta_name]])
    final_physics_losses = np.array([result['physics_loss'] for result in ablation_results[beta_name]])
    final_l2_errors = np.array([result['l2_relative_error'] for result in ablation_results[beta_name]])
    print(
        f"  beta1={beta_cfg['beta1']:g}, beta2={beta_cfg['beta2']:g}: "
        f"BC Loss = {np.mean(final_bc_losses):.4e} ± {np.std(final_bc_losses):.4e}, "
        f"Physics Loss = {np.mean(final_physics_losses):.4e} ± {np.std(final_physics_losses):.4e}, "
        f"L2 Rel Error = {np.mean(final_l2_errors):.4e} ± {np.std(final_l2_errors):.4e}"
    )

print("\n" + "─"*70)
print("CAPTION SUGGESTION FOR YOUR PAPER:")
print("─"*70)
caption = (
    f"Ablation study: Boundary condition loss over training iterations for"
    f" different Adam moment parameters $(\\beta_1, \\beta_2)$ in AdaFL-PINN."
    f" The PI-control gains are fixed at $K_p={Kp_FIXED:.0f}$ and $K_i={Ki_FIXED:g}$."
    f" Lines represent mean over"
    f" {NUM_SEEDS} independent random seeds; shaded regions show ±1 standard"
    f" deviation."
)
print(caption)
print("─"*70 + "\n")
