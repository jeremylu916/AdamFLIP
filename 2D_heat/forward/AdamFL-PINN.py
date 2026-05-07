
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
def AdamFL_PINN(f, h, df, dh, x_start, Kp, Ki=0, eta=0.01, max_iter=1000, tol=1e-6, weight_decay=0.0):
    x = np.array(x_start, dtype=np.float32)
    history = []
    Integral = 0.0

    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    m, v, t = np.zeros_like(x), np.zeros_like(x), 0
    
    pbar = trange(max_iter, desc="AdaFL Training")
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
        
        if iteration % 100 == 0:
            set_flat_params(pinn_model.net, x) 
            
            true_f_val = f(x) # Physics Loss
            loss_ic_val, loss_bc_val = pinn_model.get_ic_bc_losses()
            true_h_val_combined = loss_ic_val + loss_bc_val
            KKT_gap = np.max([np.linalg.norm(KKT_grad), np.abs(true_h_val_combined)])

            #pbar.write(f"Iter: {iteration:5d}, Physics Loss: {true_f_val:.4e}, IC Loss: {loss_ic_val:.4e}, BC Loss: {loss_bc_val:.4e}, KKT Gap: {KKT_gap:.4e}")
            pbar.set_postfix({
                'physics_loss': f'{true_f_val:.2e}', 
                'ic_loss': f'{loss_ic_val:.2e}',
                'bc_loss': f'{loss_bc_val:.2e}',
                'KKT_gap': f'{KKT_gap:.2e}'
            })
            history.append((true_f_val, KKT_gap, loss_ic_val, loss_bc_val))

        if step_size < tol:
            print(f"\nConvergence at iter {iteration}: Step size ({step_size:.2e}) < tol ({tol:.2e}).")
            break

        x = x_new
        
        if np.isnan(x).any():
            print("\nError: NaN values detected. Stopping.")
            break
            
    return x, history





 # Hyperparameters
NUM_ITERATIONS = 30000
NUM_POINTS_PDE_MASTER = 20000 # Generate a large master set of points
NUM_POINTS_PDE = 2000 # Points to use per iteration
NUM_POINTS_BC = 500
NUM_POINTS_IC = 500

# Loss weights
loss_weights_g = {'ic': 10.0, 'bc': 10.0, 'pde': 1.0}

# Generate master data *once*
pde_data_g_master, ic_data_g, bc_data_g = generate_training_data(
    NUM_POINTS_PDE_MASTER, NUM_POINTS_BC, NUM_POINTS_IC
)

# Initialize the model
pinn_model = PhysicsInformedNN(pde_data_g_master, ic_data_g, bc_data_g)

# --- Training Setup ---
Kp = np.array([[500.0]])
Ki = 0.01
eta = 1e-4
tol = 1e-8
max_iter = NUM_ITERATIONS

full_history = []

def boundary_loss(model):
    return model.data_loss()

def residual_loss(model):
    return model.pde_loss()

adam_pretrain_iters = 2000
adam_lr = 1e-3

print(f"\n--- Adam warm-start ({adam_pretrain_iters} iters) ---")
if hasattr(pinn_model, "X_u_master") and hasattr(pinn_model, "X_f_master") and hasattr(pinn_model, "u_master"):
    pinn_model.set_data_for_stage(
        pinn_model.X_u_master,
        pinn_model.u_master,
        pinn_model.X_f_master,
    )
else:
    pinn_model.set_data_for_stage(t_max=1.0)
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

x_final, history = AdamFL_PINN(
    f, h, df, dh,
    x_current, Kp, Ki=Ki, eta=eta,
    max_iter=max_iter, tol=tol
)
full_history.extend(history)

os.makedirs("./training_data", exist_ok=True)
np.savez(
    "./training_data/AdamFL_PINN_loss_history.npz",
    epochs=np.array([i * 100 for i in range(len(history))]),
    physics_loss=np.array([h[0] for h in history]),
    kkt_gap=np.array([h[1] for h in history]),
    ic_loss=np.array([h[2] for h in history]),
    bc_loss=np.array([h[3] for h in history]),
)
print("Saved loss history to ./training_data/AdamFL_PINN_loss_history.npz")






# --- Print Final Training Losses (on full domain) ---
print("\n--- Final Training Losses (Full Domain) ---")
pinn_model.set_data_for_stage(t_max=1.0) # Ensure model is set to full domain
final_physics_loss = f(x_current)
final_ic_loss, final_bc_loss = pinn_model.get_ic_bc_losses()
    
print(f"Final Physics Loss (Objective): {final_physics_loss:.4e}")
print(f"Final Initial Condition Loss: {final_ic_loss:.4e}")
print(f"Final Boundary Condition Loss: {final_bc_loss:.4e}")

# --- 7. EVALUATION AND PLOTTING ---


def plot_solution_comparison(model, t_value):
    model.net.eval()
    alpha = 0.1
    x = torch.linspace(0, 1, 100, device=device)
    y = torch.linspace(0, 1, 100, device=device)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    T = torch.full(X.shape, t_value, device=device)
    input_tensor = torch.stack([X.flatten(), Y.flatten(), T.flatten()], dim=1)
    with torch.no_grad():
        U_pred = model.net(input_tensor).reshape(X.shape)
    
    exp_term = math.exp(-2 * np.pi**2 * alpha * t_value)
    U_exact = exp_term * torch.sin(torch.pi * X) * torch.sin(torch.pi * Y)
    Error = torch.abs(U_pred - U_exact)
    
    X_np, Y_np = X.cpu().numpy(), Y.cpu().numpy()
    U_pred_np = U_pred.cpu().numpy()
    U_exact_np = U_exact.cpu().numpy()
    Error_np = Error.cpu().numpy()

    # os.makedirs("./mat_files", exist_ok=True)
    # savemat(f"./mat_files/sofl_solution_comparison_t{t_value}.mat", {
    #     "X": X_np, "Y": Y_np, "U_pred": U_pred_np,
    #     "U_exact": U_exact_np, "Error": Error_np
    # })

    # Save data into ./data for external plotting
    os.makedirs("./data", exist_ok=True)
    savemat(f"./data/AdamFL_solution_t{t_value}.mat", {
        "X": X_np,
        "Y": Y_np,
        "U_pred": U_pred_np,
        "U_exact": U_exact_np,
        "Error": Error_np
    })
    print(f"Successfully saved data to ./data/AdamFL_solution_t{t_value}.mat")

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'AdamFL-PINN Comparison at t = {t_value}', fontsize=16)
    v_min = min(U_pred_np.min(), U_exact_np.min())
    v_max = max(U_pred_np.max(), U_exact_np.max())
    c1 = ax1.pcolormesh(X_np, Y_np, U_pred_np, cmap='jet', vmin=v_min, vmax=v_max, shading='auto')
    fig.colorbar(c1, ax=ax1); ax1.set_title("Prediction"); ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax1.axis('square')
    c2 = ax2.pcolormesh(X_np, Y_np, U_exact_np, cmap='jet', vmin=v_min, vmax=v_max, shading='auto')
    fig.colorbar(c2, ax=ax2); ax2.set_title("Exact Solution"); ax2.set_xlabel("x"); ax2.axis('square')
    c3 = ax3.pcolormesh(X_np, Y_np, Error_np, cmap='Reds', shading='auto')
    fig.colorbar(c3, ax=ax3); ax3.set_title("Absolute Error"); ax3.set_xlabel("x"); ax3.axis('square')
    plt.subplots_adjust(wspace=0.1, hspace=0.1, top=0.88)
    os.makedirs("./figures", exist_ok=True)
    fig.savefig(f"./figures/adamfl_comparison_t{t_value}.png", dpi=300, bbox_inches='tight')
    plt.show()

plot_solution_comparison(pinn_model, t_value=0.5)
plot_solution_comparison(pinn_model, t_value=1.0)

def calculate_l2_relative_error(model):
    model.net.eval()
    alpha = 0.1
    x = torch.linspace(0, 1, 100, device=device)
    y = torch.linspace(0, 1, 100, device=device)
    t = torch.linspace(0, 1, 100, device=device)
    X, Y, T = torch.meshgrid(x, y, t, indexing='ij')
    input_tensor = torch.stack([X.flatten(), Y.flatten(), T.flatten()], dim=1)
    with torch.no_grad():
        U_pred = model.net(input_tensor).reshape(X.shape)
        
    exp_term = torch.exp(-2 * torch.pi**2 * alpha * T)
    U_exact = exp_term * torch.sin(torch.pi * X) * torch.sin(torch.pi * Y)
    
    mse_error = torch.mean((U_pred - U_exact) ** 2) 
    l2_error = torch.linalg.norm(U_pred - U_exact) / torch.linalg.norm(U_exact)
    print(f"\n--- Final Test Metrics ---")
    print(f"L2 Relative Error: {l2_error.item():.6e}")
    print(f"Mean Squared Error (MSE): {mse_error.item():.6e}")

calculate_l2_relative_error(pinn_model)




def plot_optimization_history_4_plots(*Histories, legends=None, linewidth=1, fontsize=20, ticksize=15, legendsize=15, savename=None):
    """Plots KKT Gap, Boundary Violate |h1(x)|, and Physics Violate |h2(x)|."""
    plt.figure(figsize=(18, 5)) 
    
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

    # --- PLOT 2: Boundary Condition Loss (h1) ---
    plt.subplot(1, 3, 2)
    for i, history in enumerate(Histories):
        label = legends[i] if legends is not None and i < len(legends) else f'Run {i+1}'
        _, _, _, bc_violations = zip(*history)
        plt.plot(bc_violations, label=label, linewidth=linewidth)
        
    plt.xlabel('Iteration (x100)', fontsize=fontsize)
    plt.ylabel('Boundary Violate |h1(x)|', fontsize=fontsize)
    plt.yscale('log'); plt.legend(fontsize=legendsize)
    plt.title('Boundary Condition Loss', fontsize=fontsize)
    plt.grid(True); plt.xticks(fontsize=ticksize); plt.yticks(fontsize=ticksize)

    # --- PLOT 3: Physics Loss (h2) ---
    plt.subplot(1, 3, 3)
    for i, history in enumerate(Histories):
        label = legends[i] if legends is not None and i < len(legends) else f'Run {i+1}'
        f_values, _, _, _ = zip(*history)
        plt.plot(f_values, label=label, linewidth=linewidth)

    plt.xlabel('Iteration (x100)', fontsize=fontsize)
    plt.ylabel('Physics Violate |h2(x)|', fontsize=fontsize)
    plt.legend(fontsize=legendsize)
    plt.title('Physics Loss', fontsize=fontsize)
    plt.grid(True); plt.xticks(fontsize=ticksize); plt.yticks(fontsize=ticksize); plt.yscale('log')

    plt.tight_layout()
    if savename:
        dirpath = os.path.dirname(savename)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        plt.savefig(savename, bbox_inches='tight')
    plt.show()



