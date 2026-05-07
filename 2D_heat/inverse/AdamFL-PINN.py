import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import time
from scipy.io import savemat
# --- SOFL Imports ---
from scipy.linalg import solve
from tqdm import trange


seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- 1. SET UP DEVICE FOR GPU OR CPU ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device} 🚀")


# --- Ground Truth (for inverse problem) ---
ALPHA_TRUE = 0.1

# --- MODIFIED: Global PINN model variable for SOFL wrappers ---
pinn_model = None

# --- MODIFIED: Global variables for training data (used by SOFL wrappers) ---
x_pde_g, y_pde_g, t_pde_g = None, None, None
x_ic_g, y_ic_g, t_ic_g, u_ic_exact_g = None, None, None, None
x_bc_g, y_bc_g, t_bc_g, u_bc_exact_g = None, None, None, None
x_data_g, y_data_g, t_data_g, u_data_exact_g = None, None, None, None
mse_loss_g = nn.MSELoss() # Global loss function

# --- MODIFIED: PINN class with loss methods for SOFL ---
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
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )
        # alpha is a trainable parameter
        self.alpha = nn.Parameter(torch.tensor([initial_alpha_guess], device=device, dtype=torch.float32))

    def forward(self, x):
        return self.net(x)

    # --- NEW: Loss functions for SOFL ---
    def loss_data(self):
        """ Objective Function f(x) """
        u_pred_data = self(torch.cat([x_data_g, y_data_g, t_data_g], dim=1))
        return mse_loss_g(u_pred_data, u_data_exact_g)
    
    def loss_pde(self):
        """ Constraint h_1(x) """
        input_data = torch.cat([x_pde_g, y_pde_g, t_pde_g], dim=1)
        u = self(input_data)
        
        grads = torch.autograd.grad(u, [x_pde_g, y_pde_g, t_pde_g], grad_outputs=torch.ones_like(u), create_graph=True)
        u_x, u_y, u_t = grads[0], grads[1], grads[2]

        u_xx = torch.autograd.grad(u_x, x_pde_g, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y_pde_g, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
        
        residual = u_t - self.alpha * (u_xx + u_yy)
        return mse_loss_g(residual, torch.zeros_like(residual))

    def loss_bc(self):
        """ Constraint h_2(x) """
        u_pred_bc = self(torch.cat([x_bc_g, y_bc_g, t_bc_g], dim=1))
        return mse_loss_g(u_pred_bc, u_bc_exact_g)
    
    def loss_ic(self):
        """ Constraint h_3(x) """
        u_pred_ic = self(torch.cat([x_ic_g, y_ic_g, t_ic_g], dim=1))
        return mse_loss_g(u_pred_ic, u_ic_exact_g)
    

# Define the initial condition
def initial_condition(x, y):
    return torch.sin(torch.pi * x) * torch.sin(torch.pi * y)

# Helper function for the analytical solution (to generate data)
def analytical_solution(x, y, t, alpha):
    return torch.sin(torch.pi * x) * torch.sin(torch.pi * y) * torch.exp(-2 * torch.pi**2 * alpha * t)

# Data generation (unchanged from inverse problem)
def generate_training_data(num_points_pde, num_points_bc, num_points_ic, num_points_data):
    # PDE (collocation) points (need gradients)
    x_pde = torch.rand(num_points_pde, 1, requires_grad=True, device=device)
    y_pde = torch.rand(num_points_pde, 1, requires_grad=True, device=device)
    t_pde = torch.rand(num_points_pde, 1, requires_grad=True, device=device)
    
    # Initial Condition points (no gradients needed)
    x_ic = torch.rand(num_points_ic, 1, device=device)
    y_ic = torch.rand(num_points_ic, 1, device=device)
    t_ic = torch.zeros(num_points_ic, 1, device=device)
    u_ic_exact = initial_condition(x_ic, y_ic)
    
    # Boundary Condition points (no gradients needed)
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

    # "Sensor data" points
    x_data = torch.rand(num_points_data, 1, device=device)
    y_data = torch.rand(num_points_data, 1, device=device)
    t_data = torch.rand(num_points_data, 1, device=device) # t > 0
    u_data_exact = analytical_solution(x_data, y_data, t_data, ALPHA_TRUE)

    return (x_pde, y_pde, t_pde,
            x_ic, y_ic, t_ic, u_ic_exact,
            x_bc, y_bc, t_bc, u_bc_exact,
            x_data, y_data, t_data, u_data_exact)


# --- MODIFIED: SOFL Helper Functions (operating on pinn_model.parameters()) ---
def get_flat_params(model):
    """Gets all model parameters (net + alpha) as a flat NumPy vector."""
    return np.concatenate([p.data.cpu().numpy().flatten() for p in model.parameters()])

def get_flat_grads(model):
    """Gets all model gradients (net + alpha) as a flat NumPy vector."""
    return np.concatenate([p.grad.cpu().numpy().flatten() if p.grad is not None else np.zeros(p.numel()) for p in model.parameters()])

def set_flat_params(model, flat_params):
    """Sets all model parameters (net + alpha) from a flat NumPy vector."""
    flat_params = flat_params.astype(np.float32)
    pointer = 0
    for p in model.parameters():
        num_params = p.numel()
        p.data = torch.from_numpy(flat_params[pointer:pointer + num_params]).view_as(p).to(device)
        pointer += num_params

# --- MODIFIED: Wrapper functions for SOFL (New Objective/Constraints) ---
def f(x):
    """Objective: Minimize Data loss."""
    set_flat_params(pinn_model, x)
    return pinn_model.loss_data().item()

def h(x):
    """Constraints: [PDE, BC, IC] losses must be zero."""
    set_flat_params(pinn_model, x)
    loss_p = pinn_model.loss_pde().item()
    loss_b = pinn_model.loss_bc().item()
    loss_i = pinn_model.loss_ic().item()
    # Return a 3-element vector
    return np.array([loss_p, loss_b, loss_i])

def df(x):
    """Gradient of the objective (Data loss)."""
    set_flat_params(pinn_model, x)
    pinn_model.zero_grad()
    loss = pinn_model.loss_data()
    loss.backward(retain_graph=True) # Keep graph for dh
    return get_flat_grads(pinn_model)

def dh(x):
    """Jacobian of the constraints [PDE, BC, IC]."""
    set_flat_params(pinn_model, x)
    
    # Grad for h_1 (PDE)
    pinn_model.zero_grad()
    loss_pde = pinn_model.loss_pde()
    loss_pde.backward(retain_graph=True)
    grad_pde = get_flat_grads(pinn_model)
    
    # Grad for h_2 (BC)
    pinn_model.zero_grad()
    loss_bc = pinn_model.loss_bc()
    loss_bc.backward(retain_graph=True)
    grad_bc = get_flat_grads(pinn_model)
    
    # Grad for h_3 (IC)
    pinn_model.zero_grad()
    loss_ic = pinn_model.loss_ic()
    loss_ic.backward(retain_graph=True) # Keep graph for next iter
    grad_ic = get_flat_grads(pinn_model)
    
    # Stack grads into a (3, N) Jacobian
    return np.stack([grad_pde, grad_bc, grad_ic])


# --- MODIFIED: SOFL Optimizer (Adapted for this problem) ---
def AdamFL_PINN(f, h, df, dh, x_start, Kp, Ki=0, eta=0.01, max_iter=1000, tol=1e-6, weight_decay=0.0):
    # --- Import global data variables ---
    global x_pde_g, y_pde_g, t_pde_g
    global x_ic_g, y_ic_g, t_ic_g, u_ic_exact_g
    global x_bc_g, y_bc_g, t_bc_g, u_bc_exact_g
    global x_data_g, y_data_g, t_data_g, u_data_exact_g
    
    x = np.array(x_start, dtype=np.float32)
    history = []
    
    # --- MODIFIED: Integral term must match constraint dimensions ---
    Integral = np.zeros(Kp.shape[0], dtype=np.float32)  # (3,)

    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    m, v, t = np.zeros_like(x), np.zeros_like(x), 0

    # --- Prepare log file ---
    os.makedirs("./training_logs", exist_ok=True)
    loss_file_path = "./training_logs/SOFL-PINN_inverse_training_loss.txt"
    with open(loss_file_path, "w") as log_file:  # <-- renamed from f → log_file
        log_file.write("Iter,Obj_Loss,PDE_Loss,BC_Loss,IC_Loss,KKT_Gap,Alpha\n")
        log_file.flush()

        pbar = trange(max_iter, desc="AdamFL-PINN")
        for iteration in pbar:
            # --- MODIFIED: Unpack all 8 data tuples into global scope ---
            (x_pde_g, y_pde_g, t_pde_g,
             x_ic_g, y_ic_g, t_ic_g, u_ic_exact_g,
             x_bc_g, y_bc_g, t_bc_g, u_bc_exact_g,
             x_data_g, y_data_g, t_data_g, u_data_exact_g) = generate_training_data(
                 NUM_POINTS_PDE, NUM_POINTS_BC, NUM_POINTS_IC, NUM_POINTS_DATA
             )

            # Set params for this iteration's calculations
            set_flat_params(pinn_model, x)

            true_grad_f = df(x)       # (N,)
            true_J_h = dh(x)          # (3, N)
            true_h_x = h(x)           # (3,)

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

            # --- Gradient clipping ---
            clip_value = 10.0
            grad_norm = np.linalg.norm(KKT_grad)
            if grad_norm > clip_value:
                KKT_grad = KKT_grad * (clip_value / grad_norm)

            # --- Adam update ---
            t += 1
            m = beta1 * m + (1 - beta1) * KKT_grad
            v = beta2 * v + (1 - beta2) * (KKT_grad**2)
            m_hat = m / (1 - beta1**t)
            v_hat = v / (1 - beta2**t)
            step_vector = eta * m_hat / (np.sqrt(v_hat) + epsilon)
            x_new = x - step_vector
            step_size = np.linalg.norm(step_vector)

            # --- Logging every 100 iterations ---
            if iteration % 100 == 0 or iteration == max_iter - 1:
                set_flat_params(pinn_model, x)
                loss_data_val = f(x)
                h_vals = h(x)
                loss_pde_val, loss_bc_val, loss_ic_val = h_vals
                KKT_gap = np.max([np.linalg.norm(KKT_grad), np.sum(np.abs(h_vals))])
                current_alpha = pinn_model.alpha.item()

                log_msg = (
                    f"[Iter {iteration:5d}] "
                    f"Obj: {loss_data_val:8.3e} | "
                    f"PDE: {loss_pde_val:8.3e} | "
                    f"BC: {loss_bc_val:8.3e} | "
                    f"IC: {loss_ic_val:8.3e} | "
                    f"KKT: {KKT_gap:8.3e} | "
                    f"Alpha: {current_alpha:6.4f}"
                )

                # Print to console
                # print(log_msg)

                # Write to file (comma-separated)
               # log_file.write(f"{iteration},{loss_data_val:.6e},{loss_pde_val:.6e},{loss_bc_val:.6e},{loss_ic_val:.6e},{KKT_gap:.6e},{current_alpha:.6f}\n")
               # log_file.flush()  # Force write to disk so you can monitor progress in real time

                # Update tqdm bar
                pbar.set_postfix({
                    'Obj': f'{loss_data_val:.2e}',
                    'KKT': f'{KKT_gap:.2e}',
                    'Alpha': f'{current_alpha:.4f}'
                })

                # Save to history
                history.append((loss_data_val, KKT_gap, loss_ic_val, loss_bc_val, loss_pde_val))

            # --- Check convergence ---
            if step_size < tol:
                print(f"\nConvergence at iter {iteration}: Step size ({step_size:.2e}) < tol ({tol:.2e}).")
                break

            x = x_new
            if np.isnan(x).any():
                print("\nError: NaN values detected. Stopping.")
                break

    return x, history




# Hyperparameters
NUM_POINTS_PDE = 2000
NUM_POINTS_BC = 500
NUM_POINTS_IC = 500
NUM_POINTS_DATA = 1000 # "Sensor" data

# --- MODIFIED: Instantiate global model ---
pinn_model = PINN(initial_alpha_guess=1.0).to(device)

# --- MODIFIED: Set up and run SOFL Optimizer ---
# --- Warm Start with Adam ---

# Adam training hyperparameters
ADAM_EPOCHS = 2000
ADAM_LR = 5e-3
NUM_POINTS_PDE = 2000
NUM_POINTS_BC = 500
NUM_POINTS_IC = 500
NUM_POINTS_DATA = 100

print(f"Starting Adam warm start training for {ADAM_EPOCHS} epochs...")

# Loss weights
loss_weights = {'ic': 1.0, 'bc': 1.0, 'pde': 0.1, 'data': 100.0}

# Adam optimizer
adam_optimizer = optim.Adam(pinn_model.parameters(), lr=ADAM_LR)

# Training loop for Adam warm start
adam_start_time = time.time()
adam_pbar = trange(ADAM_EPOCHS, desc="AdamFL warmup")
for epoch in adam_pbar:
    adam_optimizer.zero_grad()
    
    # Generate training data
    (x_pde_g, y_pde_g, t_pde_g,
     x_ic_g, y_ic_g, t_ic_g, u_ic_exact_g,
     x_bc_g, y_bc_g, t_bc_g, u_bc_exact_g,
     x_data_g, y_data_g, t_data_g, u_data_exact_g) = generate_training_data(
         NUM_POINTS_PDE, NUM_POINTS_BC, NUM_POINTS_IC, NUM_POINTS_DATA)
    
    # Compute losses
    loss_pde = pinn_model.loss_pde()
    loss_bc = pinn_model.loss_bc()
    loss_ic = pinn_model.loss_ic()
    loss_data = pinn_model.loss_data()
    
    # Total loss
    total_loss = loss_pde + loss_bc + loss_ic + loss_data
    
    # Backward and optimize
    total_loss.backward()
    adam_optimizer.step()
    
    adam_pbar.set_postfix({
        "loss": f"{total_loss.item():.2e}",
        "alpha": f"{pinn_model.alpha.item():.4f}"
    })

    # Print progress
    if epoch % 300 == 0:
        print(f"Adam Epoch {epoch}/{ADAM_EPOCHS}, Loss: {total_loss.item():.6e}, Alpha: {pinn_model.alpha.item():.4f}")

adam_end_time = time.time()
print(f"Adam warm start finished in {adam_end_time - adam_start_time:.2f} seconds.")
print(f"Learned alpha after Adam warm start: {pinn_model.alpha.item():.6f}")

print("Starting AdaFL Optimizer...")

# SOFL Hyperparameters
SOFL_MAX_ITER = 20000
SOFL_ETA = 5e-4  # Adam learning rate

# Kp must be (3, 3) for 3 constraints.
# We set a diagonal matrix, weighting each constraint equally.
# You can tune these weights! e.g., Kp_pde, Kp_bc, Kp_ic
Kp_pde = 100.0  # Gain for physics constraint
Kp_bc = 500.0   # Gain for boundary constraint
Kp_ic = 500.0   # Gain for initial condition constraint

SOFL_KP = np.diag([Kp_pde, Kp_bc, Kp_ic])
SOFL_KI = 0.0 # No integral control for simplicity

x_start = get_flat_params(pinn_model)

sofl_start_time = time.time()
x_final, history = AdamFL_PINN(
    f, h, df, dh, x_start,
    Kp=SOFL_KP,
    Ki=SOFL_KI,
    eta=SOFL_ETA,
    max_iter=SOFL_MAX_ITER,
    tol=1e-8,
    weight_decay=1e-6
)
sofl_end_time = time.time()
print(f"\nSOFL training finished in {sofl_end_time - sofl_start_time:.2f} seconds.")

# Save AdaFL loss history (every 100 iters, from `history`)
# os.makedirs("./training_data", exist_ok=True)
# np.savez(
#     "./training_data/AdaFL_PINN_loss_history.npz",
#     epochs=np.array([i * 100 for i in range(len(history))]),
#     data_loss=np.array([h[0] for h in history]),
#     kkt_gap=np.array([h[1] for h in history]),
#     ic_loss=np.array([h[2] for h in history]),
#     bc_loss=np.array([h[3] for h in history]),
#     physics_loss=np.array([h[4] for h in history]),
# )
# print("Saved loss history to ./training_data/AdaFL_PINN_loss_history.npz")

# Set model to the final trained parameters
set_flat_params(pinn_model, x_final)

# --- Run Validation and Plotting ---

def plot_solution_comparison(model, t_value):
        model.eval()
        learned_alpha = model.alpha.item()
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
        X_np, Y_np = X.cpu().numpy(), Y.cpu().numpy()
        U_pred_np = U_pred.cpu().numpy()
        U_exact_np = U_exact.cpu().numpy()
        Error_np = Error.cpu().numpy()
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
        fig.suptitle(f't = {t_value}', fontsize=16)
        v_min = min(U_pred_np.min(), U_exact_np.min())
        v_max = max(U_pred_np.max(), U_exact_np.max())
        c1 = ax1.pcolormesh(X_np, Y_np, U_pred_np, cmap='jet', vmin=v_min, vmax=v_max, shading='auto')
        fig.colorbar(c1, ax=ax1); ax1.set_title("Prediction"); ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax1.axis('square')
        c2 = ax2.pcolormesh(X_np, Y_np, U_exact_np, cmap='jet', vmin=v_min, vmax=v_max, shading='auto')
        fig.colorbar(c2, ax=ax2); ax2.set_title("Exact Solution"); ax2.set_xlabel("x"); ax2.axis('square')
        c3 = ax3.pcolormesh(X_np, Y_np, Error_np, cmap='Reds', shading='auto')
        fig.colorbar(c3, ax=ax3); ax3.set_title("Absolute Error"); ax3.set_xlabel("x"); ax3.axis('square')
        plt.subplots_adjust(wspace=0.1, hspace=0.1, top=0.88)
        fig.savefig(f"./figures/pinn_AdamFL_inverse_comparison_t{t_value}.png", dpi=300, bbox_inches='tight')
        plt.show()

plot_solution_comparison(pinn_model, t_value=0.5)
plot_solution_comparison(pinn_model, t_value=1.0)

# --- NEW: Plotting function for 5 history values ---
def plot_optimization_history_5_plots(*Histories, legends=None, linewidth=1, fontsize=20, ticksize=15, legendsize=15, savename=None):
    """Plots Data Loss, KKT Gap, IC Loss, BC Loss, and PDE Loss."""
    plt.figure(figsize=(35, 5)) 
    
    # --- PLOT 1: Data Loss (Objective) ---
    plt.subplot(1, 5, 1)
    for i, history in enumerate(Histories):
        label = legends[i] if legends is not None and i < len(legends) else f'Run {i+1}'
        f_values, _, _, _, _ = zip(*history)
        plt.plot(f_values, label=label, linewidth=linewidth)

    plt.xlabel('Iteration (x100)', fontsize=fontsize)
    plt.ylabel('$f(x)$ (Data Loss)', fontsize=fontsize)
    plt.legend(fontsize=legendsize)
    plt.title('Objective (Data Loss)', fontsize=fontsize)
    plt.grid(True); plt.xticks(fontsize=ticksize); plt.yticks(fontsize=ticksize); plt.yscale('log')

    # --- PLOT 2: KKT Gap ---
    # plt.subplot(1, 5, 2)
    # for i, history in enumerate(Histories):
    #     label = legends[i] if legends is not None and i < len(legends) else f'Run {i+1}'
    #     _, KKT_gaps, _, _, _ = zip(*history)
    #     plt.plot(KKT_gaps, label=label, linewidth=linewidth)
    
    # plt.xlabel('Iteration (x100)', fontsize=fontsize)
    # plt.ylabel('KKT Gap', fontsize=fontsize)
    # plt.yscale('log'); plt.legend(fontsize=legendsize)
    # plt.title('KKT Gap', fontsize=fontsize)
    # plt.grid(True); plt.xticks(fontsize=ticksize); plt.yticks(fontsize=ticksize)

    # --- PLOT 3: Initial Condition Loss ---
    plt.subplot(1, 5, 2)
    for i, history in enumerate(Histories):
        label = legends[i] if legends is not None and i < len(legends) else f'Run {i+1}'
        _, _, ic_violations, _, _ = zip(*history)
        plt.plot(ic_violations, label=label, linewidth=linewidth)
        
    plt.xlabel('Iteration (x100)', fontsize=fontsize)
    plt.ylabel('IC Loss', fontsize=fontsize)
    plt.yscale('log'); plt.legend(fontsize=legendsize)
    plt.title('Constraint (IC Loss)', fontsize=fontsize)
    plt.grid(True); plt.xticks(fontsize=ticksize); plt.yticks(fontsize=ticksize)

    # --- PLOT 4: Boundary Condition Loss ---
    plt.subplot(1, 5, 3)
    for i, history in enumerate(Histories):
        label = legends[i] if legends is not None and i < len(legends) else f'Run {i+1}'
        _, _, _, bc_violations, _ = zip(*history)
        plt.plot(bc_violations, label=label, linewidth=linewidth)
        
    plt.xlabel('Iteration (x100)', fontsize=fontsize)
    plt.ylabel('BC Loss', fontsize=fontsize)
    plt.yscale('log'); plt.legend(fontsize=legendsize)
    plt.title('Constraint (BC Loss)', fontsize=fontsize)
    plt.grid(True); plt.xticks(fontsize=ticksize); plt.yticks(fontsize=ticksize)

    # --- PLOT 5: PDE Loss ---
    plt.subplot(1, 5, 4)
    for i, history in enumerate(Histories):
        label = legends[i] if legends is not None and i < len(legends) else f'Run {i+1}'
        _, _, _, _, pde_violations = zip(*history)
        plt.plot(pde_violations, label=label, linewidth=linewidth)
        
    plt.xlabel('Iteration (x100)', fontsize=fontsize)
    plt.ylabel('PDE Loss', fontsize=fontsize)
    plt.yscale('log'); plt.legend(fontsize=legendsize)
    plt.title('Constraint (PDE Loss)', fontsize=fontsize)
    plt.grid(True); plt.xticks(fontsize=ticksize); plt.yticks(fontsize=ticksize)

    plt.tight_layout()
    if savename:
        dirpath = os.path.dirname(savename)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        plt.savefig(savename, bbox_inches='tight')
    plt.show()

# Plot SOFL loss history
os.makedirs("./figures", exist_ok=True)


def calculate_l2_relative_error(model):
        model.eval()
        x = torch.linspace(0, 1, 100, device=device)
        y = torch.linspace(0, 1, 100, device=device)
        t = torch.linspace(0, 1, 100, device=device)
        X, Y, T = torch.meshgrid(x, y, t, indexing='ij')
        input_tensor = torch.stack([X.flatten(), Y.flatten(), T.flatten()], dim=1)
        with torch.no_grad():
            U_pred = model(input_tensor).reshape(X.shape)
        exp_term = torch.exp(-2 * torch.pi**2 * ALPHA_TRUE * T)
        U_exact = exp_term * torch.sin(torch.pi * X) * torch.sin(torch.pi * Y)
        mse_error = torch.mean((U_pred - U_exact) ** 2)
        l2_error = torch.linalg.norm(U_pred - U_exact) / torch.linalg.norm(U_exact)
        learned_alpha = model.alpha.item()
        alpha_error = abs(learned_alpha - ALPHA_TRUE)
        print(f"\n--- Final Model Evaluation (SOFL) ---")
        print(f"True Alpha:       {ALPHA_TRUE:.6f}")
        print(f"Learned Alpha:    {learned_alpha:.6f}")
        print(f"Absolute Alpha Error: {alpha_error:.6e}")
        print(f"L2 Relative Error (u): {l2_error.item():.6e}")
        print(f"Mean Squared Error (u): {mse_error.item():.6e}")

calculate_l2_relative_error(pinn_model)


# --- PDE residual function for test evaluation ---
def pde_residual(x, y, t, model):
    """Compute PDE residual f = u_t - alpha*(u_xx + u_yy)"""
    input_data = torch.cat([x, y, t], dim=1)
    u = model(input_data)
    
    grads = torch.autograd.grad(u, [x, y, t], grad_outputs=torch.ones_like(u), create_graph=True)
    u_x, u_y, u_t = grads[0], grads[1], grads[2]

    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
    
    residual = u_t - model.alpha * (u_xx + u_yy)
    return residual

# --- Calculate final testing losses ---
def calculate_test_losses(model):
    model.eval()
    mse_loss = nn.MSELoss()
    
    # Create full test grid
    x_test = torch.linspace(0, 1, 100, device=device)
    y_test = torch.linspace(0, 1, 100, device=device) 
    t_test = torch.linspace(0, 1, 100, device=device)
    X_test, Y_test, T_test = torch.meshgrid(x_test, y_test, t_test, indexing='ij')
    
    # Flatten for model input
    x_flat = X_test.flatten().unsqueeze(1)
    y_flat = Y_test.flatten().unsqueeze(1)
    t_flat = T_test.flatten().unsqueeze(1)
    
    # Get learned alpha
    learned_alpha = model.alpha
    
    # --- Testing Physics Loss (MSE of residual on full grid) ---
    x_flat_req = x_flat.clone().detach().requires_grad_(True)
    y_flat_req = y_flat.clone().detach().requires_grad_(True)
    t_flat_req = t_flat.clone().detach().requires_grad_(True)
    
    residual_test = pde_residual(x_flat_req, y_flat_req, t_flat_req, model)
    physics_test_loss = mse_loss(residual_test, torch.zeros_like(residual_test)).item()
    
    # --- Testing Boundary Loss (MSE on x=0, x=1, y=0, y=1) ---
    # Boundary points: x=0 or x=1 or y=0 or y=1
    boundary_mask = ((X_test == 0) | (X_test == 1) | (Y_test == 0) | (Y_test == 1)).flatten()
    x_bc_test = x_flat[boundary_mask]
    y_bc_test = y_flat[boundary_mask]
    t_bc_test = t_flat[boundary_mask]
    
    with torch.no_grad():
        u_pred_bc = model(torch.cat([x_bc_test, y_bc_test, t_bc_test], dim=1))
    u_bc_true = torch.zeros_like(u_pred_bc)
    boundary_test_loss = mse_loss(u_pred_bc, u_bc_true).item()
    
    # --- Testing Initial Loss (MSE at t=0) ---
    initial_mask = (T_test == 0).flatten()
    x_ic_test = x_flat[initial_mask]
    y_ic_test = y_flat[initial_mask]
    t_ic_test = t_flat[initial_mask]
    
    with torch.no_grad():
        u_pred_ic = model(torch.cat([x_ic_test, y_ic_test, t_ic_test], dim=1))
    u_ic_true = initial_condition(x_ic_test, y_ic_test)
    initial_test_loss = mse_loss(u_pred_ic, u_ic_true).item()
    
    # --- Testing Data Loss (MSE on observational data points) ---
    # Generate same data points as in training
    num_points_data = 100  # Same as training
    torch.manual_seed(42)  # For reproducibility
    x_data_test = torch.rand(num_points_data, 1, device=device)
    y_data_test = torch.rand(num_points_data, 1, device=device)
    t_data_test = torch.rand(num_points_data, 1, device=device)
    u_data_true = analytical_solution(x_data_test, y_data_test, t_data_test, ALPHA_TRUE)
    
    with torch.no_grad():
        u_pred_data = model(torch.cat([x_data_test, y_data_test, t_data_test], dim=1))
    data_test_loss = mse_loss(u_pred_data, u_data_true).item()
    
    print("- Final Test Losses ---")
    print(f"Test Physics Loss (MSE of residual): {physics_test_loss:.6e}")
    print(f"Test Boundary Loss (MSE): {boundary_test_loss:.6e}")
    print(f"Test Initial Loss (MSE): {initial_test_loss:.6e}")
    print(f"Test Data Loss (MSE): {data_test_loss:.6e}")

calculate_test_losses(pinn_model)


