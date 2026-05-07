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

# --- 1. SET SEED FOR REPRODUCIBILITY ---
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # for multi-GPU
    # These two lines are often needed for full reproducibility on GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



# --- 1. SET UP DEVICE FOR GPU OR CPU ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device} 🚀")


ALPHA_TRUE = 0.1

# --- MODIFIED: PINN class now includes alpha as a trainable parameter ---
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
        # --- MODIFIED: Set alpha as a trainable nn.Parameter ---
        # We use .to(device) here to ensure the parameter is on the correct device from the start
        self.alpha = nn.Parameter(torch.tensor([initial_alpha_guess], device=device, dtype=torch.float32))

    def forward(self, x):
        return self.net(x)


# Define the initial condition (unchanged)
def initial_condition(x, y):
    return torch.sin(torch.pi * x) * torch.sin(torch.pi * y)

# --- MODIFIED: Helper function for the analytical solution ---
# We use this to generate the "true" data for the inverse problem
def analytical_solution(x, y, t, alpha):
    return torch.sin(torch.pi * x) * torch.sin(torch.pi * y) * torch.exp(-2 * torch.pi**2 * alpha * t)

# --- MODIFIED: PDE residual now accepts alpha as an argument ---
def pde(x, y, t, model, learned_alpha):
    input_data = torch.cat([x, y, t], dim=1)
    u = model(input_data)
    
    # Use torch.autograd.grad to compute derivatives
    grads = torch.autograd.grad(u, [x, y, t], grad_outputs=torch.ones_like(u), create_graph=True)
    u_x, u_y, u_t = grads[0], grads[1], grads[2]

    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
    
    # --- MODIFIED: Use the passed 'learned_alpha' instead of a hard-coded value ---
    return u_t - learned_alpha * (u_xx + u_yy)

# --- MODIFIED: Data generation now includes "sensor data" points ---
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

    # --- MODIFIED: Add "sensor data" points (no gradients needed) ---
    # These are points where we "know" the solution.
    # We generate them using the TRUE alpha.
    x_data = torch.rand(num_points_data, 1, device=device)
    y_data = torch.rand(num_points_data, 1, device=device)
    t_data = torch.rand(num_points_data, 1, device=device) # t > 0
    u_data_exact = analytical_solution(x_data, y_data, t_data, ALPHA_TRUE)

    return (x_pde, y_pde, t_pde,
            x_ic, y_ic, t_ic, u_ic_exact,
            x_bc, y_bc, t_bc, u_bc_exact,
            x_data, y_data, t_data, u_data_exact) # <-- MODIFIED: return new data


# --- MODIFIED: Main training loop for inverse problem ---
def train_pinn(model, num_iterations, num_points_pde, num_points_bc, num_points_ic, num_points_data, weights):
    optimizer = optim.Adam(model.parameters(), lr=5e-3)
    mse_loss = nn.MSELoss()

    # --- MODIFIED: Get all weights ---
    w_ic, w_bc, w_pde, w_data = weights['ic'], weights['bc'], weights['pde'], weights['data']

    start_time = time.time()  # Start timing

    # Open a file to save the losses
    os.makedirs("./training_logs", exist_ok=True)
    loss_file_path = "./training_logs/ADAM_PINN_inverse_training_loss.txt"
    os.makedirs("./training_data", exist_ok=True)
    loss_epochs = []
    loss_ic_history = []
    loss_bc_history = []
    loss_pde_history = []
    loss_data_history = []
    with open(loss_file_path, "w") as f:
        # --- MODIFIED: Updated log header ---
        f.write("Epoch,Loss_IC,Loss_BC,Loss_PDE,Loss_Data,Total_Loss,Learned_Alpha\n")
        f.flush()
        for iteration in range(num_iterations):
            optimizer.zero_grad()

            # --- MODIFIED: Get new data points ---
            (x_pde, y_pde, t_pde, 
             x_ic, y_ic, t_ic, u_ic_exact,
             x_bc, y_bc, t_bc, u_bc_exact,
             x_data, y_data, t_data, u_data_exact) = generate_training_data(num_points_pde, num_points_bc, num_points_ic, num_points_data)

            # --- MODIFIED: Get the current learned alpha from the model ---
            learned_alpha = model.alpha
            
            # Initial condition loss (unchanged)
            u_pred_ic = model(torch.cat([x_ic, y_ic, t_ic], dim=1))
            loss_ic = mse_loss(u_pred_ic, u_ic_exact)

            # Boundary condition loss (unchanged)
            u_pred_bc = model(torch.cat([x_bc, y_bc, t_bc], dim=1))
            loss_bc = mse_loss(u_pred_bc, u_bc_exact)

            # PDE residual loss (--- MODIFIED: pass learned_alpha ---)
            residual = pde(x_pde, y_pde, t_pde, model, learned_alpha)
            loss_pde = mse_loss(residual, torch.zeros_like(residual))

            # --- MODIFIED: Add new "data" loss ---
            u_pred_data = model(torch.cat([x_data, y_data, t_data], dim=1))
            loss_data = mse_loss(u_pred_data, u_data_exact)

            # --- MODIFIED: Total weighted loss (includes data loss) ---
            # Also fixed the original code to use the weights
            # loss = (w_ic * loss_ic + 
            #         w_bc * loss_bc + 
            #         w_pde * loss_pde + 
            #         w_data * loss_data)
            loss = loss_ic + loss_bc + loss_pde + loss_data

            loss.backward()
            optimizer.step()

            # Log and save losses every 100 iterations
            if iteration % 100 == 0:
                elapsed_time = time.time() - start_time
                # --- MODIFIED: Print and log the learned alpha ---
                print(f"Iteration: {iteration:5d}, Loss: {loss.item():.4e}, Learned alpha: {learned_alpha.item():.4f}, Time: {elapsed_time:.2f}s")
                f.write(f"{iteration},{loss_ic.item():.6e},{loss_bc.item():.6e},{loss_pde.item():.6e},{loss_data.item():.6e},{loss.item():.6e},{learned_alpha.item():.6e}\n")
                loss_epochs.append(iteration)
                loss_ic_history.append(loss_ic.item())
                loss_bc_history.append(loss_bc.item())
                loss_pde_history.append(loss_pde.item())
                loss_data_history.append(loss_data.item())




# --- Main Execution ---
if __name__ == "__main__":
    # Hyperparameters
    NUM_ITERATIONS = 20000
    NUM_POINTS_PDE = 2000
    NUM_POINTS_BC = 500
    NUM_POINTS_IC = 500
    # --- MODIFIED: Add number of "sensor data" points ---
    NUM_POINTS_DATA = 1000

    # --- MODIFIED: Loss weights (added data, and weighted it heavily) ---
    loss_weights = {'ic': 1.0, 'bc': 1.0, 'pde': 0.1, 'data': 100.0}

    # --- MODIFIED: Instantiate model with initial guess ---
    initial_alpha_guess = 1.0
    print("***** Our initial kappa guess****\n", initial_alpha_guess)
    pinn_model = PINN(initial_alpha_guess=1.0).to(device)
    
    # --- MODIFIED: Pass new argument to training function ---
    train_pinn(pinn_model, NUM_ITERATIONS, 
               NUM_POINTS_PDE, NUM_POINTS_BC, NUM_POINTS_IC, NUM_POINTS_DATA, 
               loss_weights)
    


# --- MODIFIED: Plotting function updated to show learned alpha ---
def plot_solution_comparison(model, t_value):
    model.eval()
    
    # --- MODIFIED: Get the *learned* alpha from the model ---
    learned_alpha = model.alpha.item()

    # Create grid on the specified device
    x = torch.linspace(0, 1, 100, device=device)
    y = torch.linspace(0, 1, 100, device=device)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    T = torch.full(X.shape, t_value, device=device)

    # PINN Prediction
    input_tensor = torch.stack([X.flatten(), Y.flatten(), T.flatten()], dim=1)
    with torch.no_grad():
        U_pred = model(input_tensor).reshape(X.shape)

    # Analytical Solution (--- MODIFIED: Use ALPHA_TRUE for ground truth) ---
    exp_term = np.exp(-2 * np.pi**2 * ALPHA_TRUE * t_value)
    # --- MODIFIED: Need to move X and Y to CPU for numpy operations ---
    U_exact = exp_term * torch.sin(np.pi * X.cpu()) * torch.sin(np.pi * Y.cpu())
    # --- MODIFIED: Move U_exact to the correct device for error calculation ---
    U_exact = U_exact.to(device)

    # Error
    Error = torch.abs(U_pred - U_exact)

    # Move tensors to CPU for plotting
    X_np, Y_np = X.cpu().numpy(), Y.cpu().numpy()
    U_pred_np = U_pred.cpu().numpy()
    U_exact_np = U_exact.cpu().numpy()
    Error_np = Error.cpu().numpy()

    # Save data
    os.makedirs("./data", exist_ok=True)
    savemat(f"./data/Adam_PINN_solution_t{t_value}.mat", {
        "X": X_np,
        "Y": Y_np,
        "U_pred": U_pred_np,
        "U_exact": U_exact_np,
        "Error": Error_np,
        "learned_alpha": learned_alpha,
        "true_alpha": ALPHA_TRUE
    })

    # Plotting
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    # --- MODIFIED: Update title to show learned alpha ---
    fig.suptitle(f'Comparison at t = {t_value}' , fontsize=16)

    # Shared color range
    v_min = min(U_pred_np.min(), U_exact_np.min())
    v_max = max(U_pred_np.max(), U_exact_np.max())

    # Prediction
    c1 = ax1.pcolormesh(X_np, Y_np, U_pred_np, cmap='jet', vmin=v_min, vmax=v_max, shading='auto')
    fig.colorbar(c1, ax=ax1)
    ax1.set_title("Prediction")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.axis('square')

    # Exact
    c2 = ax2.pcolormesh(X_np, Y_np, U_exact_np, cmap='jet', vmin=v_min, vmax=v_max, shading='auto')
    fig.colorbar(c2, ax=ax2)
    ax2.set_title("Exact Solution")
    ax2.set_xlabel("x")
    ax2.axis('square')

    # Error
    c3 = ax3.pcolormesh(X_np, Y_np, Error_np, cmap='Reds', shading='auto')
    fig.colorbar(c3, ax=ax3)
    ax3.set_title("Absolute Error")
    ax3.set_xlabel("x")
    ax3.axis('square')

    plt.subplots_adjust(wspace=0.1, hspace=0.1, top=0.88)
    
    os.makedirs("./figures", exist_ok=True)
    fig.savefig(f"./figures/pinn_inverse_comparison_t{t_value}.png", dpi=300, bbox_inches='tight')
    plt.show()


# Plot results
plot_solution_comparison(pinn_model, t_value=0.5)
plot_solution_comparison(pinn_model, t_value=1.0)


# --- MODIFIED: L2 Error calculation updated to report on alpha ---
def calculate_l2_relative_error(model):
    model.eval()
    x = torch.linspace(0, 1, 100, device=device)
    y = torch.linspace(0, 1, 100, device=device)
    t = torch.linspace(0, 1, 100, device=device)
    X, Y, T = torch.meshgrid(x, y, t, indexing='ij')
    input_tensor = torch.stack([X.flatten(), Y.flatten(), T.flatten()], dim=1)
    
    with torch.no_grad():
        U_pred = model(input_tensor).reshape(X.shape)
        
    # --- MODIFIED: Use ALPHA_TRUE for the exact solution ---
    exp_term = torch.exp(-2 * torch.pi**2 * ALPHA_TRUE * T) # Use torch.pi and TRUE alpha
    U_exact = exp_term * torch.sin(torch.pi * X) * torch.sin(torch.pi * Y) # Use torch.pi
    
    mse_error = torch.mean((U_pred - U_exact) ** 2)
    l2_error = torch.linalg.norm(U_pred - U_exact) / torch.linalg.norm(U_exact)
    
    # --- MODIFIED: Print stats about alpha ---
    learned_alpha = model.alpha.item()
    alpha_error = abs(learned_alpha - ALPHA_TRUE)
    
    print(f"\n--- Final Model Evaluation ---")
    print(f"True Alpha:       {ALPHA_TRUE:.6f}")
    print(f"Learned Alpha:    {learned_alpha:.6f}")
    print(f"Absolute Alpha Error: {alpha_error:.6e}")
    print(f"L2 Relative Error (u): {l2_error.item():.6e}")
    print(f"Mean Squared Error (u): {mse_error.item():.6e}")

calculate_l2_relative_error(pinn_model)


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
    
    residual_test = pde(x_flat_req, y_flat_req, t_flat_req, model, learned_alpha)
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
    
    print(" Final Test Losses ---")
    print(f"Test Physics Loss (MSE of residual): {physics_test_loss:.6e}")
    print(f"Test Boundary Loss (MSE): {boundary_test_loss:.6e}")
    print(f"Test Initial Loss (MSE): {initial_test_loss:.6e}")
    print(f"Test Data Loss (MSE): {data_test_loss:.6e}")

calculate_test_losses(pinn_model)