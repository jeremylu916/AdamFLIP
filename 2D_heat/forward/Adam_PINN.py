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

# UPDATED PINN class with a deeper network
class PINN(nn.Module):
    def __init__(self):
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

    def forward(self, x):
        return self.net(x)

# Define the initial and boundary conditions
def initial_condition(x, y):
    return torch.sin(torch.pi * x) * torch.sin(torch.pi * y)

# Define the PDE residual
def pde(x, y, t, model):
    input_data = torch.cat([x, y, t], dim=1)
    u = model(input_data)
    
    # Use torch.autograd.grad to compute derivatives
    grads = torch.autograd.grad(u, [x, y, t], grad_outputs=torch.ones_like(u), create_graph=True)
    u_x, u_y, u_t = grads[0], grads[1], grads[2]

    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
    
    alpha = 0.1 
    return u_t - alpha * (u_xx + u_yy)

# --- 2. MODIFY DATA GENERATION TO USE THE SELECTED DEVICE ---
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

    return (x_pde, y_pde, t_pde, 
            x_ic, y_ic, t_ic, u_ic_exact,
            x_bc, y_bc, t_bc, u_bc_exact)

# Main training loop
# def train_pinn(model, num_iterations, num_points_pde, num_points_bc, num_points_ic, weights):
#     optimizer = optim.Adam(model.parameters(), lr=5e-4)
#     #scheduler = lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.9)
#     mse_loss = nn.MSELoss()

#     w_ic, w_bc, w_pde = weights['ic'], weights['bc'], weights['pde']

#     start_time = time.time()  # Start timing

#     for iteration in range(num_iterations):
#         optimizer.zero_grad()

#         # Data is already generated on the correct device
#         (x_pde, y_pde, t_pde, 
#          x_ic, y_ic, t_ic, u_ic_exact,
#          x_bc, y_bc, t_bc, u_bc_exact) = generate_training_data(num_points_pde, num_points_bc, num_points_ic)

#         # Initial condition loss
#         u_pred_ic = model(torch.cat([x_ic, y_ic, t_ic], dim=1))
#         loss_ic = mse_loss(u_pred_ic, u_ic_exact)

#         # Boundary condition loss
#         u_pred_bc = model(torch.cat([x_bc, y_bc, t_bc], dim=1))
#         loss_bc = mse_loss(u_pred_bc, u_bc_exact)

#         # PDE residual loss
#         residual = pde(x_pde, y_pde, t_pde, model)
#         loss_pde = mse_loss(residual, torch.zeros_like(residual))

#         # Total weighted loss
#         loss = w_ic * loss_ic + w_bc * loss_bc + w_pde * loss_pde

#         loss.backward()
#         optimizer.step()
#        # scheduler.step()

#         if iteration % 1000 == 0:
#             elapsed_time = time.time() - start_time  # Calculate elapsed time
#             #print(f"Iteration: {iteration:5d}, Loss: {loss.item():.4e}, LR: {scheduler.get_last_lr()[0]:.4e}, Time Elapsed: {elapsed_time:.2f}s")
#             print(f"Iteration: {iteration:5d}, Loss: {loss.item():.4e}, Time Elapsed: {elapsed_time:.2f}s")

# Main training loop
def train_pinn(model, num_iterations, num_points_pde, num_points_bc, num_points_ic, weights):
    optimizer = optim.Adam(model.parameters(), lr=5e-3)
    mse_loss = nn.MSELoss()

    w_ic, w_bc, w_pde = weights['ic'], weights['bc'], weights['pde']

    start_time = time.time()  # Start timing

    # Open a file to save the losses
    os.makedirs("./training_logs", exist_ok=True)
    loss_file_path = "./training_logs/ADAM-PINN_training_loss.txt"
    with open(loss_file_path, "w") as f:
        f.write("Epoch,Loss_IC,Loss_BC,Loss_PDE,Total_Loss\n")  # Write the header

        for iteration in range(num_iterations):
            optimizer.zero_grad()

            # Data is already generated on the correct device
            (x_pde, y_pde, t_pde, 
             x_ic, y_ic, t_ic, u_ic_exact,
             x_bc, y_bc, t_bc, u_bc_exact) = generate_training_data(num_points_pde, num_points_bc, num_points_ic)

            # Initial condition loss
            u_pred_ic = model(torch.cat([x_ic, y_ic, t_ic], dim=1))
            loss_ic = mse_loss(u_pred_ic, u_ic_exact)

            # Boundary condition loss
            u_pred_bc = model(torch.cat([x_bc, y_bc, t_bc], dim=1))
            loss_bc = mse_loss(u_pred_bc, u_bc_exact)

            # PDE residual loss
            residual = pde(x_pde, y_pde, t_pde, model)
            loss_pde = mse_loss(residual, torch.zeros_like(residual))

            # Total weighted loss
            loss =  loss_ic +  loss_bc +  loss_pde

            loss.backward()
            optimizer.step()

            # Log and save losses every 100 iterations
            if iteration % 100 == 0:
                elapsed_time = time.time() - start_time  # Calculate elapsed time
                print(f"Iteration: {iteration:5d}, Loss: {loss.item():.4e}, Time Elapsed: {elapsed_time:.2f}s")
                f.write(f"{iteration},{loss_ic.item():.6e},{loss_bc.item():.6e},{loss_pde.item():.6e},{loss.item():.6e}\n")



# --- Main Execution ---
if __name__ == "__main__":
    # Hyperparameters
    NUM_ITERATIONS = 30000
    NUM_POINTS_PDE = 2000
    NUM_POINTS_BC = 500
    NUM_POINTS_IC = 500

    # Loss weights
    loss_weights = {'ic': 10.0, 'bc': 10.0, 'pde': 1.0}

    # --- 3. MOVE THE MODEL TO THE SELECTED DEVICE ---
    pinn_model = PINN().to(device)
    train_pinn(pinn_model, NUM_ITERATIONS, NUM_POINTS_PDE, NUM_POINTS_BC, NUM_POINTS_IC, loss_weights)
    
# --- 4. MODIFY PLOTTING TO USE THE DEVICE AND MOVE RESULTS TO CPU ---
def plot_solution_comparison(model, t_value):
    model.eval()
    alpha = 0.1

    # Create grid on the specified device
    x = torch.linspace(0, 1, 100, device=device)
    y = torch.linspace(0, 1, 100, device=device)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    T = torch.full(X.shape, t_value, device=device)

    # PINN Prediction
    input_tensor = torch.stack([X.flatten(), Y.flatten(), T.flatten()], dim=1)
    with torch.no_grad():
        U_pred = model(input_tensor).reshape(X.shape)

    # Analytical Solution
    exp_term = np.exp(-2 * np.pi**2 * alpha * t_value)
    U_exact = exp_term * torch.sin(np.pi * X) * torch.sin(np.pi * Y)

    # Error
    Error = torch.abs(U_pred - U_exact)

    # Move tensors to CPU for plotting
    X_np, Y_np = X.cpu().numpy(), Y.cpu().numpy()
    U_pred_np = U_pred.cpu().numpy()
    U_exact_np = U_exact.cpu().numpy()
    Error_np = Error.cpu().numpy()

    # Save data
    os.makedirs("./mat_files", exist_ok=True)
    savemat(f"./mat_files/solution_comparison_t{t_value}.mat", {
        "X": X_np,
        "Y": Y_np,
        "U_pred": U_pred_np,
        "U_exact": U_exact_np,
        "Error": Error_np
    })

    # Plotting
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Comparison at t = {t_value}', fontsize=16)

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

    # Adjust layout spacing (reduce blank space)
    plt.subplots_adjust(wspace=0.1, hspace=0.1, top=0.88)
    
    os.makedirs("./figures", exist_ok=True)
    fig.savefig(f"./figures/pinn_comparison_t{t_value}.png", dpi=300, bbox_inches='tight')
    plt.show()


# The model is already on the GPU (if available), so this will run on the device
plot_solution_comparison(pinn_model, t_value=0.5)
plot_solution_comparison(pinn_model, t_value=1.0)



def calculate_l2_relative_error(model):
        model.eval()
        alpha = 0.1
        x = torch.linspace(0, 1, 100, device=device)
        y = torch.linspace(0, 1, 100, device=device)
        t = torch.linspace(0, 1, 100, device=device)
        X, Y, T = torch.meshgrid(x, y, t, indexing='ij')
        input_tensor = torch.stack([X.flatten(), Y.flatten(), T.flatten()], dim=1)
        with torch.no_grad():
            U_pred = model(input_tensor).reshape(X.shape)
            
        # --- CORRECTED ANALYTICAL SOLUTION ---
        exp_term = torch.exp(-2 * torch.pi**2 * alpha * T) # Use torch.pi
        U_exact = exp_term * torch.sin(torch.pi * X) * torch.sin(torch.pi * Y) # Use torch.pi
        mse_error = torch.mean((U_pred - U_exact) ** 2) 
        l2_error = torch.linalg.norm(U_pred - U_exact) / torch.linalg.norm(U_exact)
        print(f"\n--- Final L2 Relative Error ---")
        print(f"L2 Relative Error: {l2_error.item():.6e}")
        print(f"Mean Squared Error (MSE): {mse_error.item():.6e}")
calculate_l2_relative_error(pinn_model)


def calculate_final_losses(model, num_points_pde, num_points_bc, num_points_ic, weights):
    """
    Calculates the final unweighted IC, BC, and PDE losses on a new, 
    randomly sampled batch of points after training is complete.
    """
    print("\n--- Calculating Final Losses on a New Batch ---")
    model.eval()  # Set model to evaluation mode
    mse_loss = nn.MSELoss()

    w_ic, w_bc, w_pde = weights['ic'], weights['bc'], weights['pde']

    # Generate a new batch of validation data
    (x_pde, y_pde, t_pde,
     x_ic, y_ic, t_ic, u_ic_exact,
     x_bc, y_bc, t_bc, u_bc_exact) = generate_training_data(num_points_pde, num_points_bc, num_points_ic)

    # IC and BC losses can be computed without gradients
    with torch.no_grad():
        # Initial condition loss
        u_pred_ic = model(torch.cat([x_ic, y_ic, t_ic], dim=1))
        loss_ic = mse_loss(u_pred_ic, u_ic_exact)

        # Boundary condition loss
        u_pred_bc = model(torch.cat([x_bc, y_bc, t_bc], dim=1))
        loss_bc = mse_loss(u_pred_bc, u_bc_exact)

    # PDE residual loss requires gradients for the pde function itself.
    # The input points (x_pde, y_pde, t_pde) from generate_training_data
    # already have requires_grad=True, so we don't need to wrap this in no_grad().
    residual = pde(x_pde, y_pde, t_pde, model)
    loss_pde = mse_loss(residual, torch.zeros_like(residual))

    # Total weighted loss
    loss = w_ic * loss_ic + w_bc * loss_bc + w_pde * loss_pde

    print(f"Final Initial Condition Loss (unweighted): {loss_ic.item():.6e}")
    print(f"Final Boundary Condition Loss (unweighted): {loss_bc.item():.6e}")
    print(f"Final Physics (PDE) Loss (unweighted):   {loss_pde.item():.6e}")
    print(f"Final Total Weighted Loss:                {loss.item():.6e}")

    model.train() # Set model back to train mode, just in case

# --- CALL THE NEW FUNCTION AT THE END ---
# We need the hyperparameters, so we call it here.
# Note: This will use the hyperparameter variables defined in the __name__ == "__main__" block
# because the model was trained with them.
calculate_final_losses(pinn_model, NUM_POINTS_PDE, NUM_POINTS_BC, NUM_POINTS_IC, loss_weights)
