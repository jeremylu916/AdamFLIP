import torch
import torch.nn as nn
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os

# --- Re-use necessary components from your training script ---

# Setup device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# PINN Class (exactly as before)
class PhysicsInformedNN():
    def __init__(self, X_u, u, X_f):
        # Move tensors to the selected device
        self.x_u = torch.tensor(X_u[:, 0:1], dtype=torch.float32, requires_grad=True).to(device)
        self.t_u = torch.tensor(X_u[:, 1:2], dtype=torch.float32, requires_grad=True).to(device)
        self.x_f = torch.tensor(X_f[:, 0:1], dtype=torch.float32, requires_grad=True).to(device)
        self.t_f = torch.tensor(X_f[:, 1:2], dtype=torch.float32, requires_grad=True).to(device)
        self.u = torch.tensor(u, dtype=torch.float32).to(device)
        self.null = torch.zeros((self.x_f.shape[0], 1), device=device)
        self.create_net()
        self.loss = nn.MSELoss()

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
        return self.net(torch.hstack((x, t)))

    def net_f(self, x, t):
        u = self.net_u(x, t)
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True)[0]
        v = 0.01 / np.pi
        f = u_t + (u * u_x) - (v * u_xx)
        return f

def restore(restore_path):
    """Restores the state of the network and optimizer."""
    return torch.load(restore_path)

# --- New Function to Load Model and Plot Residual ---

def load_adam_model_and_plot_residual(model_path, data_path):
    """
    Loads a trained PINN model, calculates the PDE residual (f_pred)
    on the test grid, and plots it as a heatmap.
    """
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return

    print("--- Loading Data and Model ---")
    # Load exact solution grid for evaluation
    data = scipy.io.loadmat(data_path)
    t_exact = data['t'].flatten()[:, None]
    x_exact = data['x'].flatten()[:, None]
    T, X = np.meshgrid(t_exact, x_exact)
    
    # We need to initialize the PINN class with some dummy data
    # The actual data used for prediction will be the exact grid
    dummy_X_u = np.zeros((1, 2))
    dummy_u = np.zeros((1, 1))
    dummy_X_f = np.zeros((1, 2))
    pinn = PhysicsInformedNN(dummy_X_u, dummy_u, dummy_X_f)

    # Load the trained model's state dictionary
    checkpoint = restore(model_path)
    pinn.net.load_state_dict(checkpoint['state_dict'])
    pinn.net.eval()  # Set the model to evaluation mode
    print("Model loaded successfully.")

    print("\n--- Calculating PDE Residual (f_pred) ---")
    # Prepare the full test grid as tensors that require gradients
    x_full = torch.tensor(X.flatten()[:, None], dtype=torch.float32, requires_grad=True).to(device)
    t_full = torch.tensor(T.flatten()[:, None], dtype=torch.float32, requires_grad=True).to(device)

    # Calculate the residual f(x,t) using the model's net_f method
    # This must be done outside of a torch.no_grad() context
    f_pred_tensor = pinn.net_f(x_full, t_full)

    # Move the result to CPU and reshape for plotting
    f_pred = f_pred_tensor.detach().cpu().numpy()
    f_pred = f_pred.reshape(X.shape)
    f_pred = f_pred / np.max(np.abs(f_pred))
    
    print("Residual calculation complete.")

    print("\n--- Plotting the Residual Heatmap ---")
    # Ensure the output directory exists
    os.makedirs("./figures/solution", exist_ok=True)
    
    plt.rcParams['font.size'] = '15'
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    
    # Use a diverging colormap to highlight positive/negative values
    # Center the color bar at zero
    # vmax = np.max(np.abs(f_pred))
    # img_handle = ax.imshow(f_pred, interpolation='nearest', cmap='seismic',
    #                        extent=[t_exact.min(), t_exact.max(), x_exact.min(), x_exact.max()],
    #                        origin='lower', aspect='auto', vmin=-vmax, vmax=vmax)
    
    # ax.set_xlabel(r"$t$")
    # ax.set_ylabel(r"$x$")
    # ax.set_title(r"PDE Residual $f(x,t)$")
    
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.10)
    # cbar = fig.colorbar(img_handle, cax=cax)
    # cbar.set_label(r"Residual Value", rotation=270, labelpad=20)
    # cbar.ax.tick_params(labelsize=10)
    # --- Plotting Physics Residual (f_pred) ---
    print("\n--- Plotting Physics Residual (f_pred) ---")

    # Create a grid of points for evaluation
    x_eval = np.linspace(-1, 1, 100)  # 100 points in the x-direction
    t_eval = np.linspace(0, 1, 100)   # 100 points in the t-direction
    T_eval, X_eval = np.meshgrid(t_eval, x_eval)  # Create a 2D grid for x and t
    X_f_eval = np.hstack((X_eval.flatten()[:, None], T_eval.flatten()[:, None]))

    # Convert to PyTorch tensors with requires_grad=True
    X_f_eval_tensor = torch.tensor(X_f_eval, dtype=torch.float32, requires_grad=True).to(device)

    # Evaluate f_pred on the grid
    f_pred_tensor = pinn.net_f(
        X_f_eval_tensor[:, 0:1], X_f_eval_tensor[:, 1:2]
    )  # Evaluate the physics residual
    f_pred = f_pred_tensor.detach().cpu().numpy().reshape(X_eval.shape)  # Reshape to match the grid

    # Normalize the residual values to [-1, 1]
    f_pred = f_pred / np.max(np.abs(f_pred))

    # Plot the physics residual
    plt.figure(figsize=(10, 6))
    contour = plt.contourf(T_eval, X_eval, abs(f_pred), levels=100, cmap='jet')  # Use T_eval and X_eval
    plt.colorbar(contour, label='Physics Residual (f_pred)')
    plt.xlabel('t')
    plt.ylabel('x')
    plt.title('ADAM_Physics Residual (f_pred) Over x and t')
    plt.savefig('ADAM_physics_residual_f_pred.png', dpi=300)
    plt.show()

# --- Main execution ---
if __name__ == "__main__":
    MODEL_SAVE_PATH = "./model/best-model-adam.pth"
    DATA_PATH = 'burgers_shock.mat'
    load_adam_model_and_plot_residual(MODEL_SAVE_PATH, DATA_PATH)