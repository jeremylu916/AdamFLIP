# '''
# This part I'm using the adam to train PINN for Burgers
# '''


# import torch
# import torch.nn as nn
# import numpy as np
# from tqdm import trange
# import random as rm
# import scipy.io
# import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# import os

# # Note: The torch.distributed setup is no longer needed for the Adam optimizer
# # but is left here in case you switch back. It does not harm to run it.


# def save(net, optimizer, save_path):
#     """Saves the state of the network and optimizer."""
#     state = {'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict()}
#     torch.save(state, save_path)

# def restore(restore_path):
#     """Restores the state of the network and optimizer."""
#     return torch.load(restore_path)

# # Setup device
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(f"We are using device: {device}")

# # Set random seed for reproducibility
# seed = 42
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# np.random.seed(seed)
# rm.seed(seed)

# class PhysicsInformedNN():
#     """
#     A class for Physics-Informed Neural Networks.
#     """
#     def __init__(self, X_u, u, X_f):
#         # Move tensors to the selected device
#         self.x_u = torch.tensor(X_u[:, 0:1], dtype=torch.float32, requires_grad=True).to(device)
#         self.t_u = torch.tensor(X_u[:, 1:2], dtype=torch.float32, requires_grad=True).to(device)
#         self.x_f = torch.tensor(X_f[:, 0:1], dtype=torch.float32, requires_grad=True).to(device)
#         self.t_f = torch.tensor(X_f[:, 1:2], dtype=torch.float32, requires_grad=True).to(device)
#         self.u = torch.tensor(u, dtype=torch.float32).to(device)
#         self.null = torch.zeros((self.x_f.shape[0], 1), device=device)

#         # Create the neural network
#         self.create_net()
        
#         # Mean Squared Error Loss
#         self.loss = nn.MSELoss()

#     def create_net(self):
#         """Creates the neural network architecture and moves it to the device."""
#         self.net = nn.Sequential(
#             nn.Linear(2, 20), nn.Tanh(),
#             nn.Linear(20, 20), nn.Tanh(),
#             nn.Linear(20, 20), nn.Tanh(),
#             nn.Linear(20, 20), nn.Tanh(),
#             nn.Linear(20, 20), nn.Tanh(),
#             nn.Linear(20, 20), nn.Tanh(),
#             nn.Linear(20, 20), nn.Tanh(),
#             nn.Linear(20, 20), nn.Tanh(),
#             nn.Linear(20, 20), nn.Tanh(),
#             nn.Linear(20, 1)
#         ).to(device)

#     def net_u(self, x, t):
#         """Predicts u(x, t)"""
#         u = self.net(torch.hstack((x, t)))
#         return u

#     def net_f(self, x, t):
#         """Computes the residual of the PDE."""
#         u = self.net_u(x, t)
        
#         u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
#         u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
#         u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True)[0]

#         # Burger's equation residual
#         v = 0.01 / np.pi
#         f = u_t + (u * u_x) - (v * u_xx)
#         return f

# # --- Problem setup ---
# N_u = 100
# N_f = 10000

# # Boundary conditions
# x_upper = np.ones((N_u // 4, 1), dtype=float)
# x_lower = -np.ones((N_u // 4, 1), dtype=float)
# t_zero = np.zeros((N_u // 2, 1), dtype=float)
# t_upper = np.random.rand(N_u // 4, 1)
# t_lower = np.random.rand(N_u // 4, 1)
# x_zero = -1 + 2 * np.random.rand(N_u // 2, 1)

# X_upper = np.hstack((x_upper, t_upper))
# X_lower = np.hstack((x_lower, t_lower))
# X_zero = np.hstack((x_zero, t_zero))
# X_u_train = np.vstack((X_upper, X_lower, X_zero))

# # Initial conditions
# u_upper = np.zeros((N_u // 4, 1), dtype=float)
# u_lower = np.zeros((N_u // 4, 1), dtype=float)
# u_zero = -np.sin(np.pi * x_zero)
# u_train = np.vstack((u_upper, u_lower, u_zero))

# # Shuffle training data
# index = np.arange(N_u)
# np.random.shuffle(index)
# X_u_train = X_u_train[index, :]
# u_train = u_train[index, :]

# # Collocation points
# X_f_train = np.random.uniform([-1, 0], [1, 1], size=(N_f, 2))
# X_f_train = np.vstack((X_f_train, X_u_train))

# # --- Load exact solution ---
# data = scipy.io.loadmat('burgers_shock.mat')
# t_exact = data['t'].flatten()[:, None]
# x_exact = data['x'].flatten()[:, None]
# Exact = np.real(data['usol'])
# T, X = np.meshgrid(t_exact, x_exact)
# X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
# u_truth = Exact.flatten()[:, None]

# pinn = PhysicsInformedNN(X_u_train, u_train, X_f_train)

# # --- Loss functions ---
# def residual_loss(model):
#     f_pred = model.net_f(model.x_f, model.t_f)
#     return model.loss(f_pred, model.null)

# def boundary_loss(model):
#     u_pred = model.net_u(model.x_u, model.t_u)
#     return model.loss(u_pred, model.u)

# # --- CHANGE 1: Replaced custom optimizer with standard torch.optim.Adam ---
# optimizer = torch.optim.Adam(pinn.net.parameters(), lr=1e-3)

# # --- Training Loop ---
# epochs = 5000 # Increased epochs for better convergence with Adam
# best_loss = np.inf
# loss_history = []

# pbar = trange(epochs, desc="Training")
# pinn.net.train()

# for epoch in pbar:
#     optimizer.zero_grad()

#     loss_res = residual_loss(pinn)
#     loss_bcs = boundary_loss(pinn)
    
#     loss = loss_res + loss_bcs 

#     loss.backward()
    
#     # --- CHANGE 2: Reverted to standard optimizer step within a no_grad context ---
#     # The workaround is no longer needed for the Adam optimizer.
#     optimizer.step()

#     current_loss = loss.item()

#     if (epoch + 1) % 100 == 0:
#         loss_history.append((epoch + 1, current_loss))

#     if current_loss < best_loss:
#         best_loss = current_loss
#         save(pinn.net, optimizer, save_path="./model/best-model-adam.pth")

#     pbar.set_postfix({'Loss': f'{current_loss:.4e}', 'Best-Loss': f'{best_loss:.4e}'})

# np.save("./training_loss_history/loss_history_adam.npy", np.array(loss_history))



# print("\n--- Evaluation ---")
# check_point = restore("./model/best-model-adam.pth")
# pinn.net.load_state_dict(check_point['state_dict'])
# pinn.net.eval()

# pinn.x_f.requires_grad_(True)
# pinn.t_f.requires_grad_(True)

# # --- Compute individual loss terms ---
# with torch.no_grad():
#     loss_boundary = boundary_loss(pinn).item()

# # For residual loss, we CANNOT use torch.no_grad() because we need autograd
# loss_residual = residual_loss(pinn).item()

# print(f"Physics Loss (Residual): {loss_residual:.6f}")
# print(f"Boundary Loss (Constraint): {loss_boundary:.6f}")

# # --- Full prediction on the test grid ---
# x_full = torch.from_numpy(X_star[:, 0:1]).float().to(device)
# t_full = torch.from_numpy(X_star[:, 1:2]).float().to(device)

# with torch.no_grad():
#     u_pred = pinn.net_u(x_full, t_full).cpu().numpy()

# # --- Compute errors ---
# l2_mse = np.mean((u_truth - u_pred) ** 2)
# print(f"Final MSE: {l2_mse:.6f}")

# relative_l2_error = np.linalg.norm(u_truth - u_pred, 2) / np.linalg.norm(u_truth, 2)
# print(f"Final Relative L2 Error: {relative_l2_error:.6f}")

# # --- Plotting ---
# print("\nPlotting the solution ...")
# os.makedirs("./figures/solution", exist_ok=True)

# x_plot = torch.from_numpy(x_exact).float().to(device)
# t_plot = torch.from_numpy(t_exact).float().to(device)

# X_grid_plot, T_grid_plot = torch.meshgrid(x_plot.squeeze(), t_plot.squeeze(), indexing='ij')
# xcol_pred = X_grid_plot.reshape(-1, 1)
# tcol_pred = T_grid_plot.reshape(-1, 1)

# with torch.no_grad():
#     usol_pred = pinn.net_u(xcol_pred, tcol_pred)
# Unp_pred = usol_pred.reshape(x_plot.numel(), t_plot.numel()).cpu().numpy()

# xnp_plot = x_plot.cpu().numpy().squeeze()
# tnp_plot = t_plot.cpu().numpy().squeeze()

# # --- Plot predicted solution ---
# print("Generating solution plot...")
# plt.rcParams['font.size'] = '15'
# fig = plt.figure(figsize=(5,6))
# ax = fig.add_subplot(111)
# plt.xlabel(r"$t$")
# plt.ylabel(r"$x$")
# plt.title(r"ADAM-PINN Approximation $u(x,t)$")

# img_handle = ax.imshow(Unp_pred, interpolation='nearest', cmap='rainbow',
#                        extent=[tnp_plot.min(), tnp_plot.max(), xnp_plot.min(), xnp_plot.max()],
#                        origin='lower', aspect='auto', vmin=-1.0, vmax=1.0)
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.10)
# cbar = fig.colorbar(img_handle, cax=cax)
# cbar.ax.tick_params(labelsize=10)

# output_filename = "./figures/solution/ADAM_PINN_solution.png"
# plt.savefig(output_filename, dpi=300, bbox_inches='tight', pad_inches=0.1)
# print(f"Solution plot saved: {output_filename}")
# plt.show()

# # --- Plot error heatmap ---
# print("Generating error plot...")
# error_field = (abs(u_truth.reshape(Unp_pred.shape) - Unp_pred))

# fig = plt.figure(figsize=(5,6))
# ax = fig.add_subplot(111)
# plt.xlabel(r"$t$")
# plt.ylabel(r"$x$")
# plt.title(r"Error")

# img_handle = ax.imshow(error_field, interpolation='nearest', cmap='bwr',
#                        extent=[tnp_plot.min(), tnp_plot.max(), xnp_plot.min(), xnp_plot.max()],
#                        origin='lower', aspect='auto')
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.10)
# cbar = fig.colorbar(img_handle, cax=cax)
# cbar.ax.tick_params(labelsize=10)

# output_filename = "./figures/solution/ADAM_PINN_error.png"
# plt.savefig(output_filename, dpi=300, bbox_inches='tight', pad_inches=0.1)
# print(f"Error plot saved: {output_filename}")
# plt.show()



import torch
import torch.nn as nn
import numpy as np
from tqdm import trange
import random as rm
import scipy.io
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os

def save_model(model, path):
    """Saves the model's state dictionary."""
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")
    
def load_model(model, path):
    """Loads the model's state dictionary."""
    model.load_state_dict(torch.load(path))
    model.eval()
    print(f"Model loaded from {path}")

# Setup device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"We are using device: {device}")

# Set random seed for reproducibility
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
rm.seed(seed)



# --- NEW: Enforce deterministic behavior on GPU ---
# This is crucial for reproducibility
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False





class PhysicsInformedNN():
    """
    A class for Physics-Informed Neural Networks.
    """
    def __init__(self, X_u, u, X_f):
        # Boundary data
        self.x_u = torch.tensor(X_u[:, 0:1], dtype=torch.float32, requires_grad=True).to(device)
        self.t_u = torch.tensor(X_u[:, 1:2], dtype=torch.float32, requires_grad=True).to(device)
        self.u = torch.tensor(u, dtype=torch.float32).to(device)

        # Initial physics data (will be updated during causal training)
        self.set_data_for_stage(X_f)

        # Create the neural network
        self.create_net()
        
        # Mean Squared Error Loss
        self.loss = nn.MSELoss()

    # --- NEW METHOD to update data for causal stages ---
    def set_data_for_stage(self, X_f_stage):
        """Updates the collocation points for the current training stage."""
        self.x_f = torch.tensor(X_f_stage[:, 0:1], dtype=torch.float32, requires_grad=True).to(device)
        self.t_f = torch.tensor(X_f_stage[:, 1:2], dtype=torch.float32, requires_grad=True).to(device)
        self.null = torch.zeros((self.x_f.shape[0], 1), device=device)

    def create_net(self):
        """Creates the neural network architecture and moves it to the device."""
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
        """Predicts u(x, t)"""
        u = self.net(torch.hstack((x, t)))
        return u

    def net_f(self, x, t):
        """Computes the residual of the PDE."""
        x.requires_grad_(True); t.requires_grad_(True)
        u = self.net_u(x, t)
        
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True)[0]

        # Burger's equation residual
        v = 0.01 / np.pi
        f = u_t + (u * u_x) - (v * u_xx)
        return f

# --- Problem setup ---
N_u = 100
N_f = 10000

# Boundary conditions
x_upper = np.ones((N_u // 4, 1), dtype=float)
x_lower = -np.ones((N_u // 4, 1), dtype=float)
t_zero = np.zeros((N_u // 2, 1), dtype=float)
t_upper = np.random.rand(N_u // 4, 1)
t_lower = np.random.rand(N_u // 4, 1)
x_zero = -1 + 2 * np.random.rand(N_u // 2, 1)

X_upper = np.hstack((x_upper, t_upper))
X_lower = np.hstack((x_lower, t_lower))
X_zero = np.hstack((x_zero, t_zero))
X_u_train = np.vstack((X_upper, X_lower, X_zero))

# Initial conditions
u_upper = np.zeros((N_u // 4, 1), dtype=float)
u_lower = np.zeros((N_u // 4, 1), dtype=float)
u_zero = -np.sin(np.pi * x_zero)
u_train = np.vstack((u_upper, u_lower, u_zero))

# Shuffle training data
index = np.arange(N_u)
np.random.shuffle(index)
X_u_train = X_u_train[index, :]
u_train = u_train[index, :]

# Collocation points (master set for the entire domain)
X_f_master = np.random.uniform([-1, 0], [1, 1], size=(N_f, 2))
X_f_master = np.vstack((X_f_master, X_u_train))



# Initialize PINN with the full dataset (it will be immediately updated by the first stage)
pinn = PhysicsInformedNN(X_u_train, u_train, X_f_master)

# --- Loss functions ---
def residual_loss(model):
    f_pred = model.net_f(model.x_f, model.t_f)
    return model.loss(f_pred, model.null)

def boundary_loss(model):
    u_pred = model.net_u(model.x_u, model.t_u)
    return model.loss(u_pred, model.u)

# --- Standard torch.optim.Adam optimizer ---
optimizer = torch.optim.Adam(pinn.net.parameters(), lr=1e-3)

# --- Causal Training Loop ---
# Define stages: (t_max, iterations_for_this_stage)
stages = [(0.33, 10000), (0.66, 10000), (1.0, 10000)]
best_loss = np.inf
full_loss_history = []
global_step = 0

for i, (t_max, iters) in enumerate(stages):
    print(f"\n--- Causal Training: Stage {i+1}/{len(stages)} (t <= {t_max}) ---")

    # Filter master collocation points for the current stage
    X_f_stage = X_f_master[X_f_master[:, 1] <= t_max]
    pinn.set_data_for_stage(X_f_stage)
    print(f"Using {len(X_f_stage)} collocation points for this stage.")

    pbar = trange(iters, desc=f"Training Stage {i+1}")
    pinn.net.train()

    for epoch in pbar:
        optimizer.zero_grad()

        loss_res = residual_loss(pinn)
        loss_bcs = boundary_loss(pinn)
        
        loss = loss_res + loss_bcs

        loss.backward()
        optimizer.step()

        current_loss = loss.item()
        global_step += 1

        if global_step % 100 == 0:
            full_loss_history.append((global_step, current_loss))

        if current_loss < best_loss:
            best_loss = current_loss
           

        pbar.set_postfix({'Loss': f'{current_loss:.4e}', 'Best-Loss': f'{best_loss:.4e}'})

print("training completed.")
save_model(pinn.net, "./model/best-model-adam-causal.pth")
# Save the complete loss history
os.makedirs("./training_loss_history", exist_ok=True)
np.save("./training_loss_history/loss_history_adam_causal.npy", np.array(full_loss_history))
# write code to show training loss is saved in the directed folder
print("training loss history saved.")




# --- Evaluation ---
print("\n--- Evaluation ---")
load_model(pinn.net, "./model/best-model-adam-causal.pth")
pinn.net.eval()


# --- Load exact solution ---
data = scipy.io.loadmat('burgers_shock.mat')
t_exact = data['t'].flatten()[:, None]
x_exact = data['x'].flatten()[:, None]
Exact = np.real(data['usol'])
T, X = np.meshgrid(t_exact, x_exact)
X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
u_truth = Exact.flatten()[:, None]

# x, t testing data
x_test = torch.from_numpy(X_star[:, 0:1]).float().to(device)
t_test= torch.from_numpy(X_star[:, 1:2]).float().to(device)

# calculate testing residual loss
f_pred_test = pinn.net_f(x_test, t_test) # calculate f_pred on the test data
null_f_test = torch.zeros_like(f_pred_test)
test_physics_loss = pinn.loss(f_pred_test, null_f_test).item()
print(f"Testing Physics Loss: {test_physics_loss:.6f}")



with torch.no_grad():
    u_pred = pinn.net_u(x_test, t_test).cpu().numpy()

# --- Compute errors ---
mse = np.mean((u_truth - u_pred) ** 2)
print(f"Final MSE: {mse:.6f}")

relative_l2_error = np.linalg.norm(u_truth - u_pred, 2) / np.linalg.norm(u_truth, 2)
print(f"Final Relative L2 Error: {relative_l2_error:.6f}")





# --- Plotting ---
print("\nPlotting the solution ...")
os.makedirs("./figures/solution", exist_ok=True)

x_plot = torch.from_numpy(x_exact).float().to(device)
t_plot = torch.from_numpy(t_exact).float().to(device)

X_grid_plot, T_grid_plot = torch.meshgrid(x_plot.squeeze(), t_plot.squeeze(), indexing='ij')
xcol_pred = X_grid_plot.reshape(-1, 1)
tcol_pred = T_grid_plot.reshape(-1, 1)

with torch.no_grad():
    usol_pred = pinn.net_u(xcol_pred, tcol_pred)
Unp_pred = usol_pred.reshape(x_plot.numel(), t_plot.numel()).cpu().numpy()

xnp_plot = x_plot.cpu().numpy().squeeze()
tnp_plot = t_plot.cpu().numpy().squeeze()

# --- Plot predicted solution ---
print("Generating solution plot...")
plt.rcParams['font.size'] = '15'
fig = plt.figure(figsize=(5,6))
ax = fig.add_subplot(111)
plt.xlabel(r"$t$")
plt.ylabel(r"$x$")
plt.title(r"Causal ADAM-PINN $u(x,t)$")

img_handle = ax.imshow(Unp_pred, interpolation='nearest', cmap='rainbow',
                       extent=[tnp_plot.min(), tnp_plot.max(), xnp_plot.min(), xnp_plot.max()],
                       origin='lower', aspect='auto', vmin=-1.0, vmax=1.0)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.10)
cbar = fig.colorbar(img_handle, cax=cax)
cbar.ax.tick_params(labelsize=10)

output_filename = "./figures/solution/Causal_ADAM_PINN_solution.png"
plt.savefig(output_filename, dpi=300, bbox_inches='tight', pad_inches=0.1)
print(f"Solution plot saved: {output_filename}")
plt.show()

# --- Plot error heatmap ---
print("Generating error plot...")
error_field = (abs(u_truth.reshape(Unp_pred.shape) - Unp_pred))

fig = plt.figure(figsize=(5,6))
ax = fig.add_subplot(111)
plt.xlabel(r"$t$")
plt.ylabel(r"$x$")
plt.title(r"Error")

img_handle = ax.imshow(error_field, interpolation='nearest', cmap='bwr',
                       extent=[tnp_plot.min(), tnp_plot.max(), xnp_plot.min(), xnp_plot.max()],
                       origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.10)
cbar = fig.colorbar(img_handle, cax=cax)
cbar.ax.tick_params(labelsize=10)

output_filename = "./figures/solution/Causal_ADAM_PINN_error.png"
plt.savefig(output_filename, dpi=300, bbox_inches='tight', pad_inches=0.1)
print(f"Error plot saved: {output_filename}")
plt.show()