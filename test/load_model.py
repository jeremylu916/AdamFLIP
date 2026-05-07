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
    model.net.load_state_dict(torch.load(path))  # Use model.net to load the state_dict
    model.net.eval()
    print(f"Model loaded from {path}")
    return model  # Return the model object



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
        self.X_u_stage = X_u_stage
        self.u_stage = u_stage
        self.X_f_stage = X_f_stage

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
    f_pred = model.net_f(model.X_f_stage[:, 0:1], model.X_f_stage[:, 1:2])
    null_f = torch.zeros((model.X_f_stage.shape[0], 1), device=device)
    return model.loss(f_pred, null_f)

def boundary_loss(model):
    u_pred = model.net_u(model.X_u_stage[:, 0:1], model.X_u_stage[:, 1:2])
    return model.loss(u_pred, model.u_stage)

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

N_u = 100
N_f = 10000
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

# load model from ./test_model/trained_Multi-Constraint_pinn_model.pth to pinn_mode; 
# Load the model from the specified path
pinn_model = load_model(pinn_model, './trained_pinn_model.pth')


print("\n--- Calculating Testing Physics Loss ---")
data = scipy.io.loadmat('burgers_shock.mat')
t_exact = data['t'].flatten()[:, None]
x_exact = data['x'].flatten()[:, None]
Exact = np.real(data['usol'])
T, X = np.meshgrid(t_exact, x_exact)
X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))

# 4. Convert test data to tensors, ENSURING requires_grad=True
x_test = torch.tensor(X_star[:, 0:1], dtype=torch.float32, requires_grad=True).to(device)
t_test = torch.tensor(X_star[:, 1:2], dtype=torch.float32, requires_grad=True).to(device)

# 5. Calculate the PDE residual on the test grid using the model
#    This internally uses automatic differentiation to find u_t, u_x, etc.
f_pred_test = pinn_model.net_f(x_test, t_test)

# 6. Calculate the Mean Squared Error of the residual against zero
null_f_test = torch.zeros_like(f_pred_test)
test_physics_loss = pinn_model.loss(f_pred_test, null_f_test).item()
mse_loss = pinn_model.loss(pinn_model.net_u(x_test, t_test), torch.tensor(Exact.flatten()[:, None], dtype=torch.float32).to(device)).item()
# --- Print Result ---
print("\n--- Final Test Metric ---")
print(f"Testing Physics Loss: {test_physics_loss:.6f}")
print(f"MSE Loss: {mse_loss:.6f}")


# print("\n--- Evaluation ---")
# data = scipy.io.loadmat('burgers_shock.mat')
# t_exact = data['t'].flatten()[:, None]
# x_exact = data['x'].flatten()[:, None]
# Exact = np.real(data['usol'])
# T, X = np.meshgrid(t_exact, x_exact)
# X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
# u_truth = Exact.flatten()[:, None]

# # plot the f_pred error over x and t

# def f_pred(model):
#     f_pred = model.net_f(model.X_f_full[:, 0:1], model.X_f_full[:, 1:2])
#     return f_pred

# # Evaluate and plot f_pred over x and t
# # Evaluate and plot f_pred over x and t
# # Evaluate and plot f_pred over x and t
# print("\n--- Plotting Physics Residual (f_pred) ---")

# # Create a grid of points for evaluation
# x_eval = np.linspace(-1, 1, 100)  # 100 points in the x-direction
# t_eval = np.linspace(0, 1, 100)   # 100 points in the t-direction
# X_eval, T_eval = np.meshgrid(x_eval, t_eval)
# X_f_eval = np.hstack((X_eval.flatten()[:, None], T_eval.flatten()[:, None]))

# # Convert to PyTorch tensors
# # Note: requires_grad=True is not strictly needed here since it's set inside net_f, 
# # but it's good practice for clarity.
# X_f_eval_tensor = torch.tensor(X_f_eval, dtype=torch.float32).to(device)
# x_eval_tensor = X_f_eval_tensor[:, 0:1]
# t_eval_tensor = X_f_eval_tensor[:, 1:2]

# # Evaluate f_pred on the grid.
# # Do NOT use `with torch.no_grad()` because net_f needs to compute gradients.
# f_pred_tensor = pinn_model.net_f(x_eval_tensor, t_eval_tensor)

# # Detach the tensor from the computation graph before converting to NumPy
# f_pred_values = f_pred_tensor.detach().cpu().numpy()

# # Reshape f_pred values to match the grid
# f_pred_values = f_pred_values.reshape(T_eval.shape)
# f_pred_values  =f_pred_values  / np.max(np.abs(f_pred_values ))

# # Plot the physics residual
# plt.figure(figsize=(10, 6))
# contour = plt.contourf(T_eval, X_eval, abs(f_pred_values), levels=100, cmap='jet')
# plt.colorbar(contour, label='Single Physics Residual (f_pred)')
# plt.xlabel('t')
# plt.ylabel('x')
# plt.title('Physics Residual (f_pred) Over x and t')
# plt.savefig('Single physics_residual_f_pred.png', dpi=300)
# plt.show()
