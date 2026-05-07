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
import scipy.io

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
h = np.pi / M
tau = 1 / N

t = torch.linspace(0, 1, N, device=device).requires_grad_(True)
x = torch.linspace(0, np.pi, M, device=device).requires_grad_(True)
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

num_pde = 5000
x_pde = (torch.rand(num_pde, 1, device=device) * torch.pi)
t_pde = torch.rand(num_pde, 1, device=device)

# --- 4. True function and source term ---
def u_true(x, t):
    t = t.reshape(-1, 1)
    x = x.reshape(1, -1)
    return torch.sin(x) * t ** 3

u_truth = u_true(x, t).view(-1, 1).to(device)
obs_noise = 0.01 * torch.std(u_truth)
u_truth_noisy = u_truth + torch.randn_like(u_truth) * obs_noise

def f(x, t, alpha):
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
        self.alpha = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
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
    A = torch.tril(i_minus_j)**alpha_1-2*torch.tril(i_minus_j-1)**alpha_1+torch.tril(i_minus_j-2).fill_diagonal_(0)**alpha_1
    A=A.fill_diagonal_(1)
    B=torch.matmul(A,u_hat[1:,:])
    i_minus_j_1=torch.arange(1, N, device=device).reshape(-1,1)
    c=(i_minus_j_1)**alpha_1-(i_minus_j_1 - 1)**alpha_1
    B = B - torch.matmul(c, u_hat[0, :].view(1, -1))
    a = tau**(-alpha) / torch.lgamma(2 - alpha).exp()
    d=torch.mul(a,B)
    return d[:,1:-1]

# --- 7. Model and optimizer ---
layers = [2, 50, 100, 1]
model = MLP(layers).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# --- 8. Loss functions ---
def residual_loss(model):
    u_hat = model(X_input) 
    u_hat = u_hat.reshape(N, M)
    u_nn_xx = h ** (-2) * torch.matmul(u_hat, S(M))[1:, 1:-1]
    loss = criterion(f(x, t, model.alpha), FPDE(model.alpha, u_hat) - u_nn_xx)
    return loss

def initial_loss(model):
    ic_input = torch.cat((x_ic, t_ic), dim=-1)
    u_pred_ic = model(ic_input)
    return criterion(u_pred_ic, u_ic)

def boundary_loss(model):
    bc_x0_input = torch.cat((x_bc_0, t_bc), dim=-1)
    bc_xpi_input = torch.cat((x_bc_pi, t_bc), dim=-1)
    u_pred_bc_0 = model(bc_x0_input)
    u_pred_bc_pi = model(bc_xpi_input)
    return criterion(u_pred_bc_0, u_bc) + criterion(u_pred_bc_pi, u_bc)

def obs_loss(model):
    predicted = model(X_input)
    loss = criterion(predicted, u_truth_noisy)
    return loss


import os
import time
import numpy as np

log_dir = "./training_logs"
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, "ADAM-PINN_training_loss.txt")

num_epochs = 10000
loss_history = []

os.makedirs("./training_data", exist_ok=True)
loss_epochs = []
loss_physics_history = []
loss_boundary_history = []
loss_initial_history = []
loss_obs_history = []

start_time = time.time()

with open(log_path, "w") as log_file:   # <-- renamed from f → log_file
    log_file.write("epoch\tloss_res\tloss_boundary\tloss_initial\tloss_obs\ttotal_loss\talpha\n")
    for epoch in range(num_epochs):
        loss_res = residual_loss(model)
        loss_boundary = boundary_loss(model)
        loss_initial = initial_loss(model)
        loss_obs = obs_loss(model)
        total_loss = loss_res + loss_boundary + loss_initial +  loss_obs

        optimizer.zero_grad()
        total_loss.backward(retain_graph=True)
        optimizer.step()

        loss_history.append(total_loss.item())
        # Write to file every 50 epochs or on the last epoch
        # if (epoch % 50 == 0) or (epoch + 1 == num_epochs):
        #     log_file.write(f"{epoch+1}\t{loss_res.item():.8f}\t{loss_boundary.item():.8f}\t{loss_initial.item():.8f}\t{loss_obs.item():.8f}\t{total_loss.item():.8f}\t{model.alpha.item():.8f}\n")
        #     log_file.flush()

        if epoch % 50 == 0 or epoch + 1 == num_epochs:
            print(f"Epoch {epoch+1:04d} | Res: {loss_res.item():.6e} | Boundary: {loss_boundary.item():.6e} | Initial: {loss_initial.item():.6e} | Obs: {loss_obs.item():.6e} | Total: {total_loss.item():.6e} | Alpha: {model.alpha.item():.6f}")

#         

end_time = time.time()
elapsed_time = end_time - start_time
print(f"\nTraining completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
print(f"Final alpha: {model.alpha.item()}")

# Save solution and alpha
x_values = np.linspace(0, np.pi, M)
t_values = np.linspace(0, 1, N)
u_pred = model(X_input).detach().cpu().numpy().reshape(N, M)
u_true_np = u_truth.detach().cpu().numpy().reshape(N, M)
np.savez("./data/Adam_PINN_solution.npz", u_pred=u_pred, u_true=u_true_np, x=x_values, t=t_values, alpha=model.alpha.item())

# Move model output to CPU before converting to numpy
u_pred_plot = u_pred

# Also move these if they were created on GPU
X_plot = X.detach().cpu().numpy()
T_plot = T.detach().cpu().numpy()

u_truth_plot = u_truth.detach().cpu().numpy().reshape(N, M)

fig = plt.figure(figsize=(12, 6))

ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X_plot, T_plot, u_truth_plot, cmap='viridis', alpha=0.8)
ax1.set_xlabel('$x$')
ax1.set_ylabel('$t$')
ax1.set_zlabel('$u_{\\mathrm{true}}(x, t)$')
ax1.set_title('$u_{\\mathrm{true}}(x, t) = \\sin(x) \\cdot t^3$')

ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(X_plot, T_plot, u_pred_plot, cmap='viridis', alpha=0.8)
ax2.set_xlabel('$x$')
ax2.set_ylabel('$t$')
ax2.set_zlabel('$u_{\\mathrm{approx}}(x, t)$')
ax2.set_title('$u_{\\mathrm{approx}}(x, t)$')

plt.tight_layout()
os.makedirs('./figures', exist_ok=True)
fig.savefig('./figures/Adam-PINN-solution.png', dpi=200, bbox_inches='tight')
print("Saved figure to ./figures/Adam-PINN-solution.png")
plt.show()


# Print final losses
print("Final Losses:")
print(f"Physics Loss: {residual_loss(model).item():.6e}")
print(f"Initial Loss: {initial_loss(model).item():.6e}")
print(f"Boundary Loss: {boundary_loss(model).item():.6e}")
print(f"Data Loss: {obs_loss(model).item():.6e}")

u_truth_tensor = u_truth.view(-1,1)
u_pred = model(X_input)

# Calculate MSE loss
print('MSE:', criterion(u_pred, u_truth_tensor).item())

# Calculate L2 relative error
l2_error = torch.linalg.norm(u_pred - u_truth_tensor) / torch.linalg.norm(u_truth_tensor)
print('L2 relative error:', l2_error.item())

# Save to .mat file
os.makedirs('./data', exist_ok=True)
scipy.io.savemat('./data/Adam-PINN-solution.mat', {
    'u_pred': u_pred.detach().cpu().numpy().reshape(N, M),
    'u_truth_plot': u_truth_plot,
    'x_values': x_values,
    't_values': t_values,
    'alpha': model.alpha.item()
})
print("Saved solution to ./data/Adam-PINN-solution.mat")