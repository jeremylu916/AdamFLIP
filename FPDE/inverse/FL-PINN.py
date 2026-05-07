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
import scipy.io
from tqdm import trange

# --- Reproducibility ---
seed = 41
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

def lgamma_lanczos(z):
    """Differentiable log-gamma approximation using regular tensor ops."""
    z = torch.clamp(z, min=torch.finfo(z.dtype).eps)
    shifted_z = z + 2.0
    coeffs = [
        0.9999999999998099,
        676.5203681218851,
        -1259.1392167224028,
        771.3234287776531,
        -176.6150291621406,
        12.507343278686905,
        -0.13857109526572012,
        9.984369578019572e-6,
        1.5056327351493116e-7,
    ]
    x = torch.as_tensor(coeffs[0], dtype=z.dtype, device=z.device)
    y = shifted_z - 1.0
    for i, coeff in enumerate(coeffs[1:], start=1):
        x = x + torch.as_tensor(coeff, dtype=z.dtype, device=z.device) / (y + i)
    t_lanczos = y + 7.5
    log_sqrt_2pi = 0.5 * math.log(2.0 * math.pi)
    lgamma_shifted = log_sqrt_2pi + (y + 0.5) * torch.log(t_lanczos) - t_lanczos + torch.log(x)
    return lgamma_shifted - torch.log(z) - torch.log(z + 1.0)

def positive_power(base, exponent):
    positive_base = base > 0
    return torch.where(
        positive_base,
        torch.clamp(base, min=1).pow(exponent),
        torch.zeros_like(base),
    )

def f(x, t, alpha):
    t = t.reshape(-1, 1)
    x = x.reshape(1, -1)
    gamma_4 = 6.0
    term = (gamma_4 / lgamma_lanczos(4 - alpha).exp()) * t ** (3 - alpha) * torch.sin(x) + t ** 3 * torch.sin(x)
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
        # Clip alpha to reasonable range
        self.alpha.data = torch.clamp(self.alpha.data, 0.0, 2.0)
        return a

criterion = nn.MSELoss()

# --- 6. FPDE operator ---
def FPDE(alpha, u_hat):
    alpha_1 = 1 - alpha
    i_minus_j = torch.arange(1, N, device=device).view(-1, 1) - torch.arange(0, N - 1, device=device).view(1, -1)
    A = (
        positive_power(torch.tril(i_minus_j), alpha_1)
        - 2 * positive_power(torch.tril(i_minus_j - 1), alpha_1)
        + positive_power(torch.tril(i_minus_j - 2).fill_diagonal_(0), alpha_1)
    )
    A=A.fill_diagonal_(1)
    B=torch.matmul(A,u_hat[1:,:])
    i_minus_j_1=torch.arange(1, N, device=device).reshape(-1,1)
    c = positive_power(i_minus_j_1, alpha_1) - positive_power(i_minus_j_1 - 1, alpha_1)
    B = B - torch.matmul(c, u_hat[0, :].view(1, -1))
    a = tau**(-alpha) / lgamma_lanczos(2 - alpha).exp()
    d=torch.mul(a,B)
    return d[:,1:-1]

# --- 7. Model ---
layers = [2, 50, 100, 1]
model = MLP(layers).to(device)

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

def ibc_loss(model):
    return initial_loss(model) + boundary_loss(model)

def obs_loss(model):
    predicted = model(X_input)
    loss = criterion(predicted, u_truth_noisy)
    return loss

# --- 9. SOFL Helper Functions ---
def get_flat_params(model):
    """Gets all model parameters as a flat NumPy vector."""
    return np.concatenate([p.detach().cpu().numpy().ravel() for p in model.parameters()])

def get_flat_grads(model):
    """Gets all model gradients as a flat NumPy vector."""
    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.cpu().numpy().ravel())
        else:
            grads.append(np.zeros(p.numel(), dtype=np.float32))
    return np.concatenate(grads)

def set_flat_params(model, flat_params):
    """Sets all model parameters from a flat NumPy vector."""
    flat_params = flat_params.astype(np.float32)
    pointer = 0
    for p in model.parameters():
        num_params = p.numel()
        new_values = torch.from_numpy(flat_params[pointer:pointer + num_params]).view_as(p).to(device)
        p.data.copy_(new_values)
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
            flat_grads.append(g.detach().cpu().numpy().ravel())
        else:
            flat_grads.append(np.zeros(p.numel(), dtype=np.float32))
    return np.concatenate(flat_grads)

# --- 10. SOFL Wrapper Functions ---
def f_sofl(x):
    """Objective uses observation loss."""
    set_flat_params(model, x)
    return obs_loss(model).item()

def df_sofl(x):
    """Gradient of the observation loss."""
    set_flat_params(model, x)
    loss_obs = obs_loss(model)
    model_params = list(model.parameters())
    return get_grads_from_loss(loss_obs, model_params, allow_unused=True)

def h_sofl(x):
    """Constraint function returns residual and IBC losses."""
    set_flat_params(model, x)
    loss_res_val = residual_loss(model).item()
    loss_ibc_val = ibc_loss(model).item()
    return np.array([loss_res_val, loss_ibc_val], dtype=np.float32)

def dh_sofl(x):
    """Constraint Jacobian gathers gradients for both losses."""
    set_flat_params(model, x)
    model_params = list(model.parameters())
    loss_res = residual_loss(model)
    grad_res = get_grads_from_loss(loss_res, model_params, allow_unused=True)
    loss_ibc = ibc_loss(model)
    grad_ibc = get_grads_from_loss(loss_ibc, model_params, allow_unused=True)
    return np.vstack([grad_res, grad_ibc])

# --- 11. SOFL Optimizer ---
def SOFL_Adam_PINN(f, h, df, dh, x_start, Kp, Ki=None, eta=0.01, max_iter=1000, tol=1e-6, weight_decay=0.0):
    log_dir = "./training_logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "AdaFL-PINN_training_loss.txt")
    with open(log_path, "w") as log_file:
        log_file.write("Iter\tloss_res\tloss_ibc\tloss_obs\n")

    os.makedirs("./training_data", exist_ok=True)
    loss_epochs = []
    loss_kkt_history = []
    loss_physics_history = []
    loss_boundary_history = []
    loss_data_history = []

    x = np.array(x_start, dtype=np.float32)
    if Ki is None:
        Ki = np.zeros_like(Kp, dtype=np.float32)
    history = []
    Integral = np.zeros(Kp.shape[0], dtype=np.float32)

    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    m, v, t_iter = np.zeros_like(x), np.zeros_like(x), 0

    pbar = trange(max_iter, desc="FL-PINN Training", unit="iter")
    for iteration in pbar:
        set_flat_params(model, x)

        true_grad_f = df(x)
        true_J_h = dh(x)
        true_h_x = h(x)

        JhJht = true_J_h @ true_J_h.T
        Pcontrol = -Kp @ true_h_x
        Integral = Integral + true_h_x
        Icontrol = Ki @ Integral
        rhs = Pcontrol + Icontrol + true_J_h @ true_grad_f
        I = np.eye(JhJht.shape[0], dtype=np.float32)
        lambda_ = -solve(JhJht + 1e-6 * I, rhs, assume_a='pos')

        KKT_grad = true_grad_f + true_J_h.T @ lambda_
        if weight_decay > 0:
            KKT_grad = KKT_grad + weight_decay * x

        clip_value = 10.0
        grad_norm = np.linalg.norm(KKT_grad)
        if grad_norm > clip_value:
            KKT_grad = KKT_grad * (clip_value / grad_norm)

        x_new = x - eta * KKT_grad

        if iteration % 50 == 0 or iteration + 1 == max_iter:
            obs_loss_val = f(x)
            KKT_gap = max(np.linalg.norm(KKT_grad), np.max(np.abs(true_h_x)))
            res_loss_val, ibc_loss_val = np.abs(true_h_x[0]), np.abs(true_h_x[1])
            pbar.set_postfix({
                'res_loss(c1)': f'{res_loss_val:.2e}',
                'ibc_loss(c2)': f'{ibc_loss_val:.2e}',
                'KKT_gap': f'{KKT_gap:.2e}',
            })
            history.append((obs_loss_val, KKT_gap, res_loss_val, ibc_loss_val))
            with open(log_path, "a") as log_file:
                log_file.write(f"{iteration}\t{res_loss_val:.6e}\t{ibc_loss_val:.6e}\t{obs_loss_val:.6e}\n")

        if iteration % 100 == 0:
            loss_epochs.append(iteration)
            loss_kkt_history.append(KKT_gap)
            loss_physics_history.append(res_loss_val)
            loss_boundary_history.append(boundary_loss(model).item())
            loss_data_history.append(obs_loss_val)

        if np.linalg.norm(x_new - x) < tol:
            break

        if np.isnan(x_new).any():
            print("\nError: NaN values detected. Stopping.")
            break

        x = x_new

   

    return x, history

# --- 12. Training with SOFL Optimizer ---
initial_params = get_flat_params(model)
Kp = np.array([[10.0, 0.0], [0.0, 10.0]])  # Further reduced for better balance
Ki = np.array([[0.0, 0.0], [0.0, 0.0]])  # Start with no integral gain
eta = 0.001  # Further reduced learning rate
max_iter = 10000  # Increased iterations
tol = 1e-6

start_time = time.time()
trained_params, sofl_history = SOFL_Adam_PINN(
    f=f_sofl,
    h=h_sofl,
    df=df_sofl,
    dh=dh_sofl,
    x_start=initial_params,
    Kp=Kp,
    Ki=Ki,
    eta=eta,
    max_iter=max_iter,
    tol=tol,
    weight_decay=0.0
)
set_flat_params(model, trained_params)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"\nTraining completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
print(f"Final alpha: {model.alpha.item()}")

# --- Print Final Losses ---


# Save solution and alpha
x_values = np.linspace(0, np.pi, M)
t_values = np.linspace(0, 1, N)
u_pred = model(X_input).detach().cpu().numpy().reshape(N, M)
u_true_np = u_truth.detach().cpu().numpy().reshape(N, M)
np.savez("./data/FL_PINN_solution.npz", u_pred=u_pred, u_true=u_true_np, x=x_values, t=t_values, alpha=model.alpha.item())

# Move model output to CPU before converting to numpy
u_pred_plot = u_pred

# Also move these if they were created on GPU
X_plot = X.detach().cpu().numpy()
T_plot = T.detach().cpu().numpy()

u_truth_plot = u_truth.detach().cpu().numpy().reshape(N, M)
abs_error_plot = np.abs(u_pred_plot - u_truth_plot)

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
fig.savefig('./figures/FL-PINN-solution.png', dpi=200, bbox_inches='tight')
print("Saved figure to ./figures/FL-PINN-solution.png")
plt.show()

fig_error = plt.figure(figsize=(8, 6))
ax_error = fig_error.add_subplot(111, projection='3d')
error_surface = ax_error.plot_surface(X_plot, T_plot, abs_error_plot, cmap='magma', alpha=0.9)
ax_error.set_xlabel('$x$')
ax_error.set_ylabel('$t$')
ax_error.set_zlabel('$|u_{\\mathrm{approx}} - u_{\\mathrm{true}}|$')
ax_error.set_title('FL-PINN Absolute Error')
fig_error.colorbar(error_surface, ax=ax_error, shrink=0.6, aspect=12, pad=0.1)

plt.tight_layout()
fig_error.savefig('./figures/FL-PINN-absolute-error.png', dpi=200, bbox_inches='tight')
print("Saved absolute error figure to ./figures/FL-PINN-absolute-error.png")
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
    'abs_error': abs_error_plot,
    'x_values': x_values,
    't_values': t_values,
    'alpha': model.alpha.item()
})
print("Saved solution to ./data/Adam-PINN-solution.mat")
