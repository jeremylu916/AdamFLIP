import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import ipywidgets as widgets
from ipywidgets import interact

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Parameters
nu = 0.01  # viscosity
N_coll = 10000  # Collocation points (physics)
N_ic = 1000  # Initial condition points
N_bc = 1000  # Periodic boundary points per side

# Analytical solution (benchmark)
def taylor_green_u(x, y, t):
    return -np.cos(x) * np.sin(y) * np.exp(-2 * nu * t)

def taylor_green_v(x, y, t):
    return np.sin(x) * np.cos(y) * np.exp(-2 * nu * t)

def taylor_green_p(x, y, t):
    return -0.25 * (np.cos(2 * x) + np.cos(2 * y)) * np.exp(-4 * nu * t)


# Example values
N = 50
t = 0.5
x = np.linspace(0, 2*np.pi, N)
y = np.linspace(0, 2*np.pi, N)
X, Y = np.meshgrid(x, y)

u = taylor_green_u(X, Y, t)
v = taylor_green_v(X, Y, t)
p = taylor_green_p(X, Y, t)

# Plot Velocity field
plt.figure(figsize=(6, 6))
plt.quiver(X, Y, u, v, scale=20)
plt.title('Taylor-Green Vortex (Velocity field)')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
plt.grid()
plt.show()

# Plot Pressure
plt.figure(figsize=(6, 5))
plt.imshow(p, extent=[0, 2*np.pi, 0, 2*np.pi], origin='lower', cmap='coolwarm')
plt.colorbar(label=r'Pressure ($\frac{\mathrm{N}}{\mathrm{m}^2}$)')
plt.title('Taylor-Green Vortex (Pressure)')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
plt.grid()
plt.show()


class PINN(nn.Module):
    def __init__(self, hidden_dim=64):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim), # Input: x, y, t
            nn.Tanh(), # Smooth activation function & good for regularization
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 3)  # Output: u, v, p
        )

    def forward(self, xyt):
        return self.net(xyt) # (N, 3) -> (N, 3)
    

def get_training_data(t_max=5):
    # Collocation points: x, y in [0, 2pi], t in [0, t_max]
    x = np.random.uniform(0, 2*np.pi, (N_coll, 1))
    y = np.random.uniform(0, 2*np.pi, (N_coll, 1))
    t = np.random.uniform(0, t_max, (N_coll, 1))
    xyt_coll = torch.tensor(np.hstack([x, y, t]), dtype=torch.float32, requires_grad=True).to(device)

    # Initial condition points: t = 0
    x = np.random.uniform(0, 2*np.pi, (N_ic, 1))
    y = np.random.uniform(0, 2*np.pi, (N_ic, 1))
    t = np.zeros((N_ic, 1))
    u = taylor_green_u(x, y, t)
    v = taylor_green_v(x, y, t)
    xyt_ic = torch.tensor(np.hstack([x, y, t]), dtype=torch.float32).to(device)
    uv_ic = torch.tensor(np.hstack([u, v]), dtype=torch.float32).to(device)

    # Periodic boundary points on x-boundaries: x=0 and x=2pi
    y_bx = np.random.uniform(0, 2*np.pi, (N_bc, 1))
    t_bx = np.random.uniform(0, t_max, (N_bc, 1))
    xyt_bx_l = torch.tensor(np.hstack([np.zeros((N_bc, 1)), y_bx, t_bx]), dtype=torch.float32).to(device)
    xyt_bx_r = torch.tensor(np.hstack([2*np.pi * np.ones((N_bc, 1)), y_bx, t_bx]), dtype=torch.float32).to(device)

    # Periodic boundary points on y-boundaries: y=0 and y=2pi
    x_by = np.random.uniform(0, 2*np.pi, (N_bc, 1))
    t_by = np.random.uniform(0, t_max, (N_bc, 1))
    xyt_by_b = torch.tensor(np.hstack([x_by, np.zeros((N_bc, 1)), t_by]), dtype=torch.float32).to(device)
    xyt_by_t = torch.tensor(np.hstack([x_by, 2*np.pi * np.ones((N_bc, 1)), t_by]), dtype=torch.float32).to(device)

    return xyt_coll, xyt_ic, uv_ic, xyt_bx_l, xyt_bx_r, xyt_by_b, xyt_by_t


# Physics loss
def compute_physics_loss(model, xyt_coll):
    xyt_coll = xyt_coll.clone().detach().requires_grad_(True)

    uvp = model(xyt_coll)
    u = uvp[:, 0:1]
    v = uvp[:, 1:2]
    p = uvp[:, 2:3]

    grads = torch.autograd.grad(u, xyt_coll, torch.ones_like(u), create_graph=True)[0]
    u_x = grads[:, 0:1]
    u_y = grads[:, 1:2]
    u_t = grads[:, 2:3]

    grads = torch.autograd.grad(v, xyt_coll, torch.ones_like(v), create_graph=True)[0]
    v_x = grads[:, 0:1]
    v_y = grads[:, 1:2]
    v_t = grads[:, 2:3]

    grads = torch.autograd.grad(p, xyt_coll, torch.ones_like(p), create_graph=True)[0]
    p_x = grads[:, 0:1]
    p_y = grads[:, 1:2]

    u_xx = torch.autograd.grad(u_x, xyt_coll, torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
    u_yy = torch.autograd.grad(u_y, xyt_coll, torch.ones_like(u_y), create_graph=True)[0][:, 1:2]

    v_xx = torch.autograd.grad(v_x, xyt_coll, torch.ones_like(v_x), create_graph=True)[0][:, 0:1]
    v_yy = torch.autograd.grad(v_y, xyt_coll, torch.ones_like(v_y), create_graph=True)[0][:, 1:2]

    continuity = u_x + v_y

    mom_x = u_t + u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
    mom_y = v_t + u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)

    loss_pde = (mom_x**2).mean() + (mom_y**2).mean() + (continuity**2).mean()
    return loss_pde

# Initial condition loss
def compute_ic_loss(model, xyt_ic, uv_ic):
    pred_uv = model(xyt_ic)[:, :2]
    return ((pred_uv - uv_ic) ** 2).mean()

# Periodic boundary loss
def compute_bc_loss(model, xyt_bx_l, xyt_bx_r, xyt_by_b, xyt_by_t):
    pred_l = model(xyt_bx_l)
    pred_r = model(xyt_bx_r)
    pred_b = model(xyt_by_b)
    pred_t = model(xyt_by_t)

    loss_x = ((pred_l - pred_r) ** 2).mean()
    loss_y = ((pred_b - pred_t) ** 2).mean()
    return loss_x + loss_y


xyt_coll, xyt_ic, uv_ic, xyt_bx_l, xyt_bx_r, xyt_by_b, xyt_by_t = get_training_data()

model = PINN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(15000):
    optimizer.zero_grad()
    loss_pde = compute_physics_loss(model, xyt_coll)
    loss_ic = compute_ic_loss(model, xyt_ic, uv_ic)
    loss_bc = compute_bc_loss(model, xyt_bx_l, xyt_bx_r, xyt_by_b, xyt_by_t)
    loss = loss_pde + 5 * loss_ic + loss_bc
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(
            f"Epoch {epoch} | Loss: {loss.item():.5e} | PDE: {loss_pde.item():.5e} | IC: {loss_ic.item():.5e} | BC: {loss_bc.item():.5e}"
        )


os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/model_weights.pth")


model = PINN().to(device)
model.load_state_dict(torch.load("models/model_weights.pth"))
model.eval()


def plot_comparison(model, t_val=0.5):
    N = 50
    x = np.linspace(0, 2*np.pi, N)
    y = np.linspace(0, 2*np.pi, N)
    X, Y = np.meshgrid(x, y)
    T = np.full_like(X, t_val)

    xyt = torch.tensor(np.stack([X.ravel(), Y.ravel(), T.ravel()], axis=1), dtype=torch.float32).to(device)
    with torch.no_grad():
        pred = model(xyt).cpu().numpy()
    U_pred = pred[:, 0].reshape(N, N)
    V_pred = pred[:, 1].reshape(N, N)

    U_true = taylor_green_u(X, Y, T)
    V_true = taylor_green_v(X, Y, T)

    diff_u = np.abs(U_pred - U_true)
    diff_v = np.abs(V_pred - V_true)

    os.makedirs("./data", exist_ok=True)
    scipy.io.savemat(
        f"./data/Adam_PINN_plot_comparison_t{t_val:.2f}.mat",
        {
            "x": x,
            "y": y,
            "t": t_val,
            "U_pred": U_pred,
            "V_pred": V_pred,
            "U_true": U_true,
            "V_true": V_true,
            "diff_u": diff_u,
            "diff_v": diff_v,
        },
    )

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle(f"Taylor-Green Vortex at t = {t_val}", fontsize=16)
    axs[0, 0].quiver(X, Y, U_true, V_true, scale=20)
    axs[0, 0].set_title("Analytical Velocity Field")

    axs[0, 1].quiver(X, Y, U_pred, V_pred, scale=20)
    axs[0, 1].set_title("PINN Velocity Field")

    im1 = axs[1, 0].imshow(diff_u, extent=[0, 2*np.pi, 0, 2*np.pi])
    axs[1, 0].set_title("Error in u")
    fig.colorbar(im1, ax=axs[1, 0])

    im2 = axs[1, 1].imshow(diff_v, extent=[0, 2*np.pi, 0, 2*np.pi])
    axs[1, 1].set_title("Error in v")
    fig.colorbar(im2, ax=axs[1, 1])

    plt.tight_layout()
    plt.show()

# Some examples...
plot_comparison(model, t_val=0)
plot_comparison(model, t_val=0.5)
plot_comparison(model, t_val=1)


