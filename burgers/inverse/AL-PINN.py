import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
from tqdm import trange
import random as rm
import scipy.io
import os
import time
from random import uniform
from tqdm import trange
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

parser = argparse.ArgumentParser()
parser.add_argument('--beta', default=1.0, type=float)
parser.add_argument('--lr', default=5e-3, type=float)
parser.add_argument('--lbd_lr', default=1e-3, type=float)
parser.add_argument('--EPOCH', default=20000, type=int)
parser.add_argument('--ordinal', default=0, type=int)
args = parser.parse_args()

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

device = torch.device(f"cuda:{args.ordinal}" if torch.cuda.is_available() else "cpu")
print("we are using device: {}".format(str(device)))

# initial PDE parameter guesses (will be learned inside the net)
lambda1_init = 2.0
lambda2_init = 0.0
nu = 0.01/np.pi


class PhysicsInformedNN():
    def __init__(self, X_u, u, X_f):
        # x & t from boundary conditions:
        self.x_u = torch.tensor(X_u[:, 0].reshape(-1, 1), dtype=torch.float32, requires_grad=True).to(device)
        self.t_u = torch.tensor(X_u[:, 1].reshape(-1, 1), dtype=torch.float32, requires_grad=True).to(device)

        # x & t from collocation points:
        self.x_f = torch.tensor(X_f[:, 0].reshape(-1, 1), dtype=torch.float32, requires_grad=True).to(device)
        self.t_f = torch.tensor(X_f[:, 1].reshape(-1, 1), dtype=torch.float32, requires_grad=True).to(device)

        # boundary solution:
        self.u = torch.tensor(u, dtype=torch.float32).to(device)

        # null vector to test against f:
        self.null = torch.zeros((self.x_f.shape[0], 1)).to(device)

        # initialize net:
        self.create_net()
        self.net.to(device)

        self.loss = nn.MSELoss()

        # initialize PDE parameters as trainable parameters inside the net (inverse problem)
        self.lambda1 = torch.tensor([lambda1_init], requires_grad=True).float().to(device)
        self.lambda2 = torch.tensor([lambda2_init], requires_grad=True).float().to(device)
        self.lambda1 = nn.Parameter(self.lambda1)
        self.lambda2 = nn.Parameter(self.lambda2)
        self.net.register_parameter('lambda1', self.lambda1)
        self.net.register_parameter('lambda2', self.lambda2)

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
        )

    def net_u(self, x, t):
        u = self.net(torch.hstack((x, t)))
        return u

    def net_f(self, x, t):
        lambda1 = self.lambda1
        lambda2 = self.lambda2
        u = self.net_u(x, t)
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True)[0]
        f = u_t + (lambda1 * u * u_x) - (lambda2 * u_xx)
        return f


def ibc_loss(model):
    u_prediction = model.net_u(model.x_u, model.t_u)
    return model.loss(u_prediction, model.u)


def residual_loss(model):
    f_prediction = model.net_f(model.x_f, model.t_f)
    return model.loss(f_prediction, model.null)


def observation_loss(model, u_observation, xcol, tcol):
    u_obs_torch = torch.from_numpy(u_observation).float().to(device)
    u_pred = model.net_u(xcol, tcol)
    return model.loss(u_pred, u_obs_torch)


def main():
    # data setup copied from inverse/Adam-PINN.py
    v = 0.01 / np.pi
    N_u = 100
    N_f = 10000

    x_upper = np.ones((N_u//4, 1), dtype=float)
    x_lower = np.ones((N_u//4, 1), dtype=float) * (-1)
    t_zero = np.zeros((N_u//2, 1), dtype=float)
    t_upper = np.random.rand(N_u//4, 1)
    t_lower = np.random.rand(N_u//4, 1)
    x_zero = (-1) + np.random.rand(N_u//2, 1) * (1 - (-1))

    X_upper = np.hstack((x_upper, t_upper))
    X_lower = np.hstack((x_lower, t_lower))
    X_zero = np.hstack((x_zero, t_zero))
    X_u_train = np.vstack((X_upper, X_lower, X_zero))

    index = np.arange(0, N_u)
    np.random.shuffle(index)
    X_u_train = X_u_train[index, :]

    X_f_train = np.zeros((N_f, 2), dtype=float)
    for row in range(N_f):
        x = uniform(-1, 1)
        t = uniform(0, 1)
        X_f_train[row, 0] = x
        X_f_train[row, 1] = t

    X_f_train = np.vstack((X_f_train, X_u_train))

    u_upper = np.zeros((N_u//4, 1), dtype=float)
    u_lower = np.zeros((N_u//4, 1), dtype=float)
    u_zero = -np.sin(np.pi * x_zero)
    u_train = np.vstack((u_upper, u_lower, u_zero))
    u_train = u_train[index, :]

    # observations from grid (every 50th as in Adam-PINN)
    data = scipy.io.loadmat('burgers_shock.mat')
    t_ = data['t'].flatten()[:, None]
    x_ = data['x'].flatten()[:, None]
    Exact = np.real(data['usol'])
    T, Xg = np.meshgrid(t_, x_)
    X_star = np.hstack((Xg.flatten()[:, None], T.flatten()[:, None]))
    u_truth = Exact.flatten()[:, None]
    u_truth_vec = u_truth[::50]
    u_observation = u_truth_vec

    # prepare observation input points
    x_flat = torch.linspace(-1, 1, 256)
    t_flat = torch.linspace(0, 1, 100)
    x, t = torch.meshgrid(x_flat, t_flat)
    xcol = x.reshape(-1, 1)[::50].to(device)
    tcol = t.reshape(-1, 1)[::50].to(device)

    pinn = PhysicsInformedNN(X_u_train, u_train, X_f_train)

    # create dual variables for AL enforcement of ibc and boundary
    # extract ini and bdry points from X_u_train
    X_u_np = X_u_train
    ini_mask = np.isclose(X_u_np[:,1], 0.0)
    bdry_mask = (np.isclose(X_u_np[:,0], -1.0) | np.isclose(X_u_np[:,0], 1.0))
    X_ini = X_u_np[ini_mask]
    X_bdry = X_u_np[bdry_mask]

    X_ini_t = torch.from_numpy(X_ini).float().to(device)
    X_bdry_t = torch.from_numpy(X_bdry).float().to(device)
    u_ini = torch.from_numpy((-np.sin(np.pi * X_ini[:,0:1])).astype(np.float32)).to(device)

    lbd1 = Variable(torch.FloatTensor([0]*X_ini_t.size(0)).to(device), requires_grad=True)
    lbd2 = Variable(torch.FloatTensor([0]*X_bdry_t.size(0)).to(device), requires_grad=True)

    optimizer = torch.optim.Adam([
        {'params': pinn.net.parameters(), 'lr': args.lr},
        {'params': lbd1, 'lr': args.lbd_lr},
        {'params': lbd2, 'lr': args.lbd_lr}
    ])

    # train
    pinn.net.train()
    best_state = {k: v.clone().cpu() for k, v in pinn.net.state_dict().items()}
    val_errs = []
    test_errs = []

    start_time = time.time()
    for epoch in trange(1, args.EPOCH+1, desc='Training'):
        optimizer.zero_grad()

        loss_res = residual_loss(pinn)
        loss_ics = ibc_loss(pinn)
        loss_obs = observation_loss(pinn, u_observation, xcol, tcol)

        # evaluate outputs for AL terms
        output_ini = pinn.net_u(X_ini_t[:,0:1], X_ini_t[:,1:2])
        output_bdry = pinn.net_u(X_bdry_t[:,0:1], X_bdry_t[:,1:2])

        # augmented-Lagrangian style objective
        loss = loss_res + loss_obs + args.beta * loss_ics + (lbd1 * (output_ini - u_ini).view(-1)).mean() + (lbd2 * output_bdry.view(-1)).mean()

        loss.backward()

        # ascend on lambda by flipping sign of its gradient before optimizer step
        if lbd1.grad is not None:
            lbd1.grad *= -1
        if lbd2.grad is not None:
            lbd2.grad *= -1

        optimizer.step()

        if epoch % 100 == 0:
            print('Epoch %d | Loss: %.6f | res: %.6f | ibc: %.6f | obs: %.6f | λ_PINN = [%.6f, %.6f]' %
                  (epoch, loss.item(), loss_res.item(), loss_ics.item(), loss_obs.item(), pinn.lambda1.item(), pinn.lambda2.item()))
            # quick validation and checkpoint: update best_state more frequently so trained lambdas are saved
            idx = np.random.choice(len(u_truth), len(u_truth)//10, replace=False)
            X_val = torch.from_numpy(X_star[idx]).float().to(device)
            y_val = torch.from_numpy(u_truth[idx]).float().to(device)
            with torch.no_grad():
                pred_val = pinn.net_u(X_val[:,0:1], X_val[:,1:2])
                val_err = torch.linalg.norm((pred_val - y_val),2).item() / torch.linalg.norm(y_val,2).item()
            val_errs.append(val_err)
            if len(val_errs)==1 or val_err < min(val_errs):
                best_state = {k: v.clone().cpu() for k, v in pinn.net.state_dict().items()}

        # save best by validation (use small random split of grid)
        if epoch % 500 == 0:
            # quick validation on subset
            idx = np.random.choice(len(u_truth), len(u_truth)//10, replace=False)
            X_val = torch.from_numpy(X_star[idx]).float().to(device)
            y_val = torch.from_numpy(u_truth[idx]).float().to(device)
            with torch.no_grad():
                pred_val = pinn.net_u(X_val[:,0:1], X_val[:,1:2])
                val_err = torch.linalg.norm((pred_val - y_val),2).item() / torch.linalg.norm(y_val,2).item()
            val_errs.append(val_err)
            if len(val_errs)==1 or val_err < min(val_errs):
                best_state = {k: v.clone().cpu() for k, v in pinn.net.state_dict().items()}

    elapsed = time.time() - start_time
    print(f'Training finished in {elapsed:.1f}s')

    # save best state
   

    # evaluation using best weights
    pinn.net.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    data = scipy.io.loadmat('burgers_shock.mat')
    t_ = data['t'].flatten()[:, None]
    x_ = data['x'].flatten()[:, None]
    T, Xg = np.meshgrid(t_, x_)
    X_star = np.hstack((Xg.flatten()[:, None], T.flatten()[:, None]))
    u_truth = np.real(data['usol']).flatten()[:, None]

    x_full = torch.from_numpy(X_star[:, 0:1]).float().to(device)
    t_full = torch.from_numpy(X_star[:, 1:2]).float().to(device)
    with torch.no_grad():
        u_pred = pinn.net_u(x_full, t_full).cpu().numpy()

    # compute losses
    x_full_req = x_full.clone().detach().requires_grad_(True)
    t_full_req = t_full.clone().detach().requires_grad_(True)
    f_pred_test = pinn.net_f(x_full_req, t_full_req).detach().cpu().numpy()
    physics_test_loss = np.mean(f_pred_test ** 2)

    x_star = X_star[:, 0]
    t_star = X_star[:, 1]
    boundary_mask = (np.isclose(x_star, -1.0) | np.isclose(x_star, 1.0) | np.isclose(t_star, 0.0))
    u_boundary_pred = u_pred[boundary_mask]
    u_boundary_true = u_truth[boundary_mask]
    boundary_test_loss = np.mean((u_boundary_pred - u_boundary_true) ** 2)

    initial_mask = np.isclose(t_star, 0.0)
    u_initial_pred = u_pred[initial_mask]
    u_initial_true = u_truth[initial_mask]
    initial_test_loss = np.mean((u_initial_pred - u_initial_true) ** 2)

    l2_loss_mse = np.mean((u_truth - u_pred)**2)
    rel_error = np.linalg.norm(u_truth - u_pred, 2) / np.linalg.norm(u_truth, 2)

    print(f"Test Physics Loss (MSE): {physics_test_loss:.6e}")
    print(f"Test Boundary Loss (MSE): {boundary_test_loss:.6e}")
    print(f"Test Initial Loss (MSE): {initial_test_loss:.6e}")
    print(f"Final L2 MSE: {l2_loss_mse:.6e}")
    print(f"Final Relative L2 Error: {rel_error:.6e}")

    # discovered parameters
    print('Discovered lambda1 (True=1.0):', pinn.lambda1.item())
    print('Discovered lambda2 (True=nu):', pinn.lambda2.item())

    # Data/observation loss on every 50th grid point (match training observation sampling)
    obs_idx = np.arange(0, u_truth.shape[0], 50)
    u_obs_pred = u_pred.reshape(-1, 1)[obs_idx]
    u_obs_true = u_truth.reshape(-1, 1)[obs_idx]
    data_mse = np.mean((u_obs_true - u_obs_pred) ** 2)
    data_rel_l2 = np.linalg.norm(u_obs_true - u_obs_pred, 2) / np.linalg.norm(u_obs_true, 2)
    print(f"Data/Observation Loss (MSE on sampled points): {data_mse:.6e}")
    print(f"Data/Observation Relative L2: {data_rel_l2:.6e}")

    # --- Save solution and error data ---
    os.makedirs('./data', exist_ok=True)

    x_plot = torch.linspace(-1, 1, 256).to(device)
    t_plot = torch.linspace(0, 1, 100).to(device)
    X_grid_plot, T_grid_plot = torch.meshgrid(x_plot, t_plot, indexing='ij')
    xcol_pred = X_grid_plot.reshape(-1, 1)
    tcol_pred = T_grid_plot.reshape(-1, 1)
    with torch.no_grad():
        usol_pred = pinn.net_u(xcol_pred, tcol_pred)
    Unp_pred = usol_pred.reshape(x_plot.numel(), t_plot.numel()).cpu().numpy()
    xnp_plot = x_plot.cpu().numpy()
    tnp_plot = t_plot.cpu().numpy()

    scipy.io.savemat('./data/AL_PINN_solution.mat', {'solution': Unp_pred, 'x': xnp_plot, 't': tnp_plot})
    print('Saved solution matrix to ./data/AL_PINN_solution.mat')

    print("\nPlotting the solution ...")
    os.makedirs('./figures/solution', exist_ok=True)
    plt.rcParams['font.size'] = '15'
    fig = plt.figure(figsize=(5, 6))
    ax = fig.add_subplot(111)
    plt.xlabel(r"$t$")
    plt.ylabel(r"$x$")
    plt.title("AL-PINN Solution")
    img_handle = ax.imshow(
        Unp_pred,
        interpolation='nearest',
        cmap='rainbow',
        extent=[tnp_plot.min(), tnp_plot.max(), xnp_plot.min(), xnp_plot.max()],
        origin='lower',
        aspect='auto',
        vmin=-1.0,
        vmax=1.0,
    )
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.10)
    cbar = fig.colorbar(img_handle, cax=cax)
    cbar.ax.tick_params(labelsize=10)
    output_filename = './figures/solution/AL_PINN_solution.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight', pad_inches=0.1)
    print(f'Plot saved successfully as {output_filename}')
    plt.show()

    print("Generating error plot...")
    error_grid = np.abs(np.real(data['usol']) - Unp_pred)
    scipy.io.savemat('./data/AL_PINN_error.mat', {'error': error_grid, 'x': xnp_plot, 't': tnp_plot})
    print('Saved error matrix to ./data/AL_PINN_error.mat')

    fig_err = plt.figure(figsize=(5, 6))
    ax_err = fig_err.add_subplot(111)
    plt.xlabel(r"$t$")
    plt.ylabel(r"$x$")
    plt.title(r"AL-PINN Error")
    img_handle_err = ax_err.imshow(
        error_grid,
        interpolation='nearest',
        cmap='bwr',
        extent=[tnp_plot.min(), tnp_plot.max(), xnp_plot.min(), xnp_plot.max()],
        origin='lower',
        aspect='auto',
        vmin=0.0,
        vmax=1.0,
    )
    divider_err = make_axes_locatable(ax_err)
    cax_err = divider_err.append_axes("right", size="5%", pad=0.10)
    cbar_err = fig_err.colorbar(img_handle_err, cax=cax_err)
    cbar_err.ax.tick_params(labelsize=10)
    output_filename_err = './figures/solution/AL_PINN_error.png'
    plt.savefig(output_filename_err, dpi=300, bbox_inches='tight', pad_inches=0.1)
    print(f'Error plot saved successfully as {output_filename_err}')
    plt.show()

    # save metrics alongside model state
   

if __name__ == '__main__':
    main()
