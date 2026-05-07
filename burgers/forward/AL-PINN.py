#!/usr/bin/env python3
"""Augmented-Lagrangian PINN for Burgers equation adapted to this repo's PhysicsInformedNN setup.

Usage: python AL-PINN.py --EPOCH 10000 --lr 1e-4 --lbd_lr 1e-4 --beta 0.05 --ordinal 0
"""
import os
import argparse
import copy
import numpy as np
import scipy.io
from tqdm import tqdm

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset

# --- Device set up ---
parser = argparse.ArgumentParser()
parser.add_argument('--beta', default=0.05, type=float)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--lbd_lr', default=1e-4, type=float)
parser.add_argument('--EPOCH', default=10000, type=int)
parser.add_argument('--ordinal', default=0, type=int)
args = parser.parse_args()

device = torch.device(f"cuda:{args.ordinal}" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

nu = 0.01/np.pi

def calculate_derivative(y, x):
    return torch.autograd.grad(y, x, create_graph=True, grad_outputs=torch.ones_like(y).to(device))[0]

def calculate_all_partial(u, x):
    del_u = calculate_derivative(u, x)
    u_t = del_u[:, 0]
    u_x = del_u[:, 1]
    u_xx = calculate_derivative(u_x.unsqueeze(-1), x)[:, 1]
    return u_t.view(-1,1), u_x.view(-1,1), u_xx.view(-1,1)


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


def train_epoch(u_model, lbd1, lbd2, beta, trainloader, ini_bdry_data, val_test, optimizer, loss_f):
    loss_list, loss1_list, loss2_list, loss3_list, val_list, test_list = [], [], [], [], [], []
    X_ini, u_ini, X_bdry, u_bdry = ini_bdry_data
    X_val, y_val, X_test, y_test = val_test

    for (data,) in trainloader:
        optimizer.zero_grad()
        X_v = data.to(device).requires_grad_(True)
        output = u_model(X_v)
        output_ini = u_model(X_ini)
        output_bdry = u_model(X_bdry)

        u_t, u_x, u_xx = calculate_all_partial(output, X_v)
        loss1 = loss_f(u_t + output * u_x - nu * u_xx, torch.zeros_like(u_t))
        loss2 = loss_f(output_ini - u_ini, torch.zeros_like(output_ini))
        loss3 = loss_f(output_bdry, torch.zeros_like(output_bdry))

        # augmented Lagrangian style objective (as in original snippet)
        loss = loss1 + 0.05 * loss2 + (lbd1 * (output_ini - u_ini).view(-1)).mean() + 0.05 * loss3 + (lbd2 * output_bdry.view(-1)).mean()

        loss.backward()

        # ascend on lambda by flipping sign of its gradient before optimizer step
        if lbd1.grad is not None:
            lbd1.grad *= -1
        if lbd2.grad is not None:
            lbd2.grad *= -1

        optimizer.step()

        # evaluation metrics
        with torch.no_grad():
            val_err = torch.linalg.norm((u_model(X_val) - y_val),2).item() / torch.linalg.norm(y_val,2).item()
            test_err = torch.linalg.norm((u_model(X_test) - y_test),2).item() / torch.linalg.norm(y_test,2).item()

        loss_list.append((loss1+loss2).item())
        loss1_list.append(loss1.item())
        loss2_list.append(loss2.item())
        loss3_list.append(loss3.item())
        val_list.append(val_err)
        test_list.append(test_err)

    return np.mean(loss_list), np.mean(loss1_list), np.mean(loss2_list), np.mean(loss3_list), np.mean(val_list), np.mean(test_list)


def main():
    # Use original Adam-PINN data creation (X_u_train etc.) and create PhysicsInformedNN with that data
    N_u = 100
    N_f = 20000

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

    # Collocation points
    X_f_train = np.random.uniform([-1, 0], [1, 1], size=(N_f, 2))
    X_f_train = np.vstack((X_f_train, X_u_train))

    # --- Load exact solution ---
    data = scipy.io.loadmat('burgers_shock.mat')
    t_exact = data['t'].flatten()[:, None]
    x_exact = data['x'].flatten()[:, None]
    Exact = np.real(data['usol'])
    T, X = np.meshgrid(t_exact, x_exact)
    # Use [x, t] ordering to match pinn.net_u/net_f which take (x, t)
    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    u_truth = Exact.flatten()[:, None]

    # Create model using original API
    pinn = PhysicsInformedNN(X_u_train, u_train, X_f_train)

    # Prepare lambda variables for initial and boundary (use X_zero and boundary subsets)
    # Extract initial points (t==0) and boundary points (x==xmin or x==xmax) from X_u_train
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

    # store best model weights as state_dict (avoid deepcopy on modules/tensors)
    best_model_state = {k: v.clone().cpu() for k, v in pinn.net.state_dict().items()}
    loss_f = nn.MSELoss()
    val_errs = []
    test_errs = []

    xmin, xmax = -1.0, 1.0

    os.makedirs('./figures', exist_ok=True)
    os.makedirs('./data', exist_ok=True)

    # Cache validation/test tensors once; avoid rebuilding every epoch.
    X_star_t = torch.from_numpy(X_star).float().to(device)
    u_truth_t = torch.from_numpy(u_truth).float().to(device)
    N = X_star_t.shape[0]
    n_val = N // 10
    n_test = N // 10

    for epoch in tqdm(range(args.EPOCH)):
        # Single-step epoch: one optimizer update using the full collocation set.
        optimizer.zero_grad()

        output_ini = pinn.net_u(X_ini_t[:,0:1], X_ini_t[:,1:2])
        output_bdry = pinn.net_u(X_bdry_t[:,0:1], X_bdry_t[:,1:2])

        f_pred = pinn.net_f(pinn.x_f, pinn.t_f)
        loss1 = loss_f(f_pred, pinn.null)
        loss2 = loss_f(output_ini - u_ini, torch.zeros_like(output_ini))
        loss3 = loss_f(output_bdry, torch.zeros_like(output_bdry))

        loss = loss1 + 0.05 * loss2 + (lbd1 * (output_ini - u_ini).view(-1)).mean() + 0.05 * loss3 + (lbd2 * output_bdry.view(-1)).mean()

        loss.backward()
        if lbd1.grad is not None:
            lbd1.grad *= -1
        if lbd2.grad is not None:
            lbd2.grad *= -1

        optimizer.step()

        # evaluation metrics on val/test sets
        with torch.no_grad():
            idx = np.arange(N)
            np.random.shuffle(idx)
            val_idx = idx[:n_val]
            test_idx = idx[n_val:n_val+n_test]

            X_val = X_star_t[val_idx]
            y_val = u_truth_t[val_idx]
            X_test = X_star_t[test_idx]
            y_test = u_truth_t[test_idx]

            val_err = torch.linalg.norm((pinn.net_u(X_val[:,0:1], X_val[:,1:2]) - y_val),2).item() / torch.linalg.norm(y_val,2).item()
            test_err = torch.linalg.norm((pinn.net_u(X_test[:,0:1], X_test[:,1:2]) - y_test),2).item() / torch.linalg.norm(y_test,2).item()

        loss = loss.item()
        loss1 = loss1.item()
        loss2 = loss2.item()
        loss3 = loss3.item()

        val_errs.append(val_err)
        test_errs.append(test_err)

        if np.argmin(val_errs) == epoch:
            best_model_state = {k: v.clone().cpu() for k, v in pinn.net.state_dict().items()}

        if (epoch+1) % 500 == 0:
            print(f"Epoch {epoch+1}/{args.EPOCH} | loss: {loss:.6e} | res: {loss1:.6e} | ini: {loss2:.6e} | bdry: {loss3:.6e} | val_err: {val_err:.6e} | test_err: {test_err:.6e}")

    # save best model
    os.makedirs('./training_data', exist_ok=True)
    # save best model state dict
    torch.save(best_model_state, './training_data/AL-PINN_model.pth')
    print('Saved model state dict to ./training_data/AL-PINN_model.pth')

    # evaluation and save solution plot (use [x,t] ordering to match training)
    data = scipy.io.loadmat('burgers_shock.mat')
    t_exact = data['t'].flatten()[:, None]
    x_exact = data['x'].flatten()[:, None]
    T, X = np.meshgrid(t_exact, x_exact)
    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))

    # build grid tensors (columns [x,t])
    X_grid = torch.from_numpy(X_star).float().to(device)
    # load best weights into model for evaluation
    pinn.net.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
    with torch.no_grad():
        u_pred = pinn.net_u(X_grid[:,0:1], X_grid[:,1:2]).cpu().numpy()

    usol_pred = u_pred.reshape(x_exact.size, t_exact.size)
    scipy.io.savemat('./data/AL_PINN_solution.mat', {'u_pred': usol_pred})

    # simple plot save (reuse matplotlib if available)
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(5,6))
        plt.imshow(usol_pred, interpolation='nearest', cmap='rainbow', origin='lower', aspect='auto')
        plt.colorbar()
        plt.title('AL-PINN solution')
        plt.xlabel('t')
        plt.ylabel('x')
        plt.savefig('./figures/AL-PINN_solution.png', dpi=300, bbox_inches='tight')
        print('Saved figure to ./figures/AL-PINN_solution.png')
    except Exception:
        pass

    # compute physics, initial and boundary losses on full grid
    loss_fn = nn.MSELoss()

    X_grid_req = X_grid.clone().detach().requires_grad_(True)
    f_pred = pinn.net_f(X_grid_req[:,0:1], X_grid_req[:,1:2])
    physics_loss = loss_fn(f_pred, torch.zeros_like(f_pred)).item()

    # initial loss (t == 0) -- X_star columns are [x, t]
    X_star_np = X_star
    ini_mask = np.isclose(X_star_np[:,1], 0.0)
    X_ini_eval = torch.from_numpy(X_star_np[ini_mask]).float().to(device)
    with torch.no_grad():
        u_ini_pred = pinn.net_u(X_ini_eval[:,0:1], X_ini_eval[:,1:2])
    u_ini_true = -torch.sin(np.pi * X_ini_eval[:,0:1])
    initial_loss = loss_fn(u_ini_pred, u_ini_true.to(device)).item()

    # boundary loss (x == xmin or x == xmax)
    bdry_mask = np.isclose(X_star_np[:,0], xmin) | np.isclose(X_star_np[:,0], xmax)
    X_bdry_eval = torch.from_numpy(X_star_np[bdry_mask]).float().to(device)
    with torch.no_grad():
        u_bdry_pred = pinn.net_u(X_bdry_eval[:,0:1], X_bdry_eval[:,1:2])
    u_bdry_true = torch.zeros_like(u_bdry_pred)
    boundary_loss = loss_fn(u_bdry_pred, u_bdry_true).item()

    # overall MSE and relative L2 on grid
    # ensure both are numpy arrays
    u_pred_arr = u_pred.reshape(-1,1) if isinstance(u_pred, np.ndarray) else u_pred.detach().cpu().numpy().reshape(-1,1)
    u_truth_arr = u_truth.reshape(-1,1) if isinstance(u_truth, np.ndarray) else u_truth.detach().cpu().numpy().reshape(-1,1)
    mse = np.mean((u_truth_arr - u_pred_arr)**2)
    rel_l2 = np.linalg.norm(u_truth_arr - u_pred_arr,2) / np.linalg.norm(u_truth_arr,2)

    print(f"Physics Loss (MSE of residual): {physics_loss:.6e}")
    print(f"Initial Loss (MSE): {initial_loss:.6e}")
    print(f"Boundary Loss (MSE): {boundary_loss:.6e}")
    print(f"Final MSE: {mse:.6e}")
    print(f"Final Relative L2 Error: {rel_l2:.6e}")

    # Data/observation loss (every 50th grid point as in other scripts)
    obs_idx = np.arange(0, u_truth.shape[0], 50)
    u_obs_pred = u_pred_arr.reshape(-1, 1)[obs_idx]
    u_obs_true = u_truth.reshape(-1, 1)[obs_idx]
    data_mse = np.mean((u_obs_true - u_obs_pred) ** 2)
    data_rel_l2 = np.linalg.norm(u_obs_true - u_obs_pred, 2) / np.linalg.norm(u_obs_true, 2)
    print(f"Data/Observation Loss (MSE on sampled points): {data_mse:.6e}")
    print(f"Data/Observation Relative L2: {data_rel_l2:.6e}")

    # save metrics
    # try:
    #     scipy.io.savemat('./data/AL_PINN_metrics.mat', {'physics_loss': physics_loss,
    #                                'initial_loss': initial_loss,
    #                                'boundary_loss': boundary_loss,
    #                                'mse': mse,
    #                                'rel_l2': rel_l2,
    #                                'data_mse': data_mse,
    #                                'data_rel_l2': data_rel_l2})
    #     print('Saved metrics to ./data/AL_PINN_metrics.mat')
    # except Exception:
    #     pass

    # print final best test error
    print('Best Test Error : ', test_errs[np.argmin(val_errs)])


if __name__ == '__main__':
    main()
