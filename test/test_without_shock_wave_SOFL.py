




# import torch
# import torch.nn as nn
# import numpy as np
# from tqdm import trange
# import random as rm
# import scipy.io
# from scipy.linalg import solve
# import os
# import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1 import make_axes_locatable

# # --- Setup, PINN Class, and Helper functions ---
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(f"We are using device: {device}")

# seed = 42
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# np.random.seed(seed)
# rm.seed(seed)

# def save_model(model, path):
#     """Saves the model's state dictionary."""
#     torch.save(model.state_dict(), path)
#     print(f"Model saved to {path}")

# def load_model(model, path):
#     """Loads the model's state dictionary."""
#     model.load_state_dict(torch.load(path))
#     model.eval()
#     print(f"Model loaded from {path}")

# def plot_optimization_history_2(*Histories, optimal_value=None, legends=None, linewidth=1, fontsize=20, ticksize=15, legendsize=15, savename=None):
#     plt.figure(figsize=(21, 5)) # Wider figure to accommodate 3 plots
    
#     # --- PLOT 1: Function Value ---
#     plt.subplot(1, 3, 1)
#     if optimal_value is not None:
#         plt.axhline(optimal_value, color='g', linestyle='--', label='Optimal $f(x)$', linewidth=linewidth)
    
#     for i, history in enumerate(Histories):
#         label = legends[i] if legends is not None and i < len(legends) else f'Run {i+1}'
#         f_values, _, _ = zip(*history)
#         N = len(f_values)
#         plt.plot(f_values, label=label, linewidth=linewidth)

#     plt.xlabel('Iteration (x100)', fontsize=fontsize)
#     plt.ylabel('$f(x)$', fontsize=fontsize)
#     plt.legend(fontsize=legendsize)
#     plt.title('Function Value', fontsize=fontsize)
#     plt.grid(True)
#     plt.xticks(fontsize=ticksize)
#     plt.yticks(fontsize=ticksize)

#     # --- PLOT 2: KKT Gap ---
#     plt.subplot(1, 3, 2)
#     for i, history in enumerate(Histories):
#         label = legends[i] if legends is not None and i < len(legends) else f'Run {i+1}'
#         _, KKT_gaps, _ = zip(*history)
#         plt.plot(KKT_gaps, label=label, linewidth=linewidth)
    
#     plt.xlabel('Iteration (x100)', fontsize=fontsize)
#     plt.ylabel('KKT Gap', fontsize=fontsize)
#     plt.yscale('log')
#     plt.legend(fontsize=legendsize)
#     plt.title('KKT Gap', fontsize=fontsize)
#     plt.grid(True)
#     plt.xticks(fontsize=ticksize)
#     plt.yticks(fontsize=ticksize)

#     # --- PLOT 3: Constraint Violation ---
#     plt.subplot(1, 3, 3)
#     for i, history in enumerate(Histories):
#         label = legends[i] if legends is not None and i < len(legends) else f'Run {i+1}'
#         _, _, h_violations = zip(*history)
#         plt.plot(h_violations, label=label, linewidth=linewidth)
        
#     plt.xlabel('Iteration (x100)', fontsize=fontsize)
#     plt.ylabel('Constraint Violation |h(x)|', fontsize=fontsize)
#     plt.yscale('log')
#     plt.legend(fontsize=legendsize)
#     plt.title('Constraint Violation', fontsize=fontsize)
#     plt.grid(True)
#     plt.xticks(fontsize=ticksize)
#     plt.yticks(fontsize=ticksize)

#     plt.tight_layout()
#     if savename:
#         dirpath = os.path.dirname(savename)
#         if dirpath:
#             os.makedirs(dirpath, exist_ok=True)
#         plt.savefig(savename, bbox_inches='tight')
#     plt.show()
                   

# class PhysicsInformedNN():
#     def __init__(self, X_u, u, X_f):
#         self.X_u_master = torch.tensor(X_u, dtype=torch.float32).to(device)
#         self.u_master = torch.tensor(u, dtype=torch.float32).to(device)
#         self.X_f_master = torch.tensor(X_f, dtype=torch.float32).to(device)
        
#         self.X_u_full = self.X_u_master
#         self.u_full = self.u_master
#         self.X_f_full = self.X_f_master

#         self.create_net()
#         self.loss = nn.MSELoss()

#     def create_net(self):
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

#     def set_data_for_stage(self, X_u_stage, u_stage, X_f_stage):
#         """Updates the active dataset for the current causal training stage."""
#         self.X_u_full = X_u_stage
#         self.u_full = u_stage
#         self.X_f_full = X_f_stage

#     def net_u(self, x, t):
#         return self.net(torch.hstack((x, t)))

#     def net_f(self, x, t):
#         x.requires_grad_(True)
#         t.requires_grad_(True)
#         u = self.net_u(x, t)
#         u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
#         u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
#         u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True)[0]
#         v = 0.01 / np.pi
#         f = u_t + (u * u_x) - (v * u_xx)
#         return f

# def residual_loss(model):
#     f_pred = model.net_f(model.X_f_full[:, 0:1], model.X_f_full[:, 1:2])
#     null_f = torch.zeros((model.X_f_full.shape[0], 1), device=device)
#     return model.loss(f_pred, null_f)

# def boundary_loss(model):
#     u_pred = model.net_u(model.X_u_full[:, 0:1], model.X_u_full[:, 1:2])
#     return model.loss(u_pred, model.u_full)

# def get_flat_params(model):
#     return np.concatenate([p.detach().cpu().numpy().flatten() for p in model.parameters()])

# def set_flat_params(model, flat_params):
#     flat_params = flat_params.astype(np.float32)
#     pointer = 0
#     for p in model.parameters():
#         num_params = p.numel()
#         p.data = torch.from_numpy(flat_params[pointer:pointer + num_params]).view_as(p).to(device)
#         pointer += num_params

# pinn_model = None

# # Wrapper functions for the optimizer
# def f(x):
#     set_flat_params(pinn_model.net, x)
#     return residual_loss(pinn_model).item()

# def h(x):
#     set_flat_params(pinn_model.net, x)
#     return np.array([boundary_loss(pinn_model).item()])

# def df(x):
#     set_flat_params(pinn_model.net, x)
#     pinn_model.net.zero_grad()
#     loss = residual_loss(pinn_model)
#     loss.backward()
#     return np.concatenate([p.grad.cpu().numpy().flatten() for p in pinn_model.net.parameters()])

# def dh(x):
#     set_flat_params(pinn_model.net, x)
#     pinn_model.net.zero_grad()
#     loss = boundary_loss(pinn_model)
#     loss.backward()
#     grad = np.concatenate([p.grad.cpu().numpy().flatten() for p in pinn_model.net.parameters()])
#     return grad.reshape(1, -1)

# # --- Optimizer ---
# def SOFL_Adam_PINN(f, h, df, dh, x_start, Kp, Ki=0, eta=0.01, max_iter=1000, tol=1e-6):
#     x = np.array(x_start, dtype=np.float32)
#     history = []
#     Integral = 0.0

#     beta1, beta2, epsilon = 0.9, 0.999, 1e-8
#     m, v, t = np.zeros_like(x), np.zeros_like(x), 0
    
#     pbar = trange(max_iter, desc="SOFL_Adam")
#     for iteration in pbar:
#         # Gradients are calculated on the current stage's dataset
#         true_grad_f = df(x)
#         true_J_h = dh(x)
#         true_h_x = h(x)

#         JhJht = true_J_h @ true_J_h.T
#         Pcontrol = -Kp @ true_h_x
#         Integral = Integral + true_h_x
#         Icontrol = Ki * Integral
#         rhs = Pcontrol + Icontrol + true_J_h @ true_grad_f
#         I = np.eye(np.shape(JhJht)[0])
#         lambda_ = -solve(JhJht + 1e-6 * I, rhs, assume_a='pos')

#         KKT_grad = true_grad_f + (true_J_h.T @ lambda_).flatten()
        
#         clip_value = 10.0
#         grad_norm = np.linalg.norm(KKT_grad)
#         if grad_norm > clip_value:
#             KKT_grad = KKT_grad * (clip_value / grad_norm)
#         # updata with Adam
#         t += 1
#         m = beta1 * m + (1 - beta1) * KKT_grad
#         v = beta2 * v + (1 - beta2) * (KKT_grad**2)
#         m_hat = m / (1 - beta1**t)
#         v_hat = v / (1 - beta2**t)
#         step_vector = eta * m_hat / (np.sqrt(v_hat) + epsilon)
#         x_new = x - step_vector
#         step_size = np.linalg.norm(step_vector)
        
#         if iteration % 100 == 0:
#             true_f_val = f(x)
#             true_h_val = h(x) 
#             # Calculate KKT Gap for logging
#             KKT_gap = np.max([np.linalg.norm(KKT_grad), np.max(np.abs(true_h_val))])
#             pbar.set_postfix({
#                 'physics_loss': f'{f(x):.2e}', 
#                 'boundary_loss': f'{true_h_x[0]:.2e}',
#                 'KKT_gap': f'{KKT_gap:.2e}',
#                 'Step Size': f'{step_size:.2e}'
#             })
#             history.append((true_f_val, KKT_gap, np.abs(true_h_val[0])))

#         if step_size < tol:
#             print(f"\nConvergence at iter {iteration}: Step size ({step_size:.2e}) < tol ({tol:.2e}).")
#             break

#         x = x_new
        
#         if np.isnan(x).any():
#             print("\nError: NaN values detected. Stopping.")
#             break
            
#     return x, history

# # --- Main Execution ---

# # 1. Load Data
# N_u = 100
# N_f = 10000
# x_upper = np.ones((N_u // 4, 1), dtype=np.float32); t_upper = np.random.rand(N_u // 4, 1).astype(np.float32)
# x_lower = -np.ones((N_u // 4, 1), dtype=np.float32); t_lower = np.random.rand(N_u // 4, 1).astype(np.float32)
# t_zero = np.zeros((N_u // 2, 1), dtype=np.float32); x_zero = -1 + 2 * np.random.rand(N_u // 2, 1).astype(np.float32)
# X_upper = np.hstack((x_upper, t_upper)); X_lower = np.hstack((x_lower, t_lower)); X_zero = np.hstack((x_zero, t_zero))
# X_u_train_np = np.vstack((X_upper, X_lower, X_zero))
# u_upper = np.zeros((N_u // 4, 1), dtype=np.float32); u_lower = np.zeros((N_u // 4, 1), dtype=np.float32)
# u_zero = -np.sin(np.pi * x_zero).astype(np.float32)
# u_train_np = np.vstack((u_upper, u_lower, u_zero))
# index = np.arange(N_u); np.random.shuffle(index)
# X_u_train_np = X_u_train_np[index, :]; u_train_np = u_train_np[index, :]
# X_f_train_np = np.random.uniform([-1, 0], [1, 1], size=(N_f, 2)).astype(np.float32)
# X_f_train_np = np.vstack((X_f_train_np, X_u_train_np))

# # 2. Initialize PINN model
# pinn_model = PhysicsInformedNN(X_u_train_np, u_train_np, X_f_train_np)

# # 3. Setup and Run Causal Training Curriculum
# x_current = get_flat_params(pinn_model.net)
# stages = [(0.33, 10000), (0.66, 10000), (1.0, 10000)] 

# # Use the best fixed hyperparameters from the successful Full Batch run
# Kp = np.array([[500.0]])
# Ki = 0.01
# eta = 1e-4 
# tol = 1e-8

# for i, (t_max, iters) in enumerate(stages):
#     print(f"\n--- Causal Training: Stage {i+1}/{len(stages)} (t <= {t_max}) ---")

#     # Filter the master collocation dataset for the current time window
#     X_f_stage_np = X_f_train_np[X_f_train_np[:, 1] <= t_max]
    
#     # Update the model with the current subset of data
#     pinn_model.set_data_for_stage(
#         pinn_model.X_u_master, 
#         pinn_model.u_master,
#         torch.tensor(X_f_stage_np, dtype=torch.float32).to(device)
#     )

#     # Run the optimizer for this stage, continuing from the parameters of the previous stage
#     x_current, history = SOFL_Adam_PINN(
#         f, h, df, dh,
#         x_current, Kp, Ki=Ki, eta=eta, 
#         max_iter=iters, tol=tol
#     )
# x_opt = x_current.detach().cpu().numpy() if torch.is_tensor(x_current) else x_current
# history_np = np.array(history)  # if it's numeric

# np.savez('burgers_equation.npz',
#          history=history_np,
#          x_opt=x_opt)

# x_final = x_current

# # 4. Final Evaluation using the full master dataset
# set_flat_params(pinn_model.net, x_final)
# pinn_model.set_data_for_stage(pinn_model.X_u_master, pinn_model.u_master, pinn_model.X_f_master)


# print("\n--- Optimization Finished ---")
# final_boundary_loss = boundary_loss(pinn_model).item()
# final_physics_loss = residual_loss(pinn_model).item()


# print(f"Final Physics Loss (Objective): {final_physics_loss:.4f}")
# print(f"Final Boundary Loss (Constraint): {final_boundary_loss:.4f}")
# save_model(pinn_model.net, "./trained_pinn_model.pth")

# # --- Evaluation ---
# print("\n--- Evaluation ---")
# data = scipy.io.loadmat('burgers_shock.mat')
# t_exact = data['t'].flatten()[:, None]; x_exact = data['x'].flatten()[:, None]
# Exact = np.real(data['usol']); T, X = np.meshgrid(t_exact, x_exact)
# X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
# u_truth = Exact.flatten()[:,None]

# load_model(pinn_model.net, "./trained_pinn_model.pth")
# x_full = torch.from_numpy(X_star[:, 0:1]).float().to(device)
# t_full = torch.from_numpy(X_star[:, 1:2]).float().to(device)
# with torch.no_grad():
#     u_pred = pinn_model.net_u(x_full, t_full).cpu().numpy()

# l2_loss_mse = np.mean((u_truth - u_pred)**2)
# print(f"Final L2 Loss (MSE): {l2_loss_mse:.4f}")
# error = np.linalg.norm(u_truth - u_pred, 2) / np.linalg.norm(u_truth, 2)
# print(f"Final L2 Relative Error: {error:.4f}")

# # --- Plotting ---


# print("\nPlotting the solution ...")
# os.makedirs("./figures/solution", exist_ok=True)
# x_plot = torch.linspace(-1, 1, 256).to(device); t_plot = torch.linspace(0, 1, 100).to(device)
# X_grid_plot, T_grid_plot = torch.meshgrid(x_plot, t_plot, indexing='ij')
# xcol_pred = X_grid_plot.reshape(-1, 1); tcol_pred = T_grid_plot.reshape(-1, 1)
# with torch.no_grad():
#     usol_pred = pinn_model.net_u(xcol_pred, tcol_pred)
# Unp_pred = usol_pred.reshape(x_plot.numel(), t_plot.numel()).cpu().numpy()
# xnp_plot = x_plot.cpu().numpy(); tnp_plot = t_plot.cpu().numpy()

# print("Generating plot...")
# plt.rcParams['font.size'] = '15'; fig = plt.figure(figsize=(5, 6)); ax = fig.add_subplot(111)
# plt.xlabel(r"$t$"); plt.ylabel(r"$x$"); plt.title(r"SOFL-PINN $u(x,t)$")
# img_handle = ax.imshow(Unp_pred, interpolation='nearest', cmap='rainbow',
#                        extent=[tnp_plot.min(), tnp_plot.max(), xnp_plot.min(), xnp_plot.max()],
#                        origin='lower', aspect='auto', vmin=-1.0, vmax=1.0)          
# divider = make_axes_locatable(ax); cax = divider.append_axes("right", size="5%", pad=0.10)
# cbar = fig.colorbar(img_handle, cax=cax); cbar.ax.tick_params(labelsize=10)
# output_filename = "./figures/solution/SOFL_PINN_solution.png"
# plt.savefig(output_filename, dpi=300, bbox_inches='tight', pad_inches=0.1)
# print(f"Plot saved successfully as {output_filename}")
# plt.show()

# # --- Error Plot ---
# print("Generating error plot...")
# error_grid = abs(Exact - Unp_pred)   # absolute error
# # Optionally: relative error
# # error_grid = (Exact - Unp_pred) / (np.abs(Exact) + 1e-8)

# plt.rcParams['font.size'] = '15'
# fig_err = plt.figure(figsize=(5, 6))
# ax_err = fig_err.add_subplot(111)
# plt.xlabel(r"$t$")
# plt.ylabel(r"$x$")
# plt.title(r"Error")

# img_handle_err = ax_err.imshow(
#     error_grid,
#     interpolation='nearest',
#     cmap='bwr',
#     extent=[tnp_plot.min(), tnp_plot.max(), xnp_plot.min(), xnp_plot.max()],
#     origin='lower',
#     aspect='auto',
#     vmin=0.0, vmax=1.0
# )

# divider_err = make_axes_locatable(ax_err)
# cax_err = divider_err.append_axes("right", size="5%", pad=0.10)
# cbar_err = fig_err.colorbar(img_handle_err, cax=cax_err)
# cbar_err.ax.tick_params(labelsize=10)

# output_filename_err = "./figures/solution/SOFL_PINN_error.png"
# plt.savefig(output_filename_err, dpi=300, bbox_inches='tight', pad_inches=0.1)
# print(f"Error plot saved successfully as {output_filename_err}")
# plt.show()



# # --- Ground Truth Plot ---
# print("Generating ground truth plot...")
# plt.rcParams['font.size'] = '15'
# fig_truth = plt.figure(figsize=(5, 6))
# ax_truth = fig_truth.add_subplot(111)
# plt.xlabel(r"$t$")
# plt.ylabel(r"$x$")
# plt.title(r"Ground Truth $u(x,t)$")

# img_handle_truth = ax_truth.imshow(
#     Exact,
#     interpolation='nearest',
#     cmap='rainbow',
#     extent=[tnp_plot.min(), tnp_plot.max(), xnp_plot.min(), xnp_plot.max()],
#     origin='lower',
#     aspect='auto',
#     vmin=-1.0, vmax=1.0
# )

# divider_truth = make_axes_locatable(ax_truth)
# cax_truth = divider_truth.append_axes("right", size="5%", pad=0.10)
# cbar_truth = fig_truth.colorbar(img_handle_truth, cax=cax_truth)
# cbar_truth.ax.tick_params(labelsize=10)

# output_filename_truth = "./figures/solution/truth.png"
# plt.savefig(output_filename_truth, dpi=300, bbox_inches='tight', pad_inches=0.1)
# print(f"Ground truth plot saved successfully as {output_filename_truth}")
# plt.show()


# plot_optimization_history_2(history, legends=['SOFL'], linewidth=3, savename='burgers_equation.png')


# model_path = "./trained_pinn_model.pth"
# if os.path.exists(model_path):
#     load_model(pinn_model.net, model_path)
# else:
#     print(f"Error: Model file not found at {model_path}")
#     exit()

# print("\n--- Evaluating Loaded Model on Test Data ---")
# data = scipy.io.loadmat('burgers_shock.mat')
# t_exact = data['t'].flatten()[:, None]
# x_exact = data['x'].flatten()[:, None]
# Exact = np.real(data['usol'])
# T, X = np.meshgrid(t_exact, x_exact)

# # The full space-time grid for testing
# X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
# # The true solution on the grid
# u_truth = Exact.flatten()[:,None]

# # Convert test data to tensors, enabling gradients for physics loss calculation
# x_test = torch.tensor(X_star[:, 0:1], dtype=torch.float32, requires_grad=True).to(device)
# t_test = torch.tensor(X_star[:, 1:2], dtype=torch.float32, requires_grad=True).to(device)
# u_truth_tensor = torch.tensor(u_truth, dtype=torch.float32).to(device)

# # --- METRIC CALCULATIONS ---

# # A. Calculate losses over the ENTIRE domain
# x_test_domain = torch.tensor(X_star[:, 0:1], dtype=torch.float32, requires_grad=True).to(device)
# t_test_domain = torch.tensor(X_star[:, 1:2], dtype=torch.float32, requires_grad=True).to(device)
# u_truth_domain_tensor = torch.tensor(u_truth, dtype=torch.float32).to(device)

# u_pred_domain_tensor = pinn_model.net_u(x_test_domain, t_test_domain)
# test_domain_mse = pinn_model.loss(u_pred_domain_tensor, u_truth_domain_tensor).item()

# f_pred_test = pinn_model.net_f(x_test_domain, t_test_domain)
# null_f_test = torch.zeros_like(f_pred_test)
# test_physics_loss = pinn_model.loss(f_pred_test, null_f_test).item()

# # B. Calculate loss on ONLY the boundaries of the test set
# # Find indices for t=0 (initial), x=-1 (left), and x=1 (right)
# initial_indices = np.where(X_star[:, 1] == 0)[0]
# left_boundary_indices = np.where(X_star[:, 0] == -1)[0]
# right_boundary_indices = np.where(X_star[:, 0] == 1)[0]
# boundary_indices = np.unique(np.concatenate([initial_indices, left_boundary_indices, right_boundary_indices]))

# # Filter the test data to get only the boundary points
# X_boundary_test = X_star[boundary_indices]
# u_boundary_truth = u_truth[boundary_indices]

# # Convert boundary test data to tensors
# x_boundary_test = torch.tensor(X_boundary_test[:, 0:1], dtype=torch.float32).to(device)
# t_boundary_test = torch.tensor(X_boundary_test[:, 1:2], dtype=torch.float32).to(device)
# u_boundary_truth_tensor = torch.tensor(u_boundary_truth, dtype=torch.float32).to(device)

# # Get predictions on the boundary and calculate the MSE
# u_pred_boundary_tensor = pinn_model.net_u(x_boundary_test, t_boundary_test)
# test_boundary_loss_mse = pinn_model.loss(u_pred_boundary_tensor, u_boundary_truth_tensor).item()

# # C. Calculate L2 Relative Error (uses entire domain)
# u_pred_np = u_pred_domain_tensor.detach().cpu().numpy()
# l2_relative_error = np.linalg.norm(u_truth - u_pred_np, 2) / np.linalg.norm(u_truth, 2)
# # --- Print Results ---
# print("\n--- Final Test Metrics ---")
# print(f"Testing Boundary Loss (MSE): {test_boundary_loss_mse:.6f}")
# print(f"Testing Domain Loss (MSE):   {test_domain_mse:.6f}")
# print(f"Testing Physics Loss:        {test_physics_loss:.6f}")
# print(f"l2 relative error:        {l2_relative_error:.6f}")



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
    model.load_state_dict(torch.load(path))
    model.eval()
    print(f"Model loaded from {path}")

def plot_optimization_history_2(*Histories, optimal_value=None, legends=None, linewidth=1, fontsize=20, ticksize=15, legendsize=15, savename=None):
    plt.figure(figsize=(21, 5)) 
    
    # --- PLOT 1: Function Value ---
    plt.subplot(1, 3, 1)
    if optimal_value is not None:
        plt.axhline(optimal_value, color='g', linestyle='--', label='Optimal $f(x)$', linewidth=linewidth)
    
    for i, history in enumerate(Histories):
        label = legends[i] if legends is not None and i < len(legends) else f'Run {i+1}'
        f_values, _, _ = zip(*history)
        plt.plot(f_values, label=label, linewidth=linewidth)

    plt.xlabel('Iteration (x100)', fontsize=fontsize)
    plt.ylabel('$f(x)$', fontsize=fontsize)
    plt.legend(fontsize=legendsize)
    plt.title('Function Value', fontsize=fontsize)
    plt.grid(True)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)

    # --- PLOT 2: KKT Gap ---
    plt.subplot(1, 3, 2)
    for i, history in enumerate(Histories):
        label = legends[i] if legends is not None and i < len(legends) else f'Run {i+1}'
        _, KKT_gaps, _ = zip(*history)
        plt.plot(KKT_gaps, label=label, linewidth=linewidth)
    
    plt.xlabel('Iteration (x100)', fontsize=fontsize)
    plt.ylabel('KKT Gap', fontsize=fontsize)
    plt.yscale('log')
    plt.legend(fontsize=legendsize)
    plt.title('KKT Gap', fontsize=fontsize)
    plt.grid(True)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)

    # --- PLOT 3: Constraint Violation ---
    plt.subplot(1, 3, 3)
    for i, history in enumerate(Histories):
        label = legends[i] if legends is not None and i < len(legends) else f'Run {i+1}'
        _, _, h_violations = zip(*history)
        plt.plot(h_violations, label=label, linewidth=linewidth)
        
    plt.xlabel('Iteration (x100)', fontsize=fontsize)
    plt.ylabel('Constraint Violation |h(x)|', fontsize=fontsize)
    plt.yscale('log')
    plt.legend(fontsize=legendsize)
    plt.title('Constraint Violation', fontsize=fontsize)
    plt.grid(True)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)

    plt.tight_layout()
    if savename:
        dirpath = os.path.dirname(savename)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        plt.savefig(savename, bbox_inches='tight')
    plt.show()

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

    def set_data_for_stage(self, X_u_stage, u_stage, X_f_stage):
        self.X_u_full = X_u_stage
        self.u_full = u_stage
        self.X_f_full = X_f_stage

    def net_u(self, x, t):
        return self.net(torch.hstack((x, t)))

    def net_f(self, x, t):
        x.requires_grad_(True)
        t.requires_grad_(True)
        u = self.net_u(x, t)
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True)[0]
        v = 0.01 / np.pi
        f = u_t + (u * u_x) - (v * u_xx)
        return f

def residual_loss(model):
    f_pred = model.net_f(model.X_f_full[:, 0:1], model.X_f_full[:, 1:2])
    null_f = torch.zeros((model.X_f_full.shape[0], 1), device=device)
    return model.loss(f_pred, null_f)

def boundary_loss(model):
    u_pred = model.net_u(model.X_u_full[:, 0:1], model.X_u_full[:, 1:2])
    return model.loss(u_pred, model.u_full)

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

def f(x):
    set_flat_params(pinn_model.net, x)
    return residual_loss(pinn_model).item()

def h(x):
    set_flat_params(pinn_model.net, x)
    return np.array([boundary_loss(pinn_model).item()])

def df(x):
    set_flat_params(pinn_model.net, x)
    pinn_model.net.zero_grad()
    loss = residual_loss(pinn_model)
    loss.backward()
    return np.concatenate([p.grad.cpu().numpy().flatten() for p in pinn_model.net.parameters()])

def dh(x):
    set_flat_params(pinn_model.net, x)
    pinn_model.net.zero_grad()
    loss = boundary_loss(pinn_model)
    loss.backward()
    grad = np.concatenate([p.grad.cpu().numpy().flatten() for p in pinn_model.net.parameters()])
    return grad.reshape(1, -1)

# --- Optimizer with L2 Regularization (Weight Decay) ---
def SOFL_Adam_PINN(f, h, df, dh, x_start, Kp, Ki=0, eta=0.01, max_iter=1000, tol=1e-6, weight_decay=0.0):
    x = np.array(x_start, dtype=np.float32)
    history = []
    Integral = 0.0

    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    m, v, t = np.zeros_like(x), np.zeros_like(x), 0
    
    pbar = trange(max_iter, desc="SOFL+Adam+Decay")
    for iteration in pbar:
        true_grad_f = df(x)
        true_J_h = dh(x)
        true_h_x = h(x)

        JhJht = true_J_h @ true_J_h.T
        Pcontrol = -Kp @ true_h_x
        Integral = Integral + true_h_x
        Icontrol = Ki * Integral
        rhs = Pcontrol + Icontrol + true_J_h @ true_grad_f
        I = np.eye(np.shape(JhJht)[0])
        lambda_ = -solve(JhJht + 1e-6 * I, rhs, assume_a='pos')

        KKT_grad = true_grad_f + (true_J_h.T @ lambda_).flatten()
        
        # --- NEW: Add L2 Regularization (Weight Decay) ---
        if weight_decay > 0:
            KKT_grad = KKT_grad + weight_decay * x
            
        clip_value = 10.0
        grad_norm = np.linalg.norm(KKT_grad)
        if grad_norm > clip_value:
            KKT_grad = KKT_grad * (clip_value / grad_norm)

        t += 1
        m = beta1 * m + (1 - beta1) * KKT_grad
        v = beta2 * v + (1 - beta2) * (KKT_grad**2)
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        step_vector = eta * m_hat / (np.sqrt(v_hat) + epsilon)
        x_new = x - step_vector
        step_size = np.linalg.norm(step_vector)
        
        if iteration % 100 == 0:
            true_f_val = f(x)
            true_h_val = h(x)
            KKT_gap = np.max([np.linalg.norm(KKT_grad), np.max(np.abs(true_h_val))])
            pbar.set_postfix({
                'physics_loss': f'{f(x):.2e}', 
                'boundary_loss': f'{true_h_x[0]:.2e}',
                'KKT_gap': f'{KKT_gap:.2e}'
            })
            history.append((true_f_val, KKT_gap, np.abs(true_h_val[0])))

        if step_size < tol:
            print(f"\nConvergence at iter {iteration}: Step size ({step_size:.2e}) < tol ({tol:.2e}).")
            break

        x = x_new
        
        if np.isnan(x).any():
            print("\nError: NaN values detected. Stopping.")
            break
            
    return x, history

# --- Main Execution ---

# 1. Load Data
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

# 3. Setup and Run Causal Training Curriculum
x_current = get_flat_params(pinn_model.net)
stages = [(0.33, 10000), (0.66, 10000), (1.0, 10000)]
full_history = []

Kp = np.array([[500.0]])
Ki = 0.01
eta = 1e-4 
tol = 1e-8
weight_decay = 1e-4 # --- NEW: Set a small weight decay value ---

for i, (t_max, iters) in enumerate(stages):
    print(f"\n--- Causal Training: Stage {i+1}/{len(stages)} (t <= {t_max}) ---")

    X_f_stage_np = X_f_train_np[X_f_train_np[:, 1] <= t_max]
    
    pinn_model.set_data_for_stage(
        pinn_model.X_u_master,
        pinn_model.u_master,
        torch.tensor(X_f_stage_np, dtype=torch.float32).to(device)
    )

    x_current, history_stage = SOFL_Adam_PINN(
        f, h, df, dh,
        x_current, Kp, Ki=Ki, eta=eta, 
        max_iter=iters, tol=tol, weight_decay=weight_decay
    )
    full_history.extend(history_stage)

x_final = x_current

# 4. Final Evaluation using the full master dataset
set_flat_params(pinn_model.net, x_final)
pinn_model.set_data_for_stage(pinn_model.X_u_master, pinn_model.u_master, pinn_model.X_f_master)
final_physics_loss = f(x_final)
final_boundary_loss = h(x_final)[0]

print("\n--- Optimization Finished ---")


save_model(pinn_model.net, "./test_model/trained_pinn_model.pth")

# --- Evaluation & Plotting ---
# ... (The rest of the script is the same)
# print("\n--- Evaluation ---")
# data = scipy.io.loadmat('burgers_shock.mat')
# t_exact = data['t'].flatten()[:, None]; x_exact = data['x'].flatten()[:, None]
# Exact = np.real(data['usol']); T, X = np.meshgrid(t_exact, x_exact)
# X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
# u_truth = Exact.flatten()[:,None]

# load_model(pinn_model.net, "./trained_pinn_model.pth")
# x_full = torch.from_numpy(X_star[:, 0:1]).float().to(device)
# t_full = torch.from_numpy(X_star[:, 1:2]).float().to(device)
# with torch.no_grad():
#     u_pred = pinn_model.net_u(x_full, t_full).cpu().numpy()

# l2_loss_mse = np.mean((u_truth - u_pred)**2)
# print(f"Final L2 Loss (MSE): {l2_loss_mse:.4f}")
# error = np.linalg.norm(u_truth - u_pred, 2) / np.linalg.norm(u_truth, 2)
# print(f"Final L2 Relative Error: {error:.4f}")

# --- Evaluation ---
print("\n--- Evaluation ---")
data = scipy.io.loadmat('burgers_shock.mat')
t_exact = data['t'].flatten()[:, None]
x_exact = data['x'].flatten()[:, None]
Exact = np.real(data['usol'])
T, X = np.meshgrid(t_exact, x_exact)
X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
u_truth = Exact.flatten()[:, None]

# Define the shock wave region (example: x in [-0.2, 0.2] and t in [0.3, 0.7])
shock_wave_region = (X.flatten() >= -0.2) & (X.flatten() <= 0.2) & \
                    (T.flatten() >= 0.3) & (T.flatten() <= 0.7)

# Exclude the shock wave region
X_star_no_shock = X_star[~shock_wave_region]
u_truth_no_shock = u_truth[~shock_wave_region]

# Convert to tensors
x_full = torch.from_numpy(X_star_no_shock[:, 0:1]).float().to(device)
t_full = torch.from_numpy(X_star_no_shock[:, 1:2]).float().to(device)

# Evaluate the model
with torch.no_grad():
    u_pred_no_shock = pinn_model.net_u(x_full, t_full).cpu().numpy()

# Calculate the L2 loss and relative error
l2_loss_mse_no_shock = np.mean((u_truth_no_shock - u_pred_no_shock) ** 2)
l2_relative_error_no_shock = np.linalg.norm(u_truth_no_shock - u_pred_no_shock, 2) / np.linalg.norm(u_truth_no_shock, 2)

# Print the results
print(f"Final L2 Loss (MSE) excluding shock wave: {l2_loss_mse_no_shock:.4f}")
print(f"Final L2 Relative Error excluding shock wave: {l2_relative_error_no_shock:.4f}")

def residual_loss(model):
    # Exclude shock wave region
    shock_wave_region = (model.X_f_full[:, 0] >= -0.2) & (model.X_f_full[:, 0] <= 0.2) & \
                        (model.X_f_full[:, 1] >= 0.3) & (model.X_f_full[:, 1] <= 0.7)
    X_f_no_shock = model.X_f_full[~shock_wave_region]

    # Calculate physics loss on the remaining points
    f_pred = model.net_f(X_f_no_shock[:, 0:1], X_f_no_shock[:, 1:2])
    null_f = torch.zeros((X_f_no_shock.shape[0], 1), device=device)
    return model.loss(f_pred, null_f)

def boundary_loss(model):
    # Exclude shock wave region
    shock_wave_region = (model.X_u_full[:, 0] >= -0.2) & (model.X_u_full[:, 0] <= 0.2) & \
                        (model.X_u_full[:, 1] >= 0.3) & (model.X_u_full[:, 1] <= 0.7)
    X_u_no_shock = model.X_u_full[~shock_wave_region]
    u_no_shock = model.u_full[~shock_wave_region]

    # Calculate boundary loss on the remaining points
    u_pred = model.net_u(X_u_no_shock[:, 0:1], X_u_no_shock[:, 1:2])
    return model.loss(u_pred, u_no_shock)


# Final evaluation excluding shock wave
final_physics_loss = residual_loss(pinn_model).item()
final_boundary_loss = boundary_loss(pinn_model).item()

print("\n--- Optimization Finished ---")
print(f"Final Physics Loss (Objective, excluding shock wave): {final_physics_loss:.4f}")
print(f"Final Boundary Loss (Constraint, excluding shock wave): {final_boundary_loss:.4f}")

# # --- Plotting ---


print("\nPlotting the solution ...")
os.makedirs("./figures/solution", exist_ok=True)
x_plot = torch.linspace(-1, 1, 256).to(device); t_plot = torch.linspace(0, 1, 100).to(device)
X_grid_plot, T_grid_plot = torch.meshgrid(x_plot, t_plot, indexing='ij')
xcol_pred = X_grid_plot.reshape(-1, 1); tcol_pred = T_grid_plot.reshape(-1, 1)
with torch.no_grad():
    usol_pred = pinn_model.net_u(xcol_pred, tcol_pred)
Unp_pred = usol_pred.reshape(x_plot.numel(), t_plot.numel()).cpu().numpy()
xnp_plot = x_plot.cpu().numpy(); tnp_plot = t_plot.cpu().numpy()

print("Generating plot...")
plt.rcParams['font.size'] = '15'; fig = plt.figure(figsize=(5, 6)); ax = fig.add_subplot(111)
plt.xlabel(r"$t$"); plt.ylabel(r"$x$"); plt.title(r"SOFL-PINN $u(x,t)$")
img_handle = ax.imshow(Unp_pred, interpolation='nearest', cmap='rainbow',
                       extent=[tnp_plot.min(), tnp_plot.max(), xnp_plot.min(), xnp_plot.max()],
                       origin='lower', aspect='auto', vmin=-1.0, vmax=1.0)          
divider = make_axes_locatable(ax); cax = divider.append_axes("right", size="5%", pad=0.10)
cbar = fig.colorbar(img_handle, cax=cax); cbar.ax.tick_params(labelsize=10)
output_filename = "./figures/solution/SOFL_PINN_solution.png"
plt.savefig(output_filename, dpi=300, bbox_inches='tight', pad_inches=0.1)
print(f"Plot saved successfully as {output_filename}")
plt.show()

# --- Error Plot ---
print("Generating error plot...")
error_grid = abs(Exact - Unp_pred)   # absolute error
# Optionally: relative error
# error_grid = (Exact - Unp_pred) / (np.abs(Exact) + 1e-8)

plt.rcParams['font.size'] = '15'
fig_err = plt.figure(figsize=(5, 6))
ax_err = fig_err.add_subplot(111)
plt.xlabel(r"$t$")
plt.ylabel(r"$x$")
plt.title(r"Error")

img_handle_err = ax_err.imshow(
    error_grid,
    interpolation='nearest',
    cmap='bwr',
    extent=[tnp_plot.min(), tnp_plot.max(), xnp_plot.min(), xnp_plot.max()],
    origin='lower',
    aspect='auto',
    vmin=0.0, vmax=1.0
)

divider_err = make_axes_locatable(ax_err)
cax_err = divider_err.append_axes("right", size="5%", pad=0.10)
cbar_err = fig_err.colorbar(img_handle_err, cax=cax_err)
cbar_err.ax.tick_params(labelsize=10)

output_filename_err = "./figures/solution/SOFL_PINN_error.png"
plt.savefig(output_filename_err, dpi=300, bbox_inches='tight', pad_inches=0.1)
print(f"Error plot saved successfully as {output_filename_err}")
plt.show()