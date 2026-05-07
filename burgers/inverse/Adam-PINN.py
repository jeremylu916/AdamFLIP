
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
import time
from random import uniform

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

def save(net, optimizer, save_path):
    # save nn
    state = {'state_dict': net.state_dict(),'optimizer': optimizer.state_dict()}
    torch.save(state, save_path)
    
def restore(restore_path):
    # restore nn
    return torch.load(restore_path)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("we are using device: {}".format(str(device)))


lambda1=2.0
lambda2=0.0
nu = 0.01/np.pi #diffusion coefficient
print("Te real 𝜆 = [", 1.0,nu,"]. Our initial guess will be 𝜆 _PINN= [",lambda1,lambda2,"]")


class PhysicsInformedNN():
    def __init__(self, X_u, u, X_f):
        # x & t from boundary conditions:
        self.x_u = torch.tensor(X_u[:, 0].reshape(-1, 1),
                                dtype=torch.float32,
                                requires_grad=True).to(device)
        self.t_u = torch.tensor(X_u[:, 1].reshape(-1, 1),
                                dtype=torch.float32,
                                requires_grad=True).to(device)

        # x & t from collocation points:
        self.x_f = torch.tensor(X_f[:, 0].reshape(-1, 1),
                                dtype=torch.float32,
                                requires_grad=True).to(device)
        self.t_f = torch.tensor(X_f[:, 1].reshape(-1, 1),
                                dtype=torch.float32,
                                requires_grad=True).to(device)

        # boundary solution:
        self.u = torch.tensor(u, dtype=torch.float32).to(device)

        # null vector to test against f:
        self.null = torch.zeros((self.x_f.shape[0], 1)).to(device)

        # initialize net:
        self.create_net()
        self.net.to(device)  # Move the model to the correct device
        
        self.loss = nn.MSELoss()

        # loss :
        self.ls = 0

        # iteration number:
        self.iter = 0

        self.lambda1 = torch.tensor([lambda1], requires_grad=True).float().to(device)
        self.lambda2 = torch.tensor([lambda2], requires_grad=True).float().to(device)
        ' Register lambda to optimize'
        self.lambda1 = nn.Parameter(self.lambda1)
        self.lambda2 = nn.Parameter(self.lambda2)

        self.net.register_parameter('lambda1', self.lambda1) 
        self.net.register_parameter('lambda2', self.lambda2)

    def create_net(self):
        """ net takes a batch of two inputs: (n, 2) --> (n, 1) """
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
            nn.Linear(20, 1))

    def init_weights(self, m):
        
        if type(m) == nn.Linear:
            torch.nn.init.xavier_normal_(m.weight, 0.1)
            m.bias.data.fill_(0.001)
        
    def net_u(self, x, t):
        u = self.net( torch.hstack((x, t)) )
        return u

    def net_f(self, x, t):
        lambda1=self.lambda1
        lambda2=self.lambda2
        u = self.net_u(x, t)
        
        u_t = torch.autograd.grad(
            u, t, 
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True)[0]
        
        u_x = torch.autograd.grad(
            u, x, 
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True)[0]
        
        u_xx = torch.autograd.grad(
            u_x, x, 
            grad_outputs=torch.ones_like(u_x),
            retain_graph=True,
            create_graph=True)[0]

        f = u_t + (lambda1 * u * u_x) - (lambda2 * u_xx)

        return f

   



v = 0.01 / np.pi         # constant in the diff. equation
N_u = 100                 # number of data points in the boundaries
N_f = 10000               # number of collocation points

x_upper = np.ones((N_u//4, 1), dtype=float)
x_lower = np.ones((N_u//4, 1), dtype=float) * (-1)
t_zero = np.zeros((N_u//2, 1), dtype=float)

t_upper = np.random.rand(N_u//4, 1)
t_lower = np.random.rand(N_u//4, 1)
x_zero = (-1) + np.random.rand(N_u//2, 1) * (1 - (-1))


X_upper = np.hstack( (x_upper, t_upper) )
X_lower = np.hstack( (x_lower, t_lower) )
X_zero = np.hstack( (x_zero, t_zero) )

X_u_train = np.vstack( (X_upper, X_lower, X_zero) )

index = np.arange(0, N_u)
np.random.shuffle(index)
X_u_train = X_u_train[index, :]

X_f_train = np.zeros((N_f, 2), dtype=float)
for row in range(N_f):
    x = uniform(-1, 1)  # x range
    t = uniform( 0, 1)  # t range

    X_f_train[row, 0] = x 
    X_f_train[row, 1] = t


X_f_train = np.vstack( (X_f_train, X_u_train) )


u_upper =  np.zeros((N_u//4, 1), dtype=float)
u_lower =  np.zeros((N_u//4, 1), dtype=float) 
u_zero = -np.sin(np.pi * x_zero)  


u_train = np.vstack( (u_upper, u_lower, u_zero) )

u_train = u_train[index, :]


pinn = PhysicsInformedNN(X_u_train, u_train, X_f_train)

''' 
True data
'''
x_flat = torch.linspace(-1, 1, 256)
t_flat = torch.linspace( 0, 1, 100)

# x & t grids:
x, t = torch.meshgrid(x_flat, t_flat)

N_f = 10000
total_points  = 25600

id_f = np.random.choice(total_points, N_f, replace=False)

# x & t columns:
xcol = x.reshape(-1, 1)
tcol = t.reshape(-1, 1)
xcol_ = xcol[::50]
tcol_ = tcol[::50]
# xcol_ = xcol[id_f]
# tcol_ = tcol[id_f]

data = scipy.io.loadmat('burgers_shock.mat')

t_ = data['t'].flatten()[:, None]
x_ = data['x'].flatten()[:, None]
Exact = np.real(data['usol'])


T, X = np.meshgrid(t_, x_)

X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
u_truth = Exact.flatten()[:, None]

obs_noise = 0.1

u_truth_vec = u_truth[::50]
# u_truth_vec = u_truth[id_f]
u_observation = u_truth_vec
# u_observation = u_truth_vec + np.random.randn(2560,1) * obs_noise # vector



#boundary loss and initial loss
def ibc_loss(model):
  u_prediction = model.net_u(model.x_u, model.t_u)
  u_loss = model.loss(u_prediction, model.u)
  return u_loss


#residual loss
def residual_loss(model):
  f_prediction = model.net_f(model.x_f, model.t_f)
  f_loss = model.loss(f_prediction, model.null)
  return f_loss

# observation loss
def observation_loss(model, u_observation):
    u_observation_torch = torch.from_numpy(u_observation).float().to(device)  # move to GPU
    xcol_torch = xcol_.to(device)
    tcol_torch = tcol_.to(device)
    u_predicted = model.net_u(xcol_torch, tcol_torch)
    observation_loss = model.loss(u_predicted, u_observation_torch)
    return observation_loss



# ADAM
pinn = PhysicsInformedNN(X_u_train, u_train, X_f_train)
optimizer = torch.optim.Adam(pinn.net.parameters(),lr= 5e-3)

best_loss = np.inf

pinn.net.train()
num_epochs = 20000

start_time = time.time()
for epoch in range(1, num_epochs + 1):
    optimizer.zero_grad()  
    
    # Compute the loss
    loss_res = residual_loss(pinn)
    #boundary loss
    loss_ics = ibc_loss(pinn)
    loss_obs = observation_loss(pinn, u_observation)
    loss =   loss_res +  loss_obs + 5* loss_ics
    
    # Backpropagation
    loss.backward()
    
    # Update parameters
    optimizer.step()

    # Print progress every 100 epochs
    if epoch % 100 == 0:
    # Ensure all variables are properly referenced
        loss_value = loss.item()

        # Create log entries
        log_entry = f"Epoch {epoch} | Loss: {loss_value:.5f} \n"
        lambda_entry = f"Epoch {epoch} | λ1: {pinn.lambda1.item():.5f} | λ2: {pinn.lambda2.item():.5f}\n"

        # Append loss log to ADAM_no_noise.txt
        # with open('Loss_ADAM_20%_noise.txt', 'a') as f_loss:
        #     f_loss.write(log_entry)

        # # Append lambda values to Lambda_ADAM_no_noise.txt
        # with open('Lambda_ADAM_20%_noise.txt', 'a') as f_lambda:
        #     f_lambda.write(lambda_entry)

        # Print for real-time monitoring
        
        print(
            'Epoch %d | Loss: %.5f |  λ_real = [1.0, %.5f] | λ_PINN = [%.5f, %.5f]' %
            (epoch, loss.item(),  nu, pinn.lambda1.item(), pinn.lambda2.item())
        )

# Training completion time
elapsed_time = time.time() - start_time
print(f'Training complete! Total time: {elapsed_time:.2f} seconds')



print("\n--- Optimization Finished ---")
# final_obs_loss = observation_loss(pinn, u_observation).item()
# final_ibc_loss = ibc_loss(pinn).item()
# final_res_loss = residual_loss(pinn).item()
final_lambda1 = pinn.lambda1.item()
final_lambda2 = pinn.lambda2.item()

# print(f"Final Observation Loss (Objective): {final_obs_loss:.6f}")
# print(f"Final Boundary Loss (Constraint 1): {final_ibc_loss:.6f}")
# print(f"Final Physics Loss (Constraint 2): {final_res_loss:.6f}")
print("--- Discovered PDE Parameters ---")
print(f"Discovered \u03BB\u2081 (True=1.0):   {final_lambda1:.6f}")
print(f"Discovered \u03BB\u2082 (True={nu:.6f}): {final_lambda2:.6f}")


# --- Evaluation on Full Test Data ---
data = scipy.io.loadmat('burgers_shock.mat')
t_exact = data['t'].flatten()[:, None]; x_exact = data['x'].flatten()[:, None]
Exact = np.real(data['usol']); T, X = np.meshgrid(t_exact, x_exact)
X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
u_truth = Exact.flatten()[:,None]


x_full = torch.from_numpy(X_star[:, 0:1]).float().to(device)
t_full = torch.from_numpy(X_star[:, 1:2]).float().to(device)
with torch.no_grad():
    u_pred = pinn.net_u(x_full, t_full).cpu().numpy()

# --- Testing physics loss (residual MSE on full grid) ---
x_full_req = x_full.clone().detach().requires_grad_(True)
t_full_req = t_full.clone().detach().requires_grad_(True)
f_pred_test = pinn.net_f(x_full_req, t_full_req).detach().cpu().numpy()
physics_test_loss = np.mean(f_pred_test ** 2)
print(f"Test Physics Loss (MSE of residual): {physics_test_loss:.4f}")

# --- Testing boundary loss (MSE on x=±1 or t=0) ---
x_star = X_star[:, 0]
t_star = X_star[:, 1]
boundary_mask = (np.isclose(x_star, -1.0) | np.isclose(x_star, 1.0) | np.isclose(t_star, 0.0))
u_boundary_pred = u_pred[boundary_mask]
u_boundary_true = u_truth[boundary_mask]
boundary_test_loss = np.mean((u_boundary_pred - u_boundary_true) ** 2)
print(f"Test Boundary Loss (MSE): {boundary_test_loss:.4f}")

# --- Testing initial loss (MSE at t=0) ---
initial_mask = np.isclose(t_star, 0.0)
u_initial_pred = u_pred[initial_mask]
u_initial_true = u_truth[initial_mask]
initial_test_loss = np.mean((u_initial_pred - u_initial_true) ** 2)
print(f"Test Initial Loss (MSE): {initial_test_loss:.4f}")

l2_loss_mse = np.mean((u_truth - u_pred)**2)
print(f"Final L2 Loss (MSE): {l2_loss_mse:.4f}")
error = np.linalg.norm(u_truth - u_pred, 2) / np.linalg.norm(u_truth, 2)
print(f"Final L2 Relative Error: {error:.4f}")



'''
plotting 
'''
print("\nPlotting the solution ...")
os.makedirs("./figures/solution", exist_ok=True)
x_plot = torch.linspace(-1, 1, 256).to(device); t_plot = torch.linspace(0, 1, 100).to(device)
X_grid_plot, T_grid_plot = torch.meshgrid(x_plot, t_plot, indexing='ij')
xcol_pred = X_grid_plot.reshape(-1, 1); tcol_pred = T_grid_plot.reshape(-1, 1)
with torch.no_grad():
    usol_pred = pinn.net_u(xcol_pred, tcol_pred)
Unp_pred = usol_pred.reshape(x_plot.numel(), t_plot.numel()).cpu().numpy()
xnp_plot = x_plot.cpu().numpy(); tnp_plot = t_plot.cpu().numpy()

# --- Save solution and error data to .mat files in ./data ---
os.makedirs('./data', exist_ok=True)
try:
    scipy.io.savemat('./data/ADAM_PINN_solution.mat', {'solution': Unp_pred, 'x': xnp_plot, 't': tnp_plot})
    print('Saved solution matrix to ./data/ADAM_PINN_solution.mat')
except Exception as e:
    print('Warning: failed to save solution .mat file:', e)

print("Generating plot...")
plt.rcParams['font.size'] = '15'; fig = plt.figure(figsize=(5, 6)); ax = fig.add_subplot(111)
plt.xlabel(r"$t$"); plt.ylabel(r"$x$"); plt.title("ADAM-PINN Solution")
img_handle = ax.imshow(Unp_pred, interpolation='nearest', cmap='rainbow',
                       extent=[tnp_plot.min(), tnp_plot.max(), xnp_plot.min(), xnp_plot.max()],
                       origin='lower', aspect='auto', vmin=-1.0, vmax=1.0)          
divider = make_axes_locatable(ax); cax = divider.append_axes("right", size="5%", pad=0.10)
cbar = fig.colorbar(img_handle, cax=cax); cbar.ax.tick_params(labelsize=10)
output_filename = "./figures/solution/ADAM_PINN_solution.png"
plt.savefig(output_filename, dpi=300, bbox_inches='tight', pad_inches=0.1)
print(f"Plot saved successfully as {output_filename}")
plt.show()

# --- Error Plot ---
print("Generating error plot...")
error_grid = abs(Exact - Unp_pred)   # absolute error
# Optionally: relative error
# error_grid = (Exact - Unp_pred) / (np.abs(Exact) + 1e-8)

# Save error grid to .mat file
try:
    scipy.io.savemat('./data/ADAM_PINN_error.mat', {'error': error_grid, 'x': xnp_plot, 't': tnp_plot})
    print('Saved error matrix to ./data/ADAM_PINN_error.mat')
except Exception as e:
    print('Warning: failed to save error .mat file:', e)

plt.rcParams['font.size'] = '15'
fig_err = plt.figure(figsize=(5, 6))
ax_err = fig_err.add_subplot(111)
plt.xlabel(r"$t$")
plt.ylabel(r"$x$")
plt.title(r"ADAM-PINN Error")

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

output_filename_err = "./figures/solution/ADAM_PINN_error.png"
plt.savefig(output_filename_err, dpi=300, bbox_inches='tight', pad_inches=0.1)
print(f"Error plot saved successfully as {output_filename_err}")
plt.show()