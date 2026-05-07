import argparse
import os
import random as rm
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import torch
import torch.nn as nn
from tqdm import trange


# ============================================================
# 0) Reproducibility + device
# ============================================================
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
rm.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ============================================================
# 1) Taylor-Green vortex setup (same as FL-PINN)
# ============================================================
T_MAX = 1.0
nu = 0.01

N_COLL = 10000
N_IC = 1000
N_BC = 1000


# ============================================================
# 2) Exact solution
# ============================================================
def taylor_green_u(x, y, t):
    return -torch.cos(x) * torch.sin(y) * torch.exp(-2.0 * nu * t)


def taylor_green_v(x, y, t):
    return torch.sin(x) * torch.cos(y) * torch.exp(-2.0 * nu * t)


def taylor_green_p(x, y, t):
    return -0.25 * (torch.cos(2.0 * x) + torch.cos(2.0 * y)) * torch.exp(-4.0 * nu * t)


# ============================================================
# 3) Data generation
# ============================================================
def sample_training_data():
    x_coll = 2.0 * np.pi * np.random.rand(N_COLL, 1)
    y_coll = 2.0 * np.pi * np.random.rand(N_COLL, 1)
    t_coll = T_MAX * np.random.rand(N_COLL, 1)
    X_coll = np.hstack([x_coll, y_coll, t_coll]).astype(np.float32)

    x_ic = 2.0 * np.pi * np.random.rand(N_IC, 1)
    y_ic = 2.0 * np.pi * np.random.rand(N_IC, 1)
    t_ic = np.zeros((N_IC, 1), dtype=np.float32)
    X_ic = np.hstack([x_ic, y_ic, t_ic]).astype(np.float32)

    y_bx = 2.0 * np.pi * np.random.rand(N_BC, 1)
    t_bx = T_MAX * np.random.rand(N_BC, 1)
    X_bx_l = np.hstack([np.zeros((N_BC, 1)), y_bx, t_bx]).astype(np.float32)
    X_bx_r = np.hstack([2.0 * np.pi * np.ones((N_BC, 1)), y_bx, t_bx]).astype(np.float32)

    x_by = 2.0 * np.pi * np.random.rand(N_BC, 1)
    t_by = T_MAX * np.random.rand(N_BC, 1)
    X_by_b = np.hstack([x_by, np.zeros((N_BC, 1)), t_by]).astype(np.float32)
    X_by_t = np.hstack([x_by, 2.0 * np.pi * np.ones((N_BC, 1)), t_by]).astype(np.float32)

    return X_coll, X_ic, X_bx_l, X_bx_r, X_by_b, X_by_t


# ============================================================
# 4) PINN model (same as FL-PINN)
# ============================================================
class PINN(nn.Module):
    def __init__(self, layers=(3, 64, 64, 64, 64, 3)):
        super().__init__()
        net = []
        for i in range(len(layers) - 2):
            net.append(nn.Linear(layers[i], layers[i + 1]))
            net.append(nn.Tanh())
        net.append(nn.Linear(layers[-2], layers[-1]))
        self.net = nn.Sequential(*net)

        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, xyt):
        return self.net(xyt)


class PhysicsInformedNS:
    def __init__(self, X_coll, X_ic, X_bx_l, X_bx_r, X_by_b, X_by_t, device):
        self.device = device
        self.model = PINN().to(device)
        self.mse = nn.MSELoss()

        self.X_coll = torch.tensor(X_coll, dtype=torch.float32, device=device, requires_grad=True)
        self.X_ic = torch.tensor(X_ic, dtype=torch.float32, device=device)

        self.X_bx_l = torch.tensor(X_bx_l, dtype=torch.float32, device=device)
        self.X_bx_r = torch.tensor(X_bx_r, dtype=torch.float32, device=device)
        self.X_by_b = torch.tensor(X_by_b, dtype=torch.float32, device=device)
        self.X_by_t = torch.tensor(X_by_t, dtype=torch.float32, device=device)

    def predict_uvp(self, X):
        out = self.model(X)
        return out[:, 0:1], out[:, 1:2], out[:, 2:3]

    def residual_loss(self):
        X = self.X_coll.clone().detach().requires_grad_(True)
        u, v, p = self.predict_uvp(X)

        grads_u = torch.autograd.grad(u, X, torch.ones_like(u), create_graph=True)[0]
        u_x, u_y, u_t = grads_u[:, 0:1], grads_u[:, 1:2], grads_u[:, 2:3]

        grads_v = torch.autograd.grad(v, X, torch.ones_like(v), create_graph=True)[0]
        v_x, v_y, v_t = grads_v[:, 0:1], grads_v[:, 1:2], grads_v[:, 2:3]

        grads_p = torch.autograd.grad(p, X, torch.ones_like(p), create_graph=True)[0]
        p_x, p_y = grads_p[:, 0:1], grads_p[:, 1:2]

        u_xx = torch.autograd.grad(u_x, X, torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
        u_yy = torch.autograd.grad(u_y, X, torch.ones_like(u_y), create_graph=True)[0][:, 1:2]

        v_xx = torch.autograd.grad(v_x, X, torch.ones_like(v_x), create_graph=True)[0][:, 0:1]
        v_yy = torch.autograd.grad(v_y, X, torch.ones_like(v_y), create_graph=True)[0][:, 1:2]

        continuity = u_x + v_y
        mom_x = u_t + u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
        mom_y = v_t + u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)

        zero = torch.zeros_like(continuity)
        return self.mse(continuity, zero) + self.mse(mom_x, zero) + self.mse(mom_y, zero)

    def initial_loss(self):
        x = self.X_ic[:, 0:1]
        y = self.X_ic[:, 1:2]
        t = self.X_ic[:, 2:3]
        u_true = taylor_green_u(x, y, t)
        v_true = taylor_green_v(x, y, t)

        u_pred, v_pred, _ = self.predict_uvp(self.X_ic)
        return self.mse(u_pred, u_true) + self.mse(v_pred, v_true)

    def initial_constraint_residual(self):
        """Pointwise IC residual vector for AL linear term, matching Burgers-style AL."""
        x = self.X_ic[:, 0:1]
        y = self.X_ic[:, 1:2]
        t = self.X_ic[:, 2:3]
        u_true = taylor_green_u(x, y, t)
        v_true = taylor_green_v(x, y, t)
        u_pred, v_pred, _ = self.predict_uvp(self.X_ic)
        return torch.cat([(u_pred - u_true).view(-1), (v_pred - v_true).view(-1)], dim=0)

    def periodic_boundary_loss(self):
        u_l, v_l, p_l = self.predict_uvp(self.X_bx_l)
        u_r, v_r, p_r = self.predict_uvp(self.X_bx_r)

        u_b, v_b, p_b = self.predict_uvp(self.X_by_b)
        u_t, v_t, p_t = self.predict_uvp(self.X_by_t)

        loss_x = self.mse(u_l, u_r) + self.mse(v_l, v_r) + self.mse(p_l, p_r)
        loss_y = self.mse(u_b, u_t) + self.mse(v_b, v_t) + self.mse(p_b, p_t)
        return loss_x + loss_y

    def periodic_boundary_constraint_residual(self):
        """Pointwise BC residual vector for AL linear term, matching Burgers-style AL."""
        u_l, v_l, p_l = self.predict_uvp(self.X_bx_l)
        u_r, v_r, p_r = self.predict_uvp(self.X_bx_r)
        u_b, v_b, p_b = self.predict_uvp(self.X_by_b)
        u_t, v_t, p_t = self.predict_uvp(self.X_by_t)

        return torch.cat(
            [
                (u_l - u_r).view(-1),
                (v_l - v_r).view(-1),
                (p_l - p_r).view(-1),
                (u_b - u_t).view(-1),
                (v_b - v_t).view(-1),
                (p_b - p_t).view(-1),
            ],
            dim=0,
        )


# ============================================================
# 5) AL-PINN training (same data/model setting, AL objective)
# ============================================================
def train_al_pinn(args):
    device = torch.device(f"cuda:{args.ordinal}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs("./Naiver-stokes/data", exist_ok=True)
    os.makedirs("./Naiver-stokes/figures", exist_ok=True)
    os.makedirs("./Naiver-stokes/training_data", exist_ok=True)

    X_coll, X_ic, X_bx_l, X_bx_r, X_by_b, X_by_t = sample_training_data()
    pinn_ns = PhysicsInformedNS(X_coll, X_ic, X_bx_l, X_bx_r, X_by_b, X_by_t, device)

    # Use pointwise dual variables as in burgers/forward/AL-PINN.py.
    n_ic_constraints = 2 * X_ic.shape[0]  # u and v IC constraints
    n_bc_constraints = 6 * X_bx_l.shape[0]  # (u,v,p) on x-periodic + (u,v,p) on y-periodic
    lbd_ic = torch.zeros(n_ic_constraints, device=device, requires_grad=True)
    lbd_bc = torch.zeros(n_bc_constraints, device=device, requires_grad=True)

    optimizer = torch.optim.Adam(
        [
            {"params": pinn_ns.model.parameters(), "lr": args.lr},
            {"params": [lbd_ic, lbd_bc], "lr": args.lbd_lr},
        ]
    )

    best_state = {k: v.detach().cpu().clone() for k, v in pinn_ns.model.state_dict().items()}
    best_score = np.inf

    history = {"loss_total": [], "loss_res": [], "loss_ic": [], "loss_bc": []}

    t0 = time.time()
    pbar = trange(1, args.epoch + 1, desc="AL-PINN Training", unit="iter")

    for ep in pbar:
        optimizer.zero_grad()

        loss_res = pinn_ns.residual_loss()
        loss_ic = pinn_ns.initial_loss()
        loss_bc = pinn_ns.periodic_boundary_loss()

        ic_res = pinn_ns.initial_constraint_residual()
        bc_res = pinn_ns.periodic_boundary_constraint_residual()

        # Burgers-style AL objective: MSE penalty + pointwise linear dual terms.
        loss = (
            loss_res
            + args.beta * loss_ic
            + (lbd_ic * ic_res).mean()
            + args.beta * loss_bc
            + (lbd_bc * bc_res).mean()
        )

        loss.backward()

        # Ascend on lambdas by flipping gradient sign.
        if lbd_ic.grad is not None:
            lbd_ic.grad *= -1
        if lbd_bc.grad is not None:
            lbd_bc.grad *= -1

        optimizer.step()

        loss_total_v = float(loss.item())
        loss_res_v = float(loss_res.item())
        loss_ic_v = float(loss_ic.item())
        loss_bc_v = float(loss_bc.item())

        history["loss_total"].append(loss_total_v)
        history["loss_res"].append(loss_res_v)
        history["loss_ic"].append(loss_ic_v)
        history["loss_bc"].append(loss_bc_v)

        score = loss_res_v + loss_ic_v + loss_bc_v
        if score < best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in pinn_ns.model.state_dict().items()}

        if ep % 100 == 0:
            pbar.set_postfix({"res": f"{loss_res_v:.2e}", "ic": f"{loss_ic_v:.2e}", "bc": f"{loss_bc_v:.2e}"})

    elapsed = time.time() - t0
    print(f"Training finished in {elapsed:.2f}s")

    pinn_ns.model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    torch.save(best_state, "./Naiver-stokes/data/AL_PINN_navier_stokes.pth")
    np.savez(
        "./Naiver-stokes/training_data/AL_PINN_history.npz",
        loss_total=np.array(history["loss_total"]),
        loss_res=np.array(history["loss_res"]),
        loss_ic=np.array(history["loss_ic"]),
        loss_bc=np.array(history["loss_bc"]),
    )

    return pinn_ns, history


# ============================================================
# 6) Plotting and evaluation
# ============================================================
def plot_training_history(history):
    plt.figure(figsize=(16, 4))

    plt.subplot(1, 4, 1)
    plt.plot(history["loss_total"])
    plt.yscale("log")
    plt.title("Total AL Loss")

    plt.subplot(1, 4, 2)
    plt.plot(history["loss_res"])
    plt.yscale("log")
    plt.title("Physics Loss")

    plt.subplot(1, 4, 3)
    plt.plot(history["loss_ic"])
    plt.yscale("log")
    plt.title("Initial Loss")

    plt.subplot(1, 4, 4)
    plt.plot(history["loss_bc"])
    plt.yscale("log")
    plt.title("Boundary Loss")

    plt.tight_layout()
    plt.savefig("./figures/AL_PINN_training_history.png", dpi=200, bbox_inches="tight")
    plt.show()


def evaluate_l2_uvp(model, device, t_eval=0.5, n_grid=80):
    model.eval()

    x = np.linspace(0.0, 2.0 * np.pi, n_grid)
    y = np.linspace(0.0, 2.0 * np.pi, n_grid)
    X, Y = np.meshgrid(x, y)
    T = np.full_like(X, t_eval)

    X_eval = np.stack([X.ravel(), Y.ravel(), T.ravel()], axis=1)
    X_eval_t = torch.tensor(X_eval, dtype=torch.float32, device=device)

    with torch.no_grad():
        pred = model(X_eval_t).cpu().numpy()

    U_pred = pred[:, 0].reshape(n_grid, n_grid)
    V_pred = pred[:, 1].reshape(n_grid, n_grid)
    P_pred = pred[:, 2].reshape(n_grid, n_grid)

    Xt = torch.tensor(X, dtype=torch.float32)
    Yt = torch.tensor(Y, dtype=torch.float32)
    Tt = torch.tensor(T, dtype=torch.float32)

    U_true = taylor_green_u(Xt, Yt, Tt).numpy()
    V_true = taylor_green_v(Xt, Yt, Tt).numpy()
    P_true = taylor_green_p(Xt, Yt, Tt).numpy()

    err_u = np.linalg.norm(U_pred - U_true) / (np.linalg.norm(U_true) + 1e-12)
    err_v = np.linalg.norm(V_pred - V_true) / (np.linalg.norm(V_true) + 1e-12)
    err_p = np.linalg.norm(P_pred - P_true) / (np.linalg.norm(P_true) + 1e-12)

    return float(err_u), float(err_v), float(err_p)


def save_plot_comparison_mat(model, device, t_val=0.5, n_grid=80):
    model.eval()
    x = np.linspace(0.0, 2.0 * np.pi, n_grid)
    y = np.linspace(0.0, 2.0 * np.pi, n_grid)
    X, Y = np.meshgrid(x, y)
    T = np.full_like(X, t_val)

    X_eval = np.stack([X.ravel(), Y.ravel(), T.ravel()], axis=1)
    X_eval_t = torch.tensor(X_eval, dtype=torch.float32, device=device)
    with torch.no_grad():
        pred = model(X_eval_t).cpu().numpy()

    U_pred = pred[:, 0].reshape(n_grid, n_grid)
    V_pred = pred[:, 1].reshape(n_grid, n_grid)
    Xt = torch.tensor(X, dtype=torch.float32)
    Yt = torch.tensor(Y, dtype=torch.float32)
    Tt = torch.tensor(T, dtype=torch.float32)
    U_true = taylor_green_u(Xt, Yt, Tt).numpy()
    V_true = taylor_green_v(Xt, Yt, Tt).numpy()
    diff_u = np.abs(U_pred - U_true)
    diff_v = np.abs(V_pred - V_true)

    os.makedirs("./data", exist_ok=True)
    scipy.io.savemat(
        f"./data/AL_PINN_t{t_val:.2f}.mat",
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--beta", type=float, default=0.05, help="AL penalty weight (matched to Burgers AL-PINN)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for model parameters")
    parser.add_argument("--lbd_lr", type=float, default=1e-4, help="Learning rate for dual variables")
    parser.add_argument("--epoch", type=int, default=10000, help="Number of AL training iterations")
    parser.add_argument("--ordinal", type=int, default=0, help="CUDA ordinal")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    solver, train_history = train_al_pinn(args)

    plot_training_history(train_history)
    dev = next(solver.model.parameters()).device
    for t_save in [0.00, 0.50, 1.00]:
        save_plot_comparison_mat(solver.model, dev, t_val=t_save, n_grid=80)
    errs_u, errs_v, errs_p = [], [], []
    time_slices = np.linspace(0.0, 1.0, 1000) 
    for t_slice in time_slices:
        eu, ev, ep = evaluate_l2_uvp(solver.model, dev, t_eval=t_slice)
       # print(f"t={t_slice:.1f} | Rel L2 u={eu:.6e} v={ev:.6e} p={ep:.6e}")
        errs_u.append(eu); errs_v.append(ev); errs_p.append(ep)
    # print(f"t={t_val:.2f} | L2(u)={eu:.6e}, L2(v)={ev:.6e}, L2(p)={ep:.6e}")

print("\nMean over time slices:")
print(f"L2(u) mean = {np.mean(errs_u):.6e}")
print(f"L2(v) mean = {np.mean(errs_v):.6e}")
print(f"L2(p) mean = {np.mean(errs_p):.6e}")
