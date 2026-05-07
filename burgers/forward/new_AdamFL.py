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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ============================================================
# 1) PINN model and Burgers losses
# ============================================================
class PhysicsInformedNN:
    def __init__(self, X_u, u, X_f):
        self.X_u_master = torch.tensor(X_u, dtype=torch.float32, device=device)
        self.u_master = torch.tensor(u, dtype=torch.float32, device=device)
        self.X_f_master = torch.tensor(X_f, dtype=torch.float32, device=device)

        self.X_u_full = self.X_u_master
        self.u_full = self.u_master
        self.X_f_full = self.X_f_master

        self.create_net()
        self.loss = nn.MSELoss()

        # Cache fixed index masks once; they do not change during training.
        t_col = self.X_u_full[:, 1:2]
        self.init_mask = torch.isclose(t_col, torch.zeros_like(t_col), atol=1e-8).squeeze(-1)
        x_col = self.X_u_full[:, 0:1]
        is_left = torch.isclose(x_col, -torch.ones_like(x_col), atol=1e-8)
        is_right = torch.isclose(x_col, torch.ones_like(x_col), atol=1e-8)
        self.boundary_mask = torch.logical_or(is_left, is_right).squeeze(-1)

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
            nn.Linear(20, 1),
        ).to(device)

    def net_u(self, x, t):
        return self.net(torch.hstack((x, t)))

    def net_f(self, x, t):
        x.requires_grad_(True)
        t.requires_grad_(True)
        u = self.net_u(x, t)
        u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
        u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True, retain_graph=True)[0]
        f = u_t + (u * u_x) - (0.01 / np.pi) * u_xx
        return f


def residual_loss(model):
    f_pred = model.net_f(model.X_f_full[:, 0:1], model.X_f_full[:, 1:2])
    zeros = torch.zeros((model.X_f_full.shape[0], 1), device=device)
    return model.loss(f_pred, zeros)


def boundary_loss(model):
    u_pred = model.net_u(model.X_u_full[:, 0:1], model.X_u_full[:, 1:2])
    return model.loss(u_pred, model.u_full)


def initial_loss(model):
    if torch.any(model.init_mask):
        X_init = model.X_u_full[model.init_mask]
        u_init = model.u_full[model.init_mask]
        u_pred = model.net_u(X_init[:, 0:1], X_init[:, 1:2])
        return model.loss(u_pred, u_init)
    return torch.tensor(0.0, device=device)


def boundary_only_loss(model):
    if torch.any(model.boundary_mask):
        X_b = model.X_u_full[model.boundary_mask]
        u_b = model.u_full[model.boundary_mask]
        u_pred = model.net_u(X_b[:, 0:1], X_b[:, 1:2])
        return model.loss(u_pred, u_b)
    return torch.tensor(0.0, device=device)


# ============================================================
# 2) Flat parameter utilities
# ============================================================
def get_flat_params(model):
    return np.concatenate([p.detach().cpu().numpy().ravel() for p in model.parameters()]).astype(np.float32)


def set_flat_params(model, flat_params):
    flat_params = flat_params.astype(np.float32)
    pointer = 0
    with torch.no_grad():
        for p in model.parameters():
            n = p.numel()
            vals = torch.from_numpy(flat_params[pointer:pointer + n]).view_as(p).to(device)
            p.copy_(vals)
            pointer += n


def get_flat_grads(model):
    grads = []
    for p in model.parameters():
        if p.grad is None:
            grads.append(np.zeros(p.numel(), dtype=np.float32))
        else:
            grads.append(p.grad.detach().cpu().numpy().ravel().astype(np.float32))
    return np.concatenate(grads)


# ============================================================
# 3) Objective/constraints wrappers
# ============================================================
pinn_model = None


def f(x):
    # Objective: physics residual loss.
    set_flat_params(pinn_model.net, x)
    return residual_loss(pinn_model).item()


def df(x):
    set_flat_params(pinn_model.net, x)
    pinn_model.net.zero_grad()
    loss = residual_loss(pinn_model)
    loss.backward()
    return get_flat_grads(pinn_model.net)


def h(x):
    # Constraints: h = [initial_loss, boundary_loss]
    set_flat_params(pinn_model.net, x)
    loss_i = initial_loss(pinn_model).item()
    loss_b = boundary_only_loss(pinn_model).item()
    return np.array([loss_i, loss_b], dtype=np.float32)


def dh(x):
    set_flat_params(pinn_model.net, x)

    pinn_model.net.zero_grad()
    loss_i = initial_loss(pinn_model)
    loss_i.backward(retain_graph=True)
    grad_i = get_flat_grads(pinn_model.net)

    pinn_model.net.zero_grad()
    loss_b = boundary_only_loss(pinn_model)
    loss_b.backward()
    grad_b = get_flat_grads(pinn_model.net)

    return np.vstack([grad_i, grad_b]).astype(np.float32)


# ============================================================
# 4) New AdamFL update (projected Adam + feedback term)
# ============================================================
def new_adamfl_pinn(
    f,
    h,
    df,
    dh,
    theta_start,
    K,
    eta=1e-4,
    max_iter=10000,
    tol=1e-8,
    beta1=0.9,
    beta2=0.999,
    eps=1e-8,
    reg=1e-6,
    grad_clip=10.0,
    jac_clip=1e3,
    feedback_clip=1e3,
    step_clip=10.0,
    max_backtrack=5,
    merit_tol=1e-6,
):
    theta = np.array(theta_start, dtype=np.float32)
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)

    history = []

    t_start = time.time()
    pbar = trange(max_iter, desc="new_AdamFL Training", unit="iter")
    for it in pbar:
        theta32 = theta.astype(np.float32, copy=False)
        grad_f = df(theta32).astype(np.float32, copy=False)
        J = dh(theta32).astype(np.float32, copy=False)
        h_val = h(theta32).astype(np.float32, copy=False)

        m_con = J.shape[0]
        I_m = np.eye(m_con, dtype=np.float32)

        # Clip Jacobian entries before Gram matrix construction to prevent overflow.
        J = np.clip(J, -jac_clip, jac_clip)
        JJt = J @ J.T
        # Use adaptive damping + pseudo-inverse to avoid noisy ill-conditioned warnings.
        cond_jjt = np.linalg.cond(JJt)
        reg_eff = max(reg, 1e-6 * min(cond_jjt, 1e8))
        inv_JJt = np.linalg.pinv(JJt + reg_eff * I_m, rcond=1e-6)

        # Apply projection without materializing dense P = I - J^T (JJ^T)^(-1) J.
        J_grad = J @ grad_f
        g_t = grad_f - (J.T @ (inv_JJt @ J_grad))

        g_norm = np.linalg.norm(g_t)
        if g_norm > grad_clip:
            g_t = g_t * (grad_clip / (g_norm + 1e-12))

        m = beta1 * m + (1.0 - beta1) * g_t
        v = beta2 * v + (1.0 - beta2) * (g_t ** 2)

        adam_step = m / (np.sqrt(v) + eps)
        J_adam = J @ adam_step
        adam_direction = adam_step - (J.T @ (inv_JJt @ J_adam))
        feedback = J.T @ inv_JJt @ (K @ h_val)
        fb_norm = np.linalg.norm(feedback)
        if fb_norm > feedback_clip:
            feedback = feedback * (feedback_clip / (fb_norm + 1e-12))

        # theta_{t+1} = theta_t - eta_t * (adam_direction - feedback)
        step = adam_direction - feedback
        step_norm = np.linalg.norm(step)
        if step_norm > step_clip:
            step = step * (step_clip / (step_norm + 1e-12))

        # Backtracking on merit = ||h||^2 to avoid divergence.
        merit_old = float(np.dot(h_val, h_val))
        eta_eff = float(eta)
        theta_new = theta - eta_eff * step
        accepted = False
        for _ in range(max_backtrack + 1):
            h_new = h(theta_new.astype(np.float32, copy=False)).astype(np.float32, copy=False)
            merit_new = float(np.dot(h_new, h_new))
            if np.isfinite(merit_new) and merit_new <= merit_old:
                accepted = True
                break
            eta_eff *= 0.5
            theta_new = theta - eta_eff * step

        if not accepted:
            # Skip update when no descent step is found.
            theta_new = theta.copy()

        if (it + 1) % 100 == 0:
            i_loss = float(abs(h_val[0]))
            b_loss = float(abs(h_val[1]))
            obj_val = float(abs(f(theta32)))
            kkt_gap = max(np.linalg.norm(g_t), np.max(np.abs(h_val)))
            elapsed = time.time() - t_start
            pbar.set_postfix({
                "obj": f"{obj_val:.2e}",
                "ic": f"{i_loss:.2e}",
                "bc": f"{b_loss:.2e}",
                "kkt": f"{kkt_gap:.2e}",
                "time_s": f"{elapsed:.1f}",
                "eta": f"{eta_eff:.1e}",
            })
            history.append((it + 1, obj_val, i_loss, b_loss, kkt_gap))

        step_taken = np.linalg.norm(theta_new - theta)
        if step_taken < tol and merit_old < merit_tol:
            print(f"\nConverged at iter {it}: step<{tol:.1e}, merit<{merit_tol:.1e}")
            theta = theta_new
            break

        if np.isnan(theta_new).any():
            print("\nNaN detected in parameters. Stopping.")
            break

        theta = theta_new

    return theta.astype(np.float32), history


# ============================================================
# 5) Data + training driver
# ============================================================
def build_burgers_forward_data(N_u=100, N_f=20000):
    x_upper = np.ones((N_u // 4, 1), dtype=np.float32)
    t_upper = np.random.rand(N_u // 4, 1).astype(np.float32)
    x_lower = -np.ones((N_u // 4, 1), dtype=np.float32)
    t_lower = np.random.rand(N_u // 4, 1).astype(np.float32)

    t_zero = np.zeros((N_u // 2, 1), dtype=np.float32)
    x_zero = -1 + 2 * np.random.rand(N_u // 2, 1).astype(np.float32)

    X_upper = np.hstack((x_upper, t_upper))
    X_lower = np.hstack((x_lower, t_lower))
    X_zero = np.hstack((x_zero, t_zero))
    X_u_train = np.vstack((X_upper, X_lower, X_zero)).astype(np.float32)

    u_upper = np.zeros((N_u // 4, 1), dtype=np.float32)
    u_lower = np.zeros((N_u // 4, 1), dtype=np.float32)
    u_zero = -np.sin(np.pi * x_zero).astype(np.float32)
    u_train = np.vstack((u_upper, u_lower, u_zero)).astype(np.float32)

    idx = np.arange(N_u)
    np.random.shuffle(idx)
    X_u_train = X_u_train[idx, :]
    u_train = u_train[idx, :]

    X_f_train = np.random.uniform([-1, 0], [1, 1], size=(N_f, 2)).astype(np.float32)
    X_f_train = np.vstack((X_f_train, X_u_train)).astype(np.float32)

    return X_u_train, u_train, X_f_train


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_iter", type=int, default=10000)
    parser.add_argument("--eta", type=float, default=1e-4)
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--N_u", type=int, default=100)
    parser.add_argument("--N_f", type=int, default=20000)
    return parser.parse_args()


def main():
    global pinn_model
    args = parse_args()

    os.makedirs("./training_data", exist_ok=True)
    os.makedirs("./figures", exist_ok=True)
    os.makedirs("./data", exist_ok=True)

    X_u_train, u_train, X_f_train = build_burgers_forward_data(N_u=args.N_u, N_f=args.N_f)
    pinn_model = PhysicsInformedNN(X_u_train, u_train, X_f_train)

    K = np.array([[800.0, 0.0], [0.0, 800.0]], dtype=np.float32)

    print(f"\n--- new_AdamFL stage ({args.max_iter} iters) ---")
    theta0 = get_flat_params(pinn_model.net)

    t0 = time.time()
    theta_final, history = new_adamfl_pinn(
        f=f,
        h=h,
        df=df,
        dh=dh,
        theta_start=theta0,
        K=K,
        eta=args.eta,
        max_iter=args.max_iter,
        tol=args.tol,
    )
    elapsed = time.time() - t0

    set_flat_params(pinn_model.net, theta_final)

    h_final = h(theta_final)
    f_final = f(theta_final)
    print("\n--- Optimization Finished ---")
    print(f"Final objective physics loss: {f_final:.6e}")
    print(f"Final initial constraint loss: {h_final[0]:.6e}")
    print(f"Final boundary constraint loss: {h_final[1]:.6e}")
    print(f"Training time: {elapsed:.2f}s")

    model_path = "./data/new_AdamFL_burgers_forward.pth"
    torch.save(pinn_model.net.state_dict(), model_path)

    if len(history) > 0:
        epoch, obj, ic, bc, kkt = zip(*history)
        np.savez(
            "./training_data/new_AdamFL_history.npz",
            epoch=np.array(epoch),
            objective_loss=np.array(obj),
            initial_loss=np.array(ic),
            boundary_loss=np.array(bc),
            kkt_gap=np.array(kkt),
        )

        plt.figure(figsize=(10, 4))
        plt.plot(epoch, obj, label="Physics objective")
        plt.plot(epoch, ic, label="Initial constraint")
        plt.plot(epoch, bc, label="Boundary loss")
        plt.yscale("log")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig("./figures/new_AdamFL_losses.png", dpi=200, bbox_inches="tight")
        plt.close()

    # Full-grid evaluation on burgers_shock.mat
    eval_mat = "burgers_shock.mat"
    if os.path.exists(eval_mat):
        data = scipy.io.loadmat(eval_mat)
        t_exact = data["t"].flatten()[:, None]
        x_exact = data["x"].flatten()[:, None]
        exact = np.real(data["usol"])

        T, X = np.meshgrid(t_exact, x_exact)
        X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None])).astype(np.float32)
        u_truth = exact.flatten()[:, None].astype(np.float32)

        # Re-load once to guarantee metrics correspond to persisted checkpoint.
        state = torch.load(model_path, map_location=device)
        pinn_model.net.load_state_dict(state)
        pinn_model.net.eval()

        x_full = torch.from_numpy(X_star[:, 0:1]).to(device)
        t_full = torch.from_numpy(X_star[:, 1:2]).to(device)
        with torch.no_grad():
            u_pred = pinn_model.net_u(x_full, t_full).cpu().numpy().astype(np.float32)

        # Physics residual MSE on full grid.
        x_full_req = x_full.clone().detach().requires_grad_(True)
        t_full_req = t_full.clone().detach().requires_grad_(True)
        f_pred_test = pinn_model.net_f(x_full_req, t_full_req).detach().cpu().numpy().astype(np.float32)
        physics_test_loss = float(np.mean(f_pred_test ** 2))
        print(f"Test Physics Loss (MSE of residual): {physics_test_loss:.6e}")

        # Boundary MSE on x=+-1 or t=0.
        x_star = X_star[:, 0]
        t_star = X_star[:, 1]
        boundary_mask = (
            np.isclose(x_star, -1.0)
            | np.isclose(x_star, 1.0)
            | np.isclose(t_star, 0.0)
        )
        u_boundary_pred = u_pred[boundary_mask]
        u_boundary_true = u_truth[boundary_mask]
        boundary_test_loss = float(np.mean((u_boundary_pred - u_boundary_true) ** 2))
        print(f"Test Boundary Loss (MSE): {boundary_test_loss:.6e}")

        # Initial condition MSE at t=0.
        initial_mask = np.isclose(t_star, 0.0)
        u_initial_pred = u_pred[initial_mask]
        u_initial_true = u_truth[initial_mask]
        initial_test_loss = float(np.mean((u_initial_pred - u_initial_true) ** 2))
        print(f"Test Initial Loss (MSE): {initial_test_loss:.6e}")

        l2_loss_mse = float(np.mean((u_truth - u_pred) ** 2))
        print(f"Final L2 Loss (MSE): {l2_loss_mse:.6e}")

        denom = np.linalg.norm(u_truth, 2)
        l2_rel_error = float(np.linalg.norm(u_truth - u_pred, 2) / (denom + 1e-12))
        print(f"Final L2 Relative Error: {l2_rel_error:.6e}")
    else:
        print(f"Evaluation skipped: '{eval_mat}' not found in current directory.")


if __name__ == "__main__":
    main()
