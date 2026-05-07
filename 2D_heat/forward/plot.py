import matplotlib.pyplot as plt
import pandas as pd
import os

# === 1. Load Data ===
adam_log_path = "./training_logs/ADAM-PINN_training_loss.txt"
sofl_log_path = "./training_logs/SOFL-PINN_training_loss.txt"

adam_data = pd.read_csv(adam_log_path)
sofl_data = pd.read_csv(sofl_log_path, sep="\t")

# === 2. Extract Columns ===
epochs_adam = adam_data["Epoch"]
loss_ic_adam = adam_data["Loss_IC"]
loss_bc_adam = adam_data["Loss_BC"]
loss_pde_adam = adam_data["Loss_PDE"]

epochs_sofl = sofl_data["Iter"]
loss_ic_sofl = sofl_data["IC_Loss"]
loss_bc_sofl = sofl_data["BC_Loss"]
loss_pde_sofl = sofl_data["Physics_Loss"]

# === 3. Create Figure with 3 Subplots ===
plt.figure(figsize=(21, 5))

# ----- (a) Initial Condition Loss -----
ax1 = plt.subplot(1, 3, 1)
ax1.plot(epochs_adam / 100, loss_ic_adam, label="Adam-PINN", linewidth=2)
ax1.plot(epochs_sofl / 100, loss_ic_sofl, label="SOFL-PINN", linewidth=2, linestyle="--")
ax1.set_yscale("log")
ax1.set_ylim( 1e-5, 0)  # <-- set fixed y-range (log scale requires positive lower bound)
ax1.set_xlabel("Iteration (x100)", fontsize=11)
ax1.set_ylabel("f(x) (Observation Loss)", fontsize=11)
ax1.set_title("Objective Function", fontsize=13)
ax1.grid(True, linestyle="--", alpha=0.6)
ax1.legend(frameon=False, fontsize=10)

# ----- (b) Boundary Condition Loss -----
ax2 = plt.subplot(1, 3, 2)
ax2.plot(epochs_adam / 100, loss_bc_adam, label="Adam-PINN", linewidth=2)
ax2.plot(epochs_sofl / 100, loss_bc_sofl, label="SOFL-PINN", linewidth=2, linestyle="--")
ax2.set_yscale("log")
ax2.set_ylim( 1e-5, 0)
ax2.set_xlabel("Iteration (x100)", fontsize=11)
ax2.set_ylabel("Boundary Violation |h₁(x)|", fontsize=11)
ax2.set_title("Boundary Constraint", fontsize=13)
ax2.grid(True, linestyle="--", alpha=0.6)
ax2.legend(frameon=False, fontsize=10)

# ----- (c) PDE (Physics) Loss -----
ax3 = plt.subplot(1, 3, 3)
ax3.plot(epochs_adam / 100, loss_pde_adam, label="Adam-PINN", linewidth=2)
ax3.plot(epochs_sofl / 100, loss_pde_sofl, label="SOFL-PINN", linewidth=2, linestyle="--")
ax3.set_yscale("log")
ax3.set_ylim( 1e-6, 0)
ax3.set_xlabel("Iteration (x100)", fontsize=11)
ax3.set_ylabel("Physics Violation |h₂(x)|", fontsize=11)
ax3.set_title("Physics Constraint", fontsize=13)
ax3.grid(True, linestyle="--", alpha=0.6)
ax3.legend(frameon=False, fontsize=10)

# === 4. Layout and Save ===
plt.tight_layout()
os.makedirs("./figures", exist_ok=True)
plt.savefig("./figures/training_loss_comparison.png", dpi=400, bbox_inches="tight")
plt.show()
