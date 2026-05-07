import numpy as np
import gurobipy as gp
from gurobipy import GRB
from scipy.linalg import solve
import random
from matplotlib import pyplot as plt
import os

def SQP_eq_stochastic(Dx, Dy, f,h,df,dh,f_sub, h_sub, df_sub, dh_sub, x_start, Kp, Ki=0,  eta=0.01, max_iter=1000, tol=1e-6, batch_size = 20):
    
    x = np.array(x_start)
    history = []
    Integral = 0

    for iteration in range(max_iter):
        sub_Dx, sub_Dy = sub_sample(Dx, Dy, batch_size)
        grad_f = df_sub(x,sub_Dx,sub_Dy)   
        J_h = dh_sub(x,sub_Dx,sub_Dy)
        h_x = h_sub(x,sub_Dx,sub_Dy)
        # 
        #  for Lagrange multipliers
        JhJht = J_h @ J_h.T
        Pcontrol = -Kp @ h_x
        Integral = Integral + h_x
        Icontrol = Ki * Integral
        rhs = Pcontrol +Icontrol + J_h @ grad_f
        I = np.eye(np.shape(JhJht)[0])
        lambda_ = -solve(JhJht + 0.1*I, rhs)
       
        
        # Compute KKT conditions
        if iteration % 10 == 0:
            grad_f_true = df(x)
            J_h_true = dh(x)
            h_true = h(x)
            f_true = f(x)
            KKT_grad = grad_f_true + J_h_true.T @ lambda_
            KKT_gap = np.max([np.linalg.norm(KKT_grad),np.max(np.abs(h_true))])
            history.append((f_true, KKT_gap, np.max(np.abs(h_true))))
            print('f_true:', f_true,'h_true:', h_true)

        # Update x
        x_new = x - eta * KKT_grad

        # Check for convergence
        if np.linalg.norm(x_new - x) < tol:
            break

        x = x_new

    return x, history


def SQP_ineq_stochastic(h_stochastic, df_stochastic, dh_stochastic, sub_idx_func, f, h, df, dh, x_start, Kp, Ki=0,  eta=0.01, max_iter=1000, tol=1e-6, k=1.0, beta=0.5):
    """
    Stochastic Sequential Quadratic Programming (SQP) for inequality constraints.

    Parameters:
        f_stochastic: callable
            Stochastic objective function f(x, sub_idx).
        h_stochastic: callable
            Stochastic inequality constraint function h(x, sub_idx) <= 0.
        df_stochastic: callable
            Gradient of the stochastic objective function ∇f(x, sub_idx).
        dh_stochastic: callable
            Gradient of the stochastic inequality constraint ∇h(x, sub_idx).
        x_start: ndarray
            Initial guess for the solution.
        K: ndarray
            Penalty matrix for the constraints.
        sub_idx_func: callable
            Function to generate stochastic sub-indices for sampling.
        eta: float
            Learning rate.
        max_iter: int
            Maximum number of iterations.
        tol: float
            Tolerance for convergence.
        k: float
            Penalty parameter for the constraints.

    Returns:
        x: ndarray
            Solution vector.
        f_val: float
            Final value of the objective function.
        history: list
            History of function values and constraint violations.
    """
    x = np.array(x_start)
    history = []
    Integral = 0
    IntegralJ_h = 0

    for iteration in range(max_iter):
        # Generate stochastic sub-indices
        sub_idx = sub_idx_func(Dy, Dx)

        # Compute gradients and constraints using stochastic functions
        grad_f = df_stochastic(x, sub_idx)
        J_h = dh_stochastic(x, sub_idx)
        h_x = h_stochastic(x, sub_idx)

        IntegralJ_h = beta*IntegralJ_h + (1-beta)*J_h

        # 
        #  for Lagrange multipliers
        JhJht = IntegralJ_h @ IntegralJ_h.T
        Pcontrol = -Kp @ h_x
        Integral = Integral + h_x
        Icontrol = Ki * Integral
        rhs = Pcontrol + Icontrol + IntegralJ_h @ grad_f
        lambda_ = -solve_stochastic(JhJht, rhs)
       
        # Compute KKT conditions
        KKT_grad = df(x) + dh(x).T @ lambda_
        h_x_total = h(x)
        KKT_gap = np.max([np.linalg.norm(KKT_grad), np.abs(lambda_ @ h_x_total), np.max([np.max(h_x_total), 0])])
        history.append((f(x), KKT_gap))

        # Update x
        x_new = x - eta * (grad_f + J_h.T @ lambda_)

        # Check for convergence
        if np.linalg.norm(x_new - x) < tol:
            break

        x = x_new

    return x, history

def solve_stochastic(JhJht, rhs):
    model = gp.Model()
    model.setParam('OutputFlag', 0)  # Suppress output

    # Decision variables
    lambda_ = model.addMVar(shape=JhJht.shape[0], name="lambda", lb=0.0)

    model.setObjective(0.5 * lambda_ @ JhJht @ lambda_ + rhs @ lambda_, GRB.MINIMIZE)
    model.optimize()

    if model.status != GRB.OPTIMAL:
        raise ValueError("Gurobi did not find an optimal solution")

    lambda_ = lambda_.X
    return lambda_

def sub_sample(Dx, Dy, sample_size):
    """
    Subsamples data from Dx and Dy along the N dimension.

    Parameters:
        Dx: list of lists
            Input features, a list of C lists, each containing N samples.
        Dy: list of lists
            Labels, a list of C lists, each containing N samples.
        sample_size: int
            Number of samples to subsample from each class.

    Returns:
        sub_Dx: list of lists
            Subsampled input features.
        sub_Dy: list of lists
            Subsampled labels.
    """
    sub_Dx = []
    sub_Dy = []
    for class_idx in range(len(Dx)):
        indices = random.sample(range(len(Dx[class_idx])), sample_size)
        sub_Dx.append([Dx[class_idx][i] for i in indices])
        sub_Dy.append([Dy[class_idx][i] for i in indices])
    return sub_Dx, sub_Dy
    

def plot_optimization_history(*Histories,optimal_value=None,legends=None,linewidth=1,fontsize=20,ticksize=15,legendsize=15,savename=None):
    """
    Plots the function values and constraint violations over iterations from optimization history.
    
    Parameters:
    - history: List of tuples (function_value, constraint_violation) representing history of optimization.
    - optimal_value: Optional, the optimal value of the objective function f(x). If provided, adds a reference line.
    """
    
    
    # Plotting
    plt.figure(figsize=(14, 5))
    if optimal_value is not None:
        plt.subplot(1, 2, 1)
        plt.axhline(optimal_value, color='g', linestyle='--', label='Optimal $f(x)$',linewidth=linewidth)
    for i, history in enumerate(Histories):
        label = legends[i] if legends is not None and i < len(legends) else None   
        f_values, h_violations = zip(*history)
        N = len(f_values)
        k = N//5
        plt.subplot(1, 2, 1)
        plt.plot(f_values, label=label,linewidth=linewidth)
        plt.xlabel('Iteration',fontsize=fontsize)
        plt.ylabel('$f(x)$',fontsize=fontsize)
        plt.legend(fontsize=legendsize)
        plt.title('Function Value over Iterations',fontsize=fontsize)
        plt.grid(True)
        plt.xticks(fontsize=ticksize)
        plt.xticks(range(0, N+1, k))
        plt.yticks(fontsize=ticksize)

        # Plot constraint violations
        plt.subplot(1, 2, 2)
        plt.plot(h_violations, label=label,linewidth=linewidth)
        #plt.axhline(0, color='g', linestyle='--', label='Zero Constraint Violation')
        plt.xlabel('Iteration',fontsize=fontsize)
        plt.ylabel('KKT-gap',fontsize=fontsize)
        plt.yscale('log')
        plt.legend(fontsize=legendsize)
        plt.title('KKT Gap over Iterations',fontsize=fontsize)
        plt.grid(True)
        plt.xticks(fontsize=ticksize)
        plt.xticks(range(0, N+1, k))
        plt.yticks(fontsize=ticksize)
    plt.tight_layout()
    if savename:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(savename), exist_ok=True)
        plt.savefig(savename, bbox_inches='tight')
    plt.show()
    


def plot_optimization_history_2(*Histories,optimal_value=None,legends=None,linewidth=1,fontsize=20,ticksize=15,legendsize=15,savename=None):
    """
    Plots the function values and constraint violations over iterations from optimization history.
    
    Parameters:
    - history: List of tuples (function_value, constraint_violation) representing history of optimization.
    - optimal_value: Optional, the optimal value of the objective function f(x). If provided, adds a reference line.
    """
    
    
    # Plotting
    plt.figure(figsize=(14, 5))
    if optimal_value is not None:
        plt.subplot(1, 2, 1)
        plt.axhline(optimal_value, color='g', linestyle='--', label='Optimal $f(x)$',linewidth=linewidth)
    for i, history in enumerate(Histories):
        label = legends[i] if legends is not None and i < len(legends) else None   
        f_values, KKT_gaps, h_violations = zip(*history)
        N = len(f_values)
        k = N//5
        plt.subplot(1, 3, 1)
        plt.plot(f_values, label=label,linewidth=linewidth)
        plt.xlabel('Iteration',fontsize=fontsize)
        plt.ylabel('$f(x)$',fontsize=fontsize)
        plt.legend(fontsize=legendsize)
        plt.title('Function Value over Iterations',fontsize=fontsize)
        plt.grid(True)
        plt.xticks(fontsize=ticksize)
        plt.xticks(range(0, N+1, k))
        plt.yticks(fontsize=ticksize)

        # Plot constraint violations
        plt.subplot(1, 3, 2)
        plt.plot(KKT_gaps, label=label,linewidth=linewidth)
        #plt.axhline(0, color='g', linestyle='--', label='Zero Constraint Violation')
        plt.xlabel('Iteration',fontsize=fontsize)
        plt.ylabel('KKT-gap',fontsize=fontsize)
        plt.yscale('log')
        plt.legend(fontsize=legendsize)
        plt.title('KKT Gap over Iterations',fontsize=fontsize)
        plt.grid(True)
        plt.xticks(fontsize=ticksize)
        plt.xticks(range(0, N+1, k))
        plt.yticks(fontsize=ticksize)

        # Plot constraint violations
        plt.subplot(1, 3, 3)
        plt.plot(h_violations, label=label,linewidth=linewidth)
        #plt.axhline(0, color='g', linestyle='--', label='Zero Constraint Violation')
        plt.xlabel('Iteration',fontsize=fontsize)
        plt.ylabel('Constraint Violation',fontsize=fontsize)
        plt.yscale('log')
        plt.legend(fontsize=legendsize)
        plt.title('KKT Gap over Iterations',fontsize=fontsize)
        plt.grid(True)
        plt.xticks(fontsize=ticksize)
        plt.xticks(range(0, N+1, k))
        plt.yticks(fontsize=ticksize)
    plt.tight_layout()
    if savename:
        # Ensure the directory exists if a directory is specified
        dirpath = os.path.dirname(savename)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        plt.savefig(savename, bbox_inches='tight')
    plt.show()