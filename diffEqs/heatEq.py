import numpy as np
import matplotlib.pyplot as plt
#from matplotlib.animation import FuncAnimation
from scipy.sparse import diags
from scipy.sparse.linalg import expm_multiply

import classicalMethods as cm

# (No global execution or parameter variables here to prevent side effects on import)
# ==============================================================================
# 2. Initial and Boundary Conditions (Defined Once)
# ==============================================================================
def get_initial_condition(x_grid):
    """
    Defines the initial temperature distribution at t = 0.
    Currently configured as a Gaussian pulse centered at L/2.
    """
    return np.exp(-100 * (x_grid - 0.5)**2)

def apply_boundary_conditions(u_array, t):
    """
    Applies Dirichlet boundary conditions at the endpoints. 
    Returns a copied array to prevent mutating the input profile in-place.
    """
    u_bc = u_array.copy()  # Ensure independence
    u_bc[0] = 0.0          # Left boundary: u(0, t) = 0
    u_bc[-1] = 0.0         # Right boundary: u(L, t) = 0
    return u_bc



# ==============================================================================
# 1. Grid and Domain Parameters
# ==============================================================================
L = 1.0              # Length of the rod
Nx = 51              # Number of spatial grid points
alpha = 0.01         # Thermal diffusivity
t_final = 2.0        # Final simulation time

dx = L / (Nx - 1)
x = np.linspace(0, L, Nx)

# For explicit stability: r = alpha * dt / dx^2 <= 0.5
r_stable = 0.4
dt = r_stable * (dx**2) / alpha
Nt = int(t_final / dt)
t_final = Nt * dt  # Align t_final exactly with time steps

print(f"Grid spacing (dx): {dx:.4f}")
print(f"Time step (dt):    {dt:.6f}")
print(f"Number of steps:   {Nt}")
print(f"Stability parameter (r): {alpha * dt / dx**2:.4f}\n")

# Generate the master initial temperature profile
u0_master = apply_boundary_conditions(get_initial_condition(x), 0.0)

# ==============================================================================
# 7. Execution and Verification of Independence
# ==============================================================================
# Run all schemes starting from the exact same master initial array
ftcs_snapshots, ftcs_times = cm.solve_ftcs(x, u0_master, alpha, dt, dx, Nt, apply_boundary_conditions)
btcs_snapshots, btcs_times = cm.solve_btcs(x, u0_master, alpha, dt, dx, Nt, apply_boundary_conditions)
cn_snapshots, cn_times = cm.solve_crank_nicolson(x, u0_master, alpha, dt, dx, Nt, apply_boundary_conditions)
expm_snapshots, expm_times = cm.solve_matrix_exponential(x, u0_master, alpha, dt, dx, Nt, apply_boundary_conditions)

# Verify that u0_master was not changed during executions
reference_u0 = apply_boundary_conditions(get_initial_condition(x), 0.0)
is_u0_preserved = np.array_equal(u0_master, reference_u0)

print(f"Initial condition array preserved? {is_u0_preserved} (True is correct!)")
print(f"Snapshots computed -> FTCS: {len(ftcs_snapshots)}, BTCS: {len(btcs_snapshots)}, CN: {len(cn_snapshots)}, Expm: {len(expm_snapshots)}\n")

# ==============================================================================
# 8. Graphical Comparison of the Results
# ==============================================================================
plt.figure(figsize=(10, 6))

plt.plot(x, u0_master, 'k--', label='Initial Condition (t=0)')
plt.plot(x, ftcs_snapshots[-1], 'ro-', label=f'FTCS (Explicit) at t={ftcs_times[-1]:.3f}s')
plt.plot(x, btcs_snapshots[-1], 'bs-', label=f'BTCS (Implicit) at t={btcs_times[-1]:.3f}s')
plt.plot(x, cn_snapshots[-1], 'g^-', label=f'Crank-Nicolson at t={cn_times[-1]:.3f}s')
plt.plot(x, expm_snapshots[-1], 'y*-', label=f'Matrix Exp (expm_multiply) at t={expm_times[-1]:.3f}s')

plt.xlabel('x (position)')
plt.ylabel('Temperature')
plt.title(f'Comparison of Schemes at t = {t_final:.2f} s')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
