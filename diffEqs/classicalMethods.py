import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import expm_multiply

# ==============================================================================
# 3. FTCS (Forward Time, Central Space) - Explicit Scheme
# ==============================================================================
def solve_ftcs(x_grid, u0, alpha, dt, dx, Nt, bc_func):
    """
    Solves the 1D heat equation using the FTCS explicit scheme.
    """
    u = u0.copy() 
    r = alpha * dt / dx**2
    snapshots = [u.copy()]
    times = [0.0]
    
    for n in range(Nt):
        u_new = u.copy()
        u_new[1:-1] = u[1:-1] + r * (u[2:] - 2*u[1:-1] + u[:-2])
        u_new = bc_func(u_new, (n + 1) * dt) #apply_boundary_conditions(u_new, (n + 1) * dt)
        u = u_new
        
        save_interval = max(1, Nt // 10)
        if (n + 1) % save_interval == 0 or n == Nt - 1:
            snapshots.append(u.copy())
            times.append((n + 1) * dt)
            
    return snapshots, times

# ==============================================================================
# 4. BTCS (Backward Time, Central Space) - Implicit Scheme
# ==============================================================================
def solve_btcs(x_grid, u0, alpha, dt, dx, Nt, bc_func):
    """
    Solves the 1D heat equation using the BTCS implicit scheme.
    """
    u = u0.copy()
    Nx = len(x_grid)
    r = alpha * dt / dx**2
    snapshots = [u.copy()]
    times = [0.0]
    
    A = np.zeros((Nx, Nx))
    for i in range(1, Nx - 1):
        A[i, i-1] = -r
        A[i, i] = 1 + 2*r
        A[i, i+1] = -r
    A[0, 0] = 1.0
    A[-1, -1] = 1.0
    
    for n in range(Nt):
        b = u.copy()
        b = bc_func(b, (n + 1) * dt)
        u_new = np.linalg.solve(A, b)
        u = u_new
        
        save_interval = max(1, Nt // 10)
        if (n + 1) % save_interval == 0 or n == Nt - 1:
            snapshots.append(u.copy())
            times.append((n + 1) * dt)
            
    return snapshots, times

# ==============================================================================
# 5. Crank-Nicolson - Semi-Implicit Scheme
# ==============================================================================
def solve_crank_nicolson(x_grid, u0, alpha, dt, dx, Nt, bc_func):
    """
    Solves the 1D heat equation using the Crank-Nicolson semi-implicit scheme.
    """
    u = u0.copy()
    Nx = len(x_grid)
    r = alpha * dt / dx**2
    snapshots = [u.copy()]
    times = [0.0]
    
    A = np.zeros((Nx, Nx))
    B = np.zeros((Nx, Nx))
    for i in range(1, Nx - 1):
        A[i, i-1] = -r / 2.0
        A[i, i] = 1.0 + r
        A[i, i+1] = -r / 2.0
        
        B[i, i-1] = r / 2.0
        B[i, i] = 1.0 - r
        B[i, i+1] = r / 2.0
        
    A[0, 0] = 1.0
    A[-1, -1] = 1.0
    B[0, 0] = 1.0
    B[-1, -1] = 1.0
    
    for n in range(Nt):
        b = B.dot(u)
        b = bc_func(b, (n + 1) * dt)
        u_new = np.linalg.solve(A, b)
        u = u_new
        
        save_interval = max(1, Nt // 10)
        if (n + 1) % save_interval == 0 or n == Nt - 1:
            snapshots.append(u.copy())
            times.append((n + 1) * dt)
            
    return snapshots, times

# ==============================================================================
# 6. Matrix Exponential - expm_multiply Approach
# ==============================================================================
def solve_matrix_exponential(x_grid, u0, alpha, dt, dx, Nt, bc_func):
    """
    Solves the 1D heat equation using the matrix exponential (expm_multiply) approach.
    Since boundary conditions are homogeneous Dirichlet (0), we solve the ODE system:
        d(u_interior)/dt = M * u_interior.
    """
    u = u0.copy()
    Nx = len(x_grid)
    snapshots = [u.copy()]
    times = [0.0]
    
    # Evolve the interior grid points
    u_int = u[1:-1]
    n_int = Nx - 2
    
    # Construct the tridiagonal matrix M for the interior points
    coeff = alpha / dx**2
    diagonals = [
        coeff * np.ones(n_int - 1),   # sub-diagonal
        -2 * coeff * np.ones(n_int),  # main diagonal
        coeff * np.ones(n_int - 1)    # super-diagonal
    ]
    M = diags(diagonals, [-1, 0, 1], format='csr')
    
    save_interval = max(1, Nt // 10)
    for n in range(Nt):
        # Evolve interior points by dt using expm_multiply
        u_int = expm_multiply(dt * M, u_int)
        
        # Reconstruct full solution with boundaries
        u_new = np.zeros(Nx)
        u_new[1:-1] = u_int
        u_new = bc_func(u_new, (n + 1) * dt)
        u = u_new
        
        if (n + 1) % save_interval == 0 or n == Nt - 1:
            snapshots.append(u.copy())
            times.append((n + 1) * dt)
            
    return snapshots, times

#if __name__ == "__main__":