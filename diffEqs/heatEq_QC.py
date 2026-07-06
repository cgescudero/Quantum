#disclaimer: this workflow the optimizer is trained against the exact classical solution. This makes it an excellent proof of concept for quantum state representation and optimization, but it does not eliminate the need for the classical solver.


#Step 1: Build the Laplacian

import numpy as np

def laplacian_matrix(N, alpha, dx):

    L = np.zeros((N,N))

    for i in range(N):

        L[i,i] = -2

        if i > 0:
            L[i,i-1] = 1

        if i < N-1:
            L[i,i+1] = 1

    return alpha/dx**2 * L


N = 16  #L is 16x16   tiene que ser 2**n_qubits


#Step 2: Initial condition
x = np.linspace(0,1,N)

u0 = np.exp(-50*(x-0.5)**2)

#normalization  sum |u_i|^2 = 1 . Unlike the physical temperature profile, the quantum state must always be normalized
u0 /= np.linalg.norm(u0)


#Step 3: Encode into qubits: |phi_0> = sum_i u_i*|i>

import pennylane as qml

n_qubits = 4

dev = qml.device("default.qubit",wires=n_qubits)


#The circuit begins with
@qml.qnode(dev)

def state():

    qml.AmplitudeEmbedding(
        u0,
        wires=range(n_qubits),
        normalize=True
    )

    return qml.state()


#Step 4: Variational circuit: adding trainable gates.

def ansatz(params):

    for i in range(n_qubits):

        qml.RY(params[i], wires=i)

    # Entangle adjacent qubits dynamically
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i + 1])

    for i in range(n_qubits):

        qml.RY(params[n_qubits+i], wires=i)


#Step 5: The full circuit: the optimizer adjusts params so the output state approximates the desired evolved state.
@qml.qnode(dev)

def circuit(params):

    qml.AmplitudeEmbedding(
        u0,
        wires=range(n_qubits),
        normalize=True
    )

    ansatz(params)

    return qml.state()

########################################
#Step 6: Classical target
#Compute the reference solution (ver los solvers clasicos): the normalization is needed because quantum states must have unit norm

def apply_boundary_conditions(u_array, t):
    """
    Applies Dirichlet boundary conditions at the endpoints. 
    Returns a copied array to prevent mutating the input profile in-place.
    """
    u_bc = u_array.copy()  # Ensure independence
    u_bc[0] = 0.0          # Left boundary: u(0, t) = 0
    u_bc[-1] = 0.0         # Right boundary: u(L, t) = 0
    return u_bc

from classicalMethods import solve_matrix_exponential
L = 1.0              # Length of the rod
Nx = N               # Match spatial grid points to the quantum state dimension (N = 8)
alpha = 0.01         # Thermal diffusivity
t_final = 2.0        # Final simulation time

dx = L / (Nx - 1)
x_grid = np.linspace(0, L, Nx)

# For explicit stability: r = alpha * dt / dx^2 <= 0.5
r_stable = 0.4
dt = r_stable * (dx**2) / alpha
Nt = int(t_final / dt)

expm_snapshots, expm_times = solve_matrix_exponential(x_grid, u0, alpha, dt, dx, Nt, apply_boundary_conditions)
u_exact = expm_snapshots[-1]   # expm(L*t) @ u0

u_exact /= np.linalg.norm(u_exact)
#################################################

#Step 7: Cost function

#A simple and effective choice is the state fidelity: F=∣⟨ψ_exact | ψ(θ)⟩∣^2
#When the cost reaches zero, the quantum state matches the classical reference (up to a global phase).

from pennylane import numpy as pnp

def cost(params):

    psi = circuit(params)

    fidelity = pnp.abs(pnp.sum(pnp.conj(u_exact) * psi))**2

    return 1-fidelity


#Step 8: Optimization: The optimizer should steadily increase the fidelity.

opt = qml.AdamOptimizer(0.05)

params = pnp.array(np.random.randn(2 * n_qubits), requires_grad=True)

for i in range(260):

    params = opt.step(cost, params)

    if i % 20 == 0:
        print(i, cost(params))


#Step 9: Compare solutions
psi = circuit(params)


import matplotlib.pyplot as plt
plt.plot(np.abs(psi), label="Quantum")
plt.plot(np.abs(u_exact), "--", label="Exact")
plt.legend()
plt.show()