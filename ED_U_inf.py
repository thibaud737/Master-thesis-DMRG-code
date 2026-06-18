"""
Exact Diagonalization (ED) for the Spin-1 Hubbard Chain


This code have been made by AI and approved by T. Lesquereux
"""


import numpy as np
import itertools
from scipy.sparse.linalg import eigsh

# local definitions

EMPTY = 0
A_OCC = 1
C_OCC = 2


# fermion number per site
def n_a(state):
    return (state == A_OCC)

def n_c(state):
    return (state == C_OCC)


# Jordan-Wigner sign

def jw_sign(config, i, j):
    """
    Fermionic sign between sites i and j (i < j)
    Counts total fermions between them.
    """
    if i > j:
        i, j = j, i

    parity = 0
    for k in range(i+1, j):
        if config[k] != EMPTY:
            parity ^= 1
    return -1 if parity else 1

# basis construction

def build_basis(L):
    """
    Full basis with U=inf constraint already built-in.
    Size = 3^L
    """
    return list(itertools.product([EMPTY, A_OCC, C_OCC], repeat=L))


# Hamiltonian builder

def build_hamiltonian(L, t, mu_a, mu_c, V_r):
    """
    Exact Hamiltonian matrix.
    Open boundary conditions.
    """
    basis = build_basis(L)
    dim = len(basis)
    index = {state: i for i, state in enumerate(basis)}

    H = np.zeros((dim, dim), dtype=float)

    V1, V2 = V_r

    # loop over basis states
    for p, config in enumerate(basis):

        # onsite chemical potential
        energy = 0.0
        for i in range(L):
            energy -= mu_a * n_a(config[i])
            energy -= mu_c * n_c(config[i])
        # density interactions
        for i in range(L):

            if i + 1 < L:
                energy += V1 * (
                    n_a(config[i]) * n_a(config[i+1]) +
                    n_c(config[i]) * n_c(config[i+1])
                )

            if i + 2 < L:
                energy += V2 * (
                    n_a(config[i]) * n_a(config[i+2]) +
                    n_c(config[i]) * n_c(config[i+2])
                )

        H[p, p] += energy

        # hopping terms
        for i in range(L-1):

            # ----- species a -----
            if config[i] == A_OCC and config[i+1] == EMPTY:
                new = list(config)
                new[i] = EMPTY
                new[i+1] = A_OCC
                sign = jw_sign(config, i, i+1)
                q = index[tuple(new)]
                H[p, q] += -t * sign

            if config[i] == EMPTY and config[i+1] == A_OCC:
                new = list(config)
                new[i] = A_OCC
                new[i+1] = EMPTY
                sign = jw_sign(config, i, i+1)
                q = index[tuple(new)]
                H[p, q] += -t * sign

            # ----- species c -----
            if config[i] == C_OCC and config[i+1] == EMPTY:
                new = list(config)
                new[i] = EMPTY
                new[i+1] = C_OCC
                sign = jw_sign(config, i, i+1)
                q = index[tuple(new)]
                H[p, q] += -t * sign

            if config[i] == EMPTY and config[i+1] == C_OCC:
                new = list(config)
                new[i] = C_OCC
                new[i+1] = EMPTY
                sign = jw_sign(config, i, i+1)
                q = index[tuple(new)]
                H[p, q] += -t * sign

    return H, basis

def fermion_counts(state_vec, basis, eps=1e-14):
    """
    Compute <N_a>, <N_c>, and ratio from a quantum state.

    Parameters
    ----------
    state_vec : ndarray
        State vector in ED basis (normalized or not).
    basis : list of tuples
        Basis from build_basis.
    eps : float
        Small cutoff to avoid division by zero.

    Returns
    -------
    Na, Nc, ratio
    """

    prob = np.abs(state_vec)**2

    Na = 0.0
    Nc = 0.0

    for p, config in enumerate(basis):
        weight = prob[p]

        # count particles in this basis state
        na_site = sum(1 for s in config if s == A_OCC)
        nc_site = sum(1 for s in config if s == C_OCC)

        Na += weight * na_site
        Nc += weight * nc_site

    ratio = Na / Nc if Nc > eps else np.inf

    return Na, Nc, ratio

def ground_state_counts(H, basis):
    """
    Diagonalize and compute particle numbers in ground state.
    """
    evals, evecs = eigsh(H, k=2, which="SA")#np.linalg.eigh(H)
    gs = evecs[:, 0]
    gs1= evecs[:,1]
    Na, Nc, ratio = fermion_counts(gs, basis)

    return evals[0],evals[1], Na, Nc, ratio, gs, gs1

L = 6
t = 1.0
mu_a = -1.0
mu_c = -1.0
V_r = [1.0, 1.0]

"""
H, basis = build_hamiltonian(L, t, mu_a, mu_c, V_r)

E0, Na, Nc, ratio = ground_state_counts(H, basis)

print("E0 =", E0)
print("Na =", Na)
print("Nc =", Nc)
print("Na/Nc =", ratio)
"""
"""
mu_vals = np.linspace(-2, 2, 10)

ratios = np.zeros((len(mu_vals), len(mu_vals)))
number = np.zeros((len(mu_vals), len(mu_vals)))
for i, mu_a in enumerate(mu_vals):
        for j, mu_c in enumerate(mu_vals):
            H, basis = build_hamiltonian(L, t, mu_a, mu_c, V_r)
            E0,E1, Na, Nc, r, gs, gs1 = ground_state_counts(H, basis)
            ratios[j, i] = r
            number[j,i] = Na+Nc
import matplotlib.pyplot as plt
ratios = np.array(ratios).reshape(len(mu_vals), len(mu_vals))
plt.figure(figsize=(8, 6))
plt.imshow(ratios, extent=(mu_vals[0], mu_vals[-1], mu_vals[0], mu_vals[-1]), origin='lower', aspect='auto', alpha=0.8, cmap='viridis', interpolation='nearest')
plt.colorbar(label='Ratio (a/c)')
plt.xlabel('Chemical Potential μ_a')
plt.ylabel('Chemical Potential μ_c')
plt.title('Ratio of Fermion Species c to a (Coarse Grid)')

plt.figure(figsize=(8, 6))
plt.imshow(number, extent=(mu_vals[0], mu_vals[-1], mu_vals[0], mu_vals[-1]), origin='lower', aspect='auto', alpha=0.8, cmap='viridis', interpolation='nearest')
plt.colorbar(label='Ratio (a/c)')
plt.xlabel('Chemical Potential μ_a')
plt.ylabel('Chemical Potential μ_c')
plt.title('number of Fermion (Coarse Grid)')
plt.show()


"""