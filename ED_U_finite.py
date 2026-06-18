"""
Exact Diagonalization (ED) for the  Fermi-Hubbard Chain


This code have been made by AI and approved by T. Lesquereux
"""


import numpy as np
from itertools import combinations
from scipy.sparse import dok_matrix, csr_matrix
from scipy.sparse.linalg import eigsh


# ---------------------------------------------------------
# Utilities
# ---------------------------------------------------------

def popcount(x):
    return x.bit_count()


def get_bit(state, site):
    return (state >> site) & 1


def fermionic_sign(state, i, j):
    """
    Sign for c_i^\dagger c_j acting on a bitstring state.
    Assumes i != j and particle exists at j and empty at i.
    """
    if i < j:
        mask = ((1 << j) - 1) ^ ((1 << (i + 1)) - 1)
    else:
        mask = ((1 << i) - 1) ^ ((1 << (j + 1)) - 1)

    n_between = popcount(state & mask)
    return -1 if (n_between % 2) else 1


def generate_sector(L, N):
    """
    Generate all bitstrings with exactly N particles.
    """
    states = []
    for occ in combinations(range(L), N):
        s = 0
        for site in occ:
            s |= (1 << site)
        states.append(s)
    return states


# ---------------------------------------------------------
# Basis
# ---------------------------------------------------------

def build_basis(L, Na=None, Nc=None):
    """
    Basis states are tuples (a_bits, c_bits).

    If Na,Nc are specified, work in fixed particle-number sector.
    Otherwise use full Hilbert space.
    """
    if Na is None:
        a_states = list(range(1 << L))
    else:
        a_states = generate_sector(L, Na)

    if Nc is None:
        c_states = list(range(1 << L))
    else:
        c_states = generate_sector(L, Nc)

    basis = [(a, c) for a in a_states for c in c_states]
    index = {state: i for i, state in enumerate(basis)}

    return basis, index


# ---------------------------------------------------------
# Hamiltonian
# ---------------------------------------------------------

def build_hamiltonian(
    L,
    t,
    mu_a,
    mu_c,
    U,
    V,
    Na=None,
    Nc=None,
    periodic=False,
):
    """
    Parameters
    ----------
    L : int
        number of sites

    t : float
        hopping amplitude

    mu_a, mu_c : float

    U : float
        onsite inter-species interaction

    V : dict
        {r : V_r}

    Na,Nc : int or None
        fixed particle sectors

    periodic : bool
        boundary conditions
    """

    basis, index = build_basis(L, Na, Nc)

    dim = len(basis)
    H = dok_matrix((dim, dim), dtype=np.float64)

    # hopping bonds
    bonds = [(i, i + 1) for i in range(L - 1)]
    if periodic and L > 2:
        bonds.append((L - 1, 0))

    for p, (a_bits, c_bits) in enumerate(basis):

        # ---------------------------------------
        # diagonal terms
        # ---------------------------------------
        E = 0.0

        for i in range(L):
            na = get_bit(a_bits, i)
            nc = get_bit(c_bits, i)

            E += -mu_a * na
            E += -mu_c * nc
            E += U * na * nc

        for r, Vr in V.items():
            max_i = L if periodic else L - r

            for i in range(max_i):
                j = (i + r) % L

                na_i = get_bit(a_bits, i)
                na_j = get_bit(a_bits, j)

                nc_i = get_bit(c_bits, i)
                nc_j = get_bit(c_bits, j)

                E += Vr * na_i * na_j
                E += Vr * nc_i * nc_j

        H[p, p] = E

        # ---------------------------------------
        # hopping for species a
        # ---------------------------------------
        for i, j in bonds:

            occ_j = get_bit(a_bits, j)
            occ_i = get_bit(a_bits, i)

            # j -> i
            if occ_j and not occ_i:
                sign =1 # fermionic_sign(a_bits, i, j)

                new_state = a_bits
                new_state ^= (1 << i)
                new_state ^= (1 << j)

                q = index[(new_state, c_bits)]

                H[p, q] += -t * sign

            # i -> j
            if occ_i and not occ_j:
                sign = 1 #fermionic_sign(a_bits, j, i)

                new_state = a_bits
                new_state ^= (1 << i)
                new_state ^= (1 << j)

                q = index[(new_state, c_bits)]

                H[p, q] += -t * sign

        # ---------------------------------------
        # hopping for species c
        # ---------------------------------------
        for i, j in bonds:

            occ_j = get_bit(c_bits, j)
            occ_i = get_bit(c_bits, i)

            # j -> i
            if occ_j and not occ_i:
                sign = 1#fermionic_sign(c_bits, i, j)

                new_state = c_bits
                new_state ^= (1 << i)
                new_state ^= (1 << j)

                q = index[(a_bits, new_state)]

                H[p, q] += -t * sign

            # i -> j
            if occ_i and not occ_j:
                sign = 1# fermionic_sign(c_bits, j, i)

                new_state = c_bits
                new_state ^= (1 << i)
                new_state ^= (1 << j)

                q = index[(a_bits, new_state)]

                H[p, q] += -t * sign

    return csr_matrix(H), basis


# ---------------------------------------------------------
# Ground state and first excited state
# ---------------------------------------------------------

def lowest_two_states(
    L,
    t,
    mu_a,
    mu_c,
    U,
    V,
    Na=None,
    Nc=None,
    periodic=False,
):
    """
    Returns:
        E0, psi0, E1, psi1, basis
    """

    H, basis = build_hamiltonian(
        L=L,
        t=t,
        mu_a=mu_a,
        mu_c=mu_c,
        U=U,
        V=V,
        Na=Na,
        Nc=Nc,
        periodic=periodic,
    )

    vals, vecs = eigsh(H, k=2, which="SA")

    order = np.argsort(vals)

    vals = vals[order]
    vecs = vecs[:, order]

    E0 = vals[0]
    E1 = vals[1]

    psi0 = vecs[:, 0]
    psi1 = vecs[:, 1]

    return E0, psi0, E1, psi1, basis


# ---------------------------------------------------------
# Example
# ---------------------------------------------------------
"""
if __name__ == "__main__":

    L = 6

    E0, psi0, E1, psi1, basis = lowest_two_states(
        L=L,
        t=1.0,
        mu_a=0.0,
        mu_c=0.0,
        U=2.0,
        V={1: 1., 2: 0.4},  # V1, V2
        Na=None,
        Nc=None,
        periodic=False,
    )

    print("Ground-state energy =", E0)
    print("First excited energy =", E1)
    print("Gap =", E1 - E0)
"""