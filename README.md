# DMRG for Spin-Dependent Hubbard Models

## Overview

This repository contains a Python implementation of the Density Matrix Renormalization Group (DMRG) algorithm based on Matrix Product States (MPS) and Matrix Product Operators (MPO).

The code was developed to investigate one-dimensional Hubbard-type models with spin-dependent interactions, including the constrained limit of infinite on-site repulsion and models inspired by the topological Devil's staircase observed in frustrated kagome Ising systems.

The implementation is designed to be modular, allowing users to define new Hamiltonians through custom MPO constructions while reusing the same DMRG engine.

---

## Main Features

* Matrix Product State (MPS) implementation
* Matrix Product Operator (MPO) implementation
* Finite-system DMRG algorithm
* Automatic energy minimization through local tensor optimization
* Support for custom Hamiltonians via MPO subclasses
* Convergence testing utilities
* Computation of local observables and densities
* Modular object-oriented architecture

---



### Core Files

#### `MPS.py`

Contains the `MPS` class.

Main functionalities:

* Initialization of matrix product states
* Canonicalization
* Left and right normalization
* Tensor manipulations
* Computation of expectation values, order parameters, entanglement entropy,...

---

#### `MPO.py`

Contains the `MPO` base class.

Main functionalities:

* Construction of matrix product operators
* MPO algebra utilities
* DMRG variational procedure
* Saving ground states MPS and informations into .txt or .npz files
  

---

#### `main.py`

Main execution script.

Typical tasks:

* Build a Hamiltonian MPO
* Initialize an MPS
* Run DMRG sweeps
* Compute observables
* Save results

---

#### `Conv_test.py`

Utility script used to study DMRG convergence.

Typical tests include:

* Energy convergence
* Bond-dimension dependence
* Truncation error analysis
* ED comparaison

---

#### Hamiltonian MPO Classes

Specific Hamiltonians are implemented as subclasses of the `MPO` base class.

Examples:

* Hubbard model
* Constrained models
* Spin-dependent interaction models

New Hamiltonians can be implemented by inheriting from `MPO` and defining the local MPO tensors.

---

## Usage

### Basic Example



---

mpo=Hubbard_V_int_MPO(L=50, V_r=[1,0], t=1, mu_a=0, mu_c=0)
filename = f"Hubbard_V_int_(L,D){L,D}_(t,mu_a,mu_c,V_r){t,float(mu_a),float(mu_c),mpo.V_r}.txt"
mpo.store_MPS(D=50,sweeps=3, tol=1e-8, zip=True, twosite=True)
mpo.store_zip_informations(filename)
mps,_= mpo.get_zip_MPS(filename)

print(infos["c"])
print(mpo.energy(mps))

---


## Example

A typical workflow consists of:

1. Choosing a Hamiltonian MPO.
2. (Running DMRG sweeps until convergence) and saving MPS into a file.
4. Computing observables:

   * Energy
   * Local densities
   * Correlation functions
   * Species ratios
5. Visualizing the resulting phase diagram.


---

## Notes

### Numerical Accuracy

The quality of the results depends strongly on the chosen bond dimension `χ` 
(or equivalently the tolerence above which the singular values are kept).

For critical or highly entangled states, larger bond dimensions may be required.

---

### Convergence

Always verify:

* Energy convergence
* Variance of the Hamiltonian
* Stability with respect to bond dimension

before drawing physical conclusions.

---

### Extending the Code

New Hamiltonians can be added by creating subclasses of the `MPO` class and implementing the corresponding local MPO tensors.

The DMRG engine is independent of the specific Hamiltonian and can therefore be reused without modification.

---

## References

S. R. White,
*Density Matrix Formulation for Quantum Renormalization Groups*,
Phys. Rev. Lett. **69**, 2863 (1992).

U. Schollwöck,
*The Density-Matrix Renormalization Group in the Age of Matrix Product States*,
Annals of Physics **326**, 96 (2011).

---

## License

This project is released under the GNU GENERAL PUBLIC LICENSE.
