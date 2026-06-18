# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 16:03:07 2025

@author: thibaud Lesquereux
"""
from abc import ABC, abstractmethod
import numpy as np
from MPS import MPS
from scipy.sparse.linalg import LinearOperator, eigsh, ArpackNoConvergence
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import opt_einsum as oe
import time
import sys
import os
import threading
_=np.newaxis
import glob


class MPO():
    """Matrix Product Operator (MPO) representation."""
    
    def __init__(self, L, d=2, D=3):
        """
        Initialize a random MPO with given number of sites.
        
        Parameters:
        L : length of the system
        d : dimension of the state 
        D : Bond dimension of the matrices
        """
        self.length = L
        self.state_dim = d
        self.bond_dim = D
        self.tensors = self._create_random_mpo(L,d,D)
        
        
    def _create_random_mpo(self,L, d, D):
        """
        Parameters
        ----------
        L : length of the system
        d : dimension of the state
        D : Bond dimension of the matrices

        Returns
        -------
        mps :list of L random matrices (tensor)

        """
        mpo = [np.random.randn(1, d, d, D)]
        for i in range(1, L - 1):
            mpo.append(np.random.randn(D, d, d, D))
        mpo.append(np.random.randn(D, d, d, 1))
        return mpo

    
    
    def apply_to(self, mps):
        """
        Parameters
        ----------
        take a MPS instance

        Raises
        ------
        ValueError if the MPO and the MPS don't have the same lenght
        

        Returns
        -------
        Applies MPO to MPS, returning a new transformed MPS.

        """
        if self.length != mps.length and self.state_dim != mps.state_dim:
            raise ValueError("MPS and MPO must have the same lenght and/or state dimension.")
            
        new_mps = MPS(mps.length, mps.state_dim, mps.bond_dim)
        
        
        for i in range(self.length):
            new_mps.tensors[i] = oe.contract("ijkl ,mkn->imjln", self.tensors[i] , mps.tensors[i]) # apply MPO tensor to MPS tensor
            new_mps.tensors[i]= new_mps.tensors[i].reshape(
                new_mps.tensors[i].shape[0]*new_mps.tensors[i].shape[1] ,
                new_mps.tensors[i].shape[2] ,
                new_mps.tensors[i].shape[3]*new_mps.tensors[i].shape[4]) #reshape the resulting tensor into a MPS tensor
        
        
        return new_mps  

    
    def multiply_mpo(self, other):
        """
        Parameters
        ----------
        other : a MPO instance

        Raises
        ------
        ValueError if the two MPO don't have the same lenght

        Returns
        -------
        new_mpo : the multiplication of two MPOs .

        """
        if self.length != other.length:
            raise ValueError("MPOs must have the same lenght.")
        
        #same shit as apply_to()
        new_tensors = []
        for i in range(self.length):
            new_tensor = oe.contract("ijkl ,mkno-> imjnlo", self.tensors[i], other.tensors[i])
            new_tensors.append(new_tensor.reshape(new_tensor.shape[0]*new_tensor.shape[1], new_tensor.shape[2], new_tensor.shape[3], -1))
        
        # Create the resulting MPO
        new_mpo = MPO(self.length, self.state_dim, self.bond_dim)
        new_mpo.tensors = new_tensors
        return new_mpo
    
    
    
    
    # Overload * for MPO and for MPS
    def __mul__(self, other):
        """
        Parameters
        ----------
        other : either a MPO or a MPS

        Raises
        ------
        ValueError if MPO is multiply with smt else than MPS or MPO

        Returns
        -------
        either a new MPO wich is the multiplication of the two previous MPO
        or returning a new transformed MPS after apllying the MPO to the MPS

        """
        if isinstance(other, MPS):
            # Apply MPO to MPS
            return self.apply_to(other)
        elif isinstance(other, MPO):
            # Multiply MPO with MPO (combine their tensors)
            return self.multiply_mpo(other)
        #else:
            #raise ValueError("MPO can only be multiplied with MPS or MPO.")
            
            

    
    def left_contraction(self, mps, site):
        """
        Compute the left contraction L_{a_{ℓ},a'_{ℓ},b_{ℓ}}
        
        Parameters:
        -----------
        mps : list of MPS tensors, 
        mpo : list of MPO tensors,
        site : int,  the site ℓ where the contraction ends
        works only for ℓ>1
        Returns:
        --------
        L : np.ndarray, left contraction tensor
        """
        # Initialize left contraction at the first site
        L = oe.contract('ijk, ljmn, omp -> knp', mps.tensors[0].conj(), self.tensors[0], mps.tensors[0])
        
        # Iteratively contract up to site ℓ-1
        for i in range(1, site+1):
            L = oe.contract('ijk ,ilm ,jlno ,knp->mop', L, mps.tensors[i].conj(), self.tensors[i], mps.tensors[i])
        
        return L

        
       

    def right_contraction(self, mps, site):
        """
        Compute the right contraction R_{a_{ℓ},b_{ℓ},a'_{ℓ}}
        
        Parameters:
        -----------
        mps : list of MPS tensors,
        mpo : list of MPO tensors, 
        site : int, the site ℓ where the contraction ends
    
        Returns:
        --------
        R : np.ndarray, right contraction tensor 
        """
        L = mps.length  # Total number of sites
        
        # Initialize right contraction at the last site
        R = oe.contract('ijk, ljmn, omp -> ilo ', mps.tensors[-1].conj(), self.tensors[-1], mps.tensors[-1])
        
        # Iteratively contract from site ℓ+1 down to site ℓ
        for i in range(L-2, site-1, -1):
            R = oe.contract('ijk ,ljmn , omp, knp->ilo', mps.tensors[i].conj(), self.tensors[i], mps.tensors[i], R)
            
        return R

     
  
    
    def effective_hamiltonian(self, mps, site):
        """
        calculate the effective Hmailtonian and optimze the MPS at site
        Parameters
        ----------
        mps : MPS

        Returns
        -------
        None.

        """
        start_time = time.time()

        #contraction of MPS and MPO
        R=self.right_contraction(mps, site+1)
        L=self.left_contraction(mps,site-1)
        l,d,r=mps.tensors[site].shape

        def H_eff_matvec(v):
            """
            Parameters
            ----------
            v : vector to be multiplied by the effective Hamiltonian

            Returns
            -------
            result : H_eff @ v, the result of the multiplication
            """
            #reshape the input vector to match the tensor shape
            
            v = v.reshape(l,d,r)


            # Perform the matrix-vector multiplication
            if site == self.length - 1:
                Hv = oe.contract('ijk,jlmn,kmp->il', L, self.tensors[site], v)
            
            elif site == 0:
                Hv = oe.contract('jlmn,kmp ,onp->lo', self.tensors[site], v, R)
           
            else:
                Hv = oe.contract('ijk,jlmn,kmp ,onp->ilo', L, self.tensors[site], v, R)
            # Reshape the result back to a vector
            return Hv.flatten()
        
        # Solve the generalized eigenvalue problem
        shape = mps.tensors[site].shape
        size = shape[0]*shape[1]*shape[2]

        H_eff_op = LinearOperator((size, size), matvec=H_eff_matvec)

        mid_time = time.time()
        # Solve for the ground state
        try:
            _, eigenvectors = eigsh(H_eff_op, k=1, which='SA', v0=mps.tensors[site], tol=1e-8, maxiter=1000, ncv=min(size,40))
            ground_state_vector = eigenvectors[:, 0]
        except ArpackNoConvergence as err:
            print("Lanczos did not fully converge.")

            if (err.eigenvectors is not None and err.eigenvectors.shape[1] > 0):
                print("Using partial eigenvector.")
                ground_state_vector = err.eigenvectors[:, 0]
                 
            else:
                print("No partial eigenvector available — reusing previous tensor.")
                ground_state_vector = mps.tensors[site]
                 

        
        
        
        # Reshape back to the MPS tensor format
        mps.tensors[site]=ground_state_vector.reshape(l,d,r)
        
        end_time = time.time()
        #print(f"Time taken for effective Hamiltonian site {site+1}: {mid_time-start_time:.3f}, {end_time - mid_time:.3f} seconds")
        sys.stdout.write(f"\rSite: {site+1}")
    


    def effective_2site_hamiltonian(self,mps, site, threshold=1e-8, direction = 'right', GS=None, lamda=1e2):
        """
        calculate the effective Hmailtonian and optimze the MPS at site and site+1
        Parameters
        ----------
        mps : MPS
        site: site where the optimization is made
        threshold: the threshold above which singular values asre kept
        direction: left or right sweep
        GS: the ground state MPS for 1st excited state calculation
        lamda: orthogonality penalty for 1st excited state calculation

        Returns
        -------
        None.

        """
        start_time = time.time()

        #contraction of MPS and MPO
        R=self.right_contraction(mps, site+2)
        L=self.left_contraction(mps,site-1)

        l1,d1,r1=mps.tensors[site].shape
        l2,d2,r2=mps.tensors[site+1].shape

        def H_eff_matvec(v):
            """
            Parameters
            ----------
            v : vector to be multiplied by the effective Hamiltonian

            Returns
            -------
            result : H_eff @ v, the result of the multiplication
            """
            #reshape the input vector to match the tensor shape

            v = v.reshape(l1,d1,d2,r2)
           

            # Perform the matrix-vector multiplication
            if site == self.length - 2:
                Hv = oe.contract('ijk,jmln,npoq ,klor ->imp', L, self.tensors[site],self.tensors[site+1], v)
            
            elif site == 0:
                Hv = oe.contract('jmln,npoq ,klor, sqr->mps', self.tensors[site],self.tensors[site+1], v, R)
           
            else:
                Hv = oe.contract('ijk,jmln,npoq ,klor, sqr->imps', L, self.tensors[site],self.tensors[site+1], v, R)
            # Reshape the result back to a vector
            return Hv.flatten()
        
        # Solve the generalized eigenvalue problem
        
        size = l1*d1*d2*r2

        H_eff_op = LinearOperator((size, size), matvec=H_eff_matvec)

        if GS is not None:
            
            L2= GS.left_projector(mps,site-1)
            R2= GS.right_projector(mps,site+2)
            L3= mps.left_projector(GS,site-1)
            R3= mps.right_projector(GS,site+2)
            def projector(v):
                """
                Parameters
                ----------
                v : vector to be multiplied by the Ground state projector

                Returns
                -------
                result : P_0 @ v, the result of the multiplication
                """
                #reshape the input vector to match the tensor shape

                v = v.reshape(l1,d1,d2,r2)




                # Perform the matrix-vector multiplication
                if site == self.length - 2:
                    overlap = oe.contract('ij,ikl,lmn ,jkmo ->', L2, GS.tensors[site].conj(),GS.tensors[site+1].conj(), v)
                
                elif site == 0:
                    overlap = oe.contract('ijk, klm ,njlo ,mo ->', GS.tensors[site].conj(),GS.tensors[site+1].conj(), v, R2)
            
                else:
                    overlap = oe.contract('ij ,ikl ,lmn ,jkmo, no ->', L2, GS.tensors[site].conj(),GS.tensors[site+1].conj(), v, R2)
                

                # Perform the matrix-vector multiplication
                if site == self.length - 2:
                    Pv = oe.contract('ij,jkl,lmn ->ikm', L3, GS.tensors[site],GS.tensors[site+1])
                
                elif site == 0:
                    Pv = oe.contract('jkl,lmn,on  ->kmo', GS.tensors[site],GS.tensors[site+1], R3)
            
                else:
                    Pv= oe.contract('ij,jkl,lmn,on  ->ikmo', L3, GS.tensors[site],GS.tensors[site+1], R3)
                


                Pv *= lamda * overlap
                # Reshape the result back to a vector
                return Pv.flatten()
            

            H_eff_op += LinearOperator((size, size), matvec=projector)


        mid_time = time.time()
        # Solve for the ground state
        v0= oe.contract('ijk, klm -> ijlm', mps.tensors[site], mps.tensors[site+1])
        v0=v0.flatten()
        try:
            _, eigenvectors = eigsh(H_eff_op, k=1, which='SA', v0=v0, tol=1e-8, maxiter=1000, ncv=min(size,40))
            ground_state_vector = eigenvectors[:, 0]
        except ArpackNoConvergence as err:
            print("Lanczos did not fully converge.")

            if (err.eigenvectors is not None and err.eigenvectors.shape[1] > 0):
                print("Using partial eigenvector.")
                ground_state_vector = err.eigenvectors[:, 0]
                 
            else:
                print("No partial eigenvector available — reusing previous tensor.")
                ground_state_vector = v0

        # Reshape back to the MPS tensor format     
        ground_state_vector = ground_state_vector.reshape(l1,d1,d2,r2)
        
        
        #truncated SVD
        u, s, vh = np.linalg.svd(ground_state_vector.reshape(l1*d1, d2*r2), full_matrices=False)
        chi = max(1,np.sum(s > threshold))
        u_truncated = u[:, :chi]
        s_truncated = np.diag(s[:chi])
        vh_truncated = vh[:chi, :]
        #print(chi)
        if direction == 'right':
            mps.tensors[site] = u_truncated.reshape(l1, d1, chi)

            mps.tensors[site+1] = (s_truncated @ vh_truncated).reshape(chi, d2, r2)
            #mps.left_normalize(site+1, threshold)
        if direction == 'left':
            mps.tensors[site] = (u_truncated @ s_truncated).reshape(l1, d1, chi)

            mps.tensors[site+1] = vh_truncated.reshape(chi, d2, r2)
            #mps.right_normalize(site, threshold)
        
        end_time = time.time()
        #print(f"Time taken for effective Hamiltonian site {site+1}: {mid_time-start_time:.3f}, {end_time - mid_time:.3f} seconds")
        sys.stdout.write(f"\rSite: {site+1}")



    def right_sweep(self, mps):
        """
        Perform a right sweep in the DMRG algorithm.

        Parameters:
        -----------
        mps : The MPS instance.

        Returns:
        --------
        mps : The updated MPS after the right sweep.
            """
        
        for site in range(self.length-1):
             self.effective_hamiltonian(mps,site)
             mps.left_normalize(site)
             
        
        return mps 
           
        
                
    def left_sweep(self, mps):
        """
        Perform a left sweep in the DMRG algorithm.
        Parameters
        ----------
        mps : The MPS instance.

        Returns
        -------
        mps :The updated MPS after the left sweep.

        """
        
        for site in range(self.length-1, 0,-1):
                self.effective_hamiltonian(mps,site)
                mps.right_normalize(site)
               
        return mps 
    

    
    def right_sweep2(self, mps, threshold=1e-8, GS=None, lamda=1e2):
        """
        Perform a right sweep in the DMRG algorithm.

        Parameters:
        -----------
        mps : The MPS instance.
        threshold: the threshold above which singular values asre kept
        GS: the ground state MPS for 1st excited state calculation
        lamda: orthogonality penalty for 1st excited state calculation
        Returns:
        --------
        mps : The updated MPS after the right sweep.
            """
        
        for site in range(0,self.length-2,1):
             self.effective_2site_hamiltonian(mps,site, 
                                              threshold=threshold, direction='right', GS=GS, lamda=lamda)
             
             
        
       
           
        
                
    def left_sweep2(self, mps, threshold=1e-8, GS=None, lamda=1e2):
        """
        Perform a left sweep in the DMRG algorithm.
        Parameters
        ----------
        mps : The MPS instance.
        threshold: the threshold above which singular values asre kept
        GS: the ground state MPS for 1st excited state calculation
        lamda: orthogonality penalty for 1st excited state calculation

        Returns
        -------
        mps :The updated MPS after the left sweep.

        """
        
        for site in range(self.length-2, 0,-1):
                self.effective_2site_hamiltonian(mps,site, 
                                              threshold=threshold, direction='left', GS=GS, lamda=lamda)
             
               
        return mps 


    def variational_ground_state(self,mps, max_iter=10, tol=1e-6, twosite=True, GS=None, lamda=1e2):
        """
        Parameters
        ----------
        mps : MPS
            The MPS instance.
        max_iter : int, optional
            Maximum number of iterations/sweeps (default is 10).
        tol : float, optional
            Convergence tolerance (default is 1e-6).
    
        Returns
        -------
        energy: the ground state energy
        mps : MPS
            The MPS of the ground state.
        """
        
        mps=mps.right_canonical_form(0)
        
        for i in range(max_iter):
            if twosite is True:
                
                self.right_sweep2(mps,tol/10, GS=GS, lamda=lamda)
                mps = self.left_sweep2(mps, tol/10, GS=GS, lamda=lamda)
            else:
                mps = self.right_sweep(mps)
                mps = self.left_sweep(mps)
            
            if self.convergence(mps, tol):
                print(f"\nA convergé en {i+1} sweep ")
                break
        print(f"\nvariance of the energy: {self.var_energy(mps)}") 
        return self.energy(mps), mps


    def energy(self, mps):
        """
        <ψ|H|ψ>/<ψ|ψ>
    
        Parameters:
        -----------
        mps : MPS instance."
        
        Returns
        -------
        energy of the mps w.r.t the MPO
        """
        E=oe.contract('ijk, ljmn, omp->knp', mps.tensors[0].conj(),self.tensors[0],mps.tensors[0])
        for i in range(1, self.length):
            E=oe.contract('ilo, ijk, ljmn, omp->knp',E, mps.tensors[i].conj(),self.tensors[i],mps.tensors[i])
        
        E=E.reshape(-1)[0]/(mps*mps)
        return E
    
    
    def var_energy(self,mps):
        """
        <ψ|H^2|ψ>/<ψ|ψ>-(<ψ|H|ψ>/<ψ|ψ>)^2
        
        Parameters
        ----------
        mps : MPS instance

        Returns
        -------
        variance of the energy w.r.t the MPO
        """
        #E =  self.energy(mps)
        #H2 = self*self
        #E2= (mps*H2*mps)
        #E2=E2/(mps*mps)
        E=oe.contract('ijk, ljmn, omp->knp', mps.tensors[0].conj(),self.tensors[0],mps.tensors[0])
        E2=oe.contract('ijk,ljmn, ompq,rps->knqs', mps.tensors[0].conj(),self.tensors[0].conj(),self.tensors[0],mps.tensors[0])
        for i in range(1, self.length):
            E2=oe.contract('ilor, ijk, ljmn, ompq, rps->knqs',E2, mps.tensors[i].conj(),self.tensors[i].conj(),self.tensors[i],mps.tensors[i])
            E=oe.contract('ilo, ijk, ljmn, omp->knp',E, mps.tensors[i].conj(),self.tensors[i],mps.tensors[i])
        E2=E2.reshape(-1)[0]/(mps*mps)
        E=E.reshape(-1)[0]/(mps*mps)
        return E2-E*E
        
        
         
    def __repr__(self):
        return f"MPO(length={self.length}, tensors={self.tensors})"



    def convergence(self, mps, epsilon=1e-1):
        """
        Check if the variance of the energy is less than epsilon.
    
        Parameters:
        -----------
        mps : MPS
            The matrix product state to evaluate.
        epsilon : float
            The tolerance for convergence.
    
        Returns:
        --------
        bool
            True if the variance is less than epsilon, False otherwise.
        float
            The variance of the energy.
        """
        
        # Check if the variance is less than epsilon
        var= self.var_energy(mps)
        if np.abs(var) > epsilon:
            return False
        else:
            #print(f"\nvariance of the energy: {var}") 
            return True
    



    def spectrum_single_site(self,mps, site, num_eig=5, tol=1e-8):
        """
        calculate the effective Hmailtonian and optimze the MPS at site
        Parameters
        ----------
        mps : MPS
        site: site where the spectrum is calculated
        num_eig: number of eigenvalues calculated
        tol: tolerence on Lancsoz convergence

        Returns
        -------
        None.

        """
        mps = mps.mixed_canonical_form(site)

        #contraction of MPS and MPO
        R=self.right_contraction(mps, site+1)
        L=self.left_contraction(mps,site-1)
        l,d,r=mps.tensors[site].shape

        def H_eff_matvec(v):
            """
            Parameters
            ----------
            v : vector to be multiplied by the effective Hamiltonian

            Returns
            -------
            result : H_eff @ v, the result of the multiplication
            """
            #reshape the input vector to match the tensor shape
            
            v = v.reshape(l,d,r)
            

            # Perform the matrix-vector multiplication
            if site == self.length - 1:
                Hv = oe.contract('ijk,jlmn,kmp->il', L, self.tensors[site], v)
            
            elif site == 0:
                Hv = oe.contract('jlmn,kmp ,onp->lo', self.tensors[site], v, R)
           
            else:
                Hv = oe.contract('ijk,jlmn,kmp ,onp->ilo', L, self.tensors[site], v, R)
            # Reshape the result back to a vector
            return Hv.flatten()
        
        # Solve the generalized eigenvalue problem
        shape = mps.tensors[site].shape
        size = shape[0]*shape[1]*shape[2]

        H_eff_op = LinearOperator((size, size), matvec=H_eff_matvec)

      
        # Solve for the ground state
        v0 = mps.tensors[site].flatten()
        try:
            eigenvalues, _ = eigsh(H_eff_op, k=num_eig, which='SA', v0=v0, tol=tol, maxiter=1000, ncv=min(size,40))
        except ArpackNoConvergence as err:
            print("Lanczos did not fully converge.")
            return [float('nan')]*num_eig

        return eigenvalues
                 



    def spectrum(self,mps, site, num_eig=5, tol=1e-8):
        """
        calculate the spectrum of the effective Hmailtonian
        Parameters
        ----------
        mps : MPS
        site: site where the spectrum is calculated
        num_eig: number of eigenvalues calculated
        tol: tolerence on Lancsoz convergence

        Returns
        -------
        The spectrum of the effective Hamiltonian at given site.

        """
    
        mps = mps.mixed_canonical_form(site)
        #contraction of MPS and MPO
        R=self.right_contraction(mps, site+2)
        L=self.left_contraction(mps,site-1)

        l1,d1,r1=mps.tensors[site].shape
        l2,d2,r2=mps.tensors[site+1].shape

        def H_eff_matvec(v):
            """
            Parameters
            ----------
            v : vector to be multiplied by the effective Hamiltonian

            Returns
            -------
            result : H_eff @ v, the result of the multiplication
            """
            #reshape the input vector to match the tensor shape

            v = v.reshape(l1,d1,d2,r2)
            

            # Perform the matrix-vector multiplication
            if site == self.length - 2:
                Hv = oe.contract('ijk,jmln,npoq ,klor ->imp', L, self.tensors[site],self.tensors[site+1], v)
            
            elif site == 0:
                Hv = oe.contract('jmln,npoq ,klor, sqr->mps', self.tensors[site],self.tensors[site+1], v, R)
           
            else:
                Hv = oe.contract('ijk,jmln,npoq ,klor, sqr->imps', L, self.tensors[site],self.tensors[site+1], v, R)
            # Reshape the result back to a vector
            return Hv.flatten()
        
        # Solve the generalized eigenvalue problem
        
        size = l1*d1*d2*r2

        H_eff_op = LinearOperator((size, size), matvec=H_eff_matvec)

        
        # Solve for the ground state
        v0= oe.contract('ijk, klm -> ijlm', mps.tensors[site], mps.tensors[site+1])
        v0=v0.flatten()
        try:
            eigenvalues, _ = eigsh(H_eff_op, k=num_eig, which='SA', v0=v0, tol=tol, maxiter=1000, ncv=min(size,40))
        except ArpackNoConvergence as err:
            print("Lanczos did not fully converge.")
            return [float('nan')]*num_eig

        return eigenvalues
        
     



    def MPO_to_matrix(self):
        """
        Convert the MPO to a full matrix representation.
    
        Returns:
        --------
        np.ndarray
            The full matrix representation of the MPO.
        """
        assert self.length <= 10, "MPO too large to convert to full matrix (length > 10)"
        # Start with the first tensor, reshape to 2D
        # Shape: (1, d, d, D) -> (d, d*D)
        full_matrix = self.tensors[0].reshape(self.state_dim, self.state_dim * self.bond_dim)
        
        # Iteratively contract with the next tensors
        for i in range(1, self.length):
            # Current shape: (d^i, d^i * D)
            # Contract with tensor of shape (D, d, d, D)
            # Result should be (d^(i+1), d^(i+1) * D)
            left_dim = full_matrix.shape[0]
            bond_left = full_matrix.shape[1] // left_dim
            
            # Reshape full_matrix to (d^i, d^i, D)
            full_matrix = full_matrix.reshape(left_dim, left_dim, bond_left)
        
            # Contract: (d^i, d^i, D) with (D, d, d, D) -> (d^i, d, d^i, d, D)
            full_matrix = oe.contract('ijk, kmno -> imjno', full_matrix, self.tensors[i])

    
            # Reshape to (d^(i+1), d^(i+1), D)
            
            new_left = left_dim * self.state_dim
            actual_bond_right = self.tensors[i].shape[-1]
            full_matrix = full_matrix.reshape(new_left, new_left, actual_bond_right)
            
        # Reshape to (d^(i+1), d^(i+1) * D) for next iteration
            full_matrix = full_matrix.reshape(new_left, new_left * actual_bond_right)
        
        # Final reshape to (d^L, d^L)
        dim = self.state_dim ** self.length
        full_matrix = full_matrix.reshape(dim, dim)
        
        return full_matrix
   
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
#----------------------------------------------storage functions------------------------------------
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------

    

    def chrono(self, start_event, stop_event):
        """
        Function to display a timer in the console.
        Stops when a keyboard interrupt is detected.
        """
        
        start_time = time.time()
        while not start_event.is_set():
            time.sleep(0.01)  
        while not stop_event.is_set():
            elapsed = int(time.time() - start_time)
            minutes, seconds = divmod(elapsed, 60)
            sys.stdout.write(f"\r          Elapsed time : {minutes:02d}:{seconds:02d}")
            sys.stdout.flush()
            time.sleep(1)  # Update every second
        print("\n")  # line to clear the console at the end

    
   

    

    def save_mps_to_zip_file(self, filename, mps):
        """
        Save the MPS tensors to a compressed file .npz.
        Parameters
        ----------
        filename: the file's name where the MPS is saved
        mps: the MPS to be saved

         
        """
        directory = r"/scratch/lesquere"
        full_path = os.path.join(directory, filename + ".npz")

        os.makedirs(directory, exist_ok=True)

        # Convert tensors to float32 
        tensors = [A.astype(np.float32) for A in mps.tensors]

        # Save everything in compressed format
        np.savez_compressed(
            full_path,
            length=mps.length,
            state_dim=mps.state_dim,
            bond_dim=mps.bond_dim,
            **{f"A{i}": A.astype(np.float32) for i, A in enumerate(mps.tensors)}
        )

    def save_mps_to_file(self, filename, mps):
            """
            Save the MPS tensors to a file .txt.
            Parameters
            ----------
            filename: the file's name where the MPS is saved
            mps: the MPS to be saved

            
            """
            # Create the full path
            directory = r"/scratch/lesquere"
            full_path = os.path.join(directory, filename)
        
            # Ensure the directory exists
            os.makedirs(directory, exist_ok=True)

            # Save the MPS tensors to the file
            with open(full_path, 'w') as f:
                # Write the dimensions of the MPS
                f.write(f"{mps.length,mps.state_dim,mps.bond_dim}\n")

                # Write each tensor's shape and data
                for tensor in mps.tensors:
                    f.write(f"{tensor.shape}\n")
                    np.savetxt(f, tensor.flatten(), newline=' ')
                    #flat = tensor.flatten()
                    #f.write(" ".join(map(str, flat)) + "\n")
                    f.write('\n')

    def store_MPS(self, D=100, tol=1e-5, sweeps=3, twosite=True, zip=True, mps=None, ):
        """
        Store the MPS of a given case in a file with a timer.

        Parameters:
        ----------

        D (int): Bond dimension.
        tol (float): Tolerance for the variational optimization.
        sweeps (int): Number of sweeps for the variational optimization.
        twosite (bool): Whether to use two-site optimization.
        zip_file (bool): Whether to save the file as a compressed .npz file.
        mps (MPS): Existing MPS to use. If None, a new MPS will be created.
        

        """
        start_event = threading.Event()
        stop_event = threading.Event()
        t = threading.Thread(target=self.chrono, args=(start_event, stop_event))
        t.start()
        start_event.set()
        self.data(D, tol,sweeps, twosite, zip, mps=mps)
        stop_event.set()
        t.join()

    @staticmethod
    def get_MPS(filename, directory=r"/scratch/lesquere"):
        """
        Load the MPS tensors from a file.

        Parameters:
        filename (str): The name of the file to load the MPS from.
        directory: the directory of the file

        Returns:
        MPS: The loaded MPS object.
        BOOl: True if the file exist
        """
        filename = filename.strip()
        full_path = os.path.join(directory, filename)
        
        mps = MPS(2,2,2)
        # Ensure the directory exists
        if not os.path.exists(full_path):
            return mps, False
            #raise FileNotFoundError(f"The file {filename} does not exist.")
        
    
        with open(full_path, 'r') as f:
            # Read the first line to get the dimensions
            dim = eval(f.readline().strip())

            # Read the rest of the file to get the tensors
            tensors = []

            for line in f:
                if line.startswith("Informations\n") or line.startswith("Correlation") or line.startswith("Entropy"):
                    break
                try:
                    shape = eval(line.strip())
                    line = next(f)
                    tensor = np.fromiter(map(float, line.strip().split()), dtype=float).reshape(shape)
                except:
                    return mps, False
                #num_elements = np.prod(shape)
                #values = []

                #while len(values) < num_elements:
                    #line = next(f)
                    #values.extend(map(float, line.strip().split()))

                #tensor = np.array(values).reshape(shape)
                tensors.append(tensor)

        mps=MPS(dim[0], dim[1], dim[2])
        mps.tensors=tensors
        print("MPS loaded successfully.")
        return mps, True


    def get_zip_MPS(self, filename,  directory=r"/scratch/lesquere"):
        """
        Load an MPS from a compressed .npz file.
        Parameters:
        filename (str): The name of the file to load the MPS from.
        directory: the directory of the file

        Returns:
        MPS: The loaded MPS object.
        BOOl: True if the file exist
        """
        full_path = os.path.join(directory, filename + ".npz")

        mps = MPS(2,2,2)
        # Ensure the directory exists
        if not os.path.exists(full_path):
            return mps, False

        data = np.load(full_path)

        

        # recover metadata
        length = int(data["length"])
        state_dim = int(data["state_dim"])
        bond_dim = int(data["bond_dim"])

        # recover tensors (sorted to ensure correct order)
        tensor_keys = sorted(
            [key for key in data.files if key.startswith("A")],
            key=lambda x: int(x[1:])
        )

        tensors = [data[key].astype(np.float32) for key in tensor_keys]

        # rebuild MPS (adapt to your class constructor)
        mps = MPS(length, state_dim, bond_dim)
        mps.tensors = tensors

        print("MPS loaded successfully.")
        return mps, True

   

    def get_MPS_by_prefix(self, prefix ,suffix=None, directory=r"/scratch/lesquere"):
        """
        Load all MPS files whose filename starts with the given prefix.

        Parameters:
        prefix (str): The begining of the name of the files to load the MPS from.
        suffix (str): The end of the name of the files to load the MPS from.
        directory: the directory of the file


        Returns a list of MPS objects.
        """
        if suffix:
            pattern = os.path.join(directory, prefix + "*" + suffix)
        else:
            pattern = os.path.join(directory, prefix + "*")
        files = sorted(glob.glob(pattern))

        if not files and not suffix:
            raise FileNotFoundError(f"No files starting with '{prefix}' found.")
        if not files and suffix:
                    raise FileNotFoundError(f"No files starting with '{prefix}' and ending with '{suffix}' found.")

        mps_list = []
        for path in files:
            filename = os.path.basename(path)
            mps,_= self.get_MPS(filename, directory)
            if mps is not False:
                mps_list.append(mps)

        print(f"Loaded {len(mps_list)} MPS files.")
        return mps_list






    def store_entropy(self,filename,directory = r"/scratch/lesquere"):
        """
        Save the entropy of the MPS at each site to a file.

        Parameters:
        filename (str): The name of the file to save the entropy to.
        directory (str): The directory where the file will be saved.
        """
        # Create the full path
        full_path = os.path.join(directory, filename)
        
        # Ensure the directory exists
        os.makedirs(directory, exist_ok=True)

        # Calculate entropy at each site
        mps,_=self.get_MPS(filename,directory)
        entropies = []
        for site in range(mps.length):
            entropies.append(mps.entanglement_entropy(site))

        # Check if "Entropy Data" already exists in the file
        with open(full_path, 'r') as f:
            lines = f.readlines()

        entropy_data_index = None
        for i, line in enumerate(lines):
            if line.startswith("Entropy Data"):
                entropy_data_index = i
                break

        # If "Entropy Data" exists, overwrite the next line
        if entropy_data_index is not None:
            lines[entropy_data_index + 1] = " ".join(map(str, entropies)) + "\n"
            with open(full_path, 'w') as f:
                f.writelines(lines)
                # Stop if another "Correlation" or "Entropy" section is found after this
                for j in range(entropy_data_index + 2, len(lines)):
                    if lines[j].startswith("Correlation") or line.startswith("Informations"):
                        break
        else:
            # Otherwise, append "Entropy Data" and the entropy values
            with open(full_path, 'a') as f:
                f.write("Entropy Data\n")
                f.write(" ".join(map(str, entropies)) + "\n")
                
        
        print(f"Entropy saved in {filename}")


    def store_zip_entropy(self, filename, directory=r"/scratch/lesquere"):
        """
        Compute and store entanglement entropy inside the compressed MPS file.

        Parameters:
        filename (str): The name of the file to save the entropy to.
        directory (str): The directory where the file will be saved.
        """

        full_path = os.path.join(directory, filename + ".npz")

        if not os.path.exists(full_path):
            raise FileNotFoundError(f"{full_path} not found. Save MPS first.")

        # load existing data
        data = dict(np.load(full_path))

        # rebuild MPS
        mps,_ = self.get_zip_MPS(filename, directory)

        # compute entropy
        entropies = np.array(
            [mps.entanglement_entropy(site) for site in range(mps.length)],
            dtype=np.float64
        )

        # update dictionary
        data["entropy"] = entropies

        # overwrite file (compressed)
        np.savez_compressed(full_path, **data)

        print(f"Entropy saved in {filename}.npz")


    def get_zip_entropy(self, filename, directory=r"/scratch/lesquere"):
        """
        Load entropy from compressed MPS file.

        Parameters:
        filename (str): The name of the file to get the entropy from.
        directory (str): The directory where the file is located.

        Returns:
        list: A list of entanglement entropy values at each site.
        """

        full_path = os.path.join(directory, filename + ".npz")

        if not os.path.exists(full_path):
            raise FileNotFoundError(f"{full_path} not found.")

        data = np.load(full_path)

        if "entropy" not in data:
            return False  # or raise error if you prefer

        return data["entropy"]


    def get_entropy(self,filename, directory = r"/scratch/lesquere"):
        """
        Load the entropy of the MPS from a file.

        Parameters:
        filename (str): The name of the file to load the entropy from.
        directory (str): The directory where the file is located.

        Returns:
        list: A list of entanglement entropy values at each site.
        """
        # Create the full path
        full_path = os.path.join(directory, filename)
        
        # Ensure the directory exists
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"The file {full_path} does not exist.")
        
        entropies = []
        with open(full_path, 'r') as f:
            for line in f:
                if line.startswith("Entropy Data"):
                    line = next(f) # Skip comment lines
                    for x in line.split():
                        entropies.append(float(x))
                        if line.lower().startswith("correlation") or line.startswith("Informations"):
                            break  # Stop if a correlation section is found
                
        if entropies == []:
            return False
            #raise ValueError(f"No entropy data found in {filename}.")
        return entropies



    def central_charge(self,filename,directory = r"/scratch/lesquere"):
        """
        calculate the central charge using entanglement entropy
        and Calabrese-Cardy formula
        
        Parameters:
        ----------
        mps : MPS instance
        filename (str): The name of the file to load the entropy from.
        directory (str): The directory where the file is located.

        returns:
        --------
        the central charge c and shift A
        """
        L= self.length
        S= self.get_zip_entropy(filename, directory)
        if S is False:
            self.store_zip_entropy(filename, directory)
            S= self.get_zip_entropy(filename, directory)


        if np.abs(S[0]-S[1])<1e-3:
                print("The system is gapped, the central charge is zero.")
                return 0,0
        # Fit the Calabrese-Cardy formula
        def calabrese_cardy(x, c, A):
            return (c / 6) * np.log(L/np.pi*np.sin(np.pi*x/L)) + A

        # Perform the fit
        x = np.linspace(1e-1, L-1e-1 ,100)
        
        l=np.arange(1e-1, L-1e-1)
        S= np.array(S)

        # Reduce the data to the central region
        
        mid = len(S) // 2
        half = len(S) // 3
        S1 = S[mid - half: mid + half]
        l1 = l[mid - half: mid + half]
        
        popt, _ = curve_fit(lambda x, c, A: calabrese_cardy(x, c, A), l1, S1, p0=[1, 0]) #curve_fit(calabrese_cardy, l, S, p0=(1,0))
        if popt[0] == float('inf'):
            print("failed to fit central charge")
        return popt[0], popt[1]


    def store_zip_correlation(self, filename, directory=r"/scratch/lesquere", coordinate='Z', dim=3):
        """
        Compute and store correlation function in compressed MPS file.
        (make sure the basis used for the MPO/MPS is the standard one)
        parameters:
        filename (str): The name of the file to save the entropy to.
        directory (str): The directory where the file is will be save.
        coordinate: the coordiante/direction on which the correlation is computed
        dim: the physical dimension of the MPS
        """

        full_path = os.path.join(directory, filename + ".npz")

        if not os.path.exists(full_path):
            raise FileNotFoundError(f"{full_path} not found.")

        # load existing data
        data = dict(np.load(full_path))

        # --- operator ---
        def spin_operator(coord, dim):
            s = (dim - 1) / 2
            m_vals = np.arange(s, -s-1, -1)

            Sp = np.zeros((dim, dim), dtype=complex)
            Sm = np.zeros((dim, dim), dtype=complex)
            Sz = np.diag(m_vals)

            for i, m in enumerate(m_vals[:-1]):
                coef = np.sqrt(s*(s+1) - m*(m-1))
                Sp[i, i+1] = coef
                Sm[i+1, i] = coef

            Sx = (Sp + Sm) / 2
            Sy = (Sp - Sm) / (2j)

            coord = coord.upper()
            if coord == 'X':
                return Sx
            elif coord == 'Y':
                return Sy
            elif coord == 'Z':
                return Sz
            else:
                raise ValueError("Invalid coordinate")

        S = spin_operator(coordinate, dim)
        #S=  np.array([[0,0,0],[0,1,0],[0,0,-1]])
        # --- load MPS ---
        mps,_ = self.get_zip_MPS(filename)

        # --- compute correlation ---
        corr_function = []
        center = mps.length // 2 - 1

        for i in range(center + 1, mps.length):
            val = (
                mps.two_site_corr(S, S, center, i)
                - mps.mean(S, center) * mps.mean(S, i)
            )
            corr_function.append(val)

        corr_function = np.array(corr_function, dtype=np.float64)

        # --- store with coordinate label ---
        key = f"correlation_{coordinate.upper()}"
        data[key] = corr_function

        # overwrite file
        np.savez_compressed(full_path, **data)

        print(f"Correlation {coordinate} saved in {filename}.npz")

    def get_zip_correlation(self, filename, directory=r"/scratch/lesquere", coordinate='Z'):
        """
        Load correlation function from compressed MPS file.

        parameters:
        filename (str): The name of the file to save the entropy to.
        directory (str): The directory where the file is will be save.
        coordinate: the coordiante/direction on which the correlation is computed
        dim: the physical dimension of the MPS

        returns:
        the correlations along the chain
        """

        full_path = os.path.join(directory, filename + ".npz")

        if not os.path.exists(full_path):
            raise FileNotFoundError(f"{full_path} not found.")

        data = np.load(full_path)

        key = f"correlation_{coordinate.upper()}"

        if key not in data:
            return False

        return data[key]

    def store_correlation(self,filename, directory = r"/scratch/lesquere",coordinate='Z', dim=3):
        """
        Save the correlation function C(r) to a file.
        (make sure the basis used for the MPO/MPS is the standard one)
        Parameters:
        filename (str): The name of the file to save the correlation to.
        correlation (list): The correlation function values.
        directory (str): The directory where the file will be saved.
        
        """
        #spin operator

        def spin_operator(coord, dim):
            s = (dim - 1) / 2  # spin value
            m_vals = np.arange(s, -s-1, -1)  # m = s, s-1, ..., -s

            # Initialize matrices
            Sp = np.zeros((dim, dim), dtype=complex)  # raising
            Sm = np.zeros((dim, dim), dtype=complex)  # lowering
            Sz = np.diag(m_vals)

            # Fill ladder operators
            for i, m in enumerate(m_vals[:-1]):
                coef = np.sqrt(s*(s+1) - m*(m-1))
                Sp[i, i+1] = coef
                Sm[i+1, i] = coef

            # Construct Sx, Sy
            Sx = (Sp + Sm) / 2
            Sy = (Sp - Sm) / (2j)

            coord = coord.upper()
            if coord == 'X':
                return Sx
            elif coord == 'Y':
                return Sy
            elif coord == 'Z':
                return Sz
            else:
                raise ValueError("Invalid coordinate. Choose 'X', 'Y', or 'Z'.")

        S = spin_operator(coord=coordinate, dim=dim )
        S=  np.array([[0,0,0],[0,1,0],[0,0,-1]])
        # Create the full path
        full_path = os.path.join(directory, filename)
        
        # Ensure the directory exists
        os.makedirs(directory, exist_ok=True)

        #compute the correlation function between O1 and O2
        mps,_=self.get_MPS(filename,directory)
        corr_function=[]
        r= range(1,-(-mps.length//2+1))
        for i in range(mps.length//2, mps.length):
            corr_function.append( mps.two_site_corr(S, S, mps.length//2-1, i)-mps.mean(S, mps.length//2-1)*mps.mean(S, i))

        # Check if "Correlation Data" already exists in the file
        with open(full_path, 'r') as f:
            lines = f.readlines()

        correlation_data_index = None
        for i, line in enumerate(lines):
            if line.startswith(f"Correlation {coordinate} Data"):
                correlation_data_index = i
                break

        # If "Correlation Data" exists, overwrite the next line
        if correlation_data_index is not None:
            lines[correlation_data_index + 1] = " ".join(map(str, corr_function)) + "\n"
            with open(full_path, 'w') as f:
                f.writelines(lines)
                # Stop if another "Correlation" or "Entropy" section is found after this
                for j in range(correlation_data_index + 2, len(lines)):
                    if lines[j].startswith("Correlation") or lines[j].startswith("Entropy"):
                        break
        else:
            # Otherwise, append "Correlation Data" and the Correlation values
            with open(full_path, 'a') as f:
                f.write(f"Correlation {coordinate} Data\n")
                f.write(" ".join(map(str, corr_function)) + "\n")

        print(f"Correlation {coordinate} saved in {filename}")


    
    def get_correlation(self,filename, directory = r"/scratch/lesquere",coordinate='Z'):
        """
        Load the correlation function C(r) from a file.

        Parameters:
        filename (str): The name of the file to load the correlation from.
        directory (str): The directory where the file is located.
        coordinate (str): The coordinate for which to load the correlation data ('X', 'Y', or 'Z').

        Returns:
        list: A list of correlation values.
        """
        # Create the full path
        full_path = os.path.join(directory, filename)
        
        # Ensure the directory exists
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"The file {filename} does not exist.")
        
        correlation = []
        with open(full_path, 'r') as f:
            for line in f:
                if line.startswith(f"Correlation {coordinate} Data"):
                    line = next(f) # Skip comment lines
                    for x in line.split():
                        correlation.append(float(x))
                        if line.startswith("Informations") or line.lower().startswith("entropy"):
                            break  # Stop if a correlation/Entropy section is found
        if correlation == []:
            return False
            #raise ValueError(f"No correlation data found in {filename}.")
        return correlation
    


    def store_informations(self,filename,directory = r"/scratch/lesquere", infos= None):
        """
        Save the number of a and c fermions as well as the string order parameter
          of the MPS to its file.

        Parameters:
        filename (str): The name of the file to save the informations to.
        directory (str): The directory where the file will be saved.
        infos (dict): A dictionary containing the information to save.
        """
        # Create the full path
        full_path = os.path.join(directory, filename)
        
        # Ensure the directory exists
        os.makedirs(directory, exist_ok=True)

        # Calculate informations
        mps,_=self.get_MPS(filename,directory)
        dim= mps.state_dim
        
        if infos is None:
            N_0 = mps.fermion_number(species='0',dim=dim)
            N_a = mps.fermion_number(species='a',dim=dim)
            N_c = mps.fermion_number(species='c',dim=dim)
            Order1 = mps.String_order_parameter(i=2, phase=1, dim=dim)
            Order2 = mps.String_order_parameter(i=2, phase=2, dim=dim)
            Order_half = mps.String_order_parameter(i=2, phase=1/2, dim=dim)
            c,A = self.central_charge(filename,directory)
            
            Infos = [N_0, N_a, N_c, Order1, Order2, Order_half, c, A]
        else:
            Infos = [infos["N0"], infos["Na"], infos["Nc"], infos["OP1"],
                      infos["OP2"], infos["OP0.5"], infos["c"], infos["A"]]

        # Check if "Informations" already exists in the file
        with open(full_path, 'r') as f:
            lines = f.readlines()

        infos_data_index = None
        for i, line in enumerate(lines):
            if line.startswith("Informations"):
                infos_data_index = i
                break

        # If "Informations" exists, overwrite the next line
        if infos_data_index is not None:
            lines[infos_data_index + 1] = " ".join(map(str, Infos)) + "\n"
            with open(full_path, 'w') as f:
                f.writelines(lines)
                # Stop if another "Correlation" or "Entropy" section is found after this
                for j in range(infos_data_index+ 2, len(lines)):
                    if lines[j].startswith("Correlation") or line.lower().startswith("entropy"):
                        break
        else:
            # Otherwise, append "Informations" and the informations values
            with open(full_path, 'a') as f:
                f.write("Informations\n")
                f.write(" ".join(map(str, Infos)) + "\n")
                
        
        print(f"Informations saved in {filename}")



    def store_zip_informations(self, filename, directory=r"/scratch/lesquere", infos=None):
        """
        Store computed observables inside the compressed MPS file.

        Parameters:
        filename (str): The name of the file to save the informations to.
        directory (str): The directory where the file will be saved.
        infos (dict): A dictionary containing the information to save.
        """

        full_path = os.path.join(directory, filename + ".npz")

        if not os.path.exists(full_path):
            raise FileNotFoundError(f"{full_path} not found.")

        # load existing data
        data = dict(np.load(full_path))

        # compute if not provided
        if infos is None:
            mps,_ = self.get_zip_MPS(filename)
            dim = mps.state_dim
        
            infos = {
                "N0": mps.fermion_number(species='0', dim=dim),
                "Na": mps.fermion_number(species='a', dim=dim),
                "Nc": mps.fermion_number(species='c', dim=dim),
                "OP1": mps.String_order_parameter(i=2, phase=1, dim=dim),
                "OP2": mps.String_order_parameter(i=2, phase=2, dim=dim),
                "OP0.5": mps.String_order_parameter(i=2, phase=1/2, dim=dim),
                "OP3": mps.String_order_parameter(i=2, phase=3, dim=dim),
                "OP1/3": mps.String_order_parameter(i=2, phase=1/3, dim=dim),
            }
            
            c, A = self.central_charge(filename, directory)
            infos["c"] = c
            infos["A"] = A
        
        # store as arrays 
        data["infos_keys"] = np.array(list(infos.keys()))
        data["infos_values"] = np.array(list(infos.values()), dtype=np.float64)

        # overwrite file
        np.savez_compressed(full_path, **data)

        print(f"Informations saved in {filename}.npz")


    def update_information(self,infos_name,filename, directory= r"/scratch/lesquere", zip=True):
        """Update an infos in the file of the MPS
        Parameters:
        infos_name (str):a list of the name of the info to save again, should be one of "N_0", "N_a", "N_c", "Order1", "Order2", "Order_half", "c", "A"
        filename (str): the name of the file to save the info to
        directory (str): the directory where the file is located
        """
        if zip:
            # Create the full path
            full_path = os.path.join(directory, filename+".npz")
            # Ensure the directory exists
            if not os.path.exists(full_path):
                raise FileNotFoundError(f"The file {full_path} does not exist.")
            # Get the current informations
            infos = self.get_zip_informations(filename, directory)
            if infos is False:
                self.store_zip_informations(filename, directory)
            else:
                mps,_=self.get_zip_MPS(filename,directory)
        else:            # Create the full path
            full_path = os.path.join(directory, filename)
            # Ensure the directory exists
            if not os.path.exists(full_path):
                raise FileNotFoundError(f"The file {full_path} does not exist.")
            # Get the current informations
            infos = self.get_informations(filename, directory)
            if infos is False:
                self.store_informations(filename, directory)
            else:
                mps,_=self.get_MPS(filename,directory)

        dim= mps.state_dim
        # Update the specified info
        if "N_0" in infos_name:
            infos["N0"] = mps.fermion_number(species='0',dim=dim)
        if "N_a" in infos_name:
            infos["Na"] = mps.fermion_number(species='a',dim=dim)
        if "N_c" in infos_name:
            infos["Nc"] = mps.fermion_number(species='c',dim=dim)
        if "Order1" in infos_name:
            infos["OP1"] = mps.String_order_parameter(i=2, phase=1, dim=dim)
        if "Order2" in infos_name:
            infos["OP2"] = mps.String_order_parameter(i=2, phase=2, dim=dim)
        if "Order_half" in infos_name:
            infos["OP0.5"] = mps.String_order_parameter(i=2, phase=1/2, dim=dim)
        if "Order3" in infos_name:
            infos["OP3"] = mps.String_order_parameter(i=2, phase=3, dim=dim)
        if "Order_third" in infos_name:
            infos["OP1/3"] = mps.String_order_parameter(i=2, phase=1/3, dim=dim)

        if "c" in infos_name:
            c,A = self.central_charge(filename,directory)
            infos["c"] = c
        if "A" in infos_name:
            c,A = self.central_charge(filename,directory)
            infos["A"] = A
        #else:
            #raise ValueError("Invalid info name. Choose among 'N_0', 'N_a', 'N_c', 'Order1', 'Order2', 'Order_half', 'c', 'A'.")

        # Save the updated informations back to the file
        if zip:
            self.store_zip_informations(filename, directory, infos=infos)
        else:
            self.store_informations(filename, directory, infos=infos)
        print("information updated")

    def get_zip_informations(self, filename, directory=r"/scratch/lesquere"):
        """
        Load stored observables from compressed MPS file.

        Parameters:
        filename (str): The name of the file to get the informations from.
        directory (str): The directory where the file will be loaded from.
        

        returns:
        the dictionary containing some informations about the MPS
        """

        full_path = os.path.join(directory, filename + ".npz")

        if not os.path.exists(full_path):
            raise FileNotFoundError(f"{full_path} not found.")

        data = np.load(full_path)

        if "infos_keys" not in data or "infos_values" not in data:
            return {
                "N0": float('inf'),
                "Na": float('inf'),
                "Nc": float('inf'),
                "OP1": float('inf'),
                "OP2": float('inf'),
                "OP0.5": float('inf'),
                "c": float('inf'),
                "A": float('inf'),
            }

        keys = data["infos_keys"]
        values = data["infos_values"]

        return {k: float(v) for k, v in zip(keys, values)}



    def get_informations(self,filename, directory = r"/scratch/lesquere"):
        """
        Load the informations of the MPS from a file.

        Parameters:
        filename (str): The name of the file to load the informations from.
        directory (str): The directory where the file is located.

        Returns:
        list: A list informations about the Ground state.
        """
        # Create the full path
        full_path = os.path.join(directory, filename)
        
        # Ensure the directory exists
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"The file {full_path} does not exist.")
        
        Infos = []
        with open(full_path, 'r') as f:
            for line in f:
                if line.startswith("Informations"):
                    line = next(f) # Skip comment lines
                    for x in line.split():
                        Infos.append(float(x))
                        if line.lower().startswith("correlation") or line.lower().startswith("entropy"):
                            break  # Stop if a correlation section is found
                
        if Infos == []:
            data = { "N0": float('inf'),
                    "Na": float('inf'),
                     "Nc": float('inf'),
                    "OP1": float('inf'),
                    "OP2": float('inf'),
                    "OP0.5": float('inf'),
                    "c": float('inf'),
                    "A": float('inf'),
                }

            return data
            #raise ValueError(f"No informations found in {filename}.")
        
        data = { "N0": Infos[0],
                    "Na": Infos[1],
                     "Nc": Infos[2],
                    "OP1": Infos[3],
                    "OP2": Infos[4],
                    "OP0.5": Infos[5],
                    "c": Infos[6],
                    "A": Infos[7],
                }

        return data
    



  