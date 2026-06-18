"""
Created on Wed March 4 2026

@author: 13thi
"""

import numpy as np
from MPO import MPO
from MPS import MPS
import time
import sys
import os
import threading
_=np.newaxis


class Hubbard_MPO(MPO):

    def __init__(self, L, V_r, t=1.,U=1., mu_a=0., mu_c=0.):
        """
        initialize the Hubbard MPO 

        :param L: number of sites
        :param t: hopping constant
        :param U: Coulomb repulsion onsite
        :param mu_a: chemical potential of a species
        :param mu_c: chemical potential of c species

        Warning : don't forget JW string while evaluting mean of operators
        """

        super().__init__(L, 4, 12) 
        self.t= t
        self.U = U
        self.mu_a= mu_a
        self.mu_c= mu_c
        self.R_max = len(V_r) # maximum range of interaction
        self.V_r = V_r
        self.tensors = self._initialize_Hubbard(V_r, t, U, mu_a, mu_c)


    def _initialize_Hubbard(self, V_r, t,U, mu_a, mu_c):
        """
        initialize the MPO 
        
        :param t: hopping constant
        :param U: Coulomb repulsion onsite
        :param mu_a: chemical potential of a species
        :param mu_c: chemical potential of c species

        Returns
        -------
        The MPO of the hamiltonian
        """
        
        W_1, W, W_L = self.tensor_operator(V_r, t, U, mu_a, mu_c)
        tensors = [W_1] + [W for _ in range(1, self.length-1)] + [W_L]
    
        return tensors
    

    
 

    
    def tensor_operator(self, V_r, t,U, mu_a, mu_c):
        
        I = np.eye(4)
        Zero= np.zeros((4,4))
        a = np.array([[0,0,1,0],[0,0,0,1],[0,0,0,0],[0,0,0,0]]) #Fermionic operator for species a
        a_dagger = a.T
        
        c = np.array([[0,1,0,0],[0,0,0,0],[0,0,0,1],[0,0,0,0]]) #Fermionic operator for species c
        c_dagger = c.T

        n_a = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,0],[0,0,0,0]]) #number operators
        n_c = np.array([[1,0,0,0],[0,0,0,0],[0,0,1,0],[0,0,0,0]])
        
        D=12
        W = np.zeros((D, 4, 4,D)) #tensor W i.e DxD matrix of 4x4 operators

        if len(V_r) == 2:
            V1, V2 = V_r
            V3=0
        else:
            V1, V2, V3 = V_r
        

        W[0, :, :, 0] = I
        W[1, :, :, 0] = a
        W[2, :, :, 0] = c
        W[3, :, :, 0] = a_dagger
        W[4, :, :, 0] = c_dagger
        W[5, :, :, 0] = n_a 
        W[6, :, :, 0] = n_c
        W[7, :, :, 0] = Zero
        W[8, :, :, 0] = Zero
        W[9, :, :, 0] = Zero
        W[10, :, :, 0] = Zero
        W[-1, :, :, 0] = - mu_a*n_a - mu_c*n_c + U*n_a @ n_c

        W[-1, :, :, 1] = -t*a_dagger 
        W[-1, :, :, 2] = -t*c_dagger
        W[-1, :, :, 3] = -t* a 
        W[-1, :, :, 4] = -t* c 
        W[-1, :, :, 5] = V1*n_a
        W[-1, :, :, 6] = V1*n_c
        W[-1, :, :, 7] = V2*n_a
        W[-1, :, :, 8] = V2*n_c
        W[-1, :, :, 9] = V3*n_a
        W[-1, :, :, 10] = V3*n_c
        W[-1, :, :, -1] = I
        

        W[7, :, :, 5] = I
        W[8, :, :, 6] = I
        W[9, :, :, 7] = I
        W[10, :, :, 8] = I

        # Boundary tensor
        W_1=W[_,-1,:,:,:]
        W_L  = W[:, :, :,0,_] 
        
        return W_1, W, W_L


    def data(self, D=100, tol=1e-5,sweeps=3, twosite=True, zip_file= True, mps=None):
        """   
        Store the MPS of a given case in a file.

        Parameters:
        D (int): Bond dimension.
        tol (float): Tolerance for the variational optimization.
        sweeps (int): Number of sweeps for the variational optimization.
        twosite (bool): Whether to use two-site optimization.
        zip_file (bool): Whether to save the file as a compressed .npz file.
        mps (MPS): Existing MPS to use. If None, a new MPS will be created.

        returns:
        a file containing the MPS tensors.
        """
        
        start_time = time.time()
        if mps is None:
            mps = MPS(self.length, self.state_dim, D)

        _, mps = self.variational_ground_state(mps,sweeps, tol, twosite=twosite)
        
        
        
        filename = f"Hubbard(L,D){self.length,D}_(t,U,mu_a,mu_c,V_r){self.t,self.U,self.mu_a,self.mu_c,self.V_r}.txt"
        if zip_file:
            self.save_mps_to_zip_file(filename, mps)
        else:
            self.save_mps_to_file(filename, mps)
        end_time = time.time()
        print(f"\nMPS saved as {filename},\nTime taken: {end_time - start_time:.6f} seconds")



