"""
Created on Wed Feb 11 2026

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


class Hubbard_U_inf_MPO(MPO):
    
    def __init__(self, L, V_r, t=1., mu_a=0., mu_c=0.):
        """
        initialize the Hubbard MPO 

        :param L: number of sites
        :param t: hopping constant
        :param V_r: density-density interaction
        :param mu_a: chemical potential of a species
        :param mu_c: chemical potential of c species
        """

        super().__init__(L, 3, 14) 
        self.t= t
        self.mu_a= mu_a
        self.mu_c= mu_c
        self.R_max = len(V_r) # maximum range of interaction
        self.V_r = V_r
        self.tensors = self._initialize_Hubbard_U_inf(V_r, t, mu_a, mu_c)


    def _initialize_Hubbard_U_inf(self, V_r, t, mu_a, mu_c):
        """
        initialize the MPO 
        
        :param t: hopping constant
        :param V_r: density-density interaction
        :param mu_a: chemical potential of a species
        :param mu_c: chemical potential of c species

        Returns
        -------
        The MPO of the hamiltonian
        """
        
        W_1, W, W_L = self.tensor_operator(V_r, t, mu_a, mu_c)
        #tensors = [W_1] + [W for _ in range(1, self.length-1)] + [W_L]
        tensors = [W_1] + [self.tensor_operator(V_r, t, mu_a, mu_c)[1] for _ in range(1, self.length-1)] + [W_L]
        return tensors
    

    
 

    
    def tensor_operator(self, V_r, t, mu_a, mu_c):
        """
        initialize the MPO tensors
        
        :param t: hopping constant
        :param V_r: density-density interaction
        :param mu_a: chemical potential of a species
        :param mu_c: chemical potential of c species

        Returns
        -------
        The MPO tensors of the hamiltonian
        """
        I = np.eye(3)
        Zero= np.zeros((3,3))
        S_Plus =S_plus = np.array([[0, 1, 0],[0, 0, 0],[1, 0, 0]]) 
        S_Minus = S_Plus.T
        S_z = np.array([[0, 0, 0],[0, 1, 0],[0, 0, -1]]) 
        S_z_squared = np.array([[0, 0, 0],[0, 1, 0],[0, 0, 1]])
        P_0= np.array([[1, 0, 0],[0, 0, 0],[0, 0, 0]])
        S_x= np.array([[0,0,0],[0,0,1],[0,1,0]])
        S_y= np.array([[0,0,0],[0,0,1j],[0,-1j,0]])
        c_a_dagger = np.array([[0,1,0],[0,0,0],[0,0,0]]) #Fermionic operator for species a
        c_a = c_a_dagger.T
        c_c_dagger = np.array([[0,0,1],[0,0,0],[0,0,0]]) #Fermionic operator for species c
        c_c = c_c_dagger.T
        D=14
        W = np.zeros((D, 3, 3,D)) #tensor W i.e DxD matrix of 3x3 operators

        if len(V_r) == 2:
            V1, V2 = V_r
            V3,V4=0,0
        if len(V_r) == 3:
            V1, V2, V3 = V_r
            V4=0
        if len(V_r) == 4:
            V1, V2, V3, V4= V_r

        W[0, :, :, 0] = I
        W[1, :, :, 0] = P_0 @ S_Minus
        W[2, :, :, 0] = P_0 @ S_Plus
        W[3, :, :, 0] = S_Plus @ P_0
        W[4, :, :, 0] = S_Minus @ P_0
        W[5, :, :, 0] = S_z
        W[6, :, :, 0] = S_z_squared
        W[7, :, :, 0] = Zero
        W[8, :, :, 0] = Zero
        W[9, :, :, 0] = Zero
        W[10, :, :, 0] = Zero
        W[11, :, :, 0] = Zero
        W[12, :, :, 0] = Zero
        W[-1, :, :, 0] = -(mu_a+mu_c)/2*S_z_squared -(mu_a-mu_c)/2*(S_z)
        W[-1, :, :, 1] = -t*S_Plus @ P_0
        W[-1, :, :, 2] = -t*S_Minus @ P_0    
        W[-1, :, :, 3] = -t*P_0 @ S_Minus
        W[-1, :, :, 4] = -t* P_0 @ S_Plus
        W[-1, :, :, 5] = V1/2*S_z
        W[-1, :, :, 6] = V1/2*S_z_squared
        W[-1, :, :, 7] = V2/2*S_z
        W[-1, :, :, 8] = V2/2*S_z_squared
        W[-1, :, :, 9] = V3/2*S_z
        W[-1, :, :, 10] = V3/2*S_z_squared
        W[-1, :, :, 11] = V4/2*S_z
        W[-1, :, :, 12] = V4/2*S_z_squared
        W[-1, :, :, -1] = I

        W[8, :, :, 6] = I
        W[7, :, :, 5] = I
        W[10, :, :, 8] = I
        W[9, :, :, 7] = I
        W[11, :, :, 9] = I
        W[12, :, :, 10] = I


        # Boundary tensor
        W_1=W[_,-1,:,:,:]
        W_L  = W[:, :, :,0,_] 

        return W_1, W, W_L
    


    def data(self, D=100, tol=1e-5, sweeps=3, twosite=True, zip_file= True, mps=None):
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
        
        
        filename = f"Hubbard_V_int_(L,D){self.length,D}_(t,mu_a,mu_c,V_r){self.t,self.mu_a,self.mu_c,self.V_r}.txt"

        if zip_file:
            self.save_mps_to_zip_file(filename, mps)
        else:
            self.save_mps_to_file(filename, mps)
        end_time = time.time()
        print(f"\nMPS saved as {filename},\nTime taken: {end_time - start_time:.6f} seconds")


