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


class Thorngren_MPO(MPO):

    def __init__(self, L, t=1.,U=1., mu=0., Jz=1., hx=0.):
        """
        initialize the Thorngren MPO 

        :param L: number of sites
        :param t: hopping constant
        :param U: Coulomb repulsion onsite
        :param mu: chemical potential
        :param Jz: z-component of the exchange interaction
        :param hx: transverse field

        Warning : don't forget JW string while evaluting mean of operators
        """

        super().__init__(L, 4, 7) 
        self.t= t
        self.U = U
        self.mu= mu
        self.Jz= Jz
        self.hx = hx
        self.tensors = self._initialize_Throngren_MPO( Jz, hx, t,U, mu)


    def _initialize_Throngren_MPO(self, Jz, hx, t,U, mu):
        """
        initialize the MPO 
        
        :param t: hopping constant
        :param U: Coulomb repulsion onsite
        :param mu: chemical potential
        :param Jz: z-component of the exchange interaction
        :param hx: transverse field

        Returns
        -------
        The MPO of the hamiltonian
        """
        
        W_1, W, W_L = self.tensor_operator( Jz, hx, t,U, mu)
        tensors = [W_1] + [W for _ in range(1, self.length-1)] + [W_L]
    
        return tensors
    

    
 

    
    def tensor_operator(self, Jz, hx, t,U, mu):
        
        I = np.eye(4)
        Zero= np.zeros((4,4))
        a = np.array([[0,0,1,0],[0,0,0,1],[0,0,0,0],[0,0,0,0]]) #Fermionic operator for species a
        a_dagger = a.T
        
        c = np.array([[0,1,0,0],[0,0,0,0],[0,0,0,1],[0,0,0,0]]) #Fermionic operator for species c
        c_dagger = c.T

        n_a = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,0],[0,0,0,0]]) #number operators
        n_c = np.array([[1,0,0,0],[0,0,0,0],[0,0,1,0],[0,0,0,0]])
        
        S_z = 1/2*(n_a-n_c)
        S_x =1/2*(a_dagger@ c + c_dagger@ a)
        D=7
        W = np.zeros((D, 4, 4,D)) #tensor W i.e DxD matrix of 4x4 operators

        

        W[0, :, :, 0] = I
        W[1, :, :, 0] = a
        W[2, :, :, 0] = c
        W[3, :, :, 0] = a_dagger
        W[4, :, :, 0] = c_dagger
        W[5, :, :, 0] = S_z 
        W[-1, :, :, 0] = - mu*(n_a+n_c)+hx*S_x + U*n_a @ n_c

        W[-1, :, :, 1] = -t*a_dagger 
        W[-1, :, :, 2] = -t*c_dagger
        W[-1, :, :, 3] = -t* a 
        W[-1, :, :, 4] = -t* c 
        W[-1, :, :, 5] = Jz*S_z
        W[-1, :, :, -1] = I
        

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
        
        
        
        filename = f"Thorngren(L,D){self.length,D}_(t,U,mu,Jz,hx){self.t,self.U,self.mu,self.Jz,self.hx}.txt"
        if zip_file:
            self.save_mps_to_zip_file(filename, mps)
        else:
            self.save_mps_to_file(filename, mps)
        end_time = time.time()
        print(f"\nMPS saved as {filename},\nTime taken: {end_time - start_time:.6f} seconds")


