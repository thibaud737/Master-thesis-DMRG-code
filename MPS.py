# -*- coding: utf-8 -*-
"""
edited by Thibaud Lesquereux.
"""
import numpy as np
import copy
import importlib
import opt_einsum as oe
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.sparse.linalg import LinearOperator, eigs, ArpackNoConvergence
from scipy.sparse.linalg import svds
_ = np.newaxis


class MPS:
    """Matrix Product State (MPS) representation."""
    
    def __init__(self, L, d=2, D=10):
        """
        Initialize a random MPS with given number of sites.
        
        Parameters:
         L : length of the system
         d : dimension of the state
         D : Bond dimension of the matrices

        """
        self.length = L
        self.state_dim = d
        self.bond_dim = D
        self.tensors =self._create_random_mps(L, d, D)
        
        
    def _create_random_mps(self, L, d, D):
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
        mps = [np.random.randn(1, d, D)]
        for i in range(1, L - 1):
            mps.append(np.random.randn(D, d, D))
        mps.append(np.random.randn(D, d, 1))
        return mps
    
    def _create_mps(self,L,d,D):
        """
        Parameters
        ----------
        L : length of the system
        d : dimension of the state
        D : Bond dimension of the matrices

        Returns
        -------
        mps :list of L  matrices (tensor)

        """
        mps = [np.ones((1, d, D))]
        for i in range(1, L - 1):
            mps.append(np.ones((D, d, D)))
        mps.append(np.ones((D, d, 1)))
        return mps
    
    

    def __repr__(self):
        return f"MPS(length={self.length}, tensors={self.tensors})"


    
    def apply_to(self, mpo):
        """
        Parameters
        ----------
        take a MPS instance and an MPO instance

        Raises
        ------
        ValueError if the MPO and the MPS don't have the same lenght
        

        Returns
        -------
        Applies MPO to MPS, returning a new transformed MPS.

        """
        if self.length != mpo.length and self.state_dim != mpo.state_dim:
            raise ValueError("MPS and MPO must have the same lenght and/or state dimension.")
            
        new_mps = MPS(self.length, self.state_dim, self.bond_dim)
        
        
        for i in range(self.length):
            new_mps.tensors[i] = oe.contract("mjn, ijkl->imkln", self.tensors[i].conj() , mpo.tensors[i]) # apply MPO tensor to MPS tensor
            new_mps.tensors[i]= new_mps.tensors[i].reshape(
                new_mps.tensors[i].shape[0]*new_mps.tensors[i].shape[1] ,
                new_mps.tensors[i].shape[2] ,
                new_mps.tensors[i].shape[3]*new_mps.tensors[i].shape[4]) #reshape the resulting tensor into a MPS tensor
        
        
        return new_mps


    
    def __mul__(self, other):
        """
        Parameters
        ----------
        other : either a MPO or a MPS
    
        Returns
        -------
        - Scalar product if other is an MPS instance ie <ψ|ϕ>
        - Transformed MPS in bra <ψ| if other is an MPO instance and multiplied by the MPO 
        """
        
        if isinstance(other, MPS):
            # Handle MPS * MPS case (scalar product)
            # Initialize left contraction ⟨ψ|φ⟩_0
            L = oe.contract('ijk, ijm -> km', self.tensors[0].conj(), other.tensors[0])
            for i in range(1,self.length):
                L = oe.contract('km, kij, mil ->jl ', L, self.tensors[i].conj(),other.tensors[i])
                                  
            return L[0,0]
    
        else:
            
            return self.apply_to(other)
    
        #else:
            #raise ValueError("MPS can only be multiplied with another MPS or MPO.")

    

    def left_projector(self, mps,site):
        """
        Parameters
        ----------
        mps : either a MPO or a MPS
        site : int, the site ℓ where the contraction ends
        Returns
        -------
        - Scalar product if other is an MPS instance ie <ψ|ϕ> up to site
        - Transformed MPS in bra <ψ| if other is an MPO instance and multiplied by the MPO 
        """
        
        if isinstance(mps, MPS):
            # Handle MPS * MPS case (scalar product)
            # Initialize left contraction ⟨ψ|φ⟩_0
            L = oe.contract('ijk, ijm -> km', self.tensors[0].conj(), mps.tensors[0])

            for i in range(1,site+1):
                L = oe.contract('km, kij, mil ->jl ', L, self.tensors[i].conj(),mps.tensors[i])
                                  
            return L
        else:
            print("error, mps is not an instance of MPS")
    

    def right_projector(self, mps,site):
        """
        Parameters
        ----------
        mps : either a MPO or a MPS
        site : int, the site ℓ where the contraction ends
        Returns
        -------
        - Scalar product if other is an MPS instance ie <ψ|ϕ> down to site
        - Transformed MPS in bra <ψ| if other is an MPO instance and multiplied by the MPO 
        """
        
        if isinstance(mps, MPS):
            L = mps.length  # Total number of sites
        
            # Initialize right contraction at the last site
            R = oe.contract('ijk, ljk -> il ', self.tensors[-1].conj(), mps.tensors[-1])
            
            # Iteratively contract from site ℓ+1 down to site ℓ
            for i in range(L-2, site-1, -1):
                R = oe.contract('ijk , ljm, km->il', self.tensors[i].conj(), mps.tensors[i], R)
                
            return R
        else:
            print("error, mps is not an instance of MPS")




    def left_normalize_QR(self,site):
        """
        apply left QR decomposition to the site
        """
        A = self.tensors[site]
        D_left, d, D_right = A.shape
        A = A.reshape(D_left * d, D_right)
        Q, R = np.linalg.qr(A)
        self.tensors[site] = Q.reshape(D_left, d, -1)
        if site + 1 < self.length:
            self.tensors[site + 1] = np.tensordot(R, self.tensors[site + 1], axes=(1, 0))
        
    def right_normalize_QR(self,site):
        """
        apply right QR decomposition to the site
        """
        A = self.tensors[site]
        D_left, d, D_right = A.shape
        A = A.reshape(D_left, d * D_right)
        Q, R = np.linalg.qr(A.T)
        self.tensors[site] = Q.T.reshape(-1, d, D_right)
        if site > 0:
            self.tensors[site - 1] = np.tensordot(self.tensors[site - 1], R.T, axes=(2, 0))


    def left_normalize(self, site, threshold=1e-8):
        """
        Parameters
        ----------
        mps : MPS instance
        site : int, the site ℓ where the transfomation is apply
        
        returns
        -------
        the site-th matrix in its left canonical form
        """
        A=self.tensors
        B = A[site].reshape(-1, A[site].shape[2])  # reshape left and d index
        U, S, V_t = np.linalg.svd(B, full_matrices=False) #carry out SVD
        #self.tensors[site] = U.reshape(A[site].shape[0], A[site].shape[1],-1)  #reshape  into tensor/ update mps
        #S_diag = np.diag(S)

    
        chi = max(1,np.sum(S > threshold))
        u_truncated = U[:, :chi]
        s_truncated = np.diag(S[:chi])
        vh_truncated = V_t[:chi, :]
        #print(chi)
        self.tensors[site] = u_truncated.reshape(A[site].shape[0], A[site].shape[1],chi)  #reshape  into tensor/ update mps
        #multiply S and V with the next matrix state 
        #self.tensors[site+1]=oe.contract('ij, jk, klm -> ilm', S_diag, V_t, self.tensors[site+1]) 
        self.tensors[site+1]=oe.contract('ij, jk, klm -> ilm', s_truncated, vh_truncated, self.tensors[site+1]) 

    def right_normalize(self,site, threshold=1e-8):
        """
        Parameters
        ----------
        mps : MPS instance
        site : int, the site ℓ where the transfomation is apply
        
        returns
        -------
        the site-th matrix in its right canonical form
        """
        A= self.tensors
        B = A[site].reshape(A[site].shape[0], -1)  # reshape right and d index
        U, S, V_t = np.linalg.svd(B, full_matrices=False) # carry out SVD
        #self.tensors[site] = V_t.reshape(-1, A[site].shape[1], A[site].shape[2]) #reshape  into tensor/ update mps
        S_diag = np.diag(S)
       
        chi = max(1,np.sum(S > threshold))
        u_truncated = U[:, :chi]
        s_truncated = np.diag(S[:chi])
        vh_truncated = V_t[:chi, :]

        self.tensors[site] = vh_truncated.reshape(-1, A[site].shape[1], A[site].shape[2]) #reshape  into tensor/ update mps
        #multiply U and S with the previous matrix state
        #self.tensors[site-1]= oe.contract('ijk , kl -> ijl', self.tensors[site-1], U @ S_diag)
        self.tensors[site-1]= oe.contract('ijk , kl -> ijl', self.tensors[site-1], u_truncated @ s_truncated)
       

    def left_canonical_form(self,site):
            """
            Parameters
            ----------
            mps : a list of L matrices
            site : int, the site ℓ where the transfomation ends
            
            (the matrix are in canonical form up to site-1 
             and the ℓ-th matrix is multiplied by the last S@V_t)
            
            Returns
            -------
            the MPS in its left canonical form
    
            """
            mps = copy.deepcopy(self)
            A= copy.deepcopy(self.tensors)
            for i in range(site):
                #print("li=",i)
                B = A[i].reshape(-1, A[i].shape[2])  # reshape left and d index
                U, S, V_t = np.linalg.svd(B, full_matrices=False) #carry out SVD
                mps.tensors[i] = U.reshape(A[i].shape[0], A[i].shape[1],-1)  #reshape  into tensor/ update mps
                S_diag = np.diag(S)
                
                #multiply S and V with the next matrix state
                A[i+1]=oe.contract('ij , jkl -> ikl', S_diag @ V_t, A[i+1])
                mps.tensors[i+1]=oe.contract('ij , jkl -> ikl', S_diag @ V_t, self.tensors[i+1]) 
                     
                
            return mps  
        
    
    
    def right_canonical_form(self,site):
        """
        Parameters
        ----------
        mps : a list of L matrices
        site : int, the site ℓ where the transfomation ends
        
        (the matrix are in canonical form L down to site+1 
         and the ℓ-th matrix is multiplied by the last U@S)
        
        Returns
        -------
        the MPS in its right canonical form
        """
        mps = copy.deepcopy(self) 
        A= copy.deepcopy(self.tensors)
        for i in range(self.length-1,site, -1):
            #print("ri=", i)
            
            B = A[i].reshape(A[i].shape[0], -1)  # reshape right and d index
            U, S, V_t = np.linalg.svd(B, full_matrices=False) # carry out SVD
            mps.tensors[i] = V_t.reshape(-1, A[i].shape[1], A[i].shape[2]) #reshape  into tensor/ update mps
            S_diag = np.diag(S)
            
            #multiply U and S with the previous matrix state
            A[i-1]= oe.contract('ijk , kl -> ijl', A[i-1], U @ S_diag)
            mps.tensors[i-1]= oe.contract('ijk , kl -> ijl', self.tensors[i-1], U @ S_diag)
            
        return mps
    

    

    def mixed_canonical_form(self, site):
        """
        Transforms an MPS into mixed canonical form 
    
        Parameters:
            -----------
            mps : a list of L matrices
            site : int the site ℓ 
            The site index around which the MPS is brought to mixed canonical form
    
        returns:
        --------
        The MPS in mixed canonical form.
        """
        # Apply left-canonical form up to the site
        mps=self.left_canonical_form(site)
        
        
        # Apply right-canonical form up to the site
        mps=mps.right_canonical_form(site)
        return mps

    
    def left_mps_contraction(self, site):
        """
        Compute the left contraction of the MPS up to site

        parameters:
        ----------
        site : int, the site ℓ where the contraction ends
        mps : MPS instance
        returns:
        --------
        L : the left contraction of the MPS up to site
        """
        L = oe.contract('ijk, ijm -> km', self.tensors[0].conj(), self.tensors[0])
        for i in range(1, site+1):
            L = oe.contract('km, kij, mil ->jl ', L, self.tensors[i].conj(), self.tensors[i])
            
        return L
    
    def right_mps_contraction(self, site):
        """
        Compute the right contraction of the MPS down to site
        parameters:
        ----------
        site : int, the site ℓ where the contraction ends
        mps : MPS instance
        returns:
        --------
        R : the right contraction of the MPS down to site
        """
        R = oe.contract('ijk, ljk -> il', self.tensors[-1].conj(), self.tensors[-1])
        for i in range(self.length-2, site-1, -1):
            R = oe.contract('jki, mkl, il ->jm ', self.tensors[i].conj(), self.tensors[i],R)
        return R
    
    


    def effective_N(self,site):
        """
        Compute the effective N tensor at site ℓ
        (used to solve generalized eigenvalue problem for DMRG. if MPS in mixed canonical form N=Id)
        Parameters:
        ----------
        site : int, the site ℓ where the effective N tensor is computed
        mps : MPS instance
        returns:
        --------
        N : the effective N tensor at site ℓ
        """
        # Create a Kronecker delta tensor
        delta = np.eye(self.state_dim)

        # Compute left and right MPS contractions
        psi_A = self.left_mps_contraction(site-1)
        psi_B = self.right_mps_contraction(site+1)
        
        # Compute the effective N tensor
        N = oe.contract('ij, kl, mn -> miknjl', psi_A, psi_B, delta)
        N = N.reshape(delta.shape[0]*psi_A.shape[0]*psi_B.shape[0],-1)

        if site==0:
             N = oe.contract('ij,kl -> kilj', psi_B, delta)
             N = N.reshape(delta.shape[0]*psi_B.shape[0],-1)
        if site==self.length-1:  
             N = oe.contract('ij, kl -> kilj', psi_A, delta)
             N = N.reshape(delta.shape[0]*psi_A.shape[0],-1)
        
        if N.all() != ((N + N.conj().T)/2).all():
            raise ValueError("N not hermitian")
            
        return N


        
        


    def mps_to_statevector(self):
        """
        Converts an MPS to a full state vector.

        Returns:
            psi : np.ndarray of shape (d^L,)
        """
        psi = self.tensors[0] 

        for  A in self.tensors[1:]:
            # Contract virtual bond
            psi = np.tensordot(psi, A, axes=[-1, 0]) 

        # Now shape is (1, d, d, ..., d, 1)
        psi = np.squeeze(psi)
        psi = psi.reshape(-1)  # flatten into (d^L,)
        return psi

    def full_density_matrix(self):
        """
        Compute the full density matrix ρ = |ψ⟩⟨ψ| (for a small MPS length).

        Returns:
            rho : (d^L, d^L) matrix
        """
        psi = self.mps_to_statevector()
        rho = np.outer(psi, psi.conj())
        return rho
    
    def reduced_density_matrix(self, site):
        """
        Compute the reduced density matrix rho at site ℓ of an MPS.

        Parameters:
            mps : MPS instance
            site : the site ℓ at which to compute the density matrix

        Returns:
            rho : np.ndarray of shape (d, d), the reduced density matrix at site
        """
        # Compute the left and right contractions
        L = self.left_mps_contraction(site-1)
        R= self.right_mps_contraction(site+1)
        if site==0:
            rho= oe.contract('ikl, jmn, ln ->km ',self.tensors[site], self.tensors[site].conj(), R)
        if site==self.length-1:
            rho= oe.contract('ij, ikl, jmn ->km ', L,self.tensors[site], self.tensors[site].conj())
        else:
            rho= oe.contract('ij,ikl,jmn, ln ->km ', L,self.tensors[site], self.tensors[site].conj(), R)  

        return rho.reshape(self.state_dim, self.state_dim)
        
        
        
    def entanglement_entropy(self, site):
        """
        Compute the entanglement entropy of the MPS at site ℓ using SVD

        Parameters:
        ----------
        site : int, the site ℓ where the entanglement entropy is computed
        mps : MPS instance
        returns:
        --------
        S : the entanglement entropy at site ℓ
        """
        
        mps = self.mixed_canonical_form(site)
        A = mps.tensors[site]
        B = A.reshape( -1,A.shape[2])  # reshape left and d index
        _, S,_ = np.linalg.svd(B, full_matrices=False)
        p = S**2
        p = p[p > 0]
        #print(np.sum(p))
        return -np.sum(p * np.log(p))
        

    def entanglement_entropy_function(self, plot=True):
        """
        Compute the entanglement entropy as a function of ℓ
        (this function is overloaded in phase_diagram.py and use stored data)
        Parameters:
        ----------
        mps : MPS instance
        returns:
        --------
        S : the entanglement entropy as function of ℓ
        the fit parameters of the entanglement entropy
        """
        
    
        S = []
        L=self.length
        for i in range(L):
            S.append(self.entanglement_entropy(i))
    
        
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
        
        popt, pcov = curve_fit(lambda x, c, A: calabrese_cardy(x, c, A), l1, S1, p0=[1, 0])

        if plot:
            # Plot the fit
            plt.figure(figsize=(8, 6))
            plt.plot(l, S, 'o', label="Entanglement entropy")
            plt.plot(x, calabrese_cardy(x, *popt), label=f"Fit: c={popt[0]:.2f}, A={popt[1]:.2f}", color="red")
            plt.xlabel("ℓ")
            plt.ylabel("Entanglement entropy")
            plt.title("Calabrese-Cardy Fit")
            plt.legend()
            plt.grid()
            plt.show()
        
        return S, popt[0], popt[1]
    
    



    def mean(self, O, site):
        """
        Compute the mean between a local operator O and the MPS at site ℓ
        𝑂_ℓ = ⟨ψ|O|ψ⟩

        Parameters:
        ----------
        O : operator
        site : int, the site where the correlation function is computed
        mps : MPS instance
        returns:
        --------
        the mean between O and the MPS at site site
        """
        #check if the operator have the good dimension
        if O.shape[0] != self.state_dim:
            raise ValueError("O must have the same physical dimension as the MPS")
        
        #initialize the left contraction
        if site ==0:
            L = oe.contract('ijk,jl, mln-> kn', self.tensors[0].conj(), O, self.tensors[0])
        else:
             L = oe.contract('ijk, ijm -> km', self.tensors[0].conj(), self.tensors[0])
        
        #left contraction of the MPS up to length-1
        for i in range(1, self.length):
            if i == site:
                L = oe.contract('ij, ikl, km, jmn -> ln', L, self.tensors[i].conj(), O, self.tensors[i])
            else:
                L = oe.contract('km, kij, mil ->jl ', L, self.tensors[i].conj(), self.tensors[i])
                
        return L[0,0]

    def fermion_number(self, species='a', dim=3):
        """
        Compute the total number of fermions/spins in the MPS
        """
        if dim == 3:
            
            if species == 'a':
                O = np.array([[0,0,0],[0,1,0],[0,0,0]])  # Number operator for species a
            elif species == 'c':
                O =np.array([[0,0,0],[0,0,0],[0,0,1]])  # Number operator for species c
            elif species == '0':
                O = np.array([[1,0,0],[0,0,0],[0,0,0]])  # Number operator for empty state
                
        if dim == 4:
            
            if species == 'a':
                O = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,0],[0,0,0,0]])  # Number operator for species a
            elif species == 'c':
                O =np.array([[1,0,0,0],[0,0,0,0],[0,0,1,0],[0,0,0,0]]) # Number operator for species c
            elif species == '0':
                O = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,1]])# Number operator for empty state
        
        N=0
        for i in range(self.length):
            N+= self.mean(O, i)
        return N

    def two_site_corr(self, O1, O2, site1, site2):
        """
        Compute the correlator ⟨ψ|O₁ ⊗ O₂|ψ⟩ between two operators O1 and O2 at site1 and site2

        Parameters:
        ----------
        O1 : operator 1
        O2 : operator 2
        site1 : int, the first site where O1 acts
        site2 : int, the second site where O2 acts
        mps : MPS instance
        returns:
        --------
        the 2-site correlation ⟨ψ|O₁ ⊗ O₂|ψ⟩ at site1 and site2 (respectively)
        """
        #check if site1 and site2 are different
        if site1 == site2:
            raise ValueError("site1 and site2 must be different")
        #check if the operator have the good dimension
        if O1.shape[0] != self.state_dim or O2.shape[0] != self.state_dim:
            raise ValueError("O1 and O2 must have the same physical dimension as the MPS")
    
        #initialize the left contraction
        
        if site1 ==0:
            L = oe.contract('ijk,jl, mln-> kn', self.tensors[0].conj(), O1, self.tensors[0])
        elif site2 ==0:
            L = oe.contract('ijk,jl, mln-> kn', self.tensors[0].conj(), O2, self.tensors[0])
        else:
             L = oe.contract('ijk, ijm -> km', self.tensors[0].conj(), self.tensors[0])
        #left contraction of the MPS up to length-1
        for i in range(1, self.length):
            if i == site1:
                L = oe.contract('ij, ikl, km, jmn -> ln', L, self.tensors[i].conj(), O1, self.tensors[i])
            elif i == site2:
                L = oe.contract('ij, ikl, km, jmn -> ln', L, self.tensors[i].conj(), O2, self.tensors[i])
            else:
                L = oe.contract('km, kij, mil ->jl ', L, self.tensors[i].conj(), self.tensors[i])
        return L[0,0]


    def correlation_function(self, O1, O2, plot=True):
        """
        Calculate the two-point correlation function ⟨O1_{L/2} O2_r⟩ - ⟨O1_{L/2}⟩⟨O2_r⟩ as a function of distance r.
        (used to compute the correlation in the critical phase i.e XX model. another general version is in the phase_diagram.py file)
        Parameters
        ----------
        O1 : Operator acting on site L/2.
        O2 : Operator acting on site r.
        plot : bool, If True, plot the correlation function and its fits.

        Returns
        -------
        corr_function : list of Correlation values for r = 1, ..., L//2.
        b : Power-law exponent from the fit a / r^b.
        xi : Correlation length from the fit a * exp(-r/xi) / r^b.
        """
        #compute the correlation function between O1 and O2
        corr_function=[]
        r= range(1,-(-self.length//2)+1)
        for i in range(self.length//2, self.length):
            corr_function.append( self.two_site_corr(O1, O2, self.length//2 -1, i)-self.mean(O1, self.length//2-1)*self.mean(O2, i))
           

        #plot the correlation function
        # Define the fitting function
        def fit_func_exp(r, a, b, c):
            return a * r**(-b) * np.exp(-r/c)

        def fit_func(r, a, b):
            return a * r**(-b)

        # Extract odd values of r and corresponding correlation values
        r= np.arange(1, len(corr_function)+1)
        r_odd = r[2::3]
        corr_odd = corr_function[2::3]
        
        #fit the bulk
        
        mid = len(r_odd) // 2
        half = len(r_odd) // 2
        r_odd1 = r_odd[mid - half: mid + half]
        corr_odd1 = corr_odd[mid - half: mid + half]

        # Perform the curve fitting
        popt, pcov = curve_fit(fit_func, r_odd1, corr_odd1, p0=(1, 1))

        # Perform the curve fitting for the exp function
        popt_exp, pcov_exp = curve_fit(fit_func_exp, r_odd1, corr_odd1, p0=(1, 1, 10))

        r_odd=np.abs(r_odd)
        corr_odd=np.abs(corr_odd)

        if plot:
            plt.figure(figsize=(10, 6))
            # Plot the fitted curves
            x= np.linspace(1, -(-self.length//2), 100)
            plt.plot(x, fit_func(x, *popt), label=f"Fit: $a/r^b$, $a={popt[0]:.3f}$, $b={popt[1]:.3f}$", linestyle="--")
            plt.plot(x, fit_func_exp(x, *popt_exp),label=f"Fit: $a \\cdot e^{{-x/\\xi}}/r^b,\\ a={popt_exp[0]:.3f},\\ b={popt_exp[1]:.3f},\\ \\xi={popt_exp[2]:.3f}$" , linestyle="-.")
            #plt.plot(x,-1/(np.pi)**2*1/x**2, label="Theory: $-1/(\\pi^2\\cdot r^2)$", linestyle=":")
            # Plot the original data points
            plt.plot(r, corr_function, marker='o')
            plt.legend()
            plt.xlabel("distance r")
            plt.ylabel(r"$\langle S^z_{\frac{L}{2}} S^z_{\frac{L}{2}+r} \rangle$")
            plt.title("Correlation function ")
            plt.grid()
            plt.show()
            
            
            # Create a plot of log(corr_function) as a function of log(r) with only the fits
            plt.figure(figsize=(10, 6))
            x = np.logspace(np.log10(1), np.log10(-(-self.length // 2)), 100)

            plt.loglog(r_odd,np.abs(corr_odd), marker='o')
            plt.loglog(x,fit_func(x, np.abs(popt[0]), popt[1]), label=f"Fit: $a/r^b$, $a={popt[0]:.3f}$, $b={popt[1]:.3f}$",linestyle="--")
            plt.loglog(x,fit_func_exp(x, np.abs(popt_exp[0]), popt_exp[1], popt_exp[2]), label=f"Fit: $a \\cdot e^{{-x/\\xi}}/r^b,\\ a={popt_exp[0]:.3f},\\ b={popt_exp[1]:.3f},\\ \\xi={popt_exp[2]:.3f}$", linestyle="-.")
            #plt.loglog(x, 1/(np.pi)**2*1/x**2, label="Theory: $-1/(\\pi^2 \\cdot r^2)$", linestyle=":")
            plt.legend()
            plt.xlabel("log(r)")
            plt.ylabel(r"$log\langle S^z_{\frac{L}{2}} S^z_{\frac{L}{2}+r} \rangle$")
            plt.title("Log-Log Correlation Function Fits")
            plt.grid()
            plt.show()
        
        return corr_function, popt_exp[1], popt_exp[2]




    def Neel_order_parameter(self,dim=2):
        """
        Compute the Neel order parameter for the MPS
        (this is plot in the phase_diagram.py file)
        𝑁𝑂𝑃 = 1/L sum_ℓ (⟨S^z_odd⟩-⟨S^z_even⟩)
        Parameters:
        ----------
        mps : MPS instance
        returns:
        --------
        the Neel order parameter for the MPS i.e. sum_ℓ (⟨S^z_odd⟩-⟨S^z_even⟩)/L
        """
        
        # Define the Sz operator for a single site
        if dim==2:
            Sz = np.array([[1, 0], [0, -1]]) / 2 
        if dim==3:
            Sz = np.array([[0,0,0],[0,1,0],[0,0,-1]])/2
        NOP=0     
        #calculate iteratively (⟨S^z_odd⟩-⟨S^z_even⟩)
        for i in range(self.length):
                NOP+=(-1)**(i+1)*self.mean(Sz, i)

        NOP=NOP/self.length
        return NOP
    

    def FM_order_parameter(self, dim=2):
        """
        Compute the FM order parameter for the MPS
        (this is plot in the phase_diagram.py file)
        FMOP = 1/L sum_ℓ ⟨S^z_ℓ⟩
        Parameters:
        ----------
        mps : MPS instance
        returns:
        --------
        the FM order parameter for the MPS i.e. sum_ℓ ⟨S^z_ℓ⟩/L
        """
        # Define the Sz operator for a single site
        if dim==2:
            Sz = np.array([[1, 0], [0, -1]]) / 2 

        if dim== 3:
            Sz = np.array([[0,0,0],[0,1,0],[0,0,-1]])
        FMOP=0     
        #calculate iteratively (⟨S^z_i⟩)
        for i in range(self.length):
                FMOP+=self.mean(Sz, i)

        FMOP=FMOP/self.length
        return FMOP


    def Dimer_Z_order_parameter(self,site):
        """
        Compute the Dimer Z order parameter for the MPS
        (this is plot in the phase_diagram.py file)
        DZOP =  ⟨S^z_ℓ S^z_{ℓ+1}⟩- ⟨S^z_{ℓ+1} S^z_{ℓ+2}⟩
        Parameters:
        ----------
        mps : MPS instance
        site : int, the site ℓ where the Dimer Z order parameter is computed
        returns:
        --------
        the Dimer Z order parameter 
        """
        # Define the Sz operator for a single site
        Sz = np.array([[1, 0], [0, -1]]) / 2
        if site < 0 or site+2 > self.length-1:
            raise ValueError("Site index out of bounds for Dimer Z order parameter.")
        # Calculate the two-site correlation functions
        # between Sz at site and site+1, and Sz at site+1 and site+2
        
        corr1 = self.two_site_corr(Sz, Sz, site, site+1)
        corr2 = self.two_site_corr(Sz, Sz, site+1, site+2)
        DZOP = corr1 - corr2
        """
        DZOP = 0
        # Calculate the two-site correlation functions
        for i in range(self.length-1):
            DZOP += (-1)**(i+1) * self.two_site_corr(Sz, Sz, i, i+1)
        DZOP = DZOP / self.length
        """
        return DZOP


    def Dimer_XY_order_parameter(self,site):
        """
        Compute the Dimer XY order parameter for the MPS
        (this is plot in the phase_diagram.py file)
        DXYOP = ⟨Sx_ℓ Sx_{ℓ+1} + Sy_ℓ Sy_{ℓ+1}⟩ - ⟨Sx_{ℓ+1} Sx_{ℓ+2} + Sy_{ℓ+1} Sy_{ℓ+2}⟩
        Parameters:
        ----------
        mps : MPS instance
        site : int, the site ℓ where the Dimer XY order parameter is computed
        returns:
        --------
        the Dimer XY order parameter 
        """
        # Define the Sx and Sy operators for a single site
        Sx = np.array([[0, 1], [1, 0]]) / 2 
        Sy = np.array([[0, -1j], [1j, 0]]) / 2 

        if site < 0 or site+2 > self.length-1:
            raise ValueError("Site index out of bounds for Dimer XY order parameter.")
    
        # Calculate the two-site correlation functions
        corr1 = self.two_site_corr(Sx, Sx, site, site+1) + self.two_site_corr(Sy, Sy, site, site+1)
        corr2 = self.two_site_corr(Sx, Sx, site+1, site+2) + self.two_site_corr(Sy, Sy, site+1, site+2)
        DXYOP = corr1 - corr2
        return DXYOP
        


    def String_order_parameter(self, i, phase=1, dim=3, Results=False):
        """
        compute the string order parameter define in the thesis:

        parameters:
        i: the starting point of the string
        phase: the ratio of spin down/up characerizing the phase
        dim: the physical dimension of the MPS
        Results: True if you want the list of all the string as fonction of the end point

        returns:
        the string order parameter and if Results==True:
        the list of all the string as fonction of the end point
        
        """

        if i >= self.length - 10 or self.length <= 10:
            print("Length too short for SOP calculation ")
            return float('inf')

        #operators
        if dim == 3:
            n_0 = np.array([[1,0,0],[0,0,0],[0,0,0]])
            n_a = np.array([[0,0,0],[0,1,0],[0,0,0]])
            n_c = np.array([[0,0,0],[0,0,0],[0,0,1]])

        
            S_z = n_a - n_c
            S_z_square = n_a + n_c
            z = -0.5 + (np.sqrt(3)/2)*1j
            S_x = np.array([[0,0,0],[0,0,1],[0,1,0]])
            #phase-dependent operators
            if phase == 1:
                O_i, O_f = S_z, S_z
                g_k_values = [-1]
                SO = [n_0 - S_z_square]

            elif phase == 2:
                O_i, O_f = n_a, S_z
                g_k_values = [3/2*z]
                SO = [z*S_z_square + n_0]

            elif phase == 1/2:
                O_i, O_f = n_c, S_z
                g_k_values = [-3/2*z]
                SO = [z*S_z_square + n_0]

            elif phase == 0:
                O_i, O_f = S_z, S_z
                g_k_values = [1]
                SO = [S_z_square + n_0]

            elif phase == 3:
                O_i, O_f = n_a, S_z
                g_k_values = [2j]
                SO=[np.diag([1,1j,1j])]

            elif phase == 1/3:
                O_i, O_f = n_c, S_z
                g_k_values = [-2j]
                SO=[np.diag([1,1j,1j])]
        
        if dim == 4:
            n_a = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,0],[0,0,0,0]])
            n_c = np.array([[1,0,0,0],[0,0,0,0],[0,0,1,0],[0,0,0,0]])

            S_z = n_a - n_c
            S_z_square = n_a + n_c
            z = -0.5 + (np.sqrt(3)/2)*1j
            
            #phase-dependent operators
            if phase == 1:
                O_i, O_f = S_z, S_z
                g_k_values = [-1]
                SO = [np.diag([1,-1,-1,1])]#[n_0 - S_z_square]

            elif phase == 2:
                O_i, O_f = n_a, S_z
                g_k_values = [3/2*z]
                SO = [np.diag([z.conj(),z,z,1])]

            elif phase == 1/2:
                O_i, O_f = n_c, S_z
                g_k_values = [-3/2*z]
                SO = [np.diag([z.conj(),z,z,1])]

            elif phase == 0:
                O_i, O_f = S_z, S_z
                g_k_values = [1]
                SO = [np.eye(4)]

            elif phase == 3:
                O_i, O_f = n_a, S_z
                g_k_values = [2j]
                SO=[np.diag([-1,1j,1j,1])]

            elif phase == 1/3:
                O_i, O_f = n_c, S_z
                g_k_values = [-2j]
                SO=[np.diag([-1,1j,1j,1])]


        # adjust starting point
        tries = 0
        while self.mean(O_i**2, i) < 1e-1 and i < self.length-3 and tries <= 3:
            
            tries += 1
            i += 1

        results = []

        # --- loop over different strings ---
        for n, g_k in enumerate(g_k_values):
            
            O_i_mean = self.mean(O_i**2, i)
            S_z_i_mean = self.mean(S_z_square, i)

            
            #precompute right environement
            R_env = [None] * self.length
            R_env[self.length-1] =np.eye(1)

            for site in reversed(range(self.length-1)):
                R_env[site] = oe.contract(
                    'ikl, jkn, ln -> ij',
                    self.tensors[site].conj(),
                    self.tensors[site],
                    R_env[site+1]
                )

            # initialize left contraction at i
            L = self.left_mps_contraction(i-1)

            L = oe.contract(
                'ij, ikl, km, jmn -> ln',
                L, self.tensors[i].conj(), O_i, self.tensors[i]
            )

            # sweep forward once
            for site in range(i+1,self.length-2):
                # --- density check ---
                if (S_z_i_mean * self.mean(S_z_square, site)) < 1e-8:
                    print("test3")
                    return float('inf')
            
                # build right environment ONCE per site
                R = R_env[site+1]
                
                # measure at this site
                val = oe.contract(
                    'ij, ikl, km, jmn, ln ->',
                    L, self.tensors[site].conj(), O_f, self.tensors[site], R
                )

                # propagate string operator
                L = oe.contract(
                    'ij, ikl, km, jmn -> ln',
                    L, self.tensors[site].conj(), SO[n], self.tensors[site]
                )

                # normalization
                norm = (O_i_mean * self.mean(O_f**2, site))
                if norm > 1e-5:
                    val/= norm
                results.append(g_k * val)
    
        
        O = np.mean(results)
        if Results== True:
            
            return np.abs(O.real), results
        else:
            
            return np.abs(O.real)
        

    
    def String_order_parameter_fit(self,site, phase=1, dim=3, plot=False):
        """
        fit the SOP as a function of the distance |i-j|
        
        Parameters:
        site: the site where the SOP begins
        phase: the phase for which the SOP is computed (1, 1/2 or 2)
        dim: the physical dimension of the MPS (3 or 4)
        plot: if True, plot the SOP as a function of the distance |i-j| and the fit
        returns:
        the fit parameters of the SOP as a function of the distance |i-j|
        
        """
        SOP =[]
        for i in range(site+5, self.length-3):
            SOP.append(self.SPT_order_parameter(site, i, phase=phase, dim=dim))

        def fit_func(r, O, A, eta, k, phi):
            return O + A * np.sin(k*r + phi) / r**eta
        
        r= np.arange(1, len(SOP)+1)
        SOP= np.array(SOP)   
        try:   
            popt, pcov = curve_fit(fit_func, r, SOP, p0=(0.1, 1, 0.1, np.pi/2, 0))       
        except:
            print("Fit failed")
            return float('inf')
        if plot:
            plt.figure(figsize=(8, 6))
            plt.plot(r, SOP, marker='o', label="SO parameter")
            x = np.linspace(1, len(SOP), 100)
            plt.plot(x, fit_func(x, *popt), label=f"Fit: $O + A\\cdot sin(k\\cdot r)\\cdot r^{{-\\eta}}, \\ O={popt[0]:.3f},\\ A={popt[1]:.3f}, \\eta={popt[2]:.3f},\\ k={popt[3]:.3f}$", color="red")
            plt.xlabel("distance |i-j|")
            if phase == 1/2:
                plt.ylabel(r'$O_{\frac{1}{2}}$')
                plt.title(r'String order parameter $O_{\frac{1}{2}}$ ')
            else:
                plt.ylabel(rf'$O_{phase}$')
                plt.title(rf'String order parameter $O_{phase}$ ')
            plt.legend()
            plt.grid()
            plt.show()

        return popt[0]











#------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------
#---------This part is not projective representationson MPS (It can be ignored)------
#------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------

    def local_symmetry_application(self, sym_op):
        """
        Apply a local symmetry operation sym_op to the MPS at each site
        Parameters:
        ----------
        sym_op : np.ndarray of shape (d, d), the local symmetry operation to apply
        mps : MPS instance
        returns:
        --------
        the MPS after applying the local symmetry operation at each site
        """
        tensors = []
        for i in range(self.length):
            tensors.append(oe.contract('ijk, jl -> ilk', self.tensors[i], sym_op))
        return tensors
    


    
    def test_local_symmetry(self, G):
        """
        Test if the MPS is invariant under a local symmetry operation sym_op
        Parameters:
        ----------
        G : list of np.ndarray of shape (d, d), the local symmetry operations to test
        mps : MPS instance
        returns:
        --------
        True if the MPS is invariant under the local symmetry operation, False otherwise
        """
        for sym_op in G:
            Invariant = True
            I = oe.contract('ikl, km, imn -> ln', self.tensors[0].conj(),sym_op, self.tensors[0])

            for i in range(1,self.length):
                I = oe.contract(' ij, ikl, km, jmn -> ln', I, self.tensors[i].conj(),sym_op, self.tensors[i])
            I= I[0,0]
            print("overlap =", I)
            if np.abs(np.abs(I) - self*self) > 1e-8:
                print("MPS is not invariant under the local symmetry operation")
                Invariant =False
                
            else:
                print("MPS is invariant under the local symmetry operation")
                
        if Invariant is False:
            print("MPS is not invraiant under the group symmetry")
            return False
        else:
            print("MPS is invraiant under the group symmetry")
            return True

    
    def CP_map(self,site, sym_op):
        """
        compute the completely positive map associated to the local symmetry operation sym_op on the MPS
        Parameters:
        ----------
        site : int, the site where the CP map is computed
        sym_op : np.ndarray of shape (d, d), the local symmetry operation to apply

        returns:
        the CP map associated to the local symmetry operation sym_op on the MPS at site
        """
        shape = self.tensors[site].shape
        def transfer_matrix(X):
            """
            Compute the transfer matrix of the MPS with a local symmetry operation sym_op
            Parameters:
            ----------
            sym_op : np.ndarray of shape (d, d), the local symmetry operation to apply
            mps : MPS instance
            site: int, the site where the transfer matrix is computed
            returns:
            --------
            the transfer matrix of the MPS with the local symmetry operation applied
            """
        
            #transfer matrix
            X_reshaped = X.reshape(shape[0], shape[0])
            T = np.zeros((shape[0], shape[0]), dtype=complex)
            T = oe.contract('ijk, jl, mln, kn ->im ', self.tensors[site].conj(), sym_op, self.tensors[site], X_reshaped)
                
            return T.flatten()
        
        def rtransfer_matrix(X):
            """
            Compute the right transfer matrix of the MPS with a local symmetry operation sym_op
            Parameters:
            ----------
            sym_op : np.ndarray of shape (d, d), the local symmetry operation to apply
            mps : MPS instance
            site: int, the site where the right transfer matrix is computed
            returns:
            --------
            the right transfer matrix of the MPS with the local symmetry operation applied
            """
        
            #right transfer matrix
            X_reshaped = X.reshape(shape[0], shape[0])
            T = np.zeros((shape[0], shape[0]), dtype=complex)
            T = oe.contract('ijk, jl, mln, kn ->im ', self.tensors[site], sym_op.conj().T, self.tensors[site].conj(), X_reshaped)
                
            return T.flatten()
        

        return LinearOperator((shape[0]*shape[0], shape[0]*shape[0]), matvec=transfer_matrix, rmatvec= rtransfer_matrix)


    
    def entanglement_spectrum(self, site, plot=True):
        """
        compute the entanglement spectrum of the MPS at site 
        with schmidt decompostion of the MPS at site. 
        Also tells if the spectrum is degenerate or not, which is a signature of the SPT phase.
        Parameters:
        ----------
        site : int, the site where the entanglement spectrum is computed

        returns:
        the entanglement spectrum of the MPS at site
        """
        mps = self.mixed_canonical_form(site)
        A = mps.tensors[site]
        B = A.reshape( -1,A.shape[2])  # reshape left and d index
        _, S,_ = np.linalg.svd(B, full_matrices=False)
        p = S**2
        p = p[p > 0]#1e-10]
        if plot:
            plt.figure(figsize=(8, 6))
            plt.plot(-2*np.log(p), marker='o')
            plt.xlabel("Index")
            plt.ylabel("Schmidt coefficients squared")
            plt.title(f"Entanglement Spectrum at site {site}")
            plt.grid()
            plt.show()
        _, counts = np.unique(np.round(p, decimals=7), return_counts=True) #-2*np.log(p)
        return -1*np.log(p), counts 



    def projective_representation(self, site, G):
        """
        compute the projective representation of the local symmetry group G on the MPS at site
        Parameters:
        ----------
        site : int, the site where the projective representation is computed
        G : list of np.ndarray of shape (d, d), the local symmetry operations to test

        returns:
        the projective representation of the local symmetry group G on the MPS at site
        """
        rep = []
        for sym_op in G:
            T = self.CP_map(site, sym_op)
            eigvals, eigvecs = eigs(T, k=1, which='LM', tol=1e-8, maxiter=1000)
            max_eigval = np.max(np.abs(eigvals))
            X = eigvecs[:, np.argmax(np.abs(eigvals))]
            X= X.reshape(self.tensors[site].shape[0], self.tensors[site].shape[0])
            rep.append(X)
        
        return rep
    
    def is_projective_representation(self, site, G):
        """
        Test if the representation of the local symmetry group G on the MPS at site is projective
        Parameters:
        ----------
        site : int, the site where the projective representation is tested
        G : list of np.ndarray of shape (d, d), the local symmetry operations to test

        returns:
        True if the representation of the local symmetry group G on the MPS at site is projective, False otherwise
        """
        rep = self.projective_representation(site, G)
        for i in range(len(G)):
            for j in range(i+1, len(G)):
                sym_op_ij = np.dot(G[i], G[j])
                sym_op_ji = np.dot(G[j], G[i])
                rep_ij = np.dot(rep[i], rep[j])
                rep_ji = np.dot(rep[j], rep[i])
                if not np.allclose(rep_ij, rep_ji) and not np.allclose(rep_ij, -rep_ji):
                    print("The representation is not projective")
                    return False
        print("The representation is projective")
        return True
    
    def phase_factor(self, site, G):
        """
        compute the phase factor of the projective representation of the local symmetry group G on the MPS at site
        Parameters:
        ----------
        site : int, the site where the phase factor is computed
        G : list of np.ndarray of shape (d, d), the local symmetry operations to test

        returns:
        the phase factor of the projective representation of the local symmetry group G on the MPS at site
        """
        rep = self.projective_representation(site, G)
        phase_factors = []
        for i in range(len(G)):
            for j in range(i+1, len(G)):
                sym_op_ij = np.dot(G[i], G[j])
                sym_op_ji = np.dot(G[j], G[i])
                rep_ij = np.dot(rep[i], rep[j])
                rep_ji = np.dot(rep[j], rep[i])
                if np.allclose(rep_ij, rep_ji):
                    phase_factors.append(1)
                elif np.allclose(rep_ij, -rep_ji):
                    phase_factors.append(-1)
                else:
                    print("The representation is not projective")
                    return [0]
        return phase_factors


    
    

 


    def test_injectivity_transfer_matrix(self, site,g= np.eye(3)):
        """
        Test the injectivity of the MPS by checking the eigenvalues of the transfer matrix at site ℓ
        Parameters:
        ----------
        site : int, the site ℓ where the injectivity is tested
        mps : MPS instance
        returns:
        --------
        True if the MPS is injective at site ℓ, False otherwise
        """
        mps =self.left_canonical_form(site+1)
        #T = oe.contract('ijk, jl, ilm ->km ', self.tensors[0].conj(), g, self.tensors[0])
        #for i in range(site):
            #T = oe.contract('ij, ikl, km, jmn ->ln', T, self.tensors[i].conj(), g, self.tensors[i])
    

        T2= mps.CP_map(site, np.eye(self.state_dim))
        eigvals, eigvecs = eigs(T2, k=1, which='LM', tol=1e-8, maxiter=1000)
        max_eigval = np.max(np.abs(eigvals))
       
        
        if np.abs(max_eigval-1) < 1e-8:
            print("MPS is injective at site ", site)
            return True
        else:
            print("MPS is not injective at site ", site, "max eigenvalue=", max_eigval)
            return False
        




 
    def projective_representation(self, site, G):
        """
        Compute the projective representation of a symmetry group G on the MPS
        Parameters:
        ----------
        G : list of np.ndarray of shape (d, d), the local symmetry operations corresponding to the group elements
        mps : MPS instance
        returns:
        --------
        a dictionary mapping each group element to its projective representation on the MPS
        """


        projective_reps = {}
        mps = self.left_canonical_form(self.length-1)
        shape = mps.tensors[site].shape
        for g in G:
            import copy
            mps2= copy.deepcopy(mps) 
            mps2.tensors = [A.transpose(2, 1, 0) for A in reversed(mps.tensors)]
            mps2 = mps2.left_canonical_form(mps2.length-1)
            
            T = oe.contract('ijk, ikm ->km ', mps.tensors[0].conj(), mps2.tensors[0])
            for i in range(1,site):
                T = oe.contract('ij, ikl, jkn ->ln', T, mps.tensors[i].conj(), mps2.tensors[i])

                 
            U,S, V= np.linalg.svd(T)
            
            projective_reps[tuple(g.flatten())] =  V

            
            
            #if np.abs(eigvals).max() < 1-1e-8:
                #print("eigenvalues less than 1 for g=", g, "max eigenvalue=", np.abs(eigvals).max())


            
        return projective_reps
        
       
        



    def factor_set(self,site,G):
        """
        Compute the factor set of a symmetry group G on the MPS
        Parameters:
        ----------
        G : list of np.ndarray of shape (d, d), the local symmetry operations corresponding to the group elements
        mps : MPS instance
        site: int, the site where the factor set is computed
        returns:
        --------
        a dictionary mapping pairs of group elements to their factor set values
        """

        projective_reps = self.projective_representation(site, G)
        factor_set = {}

        g1=G[1]
        g2=G[2]
        for g1 in G:
            for g2 in G:
                rep1 = projective_reps[tuple(g1.flatten())]
                rep2 = projective_reps[tuple(g2.flatten())]
                rep12 = projective_reps[tuple((g1 @ g2).flatten())]
                rep21 = projective_reps[tuple((g2 @ g1).flatten())]
                #print("unitarity", rep1 @ rep1.conj().T, "norm ", np.linalg.norm(rep1 @ rep1.conj().T))
                # Compute the factor set value for this pair of group elements
                dim= rep1.shape[0]
                #if np.linalg.norm(rep1 @ rep2 - rep12) > 1e-8:
                    #print("Projective representations do not satisfy the group multiplication for g1=", g1, "g2=", g2)
        
                factor_set[(tuple(g1.flatten()), tuple(g2.flatten()))] = np.angle(np.trace(rep1 @ rep2 @ np.linalg.inv(rep1) @ np.linalg.inv(rep2)@ rep21 @ np.linalg.inv(rep12))/dim)#np.linalg.inv(rep1)@ np.linalg.inv(rep2))/dim)
             
        return factor_set