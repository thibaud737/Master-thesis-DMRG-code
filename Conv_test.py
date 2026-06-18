# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 16:02:38 2025

@author: thibaud lesquereux
"""

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
from MPS import MPS, curve_fit
from MPO import MPO
from Hubbard_U_inf_MPO import Hubbard_U_inf_MPO
from Hubbard_V_int_MPO import Hubbard_V_int_MPO
from Hubbard_MPO import Hubbard_MPO 
from exact_diag_OBC import heisenberg_exact_diag as ED
from ED_Hubbard import global_ground_state
from ED_U_inf import build_hamiltonian, ground_state_counts






def test_conv_tolerance(L,D,t,V_r, mu_a, mu_c, U=float('inf')):
   """
   test the convergence of the enregy per site and the 2 points function 
   as a function of the number of tolerence (singular values kept)

   :param L: number of sites
   :param t: hopping constant
   :param V_r: density-density interaction
   :param mu_a: chemical potential of a species
   :param mu_c: chemical potential of c species
   :param U: Coulomb on-site repuslion
   :param plot: True if you want intermediate plots
   
   """
   tol_values = [1e-5, 1e-6, 1e-7,1e-8,1e-9]
   var=[]
   energies = []
   energies2 = []
   mps=None
   S_z =np.array([[0,0,0],[0,1,0],[0,0,-1]])
   corr =np.zeros(len(tol_values))
   corr2 =np.zeros(len(tol_values))
   if U == float('inf'):
      mpo=Hubbard_U_inf_MPO(L, V_r, t=t, mu_a=float(mu_a), mu_c=float(mu_c))
      filename = f"Hubbard_U_inf_(L,D){L,D}_(t,mu_a,mu_c,V_r){t,float(mu_a),float(mu_c),mpo.V_r}.txt"
   else:
      mpo = Hubbard_MPO(L, V_r=V_r, t=t, U=U, mu_a=float(mu_a), mu_c=float(mu_c))
      filename = f"Hubbard(L,D){L,D}_(t,U,mu_a,mu_c,V_r){mpo.t,mpo.U,float(mu_a),float(mu_c),mpo.V_r}.txt"

   for j,tol in enumerate(tol_values):
      #mps, store_data = mpo.get_zip_MPS(filename)
      #if not store_data:
      mpo.store_MPS(D, tol=0, sweeps= 1, mps= mps, zip=False)
      mps,_= mpo.get_MPS(filename)
      mpo.store_MPS(D, tol=0, sweeps= 0, mps= mps, zip=True)
      mps2,_= mpo.get_zip_MPS(filename)

      print(mps.tensors[mps.length//2].shape)
      print(mps2.tensors[mps2.length//2].shape)
      energy = mpo.energy(mps)
      energy2 = mpo.energy(mps2)
      energies.append(energy / L)
      energies2.append(energy2 / L)
      #var.append(mpo.var_energy(mps))
      a= mps.length//3
      corr[j] = mps.two_site_corr(S_z,S_z, a, 2*a)
      corr2[j] = mps2.two_site_corr(S_z,S_z, a, 2*a)
   tol_log = -np.log10(tol_values)


   plt.figure(figsize=(8,6))
   plt.plot(tol_log, corr , marker='o', label=r'uncompressed data')
   plt.plot(tol_log, corr2 , marker='x', label=r'compressed data')
   #plt.xscale('log')
   plt.xlabel(r'Tolerance $(-\log_{10}(\mathrm{tol}))$')
   plt.ylabel(r'$\langle S_{L/3}^zS_{2L/3}^z\rangle$')
   plt.title('Correlation convergence with respect to tolerance')
   plt.grid()
   plt.legend()
   #plt.show()
   plt.savefig(f'Conv_corr_vs_tol')

   plt.figure(figsize=(8,6))
   plt.plot(tol_log, energies , marker='o', label=r'uncompressed data')
   plt.plot(tol_log, energies2 , marker='x', label=r'compressed data')
   #plt.xscale('log')
   plt.xlabel(r'Tolerance $(-\log_{10}(\mathrm{tol}))$')
   plt.ylabel('Energy per site (E/L)')
   plt.title('Energy convergence with respect to tolerance')
   plt.grid()
   #plt.show()
   plt.savefig(f'Conv_E_vs_tol')
#test_conv_tolerance(L=20,D=3,t=1.,V_r=[1.,0.4], mu_a=2., mu_c=2., U=float('inf'))







def test_conv_sweep(L,D,t,V_r, mu_a, mu_c, U=float('inf')):
   """
   test the convergence of the Energy per site as a function 
   of the number of sweeps

   :param L: number of sites
   :param t: hopping constant
   :param V_r: density-density interaction
   :param mu_a: chemical potential of a species
   :param mu_c: chemical potential of c species
   :param U: Coulomb on-site repuslion
   
   
   """
   sweeps = [1,2,3,4,5]
   var=[]
   mps=None
   energies = []
   energies2 = []
   if U == float('inf'):
      mpo=Hubbard_U_inf_MPO(L, V_r, t=t, mu_a=float(mu_a), mu_c=float(mu_c))
      filename = f"Hubbard_U_inf_(L,D){L,D}_(t,mu_a,mu_c,V_r){t,float(mu_a),float(mu_c),mpo.V_r}.txt"
   else:
      mpo = Hubbard_MPO(L, V_r=V_r, t=t, U=U, mu_a=float(mu_a), mu_c=float(mu_c))
      filename = f"Hubbard(L,D){L,D}_(t,U,mu_a,mu_c,V_r){mpo.t,mpo.U,float(mu_a),float(mu_c),mpo.V_r}.txt"

   for sweep in sweeps:
      #mps, store_data = mpo.get_zip_MPS(filename)
      #if not store_data:
      mpo.store_MPS(D, tol=0, sweeps= 1, mps= mps, zip=False)
      mps,_= mpo.get_MPS(filename)
      mpo.store_MPS(D, tol=0, sweeps= 0, mps= mps, zip=True)
      mps2,_= mpo.get_zip_MPS(filename)
      energy = mpo.energy(mps)
      energy2 = mpo.energy(mps2)
      energies.append(energy / L)
      energies2.append(energy2 / L)
      #var.append(mpo.var_energy(mps))
      
   plt.figure(figsize=(8,6))
   plt.plot(sweeps, energies , marker='o', label=r'uncompressed data')
   plt.plot(sweeps, energies2 , marker='x', label=r'compressed data')
   plt.xlabel(r'Sweeps')
   plt.ylabel('Energy per site (E/L)')
   plt.title('Energy convergence with respect to sweeps')
   plt.legend()
   plt.grid()
   plt.show()
#test_conv_sweep(L=200,D=3,t=1.,V_r=[1.,0.4], mu_a=2., mu_c=2., U=float('inf'))





def conv_correlation_function_vs_tol(L,D,t, V_r, mu_a, mu_c, plot=True):
   """
   test the convergence of the 2 points function as function 
   of the tolerence (singular values kept)

   :param L: number of sites
   :param t: hopping constant
   :param V_r: density-density interaction
   :param mu_a: chemical potential of a species
   :param mu_c: chemical potential of c species
   :param U: Coulomb on-site repuslion
   :param plot: True if you want intermediate plots
   
   """
   tol_values = [1e-5, 1e-6, 1e-7,1e-8,1e-9]
   S_z =np.array([[0,0,0],[0,1,0],[0,0,-1]])
   corr =np.zeros(len(tol_values))

   mpo=Hubbard_U_inf_MPO(L, V_r, t=t, mu_a=float(mu_a), mu_c=float(mu_c))
   filename = f"Hubbard_U_inf_(L,D){L,D}_(t,mu_a,mu_c,V_r){t,float(mu_a),float(mu_c),mpo.V_r}.txt"
   
   for j, tol in enumerate(tol_values):
      

      mpo.store_MPS(D, tol=tol, zip=False)
      mps,_= mpo.get_MPS(filename)
      mpo.store_MPS(D, tol=0, sweeps= 0, mps= mps, zip=True)
      mps2,_= mpo.get_zip_MPS(filename)

      S_z_mean = [float(np.sign(mps.mean(S_z, j))) for j in range(mps.length)]
      chain =['+ ' if S_z_mean[j]>0 else '- ' for j in range(mps.length)]
      print(''.join(chain))
   
      
      a= mps.length//3
      corr1 =[mps.two_site_corr(S_z,S_z, a, a+r) for r in range(1,L//3)]
      

      
      corr[j] = mps.two_site_corr(S_z,S_z, a, 2*a) #np.max(diff)
      if plot:
         plt.figure(figsize=(8,6))
         plt.plot(range(L//3-1), corr1 , marker='o', label='correlation')
         #plt.plot(range(L//3-1), corr2, marker='x', label='Model 2')
         #plt.plot(range(L//3-1), diff, marker='x', label='difference')
         plt.xlabel('Distance')
         plt.ylabel('Correlation')
         #plt.legend()
         plt.show()

   tol_log = -np.log10(tol_values)
   plt.figure(figsize=(8,6))
   plt.plot(tol_log, corr , marker='o', label=r'$\langle S_{L/3}^zS_{2L/3}^z\rangle$')
   #plt.xscale('log')
   plt.xlabel(r'Tolerance $(-\log_{10}(\mathrm{tol}))$')
   plt.ylabel(r'$\langle S_{L/3}^zS_{2L/3}^z\rangle$')
   plt.title('Correlation convergence with respect to tolerance')
   plt.grid()
   #plt.legend()
   plt.show()

   
#conv_correlation_function_vs_tol(L=10,D=3,t=1., V_r=[1.,0.4], mu_a= 2., mu_c= 2., plot=False) #


def conv_correlation_function_vs_sweep(L,D,t, V_r, mu_a, mu_c, plot=True):

   """
   test the convergence of the 2 points function as function 
   of the number of sweeps

   :param L: number of sites
   :param t: hopping constant
   :param V_r: density-density interaction
   :param mu_a: chemical potential of a species
   :param mu_c: chemical potential of c species
   :param U: Coulomb on-site repuslion
   :param plot: True if you want intermediate plots
   
   """
    
   sweeps= [1,2,3,4,5]
   S_z =np.array([[0,0,0],[0,1,0],[0,0,-1]])
   corr =np.zeros(len(sweeps))
   corr2 =np.zeros(len(sweeps)) 
   mpo=Hubbard_U_inf_MPO(L, V_r, t=t, mu_a=float(mu_a), mu_c=float(mu_c))
   filename = f"Hubbard_U_inf_(L,D){L,D}_(t,mu_a,mu_c,V_r){t,float(mu_a),float(mu_c),mpo.V_r}.txt"
   
   mps=None
   
   for j, sweep in enumerate(sweeps):
      
      mpo.store_MPS(D, tol=0, sweeps= 1, mps= mps, zip=False)
      mps,_= mpo.get_MPS(filename)
      mpo.store_MPS(D, tol=0, sweeps= 0, mps= mps, zip=True)
      mps2,_= mpo.get_zip_MPS(filename)

      S_z_mean = [float(np.sign(mps.mean(S_z, j))) for j in range(mps.length)]
      chain =['+ ' if S_z_mean[j]>0 else '- ' for j in range(mps.length)]
      print(''.join(chain))
   
      
      a= mps.length//3
      corr1 =[mps.two_site_corr(S_z,S_z, a, a+r) for r in range(1,L//3)]
      

      
      corr[j] = mps.two_site_corr(S_z,S_z, a, 2*a) #np.max(diff)
      corr2[j] = mps2.two_site_corr(S_z,S_z, a, 2*a)
      if plot:
         plt.figure(figsize=(8,6))
         plt.plot(range(L//3-1), corr1 , marker='o', label='correlation')
         #plt.plot(range(L//3-1), corr2, marker='x', label='Model 2')
         #plt.plot(range(L//3-1), diff, marker='x', label='difference')
         plt.xlabel('Distance')
         plt.ylabel('Correlation')
         #plt.legend()
         plt.show()

   
   plt.figure(figsize=(8,6))
   plt.plot(sweeps, corr , marker='o', label=r'uncompressed data')
   plt.plot(sweeps, corr2 , marker='x', label=r'compressed data')
   plt.xlabel(r'Sweeps')
   plt.ylabel(r'$\langle S_{L/3}^zS_{2L/3}^z\rangle$')
   plt.title('Correlation convergence with respect to sweeps')
   plt.legend()
   plt.show()

   
#conv_correlation_function_vs_sweep(L=20,D=3,t=1., V_r=[1.,0.4], mu_a= 2., mu_c= 2., plot=False)




def conv_correlation_function_vs_D(L,t, V_r, mu_a, mu_c, plot=True):

   """
   test the convergence of the 2 points function as function 
   of the initial bond dimension D after 1 sweep

   :param L: number of sites
   :param t: hopping constant
   :param V_r: density-density interaction
   :param mu_a: chemical potential of a species
   :param mu_c: chemical potential of c species
   :param U: Coulomb on-site repuslion
   :param plot: True if you want intermediate plots
   
   """

   D_values = [10,20,40,80,160]# 10,20,40,80,
   S_z =np.array([[0,0,0],[0,1,0],[0,0,-1]])
   corr =np.zeros(len(D_values))
   corr2 =np.zeros(len(D_values)) 
   
   
   for j, D in enumerate(D_values):
      mpo=Hubbard_U_inf_MPO(L, V_r, t=t, mu_a=float(mu_a), mu_c=float(mu_c))
      filename = f"Hubbard_U_inf_(L,D){L,D}_(t,mu_a,mu_c,V_r){t,float(mu_a),float(mu_c),mpo.V_r}.txt"
      mps, store_data= mpo.get_MPS(filename)
      if not store_data:
         mps=None
         mpo.store_MPS(D, sweeps= 1, mps= mps, zip=False)
         mps,_= mpo.get_MPS(filename)
         mpo.store_MPS(D, sweeps= 0, mps= mps, zip=True)
      mps2,_= mpo.get_zip_MPS(filename)
      #mpo.store_MPS(D, sweeps= 1, mps= mps, zip=False)
      #mps,_= mpo.get_MPS(filename)
      #mpo.store_MPS(D, sweeps= 0, mps= mps, zip=True)
      #mps2,_= mpo.get_zip_MPS(filename)

      S_z_mean = [float(np.sign(mps.mean(S_z, j))) for j in range(mps.length)]
      chain =['+ ' if S_z_mean[j]>0 else '- ' for j in range(mps.length)]
      print(''.join(chain))
   
      
      a= mps.length//3
      corr1 =[mps.two_site_corr(S_z,S_z, a, a+r) for r in range(1,L//3)]
      

      
      corr[j] = mps.two_site_corr(S_z,S_z, a, 2*a) #np.max(diff)
      corr2[j] = mps2.two_site_corr(S_z,S_z, a, 2*a)
      if plot:
         plt.figure(figsize=(8,6))
         plt.plot(range(L//3-1), corr1 , marker='o', label='correlation')
         #plt.plot(range(L//3-1), corr2, marker='x', label='Model 2')
         #plt.plot(range(L//3-1), diff, marker='x', label='difference')
         plt.xlabel('Distance')
         plt.ylabel('Correlation')
         #plt.legend()
         plt.show()

   
   plt.figure(figsize=(8,6))
   plt.plot(D_values, corr , marker='o', label=r'uncompressed data')
   plt.plot(D_values, corr2 , marker='x', label=r'compressed data')
   plt.xlabel(r'$\chi_{initial}$')
   plt.ylabel(r'$\langle S_{L/3}^zS_{2L/3}^z\rangle$')
   plt.title('Correlation convergence with respect to initial Bond dimension after 1 sweep')
   plt.legend()
   plt.show()

   
#conv_correlation_function_vs_D(L=200,t=1., V_r=[1.,0.4], mu_a= 2., mu_c= 2., plot=False)





def test_conv_energy_L(L_max,D,t,V_r, mu_a, mu_c, U=float('inf')):
    """
    test the convergence of the energy per site with respect to the system size

      :param L: number of sites
      :param t: hopping constant
      :param V_r: density-density interaction
      :param mu_a: chemical potential of a species
      :param mu_c: chemical potential of c species
      :param U: Coulomb on-site repuslion
    """

    lengths = range(50, L_max + 1, 50)
    energies_per_L = []

    for L in lengths:
      #if the data is not already stored, store it
      if U == float('inf'):
         mpo = Hubbard_U_inf_MPO(L, V_r, t=t, mu_a=float(mu_a), mu_c=float(mu_c))
         filename = f"Hubbard_U_inf_(L,D){L,D}_(t,mu_a,mu_c,V_r){t,float(mu_a),float(mu_c),mpo.V_r}.txt"
      else:
            mpo = Hubbard_MPO(L, V_r=V_r, t=t, U=U, mu_a=float(mu_a), mu_c=float(mu_c))
            filename = f"Hubbard(L,D){L,D}_(t,U,mu_a,mu_c,V_r){mpo.t,mpo.U,float(mu_a),float(mu_c),mpo.V_r}.txt"

      mps, store_data = mpo.get_zip_MPS(filename)
      if not store_data:
         mpo.store_MPS(D)
         mps, _ = mpo.get_zip_MPS(filename)
      energy = mpo.energy(mps)
      energies_per_L.append(energy / L)
    # Fit energies_per_L to a function of the form a + b / L^n

    def fit_func(L, a, b, n):
        return a + b / (L ** n)

    popt, pcov = curve_fit(fit_func, lengths, energies_per_L, p0=[-1.0, 1.0, 1.0])
    a_fit, b_fit, n_fit = popt
    fit_energies = fit_func(np.array(lengths), *popt)
    x= np.linspace(min(lengths), max(lengths), 100)
    print(f"Fit parameters: a = {a_fit}, b = {b_fit}, n = {n_fit}")
    #plot the convergence of energy/L
    plt.figure(figsize=(8, 6))
    plt.scatter(lengths, energies_per_L, marker='o', label="Energy/L")
    plt.plot(x, fit_func(x,a_fit, b_fit, n_fit), label=f"Fit: $a + b \cdot L^{{-n}}$, n={n_fit:.2f}", linestyle='--', color="red")
    plt.xlabel("System size (L)")
    plt.ylabel("Energy per site (E/L)")
    plt.legend()
    plt.title(f"Convergence of Energy per site with system size for mu_a={mu_a}, mu_c={mu_c}")
    plt.grid(True)
    plt.show()


#test_conv_energy_L(L_max=300,D=3,t=1.,V_r=[1.,0.4], mu_a=2., mu_c=2., U=float('inf'))


def test_Hubbard_big_U_limit(L, V_r, t,U_values, mu_a, mu_c):
   """
   test the two models (bosonic and fermionic) in big U limit

   :param L: number of sites
   :param t: hopping constant
   :param V_r: density-density interaction
   :param mu_a: chemical potential of a species
   :param mu_c: chemical potential of c species
   :param U: Coulomb on-site repuslion
   """


   D=50

   energy_U_inf =[]
   energy = []
   for U in U_values:
      mpo_V= Hubbard_MPO(L, V_r=V_r, t=t,U=U, mu_a=float(mu_a), mu_c=float(mu_c))
      filename = f"Hubbard(L,D){L,D}_(t,U,mu_a,mu_c,V_r){mpo_V.t,mpo_V.U,float(mu_a),float(mu_c),mpo_V.V_r}.txt"
      mps_V , store_data= mpo_V.get_zip_MPS(filename)
      if not store_data:
         mpo_V.store_MPS(D)
      mps_V, _ = mpo_V.get_zip_MPS(filename)

      energy.append(mpo_V.energy(mps_V)/L)

      mpo_V1=Hubbard_U_inf_MPO(L, V_r, t=t, mu_a=float(mu_a), mu_c=float(mu_c))
      filename = f"Hubbard_U_inf_(L,D){L,D}_(t,mu_a,mu_c,V_r){mpo_V1.t,mpo_V1.mu_a,mpo_V1.mu_c,mpo_V1.V_r}.txt"
      mps_V1 , store_data= mpo_V1.get_zip_MPS(filename)
      if not store_data:
         mpo_V1.store_MPS(D)
         mps_V1, _ = mpo_V1.get_zip_MPS(filename)
      energy_U_inf.append(mpo_V1.energy(mps_V1)/L)

   def fit_func(L, a, b, n):
        return a + b / (L ** n)

   popt, pcov = curve_fit(fit_func, U_values, energy, p0=[-1.0, 1.0, 1.0])
   a_fit, b_fit, n_fit = popt
   fit_energies = fit_func(np.array(U_values), *popt)
   x= np.linspace(min(U_values), max(U_values), 100)
   print(f"Fit parameters: a = {a_fit}, b = {b_fit}, n = {n_fit}")
   plt.plot(x, fit_func(x,a_fit, b_fit, n_fit), label=f"Fit: $a + b \cdot U^{{-n}}$, n={n_fit:.2f}", linestyle='--', color="red")
   plt.scatter(U_values, energy, marker='o', label='DMRG U finite Energy')
   plt.plot(U_values, energy_U_inf, marker='x', label='DMRG U inf Energy')
   plt.xlabel(r'U ')
   plt.ylabel('Energy per site (E/L)')
   plt.title('Model comparaison')
   plt.legend()
   plt.grid()
   plt.show()

#test_Hubbard_big_U_limit(L=20, V_r=[1.,0.4], t=1.,U_values=np.linspace(1e2,1e3,5), mu_a=0., mu_c=0.)


def test_DMRG_ED_U_finite(L, V_r, t, mu_a_values, mu_c_values, U=float('inf')):
    """
    test many values between ED and DMRG for in the chemical potential map

      :param L: number of sites
      :param t: hopping constant
      :param V_r: density-density interaction
      :param mu_a: chemical potential of a species
      :param mu_c: chemical potential of c species
      :param U: Coulomb on-site repuslion
    """
    D =  3

    energy =np.zeros((len(mu_c_values), len(mu_a_values)) )
    norm_H_diff =np.zeros((len(mu_c_values), len(mu_a_values) ))
    N_a = np.zeros((len(mu_c_values), len(mu_a_values) ))
    N_c =np.zeros((len(mu_c_values), len(mu_a_values) ))
    for i, mu_a in enumerate(mu_a_values):
        for j, mu_c in enumerate(mu_c_values):

         if U == float('inf'):
            mpo = Hubbard_U_inf_MPO(L, V_r, t=t, mu_a=float(mu_a), mu_c=float(mu_c))
            filename = f"Hubbard_U_inf_(L,D){L,D}_(t,mu_a,mu_c,V_r){t,float(mu_a),float(mu_c),mpo.V_r}.txt"
         else:
            mpo = Hubbard_MPO(L, V_r=V_r, t=t, U=U, mu_a=float(mu_a), mu_c=float(mu_c))
            filename = f"Hubbard(L,D){L,D}_(t,U,mu_a,mu_c,V_r){mpo.t,mpo.U,float(mu_a),float(mu_c),mpo.V_r}.txt"

         mps, store_data = mpo.get_zip_MPS(filename)
         if not store_data:
            mpo.store_MPS(D)
            mps, _ = mpo.get_zip_MPS(filename)
         na = mps.fermion_number('a', dim=mps.state_dim)
         nc = mps.fermion_number('c', dim=mps.state_dim)
         state= mps.mps_to_statevector()
         H_exact,basis = build_hamiltonian(L, t=t, mu_a=mu_a, mu_c=mu_c, V_r=V_r)

         V1, V2 = V_r
         if U == float('inf'):
            #E0, (Na, Nc, basis, GS), E1, (Na1, Nc1, basis1, gs1) = ground_state_all_sectors(
                  # L, t, mu_a, mu_c, V_r)
               
            H, basis = build_hamiltonian(L, t, mu_a, mu_c, V_r)

            E0,E1, Na, Nc, r, GS, gs1 = ground_state_counts(H, basis)
         else:
            gs = global_ground_state(L, t, mu_a, mu_c, U, V1, V2)
            E0 = gs["energy"]
            Na= gs["Na"]
            Nc= gs["Nc"]
            GS = gs["state_vector"]

         if U== float('inf'):
            diff_norm = np.linalg.norm(H - H_exact)
         else:
            diff_norm =float('inf')
         norm_H_diff[j,i]= float(diff_norm) if diff_norm >1e-15 else 0
         energy[j,i]= float(mpo.energy(mps)-E0) if np.abs(mpo.energy(mps)-E0) >1e-15 else 0 
         #diff_norm_state = np.linalg.norm(state-gs)
         #diff_norm_state1 =  np.abs(np.vdot(gs, state))  #np.linalg.norm(state-gs1)
         #norm_H_diff[j,i]=  na/nc-ratio if np.abs(na/nc-ratio)>1e-5 else 0   #float(diff_norm_state) if diff_norm_state >1e-5 else 0 
         #exited_state[j,i]=float(diff_norm_state1) if diff_norm_state1 >1e-5 else 0 

         N_a[j,i] = float(na-Na) if float(na-Na) >1e-15 else 0
         N_c[j,i] = float(nc-Nc) if float(nc-Nc) >1e-15 else 0

   
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].imshow(energy, cmap='viridis', origin='lower', extent=[mu_a_values[0], mu_a_values[-1], mu_c_values[0], mu_c_values[-1]], aspect='auto')
    axes[0, 0].set_xlabel('mu_a')
    axes[0, 0].set_ylabel('mu_c')
    axes[0, 0].set_title('Energy DMRG -ED')
    fig.colorbar(axes[0, 0].images[0], ax=axes[0, 0])
    
    axes[0, 1].imshow(N_a, cmap='viridis', origin='lower', extent=[mu_a_values[0], mu_a_values[-1], mu_c_values[0], mu_c_values[-1]], aspect='auto')
    axes[0, 1].set_xlabel('mu_a')
    axes[0, 1].set_ylabel('mu_c')
    axes[0, 1].set_title(r'Fermion $a$ number difference')
    fig.colorbar(axes[0, 1].images[0], ax=axes[0, 1])
    
    axes[1, 0].imshow(N_c, cmap='viridis', origin='lower', extent=[mu_a_values[0], mu_a_values[-1], mu_c_values[0], mu_c_values[-1]], aspect='auto')
    axes[1, 0].set_xlabel('mu_a')
    axes[1, 0].set_ylabel('mu_c')
    axes[1, 0].set_title(r'Fermion $c$ number difference')
    fig.colorbar(axes[1, 0].images[0], ax=axes[1, 0])

    
    axes[1, 1].imshow(norm_H_diff, cmap='viridis', origin='lower', extent=[mu_a_values[0], mu_a_values[-1], mu_c_values[0], mu_c_values[-1]], aspect='auto')
    axes[1, 1].set_xlabel('mu_a')
    axes[1, 1].set_ylabel('mu_c')
    axes[1, 1].set_title('Hamiltonian Norm Difference')
    fig.colorbar(axes[1, 1].images[0], ax=axes[1, 1])

    plt.tight_layout()
    plt.show()



mu_a_values=np.linspace(-2, 2, 5)
mu_c_values=np.linspace(-2, 2, 5)

#test_DMRG_ED_U_finite(L=6, V_r=[1.,.0], t=1., mu_a_values=mu_a_values, mu_c_values=mu_c_values)