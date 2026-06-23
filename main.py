# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 16:02:38 2025

@author: thibaud lesquereux
"""
from cmath import phase
import matplotlib.pyplot as plt
import numpy as np
from MPS import MPS, curve_fit
from MPO import MPO
from Hubbard_U_inf_MPO import Hubbard_U_inf_MPO
from Hubbard_MPO import Hubbard_MPO 
from Thorngren_MPO import Thorngren_MPO
from ED_U_inf import build_hamiltonian, ground_state_counts
from ED_U_finite import lowest_two_states


if __name__ == "__main__":

   #simple exemple of the use of the MPS class (without Data storage)
   """
   L, d, D = 50, 3, 40
   V_r=[1.,0.]
   t=1.
   mu_a, mu_c =0., 0.
   mps=MPS(L, d, D)
   mpo=Hubbard_U_inf_MPO(L, V_r, t=t, mu_a=mu_a, mu_c=mu_c)

    
   E, mps = mpo.variational_ground_state(mps, max_iter=3, tol=1e-8, twosite=True)
   print(f"DMRG Ground Energy: {E:.6f}")
   print("var energy: ", mpo.var_energy(mps))

   """
   #------------------------------------------------------------------------------------------------
   # the following exemple uses the the MPS's data stored 
   #------------------------------------------------------------------------------------------------

   #data saving exemple
   """
   L, d, D = 50, 3, 40
   V_r=[1.,0.]
   t=1.
   mu_a, mu_c =0., 0.

   mpo = Hubbard_V_int_MPO(L, V_r, t=t, mu_a=float(mu_a), mu_c=float(mu_c))
   filename = f"Hubbard_V_int_(L,D){L,D}_(t,mu_a,mu_c,V_r){t,float(mu_a),float(mu_c),mpo.V_r}.txt"

   mps, store_data = mpo.get_zip_MPS(filename)
      if not store_data:
         mpo.store_MPS(D,sweeps=3, zip=True)
         mps, _ = mpo.get_zip_MPS(filename)

   print("var energy: ", mpo.energy(mps))
   """




#-----------------------------------------------------------------------------------------------------
# A more real example (similar to the function defined to compute some observables during the project)
# The following function calculate the entropy using Cardy-Calabrese formula as a function of length L
#-----------------------------------------------------------------------------------------------------



def entropy_vs_L(L_values, D, t,V_r, mu_a, mu_c, U=float('inf')):

   #loop over different L
   entropy = np.zeros(len(L_values))
   for i, L in enumerate(L_values):

      #differentiate the finite and infinite U cases to creat the MPO
      if U == float('inf'):
         mpo = Hubbard_U_inf_MPO(L, V_r, t=t, mu_a=float(mu_a), mu_c=float(mu_c))
         filename = f"Hubbard_U_inf_(L,D){L,D}_(t,mu_a,mu_c,V_r){t,float(mu_a),float(mu_c),mpo.V_r}.txt"
      else:
         mpo = Hubbard_MPO(L, V_r=V_r, t=t, U=U, mu_a=float(mu_a), mu_c=float(mu_c))
         filename = f"Hubbard(L,D){L,D}_(t,U,mu_a,mu_c,V_r){mpo.t,mpo.U,float(mu_a),float(mu_c),mpo.V_r}.txt"


      #Store the MPS in a .npz file if not already done
      mps, store_data = mpo.get_zip_MPS(filename)
      if not store_data:
         mpo.store_MPS(D,sweeps=1)
         mps, _ = mpo.get_zip_MPS(filename)
      
      #calculate the entanglement entropy at site L/2
      entropy[i] = mps.entanglement_entropy(mps.length//2)
   


   def fit_func(L, c, A):
    return A + c/6 * np.log(2*L/np.pi)

   # Fit using L
   popt, pcov = curve_fit(fit_func, L_values, entropy, p0=[1, 0])

   # x-axis variable actually shown in the plot
   x_plot = np.linspace(
      np.log(2*min(L_values)/np.pi),
      np.log(2*max(L_values)/np.pi),
      200
   )

   # Express fit in terms of x = log(2L/pi)
   y_fit = popt[1] + popt[0]/6 * x_plot


   #plot the results
   plt.figure(figsize=(8,6))
   plt.scatter(np.log(2*np.array(L_values)/np.pi), entropy, marker='o')

   plt.plot(
      x_plot,
      y_fit,
      '--',
      color='red',
      label=rf"Fit: $A+\frac{{c}}{{6}}\log\frac{{2L}}{{\pi}}$, "
            rf"$c={popt[0]:.3f}, A={popt[1]:.2f}$"
   )

   plt.xlabel(r'$\log(\frac{2L}{\pi})$')
   plt.ylabel('Entanglement entropy')
   plt.legend()
   plt.grid()
   plt.show()


#the values of the system sizes
L_values=[50,100,150,200]

#call the function
#entropy_vs_L(L_values, D=40, t=1.,V_r=[1.,0.], mu_a=1., mu_c=1., U=float('inf'))







