# This code is writting by Jeongmoon Choi
# email: jeongmoon2006@gmail.com

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

class harm_to_phi:
    '''
    reweighting harmonic bias simulation to linear bias simulation
    
    Args:
    phi(numpy.ndarray): beta*phi values that I want to use (N,)
    q_wham(numpy.ndarray): order parameter value that obtained from harmonic bias simulation WHAM (M,)
    F(numpy.ndarray): Free energy value from harmonic bias simulation WHAM (M,)
    '''
    
    def __init__(self, phi, q_wham, F):
        self.phi = phi
        self.q_wham = q_wham
        self.F = F
        self.prob_phi = np.zeros((len(self.phi), len(self.q_wham)))
        self.c_inv = np.zeros(len(self.phi),)
        
    def q_phi(self):
        for i in range (self.prob_phi.shape[0]):
            self.prob_phi[i,:] = np.exp(-self.F -self.q_wham*self.phi[i])
            self.c_inv[i] = np.sum(self.prob_phi[i,:])
            self.prob_phi[i,:] = self.prob_phi[i,:]/self.c_inv[i]
        
        self.q_phi = np.zeros(len(self.phi),)
        for i in range(self.prob_phi.shape[0]):
            self.q_phi[i] = np.sum(self.prob_phi[i,:]*self.q_wham)
        
        figure(figsize=(8,6), dpi = 200)
        plt.xlabel(r'$-\beta \phi$')
        plt.ylabel(r'$<q_v>_\phi$')
        plt.plot(-self.phi, self.q_phi)
        
        return self.phi, self.q_phi
    
    def suscept(self):
        self.suscept = np.zeros(len(self.phi)-1)
        for i in range(len(self.phi) - 1):
            self.suscept[i] = - (self.q_phi[i+1] - self.q_phi[i])/(self.phi[i+1] - self.phi[i])
        
        figure(figsize=(8,6), dpi=200)
        plt.plot(-self.phi[1:,], self.suscept)
        plt.xlabel(r'$-\beta \phi$')
        plt.ylabel(r'$\chi_v$')
        
        phi_star = self.phi[np.argmax(self.suscept) +1]
        
        return self.phi[1:,], self.suscept, phi_star
