# This code is writting by Jeongmoon Choi
# email: jeongmoon2006@gmail.com

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import autograd.numpy as nup
from autograd import value_and_grad
from scipy.optimize import minimize
from autograd.extend import primitive
from scipy.special import logsumexp


class binwham:
    
    '''
    Calculates free energy using bin wham.
    Code is based on Zhu, Fangqiang, and Gerhard Hummer. Journal of computational chemistry 33.4 (2012): 453-465.
    
    Pil = fi*Cil*Pl
    
    Args:
      op_star_arr(numpy.ndarray): op* matrix (N,)
      op_arr(numpy.ndarray): op matrix (simulation #, time #)
      n_bins(int): The number of bins that the probability should bin
      kappa(float): harmonic bias spring constant in kJ/mol
    '''
    
    def __init__(self, op_star_arr, op_arr, kappa, n_bins=100, T=298):
        
        self.op_star_arr = op_star_arr
        self.op_arr = op_arr
        self.n_bins = n_bins
        self.kappa = kappa
        self.T = T
        self.beta = 1/8.314/T*1000
        self.M = n_bins
        
        
        #S: number of simulations
        self.S = len(self.op_star_arr)
        
        #M: number of bins
        self.M = n_bins
        
        #Ni: number of samples in each simulation
        self.Ni = np.array([len(self.op_arr[i]) for i in range(self.S)])
        
        #Ml: number of simulations in the bin - np.ndarray(n_bins,)
        min_ = np.min(self.op_arr)
        max_ = np.max(self.op_arr)

        self.bins_1 = np.linspace(min_, max_, self.M+1) #shape (M+1,)
        self.bins_ = self.bins_1[1:] #shape (M,)
        
        digitized = np.digitize(self.op_arr, self.bins_1, right = False)
        self.Ml = np.array([(digitized == i).sum() for i in range(1, self.M +1)])
        
        #Pil: probability for each simulation on each bins - np.ndarray(S,M)
        self.Pil = np.zeros((self.S, self.M))
        for i in range(self.S):
            digitized = np.digitize(self.op_arr[i], self.bins_1, right = False)
            count = np.array([(digitized == i).sum() for i in range(1, self.M+1)])
            self.Pil[i,:] = count/len(digitized)
            
        #Wil: bias potential energy - kappa/2*(xl-ri)^2 np.ndarray((S,M))
        self.Wil = np.zeros((self.S,self.M))
        for i in range(self.S):

            ri = self.op_star_arr[i]
            self.Wil[i,0] = self.beta*self.kappa/2*np.power(self.bins_[0] - ri,2)

            for l in range(1,self.M):
                middle = (self.bins_[l] + self.bins_[l-1])/2
                self.Wil[i,l] = self.beta*self.kappa/2*np.power(middle-ri,2)


        self.Cil = np.exp(-self.Wil)

        self.gi0 = np.zeros(self.S,) + 1e-8
        

    def optimize_fn(self, gi, Ni, Ml, Cil):
        
        first_term = - (Ni*gi).sum()
        log_pl = nup.log(Ml) - alogsumexp(nup.repeat(gi[:,nup.newaxis],self.M,axis=1)-self.Wil, axis=0, b=nup.repeat(Ni[:,nup.newaxis],self.M,axis=1))
        second_term = - (Ml*log_pl).sum(axis = 0)
        

        return first_term + second_term
    
    def MLE_optimize(self):
        

        result = minimize(value_and_grad(self.optimize_fn),self.gi0,args=(self.Ni,self.Ml,self.Cil),jac=True,method='L-BFGS-B')

        if result.success:
            gi = result.x
            log_pl = nup.log(self.Ml) - alogsumexp(nup.repeat(gi[:,nup.newaxis],self.M,axis=1)-self.Wil, axis=0, b=nup.repeat(self.Ni[:,nup.newaxis],self.M,axis=1))
            F = -log_pl
            F = F - np.min(F)
            
            figure(figsize=(8,6), dpi = 200)
            plt.plot(self.bins_,F)
            plt.xlabel(r'$q$', fontsize = 15)
            plt.ylabel(r'$\beta F(q)$', fontsize=15)
            
            return gi, log_pl, F
        
        else:
            print('fail to do minimization')
            
    def driving_force(self):
        _,_,F = self.MLE_optimize()
        
        df = np.zeros(len(F)-1,)
        for i in range(len(F)-1):
            df[i] = (F[i+1] - F[i])/(self.bins_[i+1] - self.bins_[i])
        
        figure(figsize=(8,6), dpi = 200)
        plt.plot(self.bins_[1:], df)
        plt.xlabel(r'$q$', fontsize = 15)
        plt.ylabel(r'$\beta \frac{\partial F}{\partial q}$', fontsize=15)
        
        return self.bins_[1:], df
    
def alogsumexp(a, axis=None, b=None):
    
    '''
    Calculates logsumexp using the autograd numpy
    np.log(np.sum(b*np.exp(a)))
    '''
    
    if b is not None:
        if nup.any(b ==0):
            a = a + 0.
            a[b == 0] = -nup.inf
    
    max_a = nup.amax(a,axis=axis,keepdims=True)

    if b is not None:
        b = nup.asarray(b)
        tmp = b*nup.exp(a - max_a)
    else:
        tmp = nup.exp(a-max_a)

    # Suppress warning about log zero
    with nup.errstate(divide='ignore'):
        s = nup.sum(tmp, axis=axis)
    
    out = nup.log(s)
    max_a = nup.squeeze(max_a, axis = axis)

    return max_a + out
