import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

def sanity_counts(binwham_object):
    '''
    
    Generate plot for sanity check in terms of count.
    
    Args:
      binwham_object(binwham object)
    '''
    
    a = cc.Ml
    figure(figsize=(8,6), dpi = 200)
    plt.xlabel(r'$bins$', fontsize = 15)
    plt.ylabel(r'$counts$', fontsize = 15)
    plt.plot(a)
    
    figure(figsize=(8,6), dpi = 200)
    plt.xlabel(r'$bins$', fontsize = 15)
    plt.ylabel(r'$counts$', fontsize = 15)
    plt.plot(a)
    plt.ylim(0,100)
    
def kl_divergence(binwham_object,kappa=0.01,T=298,cutoff=0.1):
    '''
    
    Calculate kl divergence for binwham for each window.
    
    Args:
      binwham_object(binwham object)
      kappa(np.float): spring constant in kJ/mol
      T(np.float): absolute temperature
      cutoff(np.float): cutoff value for kl divergence
    
    '''
    cc = binwham_object
    Pil = cc.Pil
    q_bins = cc.bins_
    _,_,F = cc.MLE_optimize()
    beta = 1/298/8.314*1000
    kldivergence = np.zeros(len(Pil),)

    for i in range(len(Pil)):

        Pil_i = cc.Pil[i]
        effective_Pil_i = Pil_i[np.nonzero(Pil_i)]
        # print(np.nonzero(Pil_i))

        effective_bins = q_bins[np.nonzero(Pil_i)]
        # print(effective_bins)

        effective_F = F[np.nonzero(Pil_i)]
        # print(effective_F)

        effective_U = 1/2*kappa*beta*(q_star_arr2[i]-effective_bins)**2
        # print(effective_U)


        max_ = np.max(effective_F + effective_U)
        exp_term = np.exp(-effective_F-effective_U+max_)

        effective_Q_i = exp_term/np.sum(exp_term)

        Pi = effective_Pil_i
        Qi = effective_Q_i

        kldivergence[i] = np.sum(Pi*np.log(Pi/Qi))
        
    counts_ = np.count_nonzero(kldivergence > cutoff)
    if counts_:
        print('there are ',counts_,'values that are over cutoff value',cutoff)
    else:
        print('Each KL divergence is less than cutoff value', cutoff)
    

    figure(figsize=(8,6), dpi=200)
    windoww = list(range(len(kldivergence)))

    plt.scatter(windoww, kldivergence, marker='o')

    plt.xlabel(r'$simulation\  window$', fontsize=15)
    plt.ylabel(r'$KL\  divergence$', fontsize=15)
