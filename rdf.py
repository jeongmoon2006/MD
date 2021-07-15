# bins_, gr = rdf(p.atoms.positions, p.atoms.positions, box, 10, nbins = 100 )

import numpy as np
def rdf(pos_A, pos_B, box, max_radius, n_bins):
    '''
    Input:
      pos_A(numpy.ndarray) : (N, 3) position of A molecules or atoms 
      pos_B(numpy.ndarray) : (M, 3) position of B molecules or atoms
      box(numpy.ndarray) : (3, ) size of box in Å
      max_radius(float in Å) : maximum radius(distance from a molecule)
      n_bins(int) : number of beans
      
    output:
      bins(numpy.ndarray) : (n_bins,) array of bins
      gr(numpy.ndarray) : (n_bins, ) radial distribution function
      nr(numpy.ndarray) : (n_bins, ) number of cumulative molecules
    
    '''
    
    #rdf = dv_N*rho/dv

    #Lets first consider the case where N = M
    N = pos_A.shape[0]
    rho = N /(box[0]*box[1]*box[2]) #total molecule density
    bins = np.linspace(0, max_radius, n_bins)
    dr = bins[1] - bins[0]
#     r = np.append(bins, [max_radius + dr])
    
    #Let's compute distance between molecules (N*N,)
    dist = np.abs(pos_A[:, np.newaxis, :] - pos_B)
    ddist = np.abs(dist - box)
    dist = np.minimum(dist, ddist)
    dist = np.reshape(dist*dist, (N*N,3))
    dist = np.sum(dist,axis = 1)
    dist = dist[ dist != 0]
    dist = np.sqrt(dist)
    dist = np.sort(dist)
    
    p = 0
    i = 0
    m = 0
    nr = np.zeros(n_bins)
    rdf = np.zeros(n_bins)

    while True:
        while dist[i] > bins[p]:
            nr[p] = m
            p = p+1
            if p== len(nr):
                break
        else:
            i = i+1
            m = m+1
            if i == len(dist):
                break
        if p == len(nr):
            break
            
    for i in range (n_bins-1):
        rdf[i+1] = nr[i+1] - nr[i]

    #Let's get volume occupied by dr
    dv = np.zeros(n_bins)
    dv = 4*np.pi*(bins[1:]**2)*dr
    rdf[1:] = rdf[1:] / (dv *rho*N)
    
    return bins, rdf, nr


def rdf_average(array, fs_atom_name, sn_atom_name , n_bins):

    '''
    Input:
      array(numpy.ndarray) : (n_time_frames)  ex) [0,100,200,300...,1000]
      
    output:
      bins(numpy.ndarray) : (n_bins,) array of bins
      gr_average(numpy.ndarray) : (n_bins, ) radial distribution function
      nr_average(numpy.ndarray) : (n_bins, ) number of cumulative molecules
    
    '''
    box = np.array([23, 23, 23])
    
    
    rdf__ = np.zeros(n_bins)
    nrr = np.zeros(n_bins)
    
    for i in array:
        u.trajectory[np.int(i)]
        p = u.select_atoms(fs_atom_name)
        pp = u.select_atoms(sn_atom_name)
        bins, rdf_, nr = rdf(p.atoms.positions, pp.atoms.positions, box, 10, n_bins)
        rdf__ += rdf_
        nrr += nr
        
    return bins, rdf__/len(array), nrr/len(array)
