"""
Friends-of-Friends (FOF) for N-body simulations

Peter Creasey - Oct 2016

"""
from __future__ import absolute_import, print_function
from lizard.periodic import pad_unitcube
from scipy.spatial import Delaunay
from scipy.sparse import csr_matrix, csgraph
from numpy import square, flatnonzero, ones, zeros_like, cumsum, concatenate, \
    arange, searchsorted, bincount, sort, diff, int8, argsort, array
from lizard.log import MarkUp, null_log

def fof_groups(pos, b, log=null_log):
    """
    Friends-of-Friends on the period unit cube

    pos - (n,ndim) positions in [0,1]^ndim
    b   - linking length

    returns labels - (n,) array of integers for each connected component.

    This FoF algorithm computes the fixed radius connectivity by computing the 
    Delaunay tesselation (DT) for each link and then breaking those links that are 
    too long. 

    The reason this works is that the Relative Neighbourhood Graph (RNG) is a 
    subgraph of the DT, and so any pair of points separated by a distance R will be 
    connected by links of <R, and so it is enough to use the DT to establish
    connectivity.
    
    """
    print('Padding the unit cube', file=log)
    pad_idx, pad_pos = pad_unitcube(pos, b)
    all_pos = concatenate((pos, pad_pos), axis=0) + b
    all_pos *= 1.0/(1+2*b)
    b_scaled = b/(1+2*b)

    print('Added {:,} points, performing'.format(len(pad_idx)),
          MarkUp.OKBLUE+'Delaunay tesselation'+MarkUp.ENDC, 
          'of {:,} points'.format(len(all_pos)), file=log)
    dlny = Delaunay(all_pos)

    # construct list of links
    indptr, indices = dlny.vertex_neighbor_vertices
    idx1 = zeros_like(indices)
    idx1[indptr[1:-1]] = 1
    idx1 = cumsum(idx1)
    idx2 = indices

    print('{:,} links, disconnecting those with r>%.5f'.format(len(indices))%b, file=log)

    # find all links < b using square distance
    dist2 = square(all_pos[idx1] - all_pos[idx2]).sum(1)

    del dlny

    keep = flatnonzero(dist2<float(b_scaled*b_scaled))

    idx1, idx2 = idx1[keep], idx2[keep]

    print('{:,} links left, removing periodic images'.format(len(idx1)), file=log)
    # Make the map back to the original IDs
    old_id = arange(len(all_pos))
    old_id[len(pos):] = pad_idx
    idx1, idx2 = old_id[idx1], old_id[idx2]

    # remove repeats
    idx_sort = argsort(idx1*len(pos)+idx2)
    idx1,idx2 = idx1[idx_sort], idx2[idx_sort]
    if len(idx1)>0:
        keep = array([0] + list(flatnonzero(diff(idx1) | diff(idx2))+1), dtype=idx2.dtype)
        idx1, idx2 = idx1[keep], idx2[keep]

    # make a sparse matrix of connectivity

    print('{:,} links, building sparse matrix'.format(len(idx1)), file=log)

    indices = idx2
    indptr = searchsorted(idx1, arange(len(pos)+1))

    mat = csr_matrix((ones(len(indices), dtype=int8), indices, indptr), 
                     shape=(len(pos), len(pos)))

    print('Finding connected components',file=log)
    n_comps, labels = csgraph.connected_components(mat, directed=False)

    print('From {:,} links between {:,} points found {:,} connected components'.format(len(idx1), len(pos), n_comps), file=log)

    show_largest = min(n_comps, 3)
    npts = sort(bincount(labels))[-show_largest:]
    print('{:,} largest'.format(show_largest), MarkUp.OKBLUE+'FoF groups'+MarkUp.ENDC,
          'have', MarkUp.OKBLUE+' '.join('{:,}'.format(i) for i in npts), 
          'points'+MarkUp.ENDC, file=log)
    return labels
    
    
def test_labels():
    """ Test with some 64^3 data """
    from lizard.log import VerboseTimingLog
    log = VerboseTimingLog()
    import numpy as np
    parts = np.load('/mainvol/peter.creasey/bigdata/runs/test_const_pmkick/out/lizard_snap_134.npz')
    pos = parts['pos']
    boxsize = 5600
    nbox = len(pos)**(1.0/3.0)
    print(pos.max(axis=0), boxsize, nbox, file=log)


    labels = fof_groups(pos*(1.0/boxsize), b=0.2/nbox, log=log)

    print('labels in', labels.min(), labels.max(), file=log)
    bins = np.bincount(labels)
    
    part_lim = 20 # ignore anything with < part_lim particles
    NO_FOF = labels.max()+1
    newlab = np.where(bins[labels]<part_lim, NO_FOF, np.arange(len(bins))[labels])
    bins = bincount(newlab)
    halo_counts = sort(bins[:NO_FOF-1])
    print('halo counts', halo_counts[-10:][::-1], file=log)

    # Top 10
    idx = []
    lab_sort = np.argsort(bins[:NO_FOF-1])
    import pylab as pl
    for i in range(50):
        lab = lab_sort[-i-1]
        idx_i = np.flatnonzero(labels==lab)
        
        pl.plot(pos[idx_i][:,2], pos[idx_i][:,1], marker=',', ls='none')
    pl.xlim(0,5600)
    pl.ylim(0,5600)
    pl.show()

def test_random_dist(n=64):
    """ Random n^3 point placement """
    from lizard.log import VerboseTimingLog
    log = VerboseTimingLog()

    from numpy.random import RandomState
    rs = RandomState(seed=123)

    pos = rs.rand(3*(n**3)).reshape((n**3,3))
    fof_labels = fof_groups(pos, b=0.2/n, log=log)
if __name__=='__main__':
#    test_labels()
    test_random_dist(n=100)
