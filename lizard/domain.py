"""
Split the unit box into multiple domains.

Peter Creasey - Feb 2016

"""

from __future__ import print_function, division, unicode_literals, absolute_import
from lizard.lizard_c import *
from lizard.log import MarkUp, null_log
from time import time
import numpy as np
from scipy import ndimage

def sq_wtd_voxels(ndomains, pos, rcrit, ngrid=None, log=null_log):
    """
    
    Split the periodic unit box into voxels, then assign those voxels to ndomains
    domains such that the sum of squares of particles (an approximate measure 
    of complexity) is the same for each domain.

    ndomains - max number of domains
    pos      - (N,3) positions in [0,1)
    rcrit    - the distance to include ghosts
    [ngrid]  - split the unit box into ngrid^3 voxels
    [log]    - file-like

    return generator which yields pairs of arrays (idx, non_ghosts) indices for 
    particles and their ghosts, e.g.
    
    for idx, non_ghosts in weighted_boxes(...):
       accel[idx[non_ghosts]] = func(particle_data[idx])[non_ghosts]

    """

    if ngrid is None:
        # To make each cell non-ghost dominated would really like
        # 0.1/rcrit, but I rely a bit on adjacent cells being in 
        # the same domain.
        ngrid = int(0.15/rcrit)

    if ndomains>64:
        raise Exception('{:,}>64, maximum domains for single master'.format(ndomains))

    tot_cells = ngrid**3
    cells = get_cells(pos, ngrid, log)
    assert(cells.min()>=0)
    assert(cells.max()<tot_cells)
    counts = np.bincount(cells, minlength=tot_cells)
    cell_ids = np.arange(tot_cells)

    doms = domain_region_grow(np.reshape(counts, (ngrid,ngrid,ngrid)), ndomains)-1
    assert((doms>=0).all()) # check we covered all the domains
    cells_per_dmn = np.bincount(doms.ravel())
    tot_per_dmn = np.bincount(doms.ravel()[cells])
    print("Domain totals:", ', '.join('%d'%npts for npts in tot_per_dmn), file=log)

    cell_masks = 1<<(doms.astype(np.uint64))
    pos_masks = cell_masks.ravel()[cells].copy()

    prettyprint_cellmasks(cell_masks, log)
    # 'Bleed' the cell masks to find ghosts
    ngb_masks = adjacent_cell_masks(pos, cell_masks, rcrit, log)

    for dom, mask in enumerate(np.unique(pos_masks)):
        # find all the positions needed in this domain
        idx_domain = np.flatnonzero(ngb_masks & mask)
        # which of these are non-ghosts
        idx_non_ghosts = np.flatnonzero(np.equal(pos_masks[idx_domain], mask))

        pct_ghosts = (100*(len(idx_domain)-len(idx_non_ghosts)))//len(idx_domain)

        txt = 'Domain %d: '%dom + \
            '{:,} particles of which {:,}% are ghosts ({:,} domain cells)'.format(len(idx_domain),
                                                                                  pct_ghosts, cells_per_dmn[dom])
        # count parts+ghosts
        if pct_ghosts<50:
            print(txt, file=log)
        else:
            print(MarkUp.WARNING+txt+MarkUp.ENDC, file=log)            
        yield idx_domain, idx_non_ghosts

def prettyprint_cellmasks(doms, log):
    n = doms.shape[0]
    # squash onto 3 projections
    max_ijk = [doms.max(axis=axis) for axis in range(3)]
    min_ijk = [doms.min(axis=axis) for axis in range(3)]
    proj_dom = [np.where(max_dom==min_dom, max_dom, -1) for max_dom, min_dom in zip(max_ijk, min_ijk)]

    print('\n '+'   '.join((1+2*n)*'-' for axis in range(3))+' ', file=log)
    for j in range(n):
        print('|', end='', file=log)
        print(' | |'.join(''.join({False:MarkUp.OKBLUE+'%2d'%d+MarkUp.ENDC,
                                 True:' #'}[d<0] for d in row) for row in [proj[:,j] for proj in proj_dom]), 
              end='',file=log)
        print(' |', file=log)
    print(' '+'   '.join((1+2*n)*'-' for axis in range(3))+' \n', file=log)


def find_halos(pos, ngrid, log, level=3000):
    """
    TODO make this account for periodicity
    """
    print('Binning particles', file=log)
    cells = get_cells(pos, ngrid, log)
    count = bincount(cells, minlength=ngrid**3)
    count.shape = (ngrid,ngrid,ngrid)
    print('Count in', count.min(), count.max(), file=log)
    idx = flatnonzero(count>level)
    print('Number of cells above', level, 'is', len(idx), file=log)
    
    
    labels, num_features = ndimage.label(count>level)
    print('Number fo features', num_features, file=log)
    print('Labels in', labels.min(), labels.max(), file=log)
    locations = ndimage.find_objects(labels)

    dense_regions = []

    for i in range(num_features):
        loc = locations[i]
        hw = max(l.stop - l.start for l in loc) * 0.5 /ngrid
        hw_padded = hw + 0.0/ngrid

        ctr =[(0.5/ngrid)*(l.stop + l.start) for l in loc]
        count_i = count[loc][labels[loc]==(i+1)].sum()
        print('Count', count_i, file=log)
        dense_regions.append((count_i, ctr, hw_padded))

    # sort by number of particles in the region
    dense_regions = sorted(dense_regions, key = lambda num_ctr_hw :num_ctr_hw[0], reverse=True)

    return dense_regions

def domain_region_grow(counts, ndomain):
    """
    Make the domains via region-growing
    """
    
    # make it so maximum value is 100,000,000 (fits well inside int32)
    norm = 10000.0 / counts.max()
    wts = np.square(norm * counts).astype(int32)
#    wts = ((norm * counts)**1.0).astype(int32)
    
    tot = wts.sum(dtype=int64)
    
    sum_per_domain = tot//ndomain + 1
    
    doms = domain_regiongrow(wts, sum_per_domain)
    return doms


def load_partition_1d(counts, n_dom, split_fac):
    """
    Load split in 1d

    counts - an array of length M+2. counts[0] and counts[M+1] are edge
             (ghost) values. 
    n_dom      - >1 the total number of domains we are splitting (want an integer
             number either side of split)

    split_fac - how many fractional extra particles you are likely to 
                introduce by e-folding, used to determine how bad/good
                non-binary splitting is


    returns

    split  - a value from 1...M-1 indicating where to partition (the unghosted)
             array
    n_left - number of domains to use on left side
    pval   - some estimate of worst number of particles per/domain you will end
             up with (including ghosts)

    """
    count_sums = np.cumsum(counts)
    # If we split at n+1, how many points on left and right?
    ptsL = count_sums[1:].astype(np.float64)
    ptsR = np.empty_like(ptsL)
    ptsR[:] = count_sums[-1] 
    ptsR[1:] -= count_sums[:-2]


    # Best split of domains (+/- 1)
    left0 = np.clip(((n_dom * ptsL)/(ptsL+ptsR)).astype(np.int32), 1, n_dom-1)
    right0 = n_dom-left0

    left1 = np.minimum(left0+1,n_dom-1)
    right1 = n_dom-left1

    # whats the worst (left/right) load balance?
    p_per_proc0 = np.maximum((1+split_fac*np.log(left0))*ptsL/left0, (1+split_fac*np.log(right0))*ptsR/right0)
    p_per_proc1 = np.maximum((1+split_fac*np.log(left1))*ptsL/left1, (1+split_fac*np.log(right1))*ptsR/right1)

    idx_min0 = np.argmin(p_per_proc0)
    idx_min1 = np.argmin(p_per_proc1)
    
    if p_per_proc0[idx_min0] < p_per_proc1[idx_min1]:
        split = idx_min0
        n_left = left0[idx_min0]
        pval = p_per_proc0[idx_min0]
    else:
        split = idx_min1
        n_left = left1[idx_min1]
        pval = p_per_proc1[idx_min1]

    if split==0 or split==len(counts)-2:
        raise Exception('Tried to make a domain of pure ghosts. Something bad happened?')


    return split, n_left, pval

def bisect_anyaxis(counts, ndomains, split_fac):
    """
    For a given cuboid, search for a binary partition along any axis (e.g. 
    0-2) with which one can partition the processors and has a low average 
    value of parts/proc.
    
    counts    - an (I+2,J+2,K+2) shaped array (includes boundary)
    ndomains  - number of domains to ultimately use

    returns axis, split_idx, n_L
   
    """
    # split along any axis        
    splits = {}
    pvals = []
    for axis in range(len(counts.shape)):
        # Sum over other axes
        sum_axes = list(np.arange(len(counts.shape)))
        sum_axes.pop(axis)
        sum_axes = tuple(sum_axes)

        # split into left and right 
        counts1d = np.sum(counts, axis=sum_axes, dtype=np.int64)
    
        split_idx, n_L, pval = load_partition_1d(counts1d, ndomains, split_fac)

        splits[axis] = (split_idx, n_L)

        pvals.append(pval)

    axis = int(np.argmin(pvals))
    split_idx, n_L = splits[axis]
    return axis, split_idx, n_L


def kd_domain_split(counts_all, ndomains, log=null_log):
    """
    
    Split domain into hypercubes via recursive bisection (i.e. kd-tree).

    Nothing particularly magical about this, just recursively splitting a 
    cuboid into sub-cuboids along the longest axis and assigning a number of
    domains to each side such that the load balance approximately equal,
    with some splitting factor to describe the extra ghosts introduced via
    splitting. Should get you down to about ~36% ghosts for the worst cuboid if
    theyre clustered.


    """

    split_fac = 1.35 * (float(ndomains)/np.cumprod(counts_all.shape)[-1])**(1.0/3.0)
    print('split factor', split_fac, file=log)
    # First translate the box so 0,0,0 in best posn to minimise communication
    total_shifts = []
    for axis in range(3):
        # Sum over other axes
        sum_axes = list(np.arange(len(counts_all.shape)))
        sum_axes.pop(axis)
        sum_axes = tuple(sum_axes)

        count_ax = counts_all.sum(axis=sum_axes, dtype=np.int64)
        # amount communicated per plane
        comm = count_ax + np.roll(count_ax, 1)

        total_shifts.append(np.argmin(comm))


    for axis, r in enumerate(total_shifts):
        counts_all = np.roll(counts_all, shift=-r, axis=axis)

    print('Best shifts', total_shifts, file=log)


    # pad
    counts_pad = np.empty(tuple(v+2 for v in counts_all.shape), dtype=counts_all.dtype)
    counts_pad[1:-1,1:-1,1:-1] = counts_all
    counts_pad[1:-1,1:-1,0] = counts_pad[1:-1,1:-1, -2]
    counts_pad[1:-1,1:-1,-1] = counts_pad[1:-1,1:-1,1]
    counts_pad[1:-1,0] = counts_pad[1:-1, -2]
    counts_pad[1:-1,-1] = counts_pad[1:-1, 1]
    counts_pad[0] = counts_pad[-2]
    counts_pad[-1] = counts_pad[1]


    domain_segments = []

    doms_tosplit = [((0,0,0), counts_pad, ndomains)]

    while len(doms_tosplit):
        dom_topleft, counts, ndom = doms_tosplit.pop(0)

        if ndom==1:
            # done
            dom_shape = tuple(v-2 for v in counts.shape)
            domain_segments.append((dom_topleft, dom_shape, counts.sum(dtype=np.uint64)))
            continue

        # Bisect this domain 
        axis, split_idx, n_L = bisect_anyaxis(counts, ndom, split_fac)

        n_R = ndom-n_L

        if axis==0:
            counts_L, counts_R = counts[:split_idx+2], counts[split_idx:]
        elif axis==1:
            counts_L, counts_R = counts[:,:split_idx+2], counts[:,split_idx:]        
        elif axis==2:
            counts_L, counts_R = counts[:,:,:split_idx+2], counts[:,:,split_idx:]
        else:
            raise Exception('3d only, aaargh.')

        # add left and right domains
        doms_tosplit.append((dom_topleft, counts_L, n_L))

        # top left of right domain
        dom_R_topleft = list(dom_topleft)
        dom_R_topleft[axis] += split_idx
        dom_R_topleft = tuple(dom_R_topleft)

        doms_tosplit.append((dom_R_topleft, counts_R, n_R))


    # sort domains biggest->smallest
    domain_segments = sorted(domain_segments, key=lambda ijk_shape_pts:-ijk_shape_pts[2])

    doms = np.empty(counts_all.shape, dtype=np.int16)

    for d,(ijk, shape, tot_pts) in enumerate(domain_segments):
        segment = tuple(slice(i,i+size) for i,size in zip(ijk, shape))
        doms[segment] = d+1
        real_pts = counts_all[segment].sum(dtype=np.int64)
#        print('domain', d, 'shape', shape, '{:,} pts, {:,} total'.format(real_pts, tot_pts), file=log)

    # Undo the total shifts
    for axis, r in enumerate(total_shifts):
        doms = np.roll(doms, shift=r, axis=axis)
    
    return doms


def analyse_doms(doms, counts, log):
    """
    only used for test. Will need a version for actual run though.
    """
    dom_masks = 1<<(doms.astype(np.uint64))

    # initialise space for masks
    ngb_masks = np.zeros_like(dom_masks)

    # Fill in the masks of all the neighbouring cells
    inc = [1,0,-1] # roll increments for left, middle, right
    for i in inc:
        ri = np.roll(dom_masks, i, axis=0)
        for j in inc:
            rj = np.roll(ri, j, axis=1)
            for k in inc:
                ngb_masks |= np.roll(rj, k, axis=2)



    count_ds, count_alls, pcts = [], [], []
    
    for d in range(doms.max()+1):
        idx = np.flatnonzero(doms==d)
        idx_all = np.flatnonzero(ngb_masks&(1<<d))
        
        count_d = counts.ravel()[idx].sum()
        count_all = counts.ravel()[idx_all].sum()
        
        pct_ghosts = ((count_all - count_d)*100)//count_all
        pcts.append(pct_ghosts)
        print('Domain %2d'%d, 'has {:,} real points, {:,} total of which'.format(count_d, count_all), 
              '%d%% are ghosts'%pct_ghosts, file=log)

        count_ds.append(count_d)
        count_alls.append(count_all)



    print('Total particles {:,}, total evaluated {:,} (average ghosts {:,}%)'.format(sum(count_ds), sum(count_alls), ((sum(count_alls)-sum(count_ds))*100)//sum(count_alls)), file=log)
    print('maximum {:,} on a single proc, worst ghost percentage {:,}%'.format(max(count_alls), max(pcts)), file=log)

def test(count_file, ndomain=64):
    """
    Domain splitting on a 3d histogram of counts
    """

    from lizard.log import VerboseTimingLog
    log = VerboseTimingLog()
    print('loading counts', file=log)
    counts = np.load(count_file).astype(np.int64)
#    counts = np.ones((85,85,85),dtype=np.int64) * (256**3 / (85*85*85))
    print('kd-split to find cuboid domains', file=log)
    doms = kd_domain_split(counts, ndomain, log=log)

    print('Doms in', doms.min(), doms.max(), file=log)


    doms = doms-1
    analyse_doms(doms, counts, log)



if __name__=='__main__':
    test('/Users/pec27/temp/sim256_z0_counts.npy')
