"""
Periodicity utilities, i.e. padding the unit-cube, and also translating in 
order to minimise the amount of points needed to pad (and consequently kernel 
evalutions)

Peter Creasey - Oct 2016
"""
from __future__ import print_function, absolute_import
import numpy as np
from numpy import cumsum, argsort, floor, concatenate, argmin, zeros, \
    flatnonzero, array, empty
from lizard.log import null_log

def pad_unitcube(pos, r_pad):
    """
    For a set of points in [0,1]^ndim, find the points within r_pad of the 
    edges and repeat them on the opposite side, i.e return a set of positions
    in [-r_pad, 1+r_pad]^ndim. An array with these new positions is returned, 
    along with their corresponding indices in the original array (in case you 
    wanted to clone other properties such as their weights).

    pos   - (N,ndim) array
    r_pad - distance to edges to pad (<=0.5)
    
    returns pad_idx, pad_pos
       pad_idx - indices of the positions used to pad
       pad_pos - the padded positions in [-r_pad, 1+r_pad]
    
    """
    if r_pad>=0.5:
        raise Exception('r_pad>=0.5 too big for repeats!')

    pos = array(pos)
    npts, ndim = pos.shape

    for ax in range(ndim):
        rep_right, rep_left = zeros((ndim,),pos.dtype), zeros((ndim,),pos.dtype)
        rep_right[ax]=1
        rep_left[ax]=-1

        # check those within r_pad of 0 or 1
        rt_orig = flatnonzero(pos[:,ax]<r_pad)
        lt_orig = flatnonzero(pos[:,ax]>=(1-r_pad))

        if ax==0:
            # No results from previous padding
            pad_pos = concatenate((pos[rt_orig]+rep_right, 
                                 pos[lt_orig]+rep_left), axis=0) 
            pad_idx = concatenate((rt_orig, lt_orig))
            continue


        # Some of the *padded* positions may need to be repeated
        rt_pad = flatnonzero(pad_pos[:,ax]<r_pad)
        lt_pad = flatnonzero(pad_pos[:,ax]>=1-r_pad)

        pad_idx = concatenate((pad_idx, rt_orig, lt_orig,
                               pad_idx[rt_pad], pad_idx[lt_pad]))
            
        pad_pos = concatenate((pad_pos, 
                               pos[rt_orig]+rep_right, pos[lt_orig]+rep_left, 
                               pad_pos[rt_pad]+rep_right, pad_pos[lt_pad]+rep_left),
                              axis=0)

    return pad_idx, pad_pos


def shift_pos_optimally(pos, r_pad, log=null_log):
    """
    Shift positions for padding (in the worst case all your points are near
    the corner of your box and you repeat calculations 8x)!
    """
    shift = _optimal_shift(pos, r_pad, log)
    new_pos = pos + shift
    new_pos -= floor(new_pos)
    return new_pos
    
def _optimal_shift(pos, r_pad, log):
    """
    Find the shift for the periodic unit cube that would minimise the padding.
    """
    
    npts, ndim = pos.shape

    # +1 whenever a region starts, -1 when it finishes
    start_end = empty(npts*2, dtype=np.int32)
    start_end[:npts] = 1
    start_end[npts:] = -1

    pad_min = []
    # Go along each axis, find the point that would require least padding
    for ax in range(ndim):
        start_reg = pos[:,ax] - r_pad
        end_reg = pos[:,ax] + r_pad

        # make periodic
        start_reg -= floor(start_reg)
        end_reg -= floor(end_reg)

        # Order from 0-1, add 1 whenever we come into range of a new point, -1
        # whenever we leave
        idx_sort = argsort(concatenate([start_reg, end_reg]))
        region_change = cumsum(start_end[idx_sort])
        
        # Find the minimum
        min_chg = argmin(region_change)
        # Note since this is the minimum trough:
        #  start_end[idx_sort[min_chg]==-1  (a trough)
        #  start_end[idx_sort[min_chg+1]] == +1 (otherwise it wasnt the minimum)

        trough0 = end_reg[idx_sort[min_chg]-npts] # has to be a -1 (i.e. region end)
        if min_chg+1==2*npts:
            trough1 = start_reg[idx_sort[0]]+1 
            mid_trough = 0.5 * (trough0 + trough1)
            mid_trough -= floor(mid_trough)
            
        else:
            trough1 = start_reg[idx_sort[min_chg+1]]
            mid_trough = 0.5 * (trough0 + trough1)

        pad_min.append(mid_trough)

    shift = array([1.0-x for x in pad_min], dtype=pos.dtype)
    print("Best shift", ', '.join('%.3f'%x for x in shift), file=log)
    return shift
        
        
        
