"""
Module for calculating forces from nearest neighbours (in particular for use 
with a short-range gravitational force). The steps for using this are:

1) Convert all your positions to be within the box [0,1)
2) setup a kernel function up to some rmax (e.g. 0.1), using e.g.
>>> setup_newton(rmax=0.1)
to set up a Newtonian force.
3) Evaluate the radial kernel (non-periodic) for all particles, e.g
>>> pairs, kernel_sum = radial_kernel_evaluate(rmax, pos, wts)

The heavy-lifting is done by C-functions (see lizard_c.py) which puts all the
particles into a hash-grid for quick nearest-neighbour calculations. If you 
want to repeatedly calculate forces then you can ask for the optional sort 
indices to be returned, which allows a better ordering of the particles.

Peter Creasey - Dec 2015
"""
from __future__ import print_function, absolute_import, division, unicode_literals
from lizard.lizard_c import get_cells, lattice_kernel, lattice_setup_kernel, \
    bh_tree_walk, lattice_setup_kernel, build_octrees
from lizard.log import MarkUp as MU, null_log
from lizard.periodic import shift_pos_optimally, pad_unitcube
from numpy import argsort, empty, int32, searchsorted, unique, arange, array, \
    square, empty_like, concatenate, isscalar
import numpy as np
from time import time

def setup_newton(rmax, npts=500):
    """
    Utility function: Set up the newtownian 1/r^2 force up to rmax
    """
    r = (arange(npts)+1) * (float(rmax)/npts) # vals rmax/npts,..., rmax
    kernel = -1.0/(r*r*r) # these are multipliers for dx (magnitude r), so r^-3
    return kernel

def radial_kernel_evaluate(rmax, kernel, pos, wts, log=null_log, sort_data=False, 
                           many_ngb_approx=None):
    """
    Perform evaluation of radial kernel over neighbours. 
    Note you must set-up the linear-interpolation kernel before calling this
    function.
    
    rmax   - radius to evaluate within
    kernel - kernel table
    pos    - (N,3) array of positions
    wts    - (N,) array of weights

    [many_ngb_approx - guess for number of neighbours. If this is included and
                       large, i.e. >140, we will use combine the kernels due to
                       particles in non-adjacent cells (=monopole approximation
                       for the 1/r^2 force)]

    returns pairs, f
    
    where
    pairs - the number of pairs within rmax
    f     - An (N,3) array s.t.

    f_i = Sum_j wts_j (pos_j - pos_i) * kernel(|pos_j - pos_i|)

    """
    pos_arr = array(pos)
    num_pts = len(pos)
    if len(pos) != len(wts):
        raise Exception('Number of weights ({:,}) must be the same as number of points ({:,})'.format(len(wts),num_pts))
    
    stencil = None

    # Choose a stencil based on number of neighbours
    if many_ngb_approx is not None:
        guess_ngb = int(many_ngb_approx)
        if guess_ngb>400:
            # 7x7x7 stencil (only inner 3x3x3 direct)
            stencil = 7 
            ngrid = int(3.0/rmax)
        elif guess_ngb>140:
            # 5x5x5 stencil (inner 3x3x3 direct)
            stencil = 5 
            ngrid = int(2.0/rmax)
        else:
            # 3x3x3, all direct
            ngrid = int(1.0/rmax)
    else:
        ngrid = int(1.0/rmax) # 3x3x3 by direct summation

    # Avoid nasty hashing problems, make sure ngrid&3 == 3
    if ngrid&3!=3 and ngrid >=3:
        ngrid = (ngrid//4)*4 -1 

    print('Using hash grid of size {:,}^3 bins, binning particles.'.format(ngrid), file=log)

    cells = get_cells(pos_arr, ngrid, log)
    sort_idx, cellbin_data = _bin_id_data(cells, log)

    pos = pos_arr[sort_idx].copy()
    wts= array(wts)[sort_idx].copy()                   
    print(MU.OKBLUE+'Kernel evalations at {:,} positions'.format(num_pts)+MU.ENDC, 
          file=log)
    t0 = time()
    lattice_setup_kernel(rmax, kernel, log)
    pairs, accel = lattice_kernel(pos, cellbin_data, ngrid, masses=wts, stencil=stencil)
    t1 = time()
    dt = t1-t0
    if stencil is None:
        mean_num_ngb = pairs * 2.0 / num_pts
        print('Within r=%.4f, mean number of neighbours was'%rmax,
              MU.OKBLUE+'%.2f'%(mean_num_ngb)+MU.ENDC, file=log)
        
        print('{:,} pairs in'.format(pairs), '%.2f seconds'%dt, 
              'i.e. {:,} positions/sec, {:,} kernels/sec'.format(int(num_pts/dt), int(2*pairs/dt)), file=log)
    else:
        print('%dx%dx%d monopole approximation, so no exact count for neighbours\n'%((stencil,)*3), 
              'but {:,} force-pairs in'.format(pairs), '%.2f seconds'%dt, 
              'i.e. {:,} positions/sec, {:,} kernels/sec'.format(int(num_pts/dt), int(2*pairs/dt)), file=log)

    if sort_data:
        # return the sort index along with sorted positions and masses, and corresponding accelerations.
        # If you want to unsort you need to do it yourself
        return pairs, sort_idx, pos, wts, accel

    # indices for 'un'-sorting
    unsort = empty_like(sort_idx)
    unsort[sort_idx] = arange(num_pts)

    return pairs, accel[unsort]


def _bin_id_data(ids, log):
    """
    Helper-function to bin up the ids. 
    ids - the ID for N items
    log - write out debug data

    returns sort_idx, bin_data

    s.t.
    sort_idx - an array such that ids[sort_idx] is sorted
    bin_data - a (Q,3) array such that Q is the number of unique IDs, 
        bin_data[i,0] is the count of the i-th smallest ID and
        ids[sort_idx][bin_data[i,1]:bin_data[i,2]] would all be the i-th
        smallest ID.
               
    Since these indices are used in the C-function, in principle you can cause 
    a seg-fault if you get them wrong. Mess with this function at your own peril!
    """

    num_pos = len(ids)
    # TODO may be able to do better with optional arguments to np.unique in Numpy>1.9
    sort_idx = argsort(ids) 
    filled_cells = unique(ids)
    num_bins = len(filled_cells)
    bin_data = empty((num_bins,3), dtype=int32)

    bin_data[:,0]   = filled_cells
    bin_data[:,1]   = searchsorted(ids[sort_idx], filled_cells)
    bin_data[:-1,2] = bin_data[1:,1]
    bin_data[-1,2]  = len(ids)

    av_bin_fill = float(num_pos)/num_bins
    bin_fill = bin_data[:,2] - bin_data[:,1]
    av_fill_per_id = float(square(bin_fill).sum())/num_pos
    print('Put {:,} IDs into {:,} bins'.format(num_pos, num_bins),
          'of fill {:,}-{:,} IDs,'.format(bin_fill.min(), bin_fill.max()), 
          'average %.2f.'%av_bin_fill,
          'Average ID lives in a bin of fill %.2f IDs'%av_fill_per_id, file=log)
    return sort_idx,bin_data


def periodic_kernel(rmax, kernel, pos, wts, log=null_log):
    """
    like radial but for periodic [0,1) box.
    """
    if rmax>=0.5:
        raise Exception('Cannot have rmax greater than half the box size, could get periodic images')

    num_pts = len(pos)
    pos = array(pos)
    wts = array(wts)

    print('Finding optimal shift',file=log)
    pos = shift_pos_optimally(pos, rmax, log)
    print('Padding the unit cube', file=log)
    pad_idx, pad_pos = pad_unitcube(pos, rmax)

    print('Inserted {:,} ghost particles for periodicity'.format(len(pad_idx)),file=log)
    new_pts = concatenate((pos, pad_pos), axis=0)

    if sum(wts.shape)<=1:
        new_wts = empty(len(new_pts), dtype=wts.dtype)
        new_wts[:] = wts
    else:
        new_wts = concatenate((wts, wts[pad_idx]))

    # Scale everything to be in the new box
    scale_fac = 1.0 / (1+2*rmax) 
    new_pts += rmax
    new_pts *= scale_fac

    pairs, sort_idx, pos, wts, accel = radial_kernel_evaluate(rmax*scale_fac, kernel, new_pts, new_wts, log=log, sort_data=True)

    # unsort only the real points
    unsort = empty_like(sort_idx)
    unsort[sort_idx] = arange(len(new_pts))
    unsort = unsort[:num_pts]

    accel = accel[unsort]

    # undo the scale factor (remember dx's were all shortened)
    accel *= 1.0/scale_fac

    return pairs, accel


def radial_BH_octree_kernel_evaluate(rmax, kernel, pts, wts, theta, log=null_log, sort_data=False, bucket_size=11, force_ngrid=None):
    """
    Like radial_kernel_evaluate but puts particles into a hash-grid of octrees

    theta - opening angle criteria (theta * |cell_CoM - x| < cell_width)
    [bucket_size=11, sweet spot for switchover to direct evaluation]

    """

    if force_ngrid is None:
        ngrid = max(int(1.0/rmax), 1)

        # Avoid nasty hashing problems, make sure ngrid&3 == 3
        if ngrid&3!=3 and ngrid >=3:
            ngrid = (ngrid//4)*4 -1 
    else:
        if force_ngrid*rmax>1.0:
            raise Exception('ngrid=%d has cells smaller than rmax=%.7f'%(force_ngrid,rmax))
        ngrid = force_ngrid

    print('Using grid of size {:,}^3 bins, building octree down to buckets of size {:,}.'.format(ngrid, bucket_size), file=log)
    tree, sort_idx =  build_octrees(pts, bucket_size, ngrid, wts, log)
    print('Initialising kernel', file=log)    
    lattice_setup_kernel(rmax, kernel, log)
    print('BH kernel calculation on {:,} pts'.format(len(pts)),file=log)
    t0 = time()
    n_kernels, accel = bh_tree_walk(tree, ngrid, theta, tree.xyzw, log=log)
    dt = time() - t0
    print('Total kernels {:,} for {:,} pts at'.format(n_kernels, len(pts)),
          MU.OKBLUE+'{:,} pts/sec'.format(int(len(pts)/dt))+MU.ENDC, file=log)


    if sort_data:
        # return the sort index along with sorted positions and masses, and corresponding accelerations.
        # If you want to unsort you need to do it yourself
        return n_kernels, sort_idx, accel

    # indices for 'un'-sorting
    unsort = empty_like(sort_idx)
    unsort[sort_idx] = arange(len(pts), dtype=np.int32)

    return n_kernels, accel[unsort]

if __name__=='__main__':
    # Performance testing for the radial force
    n = 250000
    rcrit = 0.011*1.5

    from numpy.random import RandomState
    import numpy as np
    rs = RandomState(seed=123)

    pts = np.reshape(rs.rand((n*3)), (n,3))*0.25+0.5 # 1/64th of simulation volume
#    pts = np.reshape(rs.rand((n*3)), (n,3))
    mass = np.ones(n)

    kernel = setup_newton(rcrit, npts=100)

    pairs, accel = periodic_kernel(rcrit, kernel, pts, mass)
    
