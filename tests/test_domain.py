"""
Test the domain splitting
"""
from __future__ import print_function, absolute_import, division, unicode_literals
from numpy import *
from numpy.random import RandomState
from lizard.lizard_c import *
from lizard.ngb_kernel import *
from lizard.log import VerboseTimingLog
from lizard import domain
from os import path

def test_ghosts():
    """
    adjacent_cell_masks - checking ghosts combinations correct in 3x3x3
    """
    ngrid = 3 # 3x3x3 grid, blocks 0-26
    rcrit = 0.07 # anything 0.07 from neighbour cell is ghost
    pos = [(0.17,0.17,0.17), # centre of block 0
           (0.17,0.17,0.3),  # in block 0, ghost of 1
           (0.17,0.17,0.05), # in 0 ghost of block 2 (periodic)
           (0.17,0.17,0.5),  # centre of block 1
           (0.17,0.3,0.17),  # in 0 ghost of block 3
           (0.5,0.5,0.5),    # centre of block 13
           (0.29,0.29,0.29)] # in 0, ghost of 1,3,4,9,10,12, not 13, as 3*(0.043333^2) > rcrit^2


    cell_ans = (0,0,0,1,0,13,0)
    cells = get_cells(pos, ngrid)
    print('cells', cells)
    assert(tuple(cells)==cell_ans)
    # Mask 1<<n for filled cells
    cell_masks = reshape(1<<arange(27), (3,3,3))
    
    # Mark the cells and cells of which we are a ghost
    mask_ans = (0b1, 0b11, 0b101, 0b10, 0b1001, 0b10000000000000, 0b1011000011011)
    
    masks = adjacent_cell_masks(pos, cell_masks, rcrit)
    
    print('masks', masks)
    print('expected', mask_ans)
    assert(tuple(masks)==mask_ans)


def test_periodic_kernel_split():
    """
    neighbour-force calculation in a single-pass vs multiple domains
    """
    log = VerboseTimingLog()
    rs = RandomState(seed=123)
    ng = 64
    npts = ng**3
    print('Making {:,} points'.format(npts),file=log)    
    pos = reshape(rs.rand(npts*3), (npts,3))
    wts = rs.rand(npts)

    rmax = 0.9/ng
    
    ndomain = 64



    # now do kernel summation in parts and in total
    kernel = setup_newton(rmax, npts=100)
    accel = empty_like(pos)
        
    for idx, non_ghosts in domain.sq_wtd_voxels(ndomain, pos, rmax, ngrid=4, log=log):
        # acceleration in this domain
        pairs, dmn_accel = periodic_kernel(rmax, kernel, pos[idx], wts[idx], log)
        accel[idx[non_ghosts]] = dmn_accel[non_ghosts]

    print('Acceleration all in one pass', file=log)
    pairs, accel_single = periodic_kernel(rmax, kernel, pos, wts, log)
    print('Accel in', accel.min(), accel.max(), file=log)
    print('Accel single in',accel_single.min(), accel_single.max(), file=log)

    err_tol = 1e-9
    err = accel_single-accel
    rms_accel = sqrt(square(accel).sum(1).mean())
    max_err = abs(err).max() / rms_accel
    # some last-bit errors due to different summation order.
    print('Maximum error', max_err,file=log)
    if max_err>err_tol:
        idx = argmax(abs(err).max(axis=1))
        print('Particle', idx, 'Pos', pos[idx])
        print('Acceleration', accel_single[idx], 'on many procs', accel[idx], 'rms', rms_accel)
        dx = pos[idx] - pos
        # nearest image
        dx = remainder(dx+0.5,1) - 0.5
        r = sqrt(square(dx).sum(1))
        ngb = flatnonzero(r<rmax)
        

        print('Neighbours')
        for i in ngb:
            print(i, dx[i], pos[i], r[i])
    assert(max_err<err_tol)



def test_simple_domain():
    """ 
    Test domain splitting via periodic region growing
 
    Test that the biggest peaks are first, and the separated domains are 
    combined.

    Have 2x2x2 block with values
    
    4 0  |  2 2
    0 3  |  1 1
    
    and threshold is 3. Thus split into peaks

    A D  |  C C
    D B  |  D D

    (last domain D is not contiguous)
    """

    wts = (((4,0),(0,3)),((2,2),(1,1)))

    A,B,C,D = 1,2,3,4
    
    doms = (((A,D),(D,B)),((C,C),(D,D)))

    print('Doms', doms)
    res = domain_regiongrow(wts, 3)
    print(res)

    assert((res==doms).all())

def test_sim_domain():
    """
    Test a 16^3 domain from an actual simulation
    """


    # Actual redshift zero particle distribution (in 16^3 voxels)
    name = path.join(path.dirname(path.abspath(__file__)), 'particle_voxel16x16x16.npy')
    
    counts = load(name)
    print('Total number of particles', counts.sum(dtype=int64))
    doms = domain.domain_region_grow(counts, 4)
    uni_doms = tuple(unique(doms))
    print(uni_doms)
    assert(uni_doms==(1,2,3,4))



          
