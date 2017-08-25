from lizard import parallel
from lizard import p3m
from numpy.random import RandomState
import numpy as np

def test_pool():
    """ Build a small pool """
    pool, log = parallel.build_pool(4)

def test_parallel_short():
    """ Parellelised short-range force """

    n = 16
    nprocs = 2
    rs = RandomState(seed=123)

    pos = rs.rand(n**3*3)
    print(pos.min(), pos.max())
    pos.shape = (n**3,3)
    wts = 1.0
    
    # First do via pool
    pool, log = parallel.build_pool(nprocs)
    r_split = 1.0/n # guarantee 2^3 block of cells
    r_soft = 0.2/n
    short = parallel.PooledShort(nprocs, wts, pos, r_soft, r_split, pool, log)

    pooled_accel = short.get()

    # now straight
    fs = p3m.get_force_split(r_split, mode='erf', deconvolve_cic=False)
    pairs, accel = p3m.pp_accel(fs, wts, pos, r_soft)

    print('mean accel', np.sqrt(np.square(accel).sum(1).mean())) # should be ~10^4
    
    
    err = accel - pooled_accel
    err_max = np.abs(err).max()
    print('Maximum err', err_max)
    assert(err_max<1e-7)

