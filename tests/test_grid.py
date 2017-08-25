from __future__ import print_function, division, unicode_literals, absolute_import
from numpy import *
from lizard.grid import *
import sys

def test_k_func():
    """ kfunc_on_grid - fft weights """
    L = 100.0
    k0_val = 0.0 
    func = lambda x : (cos(x) - sin(x)/x)/(4*pi)

    log = sys.stdout
    
    for n in [12,13]: # good to test both odd and even cases
        res0 = kfunc_on_grid(L, n, func, k0_val)


        print('making k', file=log)
        k, kmag, inv_k2 = make_k_values(L, n)

        print('calling func', file=log)
        res1 = empty(kmag.size, dtype=float64)
        res1[1:] = func(kmag.ravel()[1:])
        res1[0] = k0_val
        res1.shape = kmag.shape
        print('done', file=log)
        # comparison
        err = abs(res1-res0).max()
        idx = argmax(abs(res1-res0))
        
        print(abs(res0).max(), 'error', err, 'idx', ((idx//(n*n))%n, (idx//n)%n, idx%n), file=log)
        if err>1e-13:

            idx = flatnonzero(abs(res1.ravel()-res0.ravel()))[0]
            print('first error', ((idx//(n*n))%n, (idx//n)%n, idx%n), file=log)
#        print('res0',res0,file=log)
#        print('res1',res1,file=log)
        assert(err<1e-13)

# you'll want to run "nosetests tests/test_grid.py" to hit this (only)
