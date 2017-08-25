from lizard.lizard_c import *
import sys

def test():
    """ Test that the linear interpolation gives the same result as scipy """
    n = 50 # grid size (n,n,n)
    npts = 10 # number of points to interpolate
    from numpy import arange
    from numpy.random import random
    from scipy.ndimage import map_coordinates
    
    grid = arange(n*n*n).astype(float64)
    grid.shape = (n,n,n)

    # Weird wrapping bug in scipy, it doesnt wrap n-d arrays properly
    coords = random(3*npts)*(n-1) 
    coords.shape = (3,npts)
    
#    coords = array([[0.5],[48.5],[0.5]])
    res = map_coords(grid, coords,sys.stdout)

    res2 = map_coordinates(grid, coords, None, order=1, mode='wrap')

    #print 'res', res
    #print 'res2', res2
    diff = abs(res2 - res)
    print('diff in', diff.min(), diff.max())
    assert(abs(diff).max()<1e-10)
    print('Test on interpolation passed')

def test_cic():
    from numpy import array, equal,all
    ngrid = 3
    pts = array([[1.2,0.6,2.5], [0.5,0.4,0.3]])
    
    masses = array([0.1,0.5])
    cic = get_cic(pts, ngrid, masses)
    expected = array([[[ 0.105,  0.045,  0.   ],
                       [ 0.07 ,  0.03 ,  0.   ],
                       [ 0.   ,  0.   ,  0.   ]],
                      
                      [[ 0.121,  0.045,  0.016],
                       [ 0.094,  0.03 ,  0.024],
                       [ 0.   ,  0.   ,  0.   ]],
                      
                      [[ 0.004,  0.   ,  0.004],
                       [ 0.006,  0.   ,  0.006],
                       [ 0.   ,  0.   ,  0.   ]]])

    err = cic-expected
    print("Diff in", err.min(), err.max())
    assert(abs(err).max()<1e-13)

    # note that Taylor expansion is only exact if 
    # particles move along an axis
    vel = array([[0.1,0.0,0.0],[0,0,-0.2]])
    from numpy import reshape
    cic_grad = get_cic(pts, ngrid, mass=masses, mom=reshape(masses, (len(vel),1))*vel)
    cic1 = get_cic(pts+vel, ngrid, mass=masses)

    err = cic1 - (cic_grad.real + cic_grad.imag)
    print('Diff in', err.min(), err.max())
    assert(abs(err).max()<1e-13)
    print('Test on CIC passed')

def test_5pt_grad():
    from numpy import array, arange, reshape, roll
    n=3
    #pts = reshape(arange(n*n*n)%7, (n,n,n))*4

    pts = array([[[ 0,  4,  8],
                 [12, 16, 20],
                 [24,  0,  4]],
                
                [[ 8, 12, 16],
                 [20, 24,  0],
                 [ 4,  8, 12]],
                
                [[16, 20, 24],
                 [ 0,  4,  8],
                 [12, 16, 20]]])

    
    grad = gradient_5pt_3d(pts)

    grad_expected = array([[[[-6, -9, -3], [-6, 12, 6], [-6, 12, -3]],
                            [[15, 18, -3], [15, -3, 6], [-6, -3, -3]],
                            [[-6, -9, -3], [-6, -9, -15], [-6, -9, 18]]],

                           [[[12, 12, -3], [12, 12, 6], [12, -9, -3]], 
                            [[-9, -3, 18], [-9, -3, -15], [-9, -3, -3]],
                            [[-9, -9, -3], [12, -9, 6], [12, 12, -3]]],
                           
                           [[[-6, -9, -3], [-6, -9, 6], [-6, -9, -3]],
                            [[-6, -3, -3], [-6, -3, 6], [15, -3, -3]],
                            [[15, 12, -3], [-6, 12, 6], [-6, 12, -3]]]])

    err = abs(grad-grad_expected).max()
    print('err on gradient', err)
    assert(err<1e-6)

    # five-point stencil for gradient implemented with numpy roll
    grad_roll = empty((n,n,n,3))
    for i in range(3):
        grad_roll[:,:,:,i] = (-roll(pts,-2,axis=i) + 8*roll(pts,-1,axis=i) - 8*roll(pts,1,axis=i) + roll(pts,2,axis=i)) * (1.0/12.0)
    
    err_roll = abs(grad-grad_roll).max()
    print('err roll', err_roll)
    assert(err_roll<1e-6)
    print('Test for gradients passed')

def test_interp3d():

    import numpy as np
    pts = np.array([[1.2,0.6,3.5], [0.5,0.4,0.3]])

    grid = np.reshape(np.arange(4*4*4*3), (4,4,4,3)) % 15 # make funny vals
    
    # Can just interpolate using the normal 1d in 3 passes
    ans0 = empty(pts.shape)
    for axis in range(3):
        ans0[:,axis] = map_coords(grid[:,:,:,axis], pts.T)
    print('Ans', ans0)
    ans1 = interp_vec3(grid, pts)
    print('Ans1', ans1)

    err = ans0 - ans1
    max_err = abs(err).max()
    print('Max diff', max_err)
    assert(max_err<1e-10)

