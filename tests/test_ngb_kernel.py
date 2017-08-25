from __future__ import print_function
from lizard.ngb_kernel import *
import numpy as np

def test_newtonian_force():
    """
    Test the newtonian force for 3 particles
    """

    pts = [(0.5,0.5,0.5), (0.5,0.6,0.5), (0.55, 0.55, 0.55)]
    mass = [1,2,3] # different masses to be sure
    rcrit = 0.12 # cutoff for neighbours - note they are all within this
    
    # Newtonian accelerations (from direct sum)
    newtonian = np.array([[ 230.94010768,  430.94010768,  230.94010768],
       [ 230.94010768, -330.94010768,  230.94010768],
       [-230.94010768,   76.98003589, -230.94010768]])

    # magnitude of accelerations
    acc_mag = np.sqrt(np.square(newtonian).sum(1))
    
    kernel = setup_newton(rcrit, npts=100) # about 0.01% accurate

    pairs, accel = radial_kernel_evaluate(rcrit, kernel, pts, mass)

    assert(pairs==3) # 3 pairs of forces

    acc_diff_mag = np.sqrt(np.square(accel-newtonian).sum(1))
    accel_err = acc_diff_mag/acc_mag
    
    print('Maximum relative error', accel_err.max())
    assert(accel_err.max()<1e-3)

def test_periodic():
    """
    Test periodic kernel evaluation
    """

    # Exactly as for test_newtonian(), except shifted so neighbouring 
    # points lie on the other side of the box
    pts = [(0.98,  0.98,  0.98), ( 0.98,  0.08,  0.98),( 0.03,  0.03,  0.03)]

    mass = [1,2,3] # different masses to be sure
    rcrit = 0.12 # cutoff for neighbours - note they are all within this
    
    # Newtonian accelerations (from direct sum)
    newtonian = array([[ 230.94010768,  430.94010768,  230.94010768],
       [ 230.94010768, -330.94010768,  230.94010768],
       [-230.94010768,   76.98003589, -230.94010768]])

    # magnitude of accelerations
    acc_mag = np.sqrt(np.square(newtonian).sum(1))
    
    print('Points', repr(pts))

    kernel = setup_newton(rcrit, npts=100)
    pairs, accel = periodic_kernel(rcrit, kernel, pts, mass)

#    assert(pairs==24) # Now have forces between ghosts
    print('Newtonian', newtonian)
    print('New', accel)
    acc_diff_mag = np.sqrt(square(accel-newtonian).sum(1))
    accel_err = acc_diff_mag/acc_mag
    
    print('Maximum relative error', accel_err.max())
    assert(accel_err.max()<1e-3)

def test_monopole():
    """
    Monopole approximation for non-adjacent neighbours

    Makes a 1/r density profile in particles and compares Newtonian accelerations 
    via direct summation and approximated
    """
    import lizard.log as lg

    rs = np.random.RandomState(seed=122)
    npts = 16**3 
    pos = np.reshape(rs.rand(npts*3)-0.5, (npts,3))
#    pos = np.reshape(np.mgrid[:16,:16,:16], (3,16**3)).T * (1.0/16) - 0.5
    wts = np.ones(npts) * (1.0/npts)
    r = np.sqrt((pos**2).sum(1))

    # r->r^3/2 scale to be 1/r
    r_split = 0.1
    pos *= np.sqrt(0.75)*r_split*np.reshape(np.sqrt(r), (npts,1))
    pos += 0.5
    pos = pos - np.floor(pos)
    
    max_rad = square(pos-0.5).sum(1).max()

    log = lg.VerboseTimingLog()

    print('Max distance of particle from centre', max_rad/r_split, 'r_split, should be 0.5', file=log)
    assert(max_rad<=r_split*0.5)

    
    kernel_pts = 1000


    kernel = setup_newton(r_split, kernel_pts)

    pairs, acc0=radial_kernel_evaluate(r_split, kernel, pos, wts, log=log) #force direct
    pairs, acc5=radial_kernel_evaluate(r_split, kernel, pos, wts, many_ngb_approx=200, log=log) #force 5x5x5
    pairs, acc7=radial_kernel_evaluate(r_split, kernel, pos, wts, many_ngb_approx=1000, log=log)#force 7x7x7
    
    acc_rms = (np.square(acc0).mean()*3)**0.5
    acc_mag = (np.square(acc0).sum(1))**0.5
    err5_mag = np.square(acc5-acc0).sum(1)**0.5
    err7_mag = np.square(acc7-acc0).sum(1)**0.5
    err5_rms = (err5_mag*err5_mag).mean()
    err7_rms = (err7_mag*err7_mag).mean()

    print('For 5x5x5, error in', err5_mag.min(), err5_mag.max(), 'err RMS/RMS', err5_rms/acc_rms, 'Max/RMS', err5_mag.max()/acc_rms, 'Max/mag', err5_mag.max()/acc_mag[np.argmax(err5_mag)], file=log)
    print('For 7x7x7, error in', err7_mag.min(), err7_mag.max(), 'err RMS/RMS', err7_rms/acc_rms, 'Max/RMS', err7_mag.max()/acc_rms, 'Max/mag', err7_mag.max()/acc_mag[np.argmax(err7_mag)], file=log)

    worst = np.argmax(err5_mag)
    print('idx of worst in 5^3', worst, 'accel 5', acc5[worst], 'exact', acc0[worst], 'accel 7', acc7[worst], file=log)
    print('\n', file=log)

    # Maximum acceleration error / RMS acc < 1.5%
    assert(err5_mag.max()/acc_rms < 0.015)
    assert(err7_mag.max()/acc_rms < 0.015)
    # RMS error / RMS acc < 1.5%
    assert(err5_rms/acc_rms < 0.015)
    assert(err7_rms/acc_rms < 0.015)
    
def test_bh_approx():
    """ Test the tree """

    # first check that when we make theta=0 (open everything) we get the same force and pairs
    rs = np.random.RandomState(seed=122)
    npts = 8**3 
    pts = np.reshape(rs.rand(npts*3), (npts,3))
    wts = rs.rand(npts)

    kernel = arange(100)/99.0
    rmax = 2.0
    theta = 0.0
    n_kernels, acc = radial_BH_octree_kernel_evaluate(rmax, kernel, pts, wts, theta)

    pairs, acc2 = radial_kernel_evaluate(rmax, kernel, pts, wts)

    print('kernels', n_kernels, 'pairs', pairs*2)
    assert(n_kernels<pairs*2) # some pairs only get 1 kernel since in same bucket
    
    acc_err = acc-acc2
    acc_err_max = abs(acc_err).max()
    print('Max error on kernels (i.e. double sum re-order)', acc_err_max)
    assert(acc_err_max<1e-12) # double sums can be in different order, so need tolerance

    # Now check when using opening angle criteria

    pairs, acc = radial_BH_octree_kernel_evaluate(rmax, kernel, pts, wts, theta=0.7)
    acc_err = np.square(acc-acc2).sum(axis=1)
    acc2_mag2 = np.square(acc2).sum(axis=1)
    idx = np.argmax(acc_err)
    acc_err_max = acc_err[idx]**0.5
    worst_err_frac = acc_err_max/(acc2_mag2[idx]**0.5)
    print('Max error on BH approx %.2f%%  delta(|acc|)=%.2f'%(100.0*worst_err_frac, acc_err_max), 'at acc[%d]='%idx, acc[idx])

    assert(worst_err_frac<0.05)



