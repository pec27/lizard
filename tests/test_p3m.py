from __future__ import print_function, division, unicode_literals, absolute_import
import lizard.p3m as p3m
from lizard.power import CosmologicalParticleModel, pspec_normalised_by_sigma8_expansion, efstathiou, hubble_closure
from lizard.grid import build_displacement, displacement_sampler_factory
from lizard.uniform import mass_field
from numpy import * #mgrid, ones, float64
from lizard.log import VerboseTimingLog
from iccpy.cgs import msun, mpc, G
from lizard import integrate 
import sys

def rms(x):
    """ RMS of magnitude of an (...,3) array """
    last_ax = len(x.shape)-1
    return sqrt(square(x.astype(float64)).sum(last_ax).mean())

def _direct_sum(pts, wts, ewald_corr=False):
    """
    sum up r^-2 force for all the points
    Not meant to be remotely, efficient, just for testing purposes with a very small number of points
    """

    acc = zeros_like(pts)
    assert(len(pts)==len(wts))

    dx_nearest = []
    for i0, (p0, w0) in enumerate(zip(pts, wts)):
        for i1, (p1, w1) in enumerate(zip(pts, wts)):
            if i0==i1:
                continue
            # relative position
            dx = array(p1) - array(p0)
            # nearest image
            dx_im = dx - floor(dx+0.5)

            r2 = square(dx_im).sum()

            inv_r3 = 1.0 / (sqrt(r2)*r2)
            
            acc_nearest = dx_im * inv_r3
            if not ewald_corr:
                acc[i0] += acc_nearest * w1                
                continue
            acc_ewald   = _ewald_correction(reshape(-dx_im, (1,3)))[0]
            acc[i0] += (acc_nearest+acc_ewald) * w1
    return acc

def test_short_only():
    """ test very close points where force ->  Newtonian """
    pts = [(0.5,0.5,0.5), (0.5,0.6,0.5), (0.55, 0.55, 0.55)]
    wts = [1,2,3]
    print('Ewald summation of forces')
    acc = _direct_sum(pts, wts, ewald_corr=False)
    
    # Newtonian (from direct sum)
    newtonian = array([[ 230.94010768,  430.94010768,  230.94010768],
       [ 230.94010768, -330.94010768,  230.94010768],
       [-230.94010768,   76.98003589, -230.94010768]])

    max_acc = sqrt(square(newtonian).sum(1).max())

    print('Acceleration', repr(acc))
    print('Newtonian', repr(newtonian))
    
    print('Total rate of change of momentum', (acc * reshape(wts, (len(wts),1))).sum(0))
    print('Hash-grid summation of forces')
    rcrit = 0.12
    kernel = p3m.setup_newton(rcrit, npts=100) # about 0.01% accurate

    pairs, accel = p3m.radial_kernel_evaluate(rcrit, kernel, pts, wts)
    print('Total force-pairs calculated', pairs)
    assert(pairs==3) # 3 pairs of forces
    
    accel_err = abs(accel-newtonian).max()/max_acc
    
    print('Maximum relative error', accel_err)
    assert(accel_err<1e-3)

def test_short_long():
    """ Combination of close (Newtonian) and far (periodic effects) pairs """
    from numpy import empty_like, arange
    print('A close pair and a far point')
    # worst-case scenario, short-pair near (0.1) splitting scale (0.105)
    pts = [(0.5,0.5,0.5), (0.5,0.6,0.5), (0.9, 0.3, 0.95)]
    wts = [1,2,3]
    print('Ewald summation of forces')
    acc = _direct_sum(pts, wts, ewald_corr=True)
    print('Short-long Acceleration', repr(acc))
    # Ewald (from direct sum)
    ewald = array([[   2.04840161,  196.8858954 ,    1.18571601],
                   [   1.63087151, -101.75612334,    0.92870922],
                   [  -1.77004821,    2.20878376,   -1.01437815]])

    max_acc = sqrt(square(ewald).sum(1).max())

    max_err = sqrt(square(ewald-acc).sum(1).max())

    assert(max_err<1e-3)
    
    print('Total rate of change of momentum', (acc * reshape(wts, (len(wts),1))).sum(0))
    print('Hash-grid summation of forces')
    fs = p3m.get_force_split(r_split=0.105, mode='erf')
    pairs, accel_short = p3m.pp_accel(fs, wts, pts, r_soft=None) # Erf-split Newtonian

    assert(pairs==1) # Only the close pair

    print('Making long range force via fft')
    long_force = p3m.PMAccel(fs)

    accel_long = long_force.accel(wts, pts)


    accel = accel_short + accel_long
    print('short+long', accel)

    accel_err = abs(accel-ewald).max()/max_acc
    
    print('short+long maximum relative error', accel_err)
    assert(accel_err<1.5e-2)

def test_long_only():
    """ Far pair with no short-range contribution """
    pos = [(0.2,0.2,0.2), (0.7,0.6,0.5)]
    mass = [1,10]
    print('Ewald summation of forces')
    acc = _direct_sum(pos, mass, ewald_corr=True)
    
    ewald = array([[  0.00000000e+00,   5.26024612e+00,   7.09533177e+00],
       [ -6.66133815e-16,  -5.26024612e-01,  -7.09533177e-01]])
    max_acc = sqrt(square(ewald).sum(1).max())

    max_err = sqrt(square(ewald-acc).sum(1).max())
    print('Acceleration', repr(acc))
    assert(max_err<1e-5)

    print('Total rate of change of momentum', (acc * reshape(mass, (len(mass),1))).sum(0))
    fs = p3m.get_force_split(r_split=0.15, mode='erf')
    long_force = p3m.PMAccel(fs)
    accel = long_force.accel(mass, pos)
    print('Accel long', accel)
    accel_err = abs(accel-ewald).max()/max_acc
    
    print('Maximum relative error', accel_err)
    assert(accel_err<2e-3)

def _ewald_correction(x, alpha=2.0):
    """
    calculate the Ewald-correction forces, i.e. the force due to all the images
    of a point particle, EXCLUDING the nearest one.

    Particle at 0,0,0, in the periodic unit cube (x,y,z in [-0.5,0.5])

    See also Hernquist, Bouchet & Suto 1991
    """
    from numpy import cumprod,empty_like,mgrid,float64,subtract,exp,flatnonzero,dot
    from scipy.special import erfc
    old_shape = x.shape
    n = cumprod(old_shape[:-1])[-1]
    x2 = x.reshape(n,3)

    force = empty_like(x2)
    r2 = square(x2).sum(1)
    mult = 1.0 / (r2 * sqrt(r2))
    for i in range(3):
        force[:,i] = x2[:,i] * mult

    N = (mgrid[:9,:9,:9]-4).reshape((3,9*9*9))

    vec = empty((n,N.shape[1],3), dtype=float64)
    for i in range(3):
        vec[:,:,i] = subtract.outer(x2[:,i],N[i])

    r = sqrt(square(vec).sum(2))
    mult = (erfc(alpha * r) + (2 * alpha / sqrt(pi)) * r * exp(-alpha * alpha * r * r)) / (r*r*r)

    for i in range(3):
        force[:,i] -= (vec[:,:,i] * mult).sum(1)

    N2 = square(N).sum(0).reshape(1,N.shape[1])
    idx = flatnonzero(N2>0)
    N = N[:,idx]
    N2 = N2[:,idx]

    N_x = dot(x2, N)

    N = N.reshape(3,1,N.shape[1])
    mult = (2.0 /  N2) * exp(-pi * pi * N2 / (alpha * alpha)) * sin(2 * pi * N_x)

    for i in range(3):
        force[:,i] -= (N[i] * mult).sum(1)

    # make the x=0 point zero, if it exists
    idx = flatnonzero(r2==0)
    force[idx,:]= 0.0

    force = force.reshape(old_shape)

    return force

def test_cosmological_vels_agree_accel():
    """
    Linearity of cosmological accelerations.

    The cosmological ICs are made in the linear regime, and thus should have
    velocities proportional to the gravitational acceleration. By making a 
    realisation and using PPPM when can calculate the gravity and confirm this.
    """
    
    
    redshift = 50
    # Small box dominated by longest modes
    boxsize = 10.0 # Mpc/h 
    H0 = 70.0 # km / s / Mpc
    a = 1.0 / (1.0+redshift)
    omegaM = 0.279
    omegaL = 0.721
    sigma8 = 0.8

    pspec = pspec_normalised_by_sigma8_expansion(efstathiou, sigma8=sigma8, a=a, omegaM=omegaM, omegaL=omegaL)

    ngrid = 30

    disp_grid = build_displacement(boxsize=boxsize, ngrid=ngrid, power_func=pspec)
    disp_sampler = displacement_sampler_factory(disp_grid, boxsize)

    # fill a box with uniform points
    xyz = (mgrid[:ngrid,:ngrid,:ngrid]+0.5) * (boxsize/ngrid)

    uni_pts = reshape(xyz, (3,ngrid**3)).T
    uni_sizes = ones(ngrid**3)* (boxsize/ngrid)

    dm_pos, dm_mass, dm_vel, dm_nums = mass_field(boxsize, uni_pts, uni_sizes, disp_sampler, a, omegaM, omegaL, H0)
    
    print('DM vel shape', dm_vel.shape, dm_mass.shape, dm_pos.shape, 'mass', dm_mass)
    rms_vel = sqrt(square(dm_vel).sum(1).mean(dtype=float64))
    print('RMS velocity for DM', rms_vel)
    
    # now do gravity
    rcrit = 2.0 / ngrid 

    log = VerboseTimingLog(insert_timings=True)
    fs = p3m.get_force_split(rcrit, mode='erf')
    pos = dm_pos * (1.0/boxsize)

    wts = empty(pos.shape[0])

    h = H0 / 100.0
    wts[:] = dm_mass * G * (msun/h) / (boxsize*mpc / h)**3
    print('Pos in',pos.min(axis=0), pos.max(axis=0))
    pairs, acc_short = p3m.pp_accel(fs, wts, pos, r_soft=None) # Erf-split Newtonian
    long_force = p3m.PMAccel(fs)
    acc_long = long_force.accel(wts, pos)
    log.close()
    acc = (acc_short + acc_long) / (a*a*a) # d^2 x / dt^2

    hubble = hubble_closure(H0*(1e5/mpc), omegaM, omegaL)

    # In linear regime of matter-dominated universe, vel = 2/3 H * acc
    eds_vel = acc * (boxsize *mpc/h) *1e-5/ hubble(a) * a * 2.0/3.0 # in km/s physical

    dm_vel *= sqrt(a) # convert Gadget vels to km/s
    print('Samples:')
    print(eds_vel[10], 'km/s velocity (EdS approx from accel)')
    print(dm_vel[10], 'km/s velocity from linear theory')
    err = rms(dm_vel - eds_vel) / rms(dm_vel)
    print('Error in', err.min(), err.max(), 'mean', err.mean())
    assert(err.mean()<0.06)

def test_evolve():
    """
    Build a particle and velocity distribution at two expansion factors using
    the linear spectrum, test it is the same as using the evolution code
    """
    ngrid = 16 # very small sim
    redshift0, redshift1 = 50, 49.0
    # Small box dominated by longest modes
    boxsize = 10.0 # Mpc/h 
    H0 = 70.0 # km / s / Mpc
    a0 = 1.0 / (1.0+redshift0)
    a1 = 1.0 / (1.0+redshift1)
    omegaM = 0.279
    omegaL = 0.721
    sigma8 = 0.8

    pspec0 = pspec_normalised_by_sigma8_expansion(efstathiou, sigma8=sigma8, a=a0, omegaM=omegaM, omegaL=omegaL)
    pspec1 = pspec_normalised_by_sigma8_expansion(efstathiou, sigma8=sigma8, a=a1, omegaM=omegaM, omegaL=omegaL)



    disp_grid0 = build_displacement(boxsize=boxsize, ngrid=ngrid, power_func=pspec0)
    disp_grid1 = build_displacement(boxsize=boxsize, ngrid=ngrid, power_func=pspec1)
    disp_sampler0 = displacement_sampler_factory(disp_grid0, boxsize)                         
    disp_sampler1 = displacement_sampler_factory(disp_grid1, boxsize)                         
    # fill a box with uniform points
    xyz = (mgrid[:ngrid,:ngrid,:ngrid]+0.5) * (boxsize/ngrid)

    uni_pts = reshape(xyz, (3,ngrid**3)).T 
    uni_sizes = ones(ngrid**3)* (boxsize/ngrid)

    # Only DM
    pos0, mass0, vel0, nums = mass_field(boxsize, uni_pts, uni_sizes, disp_sampler0, a0, omegaM, omegaL, H0)
    pos1, mass1, vel1, nums = mass_field(boxsize, uni_pts, uni_sizes, disp_sampler1, a1, omegaM, omegaL, H0)
    
    # scale for the unit box evolution

    h = H0/100.0
    spos  = pos0 / boxsize
    smass = mass0 * (msun / h) * G / (boxsize*mpc/h)**3 # such that 1/r^2 'force' in /s^2
    svel  = vel0 * (1e5 / (boxsize*mpc/h))/sqrt(a0) # fraction of unit box / sec
    r_soft = 0.25 / ngrid # 0.25x interparticle spacing

    log = VerboseTimingLog(insert_timings=True, also_stdout=True)

    gm = integrate.SingleProcGravityModel(r_split=r_soft*8, log=log)
    cosmo_mdl = CosmologicalParticleModel(gm, spos, smass, svel, r_soft=r_soft, a0=a0, omegaM=omegaM, omegaL=omegaL, H0=H0, hub_crit=0.05, log=log)
    spos, svel = list(integrate.kdk_integrate(cosmo_mdl, (a1,), log))[0]
    
    # convert back to regular units
    spos = spos * boxsize
    svel = svel * (boxsize*mpc/h/1e5) * sqrt(a1) # to gadget units
    
    delta_spos = spos-pos0
    delta_pos1 = pos1-pos0

    rms_delta_pos1 = rms(delta_pos1)
    rms_delta_ev = rms(delta_spos)
    print('Expected mean change in comoving position %.3f'%(1e3*rms_delta_pos1), 'kpc/h comoving')
    print('Actual %.3f'%(1e3*rms_delta_ev), 'kpc/h comoving')
    # difference between analytic and numerical positions
    pos_diffs = pos1-spos
    rms_diff = rms(pos_diffs)
    print('Mean difference', 1e3*rms_diff, 'kpc/h or %.3f%% of time evolution'%(100*rms_diff/rms_delta_pos1))

    assert(rms_diff < 0.2 *rms_delta_pos1) # Change in position test

    dvel_an   = vel1*sqrt(a1) - vel0*sqrt(a0) # Analytic velocity change
    dvel_num  = svel*sqrt(a1) - vel0*sqrt(a0) # numerical velocity change

    rms_dv_an = rms(dvel_an)
    rms_dv_num = rms(dvel_num)
    print('Expected change in velocity %.3f vs numerical %.3f km/s'%(rms_dv_an, rms_dv_num))
    vel_diffs = (vel1 - svel)*sqrt(a1)
    print('Mean difference', rms(vel_diffs),'km/s or %.3f%% of time evolution'%(100*rms(vel_diffs)/rms_dv_an))
    # velocities never particularly good, due to particle discretisation
    print(vel0[10]*sqrt(a0))
    print(vel1[10]*sqrt(a1))
    print(svel[10]*sqrt(a1))
    assert(rms(vel_diffs)<rms_dv_an*0.2) # but test change in velocity within 20% of expected

def test_subcube_ghosts():
    """ Test ghosts of subcubes """
    box_cen = (0.4,0.3,0.2) # centre of the subcube
    box_hw = 0.15  # half width of box
    ghost_rad = 0.1
    pos = ((0.4, 0.3, 0.2),  # centre of the sub-box
            (0.4, 0.4, 0.2), # In box, ghost of outside
            (0.4, 0.5, 0.2), # Out of box, ghost of inside
            (0.4, 0.6, 0.2), # Completely out of box
            (0.61, 0.51, 0.41), # Completely out of box, but only due to rounded corners
            (0.4, 0.3, 0.99)) # Out of box, ghost of inside, only via periodicity



    exp_in = [0,1,2,5]
    exp_in_nonghost = [0,1]
    exp_out = [1,2,3,4,5]
    exp_out_nonghost = [1,2,3,4]
    
    idx_in, idx_in_nonghost, idx_out, idx_out_nonghost = p3m.subcube_periodic_ghosts(pos, box_cen, box_hw, ghost_rad)



    assert(all(idx_in==exp_in))
    assert(all(idx_in_nonghost==exp_in_nonghost))
    assert(all(idx_out==exp_out))
    assert(all(idx_out_nonghost==exp_out_nonghost))


def test_intermediate_grid():
    """ test an additional non-periodic grid """
    pts = [(0.5,0.5,0.5), (0.5,0.8,0.5), (0.55, 0.55, 0.55)]
    wts = [1,2,3]
    print('Ewald summation of forces')
    acc = _direct_sum(pts, wts, ewald_corr=True)
    
    print('Acceleration', repr(acc))
    r_coarse = 0.105
    print('Long-Short split of forces')

    fs = p3m.get_force_split(r_split=r_coarse, mode='cubic') #, kernel_pts=200)
    pairs, accel_short = p3m.pp_accel(fs, wts, pts, r_soft=None) # Cubic-split Newtonian
    accel_long = p3m.PMAccel(fs).accel(wts, pts)
    accel = accel_short + accel_long
    print('Result', repr(accel))
    print('Long-Medium-Short split')
    ifs = p3m.IntermediateGrid(r_coarse=r_coarse, ngrid=96, centre=(0.5,0.5,0.5), hw_min=0.01)
    print('Rlong', ifs.r_coarse, 'Rfine', ifs.r_fine, 'box width', ifs.hw*2)
    idx_in, idx_in_nonghost, idx_out, idx_out_nonghost = ifs.split_in_vs_out(pts)
    print('idx in', idx_in, idx_in_nonghost)
    pos_in = array(pts)[idx_in]
    wts_in = array(wts)[idx_in]
    accel_int_pm = ifs.pm_accel(pos_in, wts_in, idx_in_nonghost)
    accel_int_pp = ifs.pp_accel(pos_in, wts_in, idx_in_nonghost)
    
    accel_int = accel_int_pm + accel_int_pp
    pos_out = array(pts)[idx_out]
    wts_out = array(wts)[idx_out]
    pairs, accel_short = p3m.pp_accel(fs, wts_out, pos_out, r_soft=None) # Cubic-split Newtonian
    


    accel_short = accel_short[idx_out_nonghost]

    # add all the components
    accel = accel_long
    acs = zeros_like(accel)
    acs[idx_out[idx_out_nonghost]] += accel_short
    aci = zeros_like(accel)
    aci[idx_in[idx_in_nonghost]] += accel_int
    print('Accel int pp', accel_int_pp)
    print('Accel int', aci)
    print('Accel short', acs)
    print('ACCel long', accel_long)
    accel += aci + acs

    print('Result', repr(accel))
    
    err = abs(accel-acc)
    print('Error',err)
    assert(err.max()<4.0) # about 1.3 % error 

def test_inter_grid_halo():
    from numpy.random import RandomState

    rs = RandomState(seed=123)

    npts = 150
    phi = 2 * pi * rs.rand(npts)
    cos_theta =2* rs.rand(npts) - 1
    sin_theta = sqrt(1-cos_theta**2)
    r = 0.5 * sqrt((arange(npts)+1.0)/npts) # 1/r profile, like centre of NFW # mass is uniformly distributed over radii for isothermal halo (rho ~1/r^2)
    
    pos = empty((npts,3))
    pos[:,0] = 0.5+r * sin_theta*cos(phi)
    pos[:,1] = 0.5+r * sin_theta*sin(phi)
    pos[:,2] = 0.5+r * cos_theta

    wts = ones(npts) * (1.0/npts)
    print('Gravity via short-long')
    r_coarse = 0.15
    r_soft = 0.01
    print('Long-Short split of forces')

    fs = p3m.get_force_split(r_split=r_coarse, mode='cubic') #, kernel_pts=200)
    pairs, accel_short = p3m.pp_accel(fs, wts, pos, r_soft)
    accel_long = p3m.PMAccel(fs).accel(wts, pos)

    accel_LS = accel_short + accel_long
    print('Total pairs {:,}'.format(pairs))
    print('Gravity via short-med-long')
    ifs = p3m.IntermediateGrid(r_coarse=r_coarse, ngrid=64, centre=(0.5,0.5,0.5), hw_min=r_coarse)
    print('Rlong', ifs.r_coarse, 'Rfine', ifs.r_fine, 'box width', ifs.hw*2)
    idx_in, idx_in_nonghost, idx_out, idx_out_nonghost = ifs.split_in_vs_out(pos)
    print('Number of points in central region', len(idx_in_nonghost))

    print('Number of points outside is {:,}, including ghosts is {:,}'.format(len(idx_out_nonghost), len(idx_out)))

    accel_int_pm = ifs.pm_accel(pos[idx_in], wts[idx_in], idx_in_nonghost)
    accel_int_pp = ifs.pp_accel(pos[idx_in], wts[idx_in], idx_in_nonghost, r_soft)
    
    accel_int = accel_int_pm + accel_int_pp
    pairs, accel_short = p3m.pp_accel(fs, wts[idx_out], pos[idx_out], r_soft)
    
    accel_short = accel_short[idx_out_nonghost]

    # add all the components
    accel_LMS = accel_long.copy()
    acs = zeros_like(accel_LMS)
    acs[idx_out[idx_out_nonghost]] += accel_short
    aci_pm = zeros_like(accel_LMS)
    aci_pm[idx_in[idx_in_nonghost]] += accel_int_pm
    aci_pp = zeros_like(accel_LMS)
    aci_pp[idx_in[idx_in_nonghost]] += accel_int_pp
    

    accel_LMS += aci_pm +aci_pp + acs

    
    print('RMS Long-short', rms(accel_LS))
    print('RMS Long-Med-Short', rms(accel_LMS))
    ix = arange(npts)#idx_in[idx_in_nonghost]
    max_acc = sqrt(max(square(accel_LS).sum(1)))
    err = accel_LMS[ix] - accel_LS[ix]
    err2 = square(err).sum(1)
    idx_max = argmax(err2)
    rad_max = sqrt(square(pos[idx_max]-0.5).sum())
    print('Position of maximum', pos[idx_max], 'radius', rad_max, 'AccelLS', accel_LS[idx_max], 'AccelLMS', accel_LMS[idx_max])
    print('M', aci_pm[idx_max], aci_pp[idx_max], 'L', accel_long[idx_max])
    print('Index of max', idx_max)
    max_error = sqrt(err2.max())

    print('Max error/max_acc', max_error/max_acc)
    assert(max_error<0.01*max_acc)
    print('RMS error/RMS acc', rms(err)/rms(accel_LS))
    assert(rms(err)<0.01*rms(accel_LS))

def test_grid_time_deriv():
    """
    Time derivative of grid force
    """
    pos0 = array([(0.5,0.5,0.5)])
    vel = array([(1.0, 2.0, 3.0)])
    dt = 0.1/96
    wts = array([1.0])
    log = sys.stdout
    ngrid, r_core = 96, 6.0/96

    fft_wts = p3m._build_fft_wts(ngrid, r_core, log, p3m._ta_cubic)

    dt_est, grid0, dgrid_dt = p3m._grid_accel_deriv(pos0, wts, vel, fft_wts, log)
   
    dt = dt_est * 1.5
    print('Estimated time', dt_est, 'used', dt)
    pos1 = pos0 + vel * dt
    pos1 = pos1 - floor(pos1)
    grid1 = p3m._grid_accel_deriv(pos1, wts, vel, fft_wts, log)[1]

    # now check grid differences
    dgrid = grid1 - grid0
    print('RMS grid change', rms(dgrid))
    exp_dgrid = dgrid_dt * dt
    print('Expected', rms(exp_dgrid))
    err_ratio = rms(exp_dgrid)/rms(dgrid) - 1
    print('Ratio error', err_ratio)
    assert(abs(err_ratio)<5e-3)
    max_diff = sqrt(square(exp_dgrid-dgrid).sum(3).max())
    err_rms =  rms(exp_dgrid - dgrid) / rms(grid0)
    print('RMS Difference / RMS grid', err_rms)
    assert(err_rms<1e-3)
    err_max = max_diff/rms(grid0)
    print('Max Diff / RMS grid', err_max)
    assert(err_max<0.02)
    


