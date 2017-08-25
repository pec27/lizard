"""
Module for things related to uniform grids and their Fourier transforms, i.e. 
making a displacement grid with a given matter power spectrum, or finding the
cloud-in-cell (gridded) density distribution for a set of particles and then
computing its Fourier modes.

Some of the functions are backed up by C-implementations for speed, and they 
can be found in grid.c

"""

from __future__ import print_function, division, unicode_literals, absolute_import
from numpy.random import RandomState
from numpy import arange, pi, square, add, sqrt, empty, float64, mgrid, floor, \
    reshape, int64, zeros, float32, bincount, digitize, array, remainder, \
    multiply, empty_like, complex128, interp, transpose, where, int32, cumprod
from numpy.fft import fftn, ifftn
from lizard.lizard_c import get_cic, gradient_5pt_3d, interp_vec3, unpack_kgrid
from lizard.log import null_log

def make_k_values(boxsize, ngrid):
    """ 
    build the grid of |k| for the given box size, in a vaguely memory efficient way
    returns k (n,) array, inv_k2 (n,n,n) array, with 0 at i=j=k=0
    """
    # 1d component of k
    k1 = arange(ngrid)
    k1[1+ngrid//2:] -= ngrid
    k1 = k1 * (2 * pi / float(boxsize))
    k2 = square(k1)
    kmag = add.outer(add.outer(k2, k2), k2)
    inv_k2 = kmag.copy()
    kmag = sqrt(kmag)
    inv_k2[0,0,0] = 1.0
    inv_k2 = 1.0 / inv_k2
    inv_k2[0,0,0] = 0.0
    return k1, kmag, inv_k2

def deconv_cic(k_dx):
    """
    Deconvolution to account for CIC (cloud-in-cell) kernel

    see also deconvolve_cic in p3m.py, however that attempts to adjust for 
    *twice* the CIC (once for CIC then once for force interpolation).

    This is some sort of *magic* Taylor expansion to the transfer function
    (k_dx/sin(k_dx))^2, since it is actually 3d (we consider only the 1d 
    magnitude) and the CIC isn't a true convolution.
    """
    u = square(k_dx*0.5).sum(1)
    # 4th term makes it worse(!)
    s = 1.0 + (1/3.0) * u + (1.0/15.0)*square(u) #+ (1/94.5)*u*square(u)
    return s

    
def deconv_5pt(k_dx):
    """
    Multiplier for modes to counteract the smoothing that is introduced
    with the 5-point gradient method

    k_dx - k*dx (or array) for the given mode

    returns m_k s.t. gradient_5pt(m_k * exp(ikx)) = ik exp(ikx)

    Where the 5pt gradient is the estimate

    f'(x) = -f(x+2dx)/12 + 2*f(x+dx)/3 - 2*f(x-dx)/3 + f(x-2dx)/12  

    which suppresses the derivative of the mode exp(ikx) by a factor:
      4 sinc(k_dx) / 3 - sinc(-2 k_dx )/3
    where sinc(t) := sin(t)/t (not the numpy definition)

    """
    k_dx2 = square(k_dx)
    k_dx4 = square(k_dx2)
    k_dx6 = k_dx4*k_dx2

    print('Maximum k_dx', k_dx.max(), pi*sqrt(3))
#    return where(k_dx<2.5,1.0+k_dx4 /30.0- k_dx6 / 252 + k_dx4*k_dx4 / 3420, 0.0)
#    return where(k_dx<pi,1.0+k_dx4 /30.0- k_dx6 / 252 + k_dx4*k_dx4 / 3420,1.03).mean(axis=1)#+k_dx4 /30.0- k_dx6 / 252 + k_dx4*k_dx4 / 3420, 0.0)
    return where(k_dx<pi,1.0+k_dx4 /30.0- k_dx6 / 252 + k_dx4*k_dx4 / 3420,1.0).max(axis=1)#+k_dx4 /30.0- k_dx6 / 252 + k_dx4*k_dx4 / 3420, 0.0)
#   This goes asymptotic as k_dx -> pi
#    from numpy import sinc
#    return where(k_dx<2.5,1.0/((4.0/3)*sinc(k_dx/pi) - sinc(-2*k_dx/pi)/3.0), 0.0)

def _pack_kgrid(L, n):
    """
    Utility function to find |k| where 
      k:= 2 pi / L * (u, v, w)
    for the sequence of u,v,w 
       (0,0,0), (1,0,0), (1,1,0), (1,1,1), (2,0,0), (2,1,0), (2,1,1),... (M-1,M-1,M-1)
    where M := 1 + n//2

    i.e. u >= v >= w >= 0 and the last index is rolling fastest.

    L - size of the box
    n - Size for (n,n,n) grid this is equivalent to (i.e. your fft-size)

    retiurns 1d array of M(M+1)(M+2)/6 k-values

    See also unpack_kgrid to restore these to an [n,n,n] grid.
    """
    mid = 1+n//2 # indices [0,1,..., m-1]
    
    vals = empty((mid*(mid+1)*(mid+2))//6, dtype=int32)
    idx =0
    sqs = square(arange(mid))
    for i in range(mid):
        for j in range(i+1):
            nk = j+1
            vals[idx:idx+nk] = i*i+j*j+sqs[:nk]
            idx += nk
    assert(idx==len(vals))
    return (2*pi/L) * sqrt(vals)

def _pack_kvec(L, n):
    """
    Utility function to find 
      k:= 2 pi / L * (u, v, w)
    for the sequence of u,v,w 
       (0,0,0), (1,0,0), (1,1,0), (1,1,1), (2,0,0), (2,1,0), (2,1,1),... (M-1,M-1,M-1)
    where M := 1 + n//2

    i.e. u >= v >= w >= 0 and the last index is rolling fastest.

    L - size of the box
    n - Size for (n,n,n) grid this is equivalent to (i.e. your fft-size)

    retiurns 2d array of (M(M+1)(M+2)/6, 3) k-vectors
    """
    mid = 1+n//2 # indices [0,1,..., m-1]
    
    vals = empty(((mid*(mid+1)*(mid+2))//6, 3), dtype=int32)
    idx = 0
    kw = arange(mid)
    for i in range(mid):
        ni = ((i+1)*(i+2))//2
        vals[idx:idx+ni,0] = i
        for j in range(i+1):
            nk = j+1
            vals[idx:idx+nk,1] = j
            vals[idx:idx+nk,2] = kw[:nk]
            idx += nk

    assert(idx==len(vals))
    return (2*pi/L) * vals

def kfunc_on_grid(L, n, func, k0_val=None, log=null_log, vecfunc=False):
    """
    Evaluate the function: 
      f[p,q,r] := func(|k|) 
    or
      f[p,q,r] := func((k_p, k_q, k_r)) where func is indept of permutations
                  (p,q,r) and k_i |-> -k_i

    on the 3d grid of k values
      k_pqr = 2*pi/L * (i_p,i_q,i_r)
    where
      i_p := p, for p<= n/2
             n-p, for p> n/2


    L      - float size of the box
    n      - integer size of the grid
    func   - the function to be evaluated
    k0_val - special value of the function at k=0 (for many functions this is 
             a special point or removable singularity)
    

    returns f - an (n,n,n) array of values

    This function is optimised to exploit the 48-fold symmetries of 
    (i_p, i_q, i_r), i.e. all the cyclic permuations (6) and i->-i (8), which
    means that even if func is slow you get ok performance.
    """

    if vecfunc:
        kvals = _pack_kvec(L,n)
    else:
        kvals = _pack_kgrid(L, n)

    if k0_val is None:
        f_vals = func(kvals)
    else:
        res = func(kvals[1:]) # 0 is always 0 term
        f_vals = empty(kvals.shape[0], dtype=res.dtype)
        f_vals[1:] = res
        f_vals[0] = k0_val

    return unpack_kgrid(n, f_vals, log)


def powerspec_bins(ngrid, boxsize):
    """
    find power spectrum bins to for a cubic grid of size ngrid^3 of fourier modes.
    Assumes the FFT convention of 0, ..., n/2, -n/2+1, ..., -1
    ngrid   - num cells on side of cube
    boxsize - size of the box in real space

    returns kmin, kmax, kbins, kvol
    kmin  - the lower bound of the bin
    kmax  - the upper bound of the bin
    kbins - index (0, ..., m) of the bin of each cell
    kvol  - the volume in k space of all modes in that bin
    """

    mid = ngrid//2
    # find the magnitude of the indices (i.e. ix**2+iy**2+iz**2 in the FFT convention)
    n1 = arange(ngrid)
    n1[1+mid:] -= ngrid
    n2 = square(n1)
    nmag = sqrt(add.outer(add.outer(n2, n2), n2)).ravel()
    
    nbins = (-1,) + tuple(arange(mid-1)+1.5) + (ngrid*2,)
    #print 'nbins', nbins
    kbins = digitize(nmag, nbins) - 1
    assert(kbins.min()==0)
    assert(kbins.max()==len(nbins)-2)
    
    # multiplier to go to k-space
    dk = 2.0 * pi / boxsize

    kmin = (array(nbins) * dk)[:-1]
    kmin[0] = 0

    kmax = (array(nbins) * dk)[1:]
    kmax[-1] = mid * dk * sqrt(3.0)
    
    kvol = bincount(kbins) * (dk * dk * dk)
    return kmin, kmax, kbins, kvol
    
def build_displacement(boxsize, ngrid, power_func, seed=12345, log=null_log):
    """ 
    Build a grid of the displacement field using the given power spectrum

    boxsize    - Size of the box, units (e.g. Mpc/h) must be consistent with
                 the power spectrum function
    ngrid      - Integer size of the grid, e.g. 32 for a 32x32x32 grid 
    power_func - Power spectrum function to be used on each k-mode, units
                 should be inverse of the box size (e.g. h/Mpc)
    [seed]     - Optional seed for the random number generator.

    returns 
    disp_grid  - (3,ngrid,ngrid,ngrid) array of displacements, with
                 disp_grid[i] giving the displacement along the i-th axis etc.
    """

    mersenne = RandomState(seed)
    ngrid3 = ngrid**3

    print('- Making k values',file=log)
    # make the k values 
    dk = 2 * pi / boxsize # Distance between the modes (k)
    # amplitudes of the double integral of density (proportional to potential)
    print('- Amplitudes of modes', file=log)
    if ngrid<32:
        #inverse k^2 for potential
        ampl_func = lambda k : sqrt(power_func(k) * dk * dk * dk) / (k*k)
        mode_ampl = kfunc_on_grid(boxsize, ngrid, ampl_func, k0_val=0.0, log=log)
    else:
        # When ngrid is large, the sampling of the power spectrum is very dense
        # (i.e. a 1-d function sampled ngrid**3 times), which can be the 
        # most expensive part for complicated functions. Instead, for ngrid>32
        # we resample the function ngrid*5 times in linear space, and 
        # interpolate from this. 
        #
        # Note that this guarantees that the nearest grid point is at most 
        # 9% away in units of dk or 4% in relative. Tests on maximum 
        # interpolation error on power spectrum give ~0.1% for large 
        # (>100 Mpc/h) boxes, up to ~0.6% for small (<<100 Mpc/h) ones.
        
        num_pts = ngrid * 5
        print('- Large ngrid={:,}^3, sampling power spectrum 5*ngrid={:,} times'.format(ngrid, num_pts), file=log)
        kmax = dk * ngrid * sqrt(0.75)
        resample_k_pts = (arange(num_pts)+1)* (kmax / num_pts)
        ampl = sqrt(power_func(resample_k_pts)*dk*dk*dk)
        print('- Interpolating power spectrum', file=log)
        #inverse k^2 for potential, multiply to account for the 5-pt differencing
#        ampl_func = lambda k : interp(k, resample_k_pts, ampl) * deconv_5pt(k*boxsize/ngrid) / (k*k)
#        mode_ampl = kfunc_on_grid(boxsize, ngrid, ampl_func, k0_val=0.0, log=log)

        ampl_func = lambda k : interp(k, resample_k_pts, ampl) / (k*k)

        mode_ampl = kfunc_on_grid(boxsize, ngrid, ampl_func, k0_val=0.0, log=log) #* kfunc_on_grid(ngrid, ngrid, deconv_5pt, k0_val=1.0, log=log, vecfunc=True)  
        
    print('- Building {:,} random numbers'.format(ngrid3*2), file=log)
    grf_k = mersenne.standard_normal(size=2*ngrid3).view(dtype=complex128)
    print('- Scaling', file=log)
    grf_k *= mode_ampl.ravel() 
    grf_k.shape = (ngrid, ngrid, ngrid)

    print('- {:,}^3 fourier transform to calculate potential'.format(ngrid), file=log)
    # 'integral' of displacement field (i.e. proportional to potential,
    # and the displacement is the gradient of this)
    int_disp = fftn(grf_k).real 

    print('- 5 point gradient to calculate displacement field', file=log)
    dx = boxsize/ngrid
    d = gradient_5pt_3d(int_disp * (1.0 / dx))

    print('- Transposing', file=log)
    disp_fld = empty((3, ngrid, ngrid, ngrid), dtype=float32)
    for i in range(3):
        disp_fld[i] = d[:,:,:,i]
    
    rms_disp = sqrt(square(disp_fld).sum(0).mean(dtype=float64))
    print('- RMS displacement %.3f kpc/h'%(rms_disp * 1e3,), file=log)
    return disp_fld

def displacement_sampler_factory(disp_grid, boxsize, boost_grid=None, boost_repeats=None, log=null_log):
    """
    Return an interp function on the displacement field, i.e.
    f(coords) where coords is an (3,N) array and returns an (N,3) array of 
    displacements.

    Optional boost grid for tiled extra displacements

    disp_grid      - (3,N,N,N) array
    boxsize       - size of box for coordinates
    boost_grid    - (M,M,M,3) array
    boost_repeats - number of times to tile in x,y,z
    TODO make everything in [0,1), as usual, and transpose.
    """

    rms_disp = sqrt(square(disp_grid).mean(dtype=float64) * 3.0)
    print('RMS displacement on grid, %6.4f kpc/h'%float(1e3*rms_disp), file=log)

    if boost_grid is None:
        def interpolator(coords):
            return interpolate_displacement_grid(coords, disp_grid, boxsize, order=1, log=log)
        return interpolator

    rms_disp = sqrt(square(boost_grid).mean(dtype=float64) * 3.0)
    print('RMS displacement on boost-grid, %6.4f kpc/h'%float(1e3*rms_disp), file=log)

    def boosted_interpolator(coords):
        M = boost_grid.shape[0] # M
        boosted_idx = coords * float(M * boost_repeats/boxsize)

        print('Indices for boost in %.2f-%.2f'%(boosted_idx.min(), boosted_idx.max()),
              'should be in 0-%d'%(M*boost_repeats), file=log)
        # wrap 
        boosted_idx = remainder(boosted_idx, M)
        return interp_vec3(boost_grid, boosted_idx.T) + interpolate_displacement_grid(coords, disp_grid, boxsize, log=log)
        
    return boosted_interpolator


def interpolate_displacement_grid(coords, disp_grid, boxsize, order=1,log=null_log):
    """ 
    Displace points via interpolation of displacement field 
    [order = 1] use linear interpolation (3=cubic spline, not implemented)

    coords    - (3,N) array of coordinates in [0,boxsize)
    disp_grid - (Ng,Ng,Ng,3) grid of displacements
    boxsize   - side length of box

    returns (N,3) array of vectors
    """
    assert(coords.shape[0]==3)

    if order!=1:
        raise Exception('Only order=1 (linear) interpolation implemented')
    
    dx = boxsize / float(disp_grid.shape[1])

    indices = coords * (1.0 / dx) # go from positions to indices

    print('Indices in', indices.min(), indices.max(), 'should be in 0-%d'%disp_grid.shape[1], file=log)

    # TODO re-organise so we dont need the wasteful transpose
    disp_grid = transpose(disp_grid, axes=(1,2,3,0))
    disps = interp_vec3(disp_grid, indices.T)
    return disps


def cic_modes(pts, boxsize, ngrid, masses=None, nonperiodic=None, log=null_log):
    """ find the power spec using cic """
    if pts.shape[0]!=3:
        raise Exception('cic_modes needs (3,n) array for point positions')

    # Wrap around the periodic box
    wrapped_pts = remainder(pts.T * (float(ngrid)/boxsize), ngrid)

    density = get_cic(wrapped_pts, ngrid, masses)

    
    density *= 1.0 / density.mean(dtype=float64)  # Bug when the mean accumulator is less than float64
    density -= 1.0 
    
        
    modes = ifftn(density)

    print('Apply deconvolution weights to account for CIC', file=log)
    modes *= kfunc_on_grid(ngrid, ngrid, deconv_cic, log=log, vecfunc=True)
    
    return modes
    
def modes_to_pspec(modes, boxsize, log=null_log):
    """ 
    From a given set of fourier modes, compute the (binned) power spectrum 
    with errors.

    modes   - (n,n,n) array (from ifftn of overdensity values)
    boxsize - size of box (e.g. in Mpc/h)
    
    returns

    kmin    - smallest k value in each bin
    kmax    - largest k value
    kvol    - volume in k space (dk^3)
    kmid    - midpoint
    pspec   - estimate of power spec at k
    perr    - error on estimate
    """

    ngrid = modes.shape[0]
    assert(modes.shape==(ngrid, ngrid,ngrid))
    kmin, kmax, kbins, kvol = powerspec_bins(ngrid, boxsize)

    wts = square(modes.ravel().real) + square(modes.ravel().imag)
    print('Summing power spectrum', wts.dtype, file=log)
    v1 = bincount(kbins, weights=wts) 
    print('bc', v1.dtype, file=log)
    powerspec = v1 * (1.0 / kvol)

    # work out error on power spectrum
    v2 = bincount(kbins, weights=square(wts))
    v0 = bincount(kbins)
    p_err = sqrt((v2*v0 - v1*v1)/(v0-1)) / kvol 

    kmid_bins = 0.5 * (kmin+kmax)

    return kmin, kmax, kvol, kmid_bins, powerspec, p_err
             
       
def displacement_boost(orig_boxsize, orig_ngrid, boost_ngrid, nrepeat, 
                       power_func, seed=12345, log=null_log):
    """ 
    Like build_displacement but for a boost grid.


    Build a boost-grid with only high frequency power, to be tiled over the
    box. For example say you have a 2048^3 basic grid, but you want the high
    frequencies of a 4096^3, then you could make a 1024^3 grid that is repeated
    4 times in each direction (1024*4=4096) and then zero out all the modes 
    that have already been inserted in the 2048.

    
    displacement_boost(orig_boxsize, orig_ngrid=2048, boost_ngrid=1024, nrepeat=4, ...)

    which can be tiled over the box of the displacement field using the given power spectrum

    orig_boxsize - Size of the original box
    orig_ngrid   - Original ngrid
    boost_ngrid  - ngrid of the boosted grid
    nrepeat      - Number of 
    power_func - Power spectrum function to be used on each k-mode, units
                 should be inverse of the box size (e.g. h/Mpc)
    [seed]     - Optional seed for the random number generator.

    returns 
    disp_grid  - (ngrid,ngrid,ngrid,3) array of displacements, with
                 disp_grid[i] giving the displacement along the i-th axis etc.
    """

    effective_res = boost_ngrid*nrepeat
    if effective_res%orig_ngrid != 0:
        raise Exception('boost_ngrid*nrepeat = {:,} must be a multiple of {:,}, the original grid size'.format(effective_res, orig_ngrid))
    print('- Building boost grid to increase resolution by a factor %d'%(effective_res//orig_ngrid),file=log)
    mersenne = RandomState(seed)
    ngrid3 = boost_ngrid**3

    boxsize = float(orig_boxsize) / nrepeat
    print('- Making k values',file=log)
    # make the k values 
    dk = 2 * pi / boxsize # Distance between the modes (k)
    # amplitudes of the double integral of density (proportional to potential)
    print('- Amplitudes of modes', file=log)
    if boost_ngrid<32:
        #inverse k^2 for potential, multiply to account for the 5-pt differencing
        ampl_func = lambda k : sqrt(power_func(k) * dk * dk * dk) / (k*k)
        mode_ampl = kfunc_on_grid(boxsize, boost_ngrid, ampl_func, k0_val=0.0, log=log)
    else:
        num_pts = boost_ngrid * 5
        print('- Large ngrid={:,}^3, sampling power spectrum 5*ngrid={:,} times'.format(boost_ngrid, num_pts), file=log)
        kmax = dk * boost_ngrid * sqrt(0.75)
        resample_k_pts = (arange(num_pts)+1)* (kmax / num_pts)
        ampl = sqrt(power_func(resample_k_pts)*dk*dk*dk)
        print('- Interpolating power spectrum', file=log)
        #inverse k^2 for potential, multiply to account for the 5-pt differencing
        ampl_func = lambda k : interp(k, resample_k_pts, ampl) / (k*k)
        mode_ampl = kfunc_on_grid(boxsize, boost_ngrid, ampl_func, k0_val=0.0, log=log) #* kfunc_on_grid(boost_ngrid, boost_ngrid, deconv_5pt, k0_val=1.0, log=log, vecfunc=True)

    # E.g. for 2048 grid we sample modes up to k=1024 dk . In the nrepeat=4 box this corresponds to k=256
    imax = orig_ngrid//(2*nrepeat)
    print('- Zero-ing modes (8 corners) :{:,} that are sampled by original {:,}^3 grid'.format(imax, orig_ngrid), file=log)

    clip_func = lambda k : where(k<(imax*dk), 0.0, 1.0)
    mode_ampl *= kfunc_on_grid(boxsize, boost_ngrid, clip_func, log=log)
    
    print('- Building {:,} random numbers'.format(ngrid3*2), file=log)
    grf_k = mersenne.standard_normal(size=2*ngrid3).view(dtype=complex128)
    print('- Scaling', file=log)
    grf_k *= mode_ampl.ravel()
    grf_k.shape = (boost_ngrid, boost_ngrid, boost_ngrid)

    print('- {:,}^3 fourier transform to calculate potential'.format(boost_ngrid), file=log)
    # 'integral' of displacement field (i.e. proportional to potential,
    # and the displacement is the gradient of this)
    int_disp = fftn(grf_k).real 

    print('- 5 point gradient to calculate displacement field', file=log)
    dx = boxsize/boost_ngrid
    disp_fld = gradient_5pt_3d(int_disp * (1.0 / dx)).astype(float32)
    print('- Computing RMS', file=log)
    rms_disp = sqrt(square(disp_fld).sum(axis=3).mean(dtype=float64))
    print('- RMS displacement %.3f kpc/h'%(rms_disp * 1e3,), file=log)
    return disp_fld

def _test_cic_speed():
    # speed test

    from numpy.random import random
    from numpy import argsort
    from lizard.lizard_c import get_cells
    from lizard.log import VerboseTimingLog



    n = 128
    pts = random(n*n*n*3)
    pts.shape = (n*n*n,3)
    masses = empty(n*n*n,dtype=float64)
    masses[:] = 2.0
    ngrid = n
    pts *= ngrid
    vel = random(n*n*n*3)
    vel.shape = pts.shape

    
    cic = get_cic(pts[:100], ngrid) #quick call to load the library
    log = VerboseTimingLog()
    print('Lizard CIC for {:,} points          '.format(pts.shape[0]),file=log)
    cic = get_cic(pts, ngrid)
    print('Lizard CIC+gradients for {:,} points'.format(pts.shape[0]), file=log)
    cic_grad = get_cic(pts, ngrid, mom=vel)

    print('Sorting ',file=log)
    cell = get_cells(pts*(1.0/ngrid), ngrid)

    idx = argsort(cell)
    pts = pts[idx].copy()
    print('Lizard CIC+gradient on {:,} sorted points'.format(pts.shape[0]), file=log)
    cic = get_cic(pts,ngrid,mom=vel)
    log.close()


def _test_5pt(n=196):
    from numpy import arange, float64, roll
    from lizard.lizard_c import gradient_5pt_3d
    from lizard.log import VerboseTimingLog

    log = VerboseTimingLog()

    v = (arange(n*n*n)%(n-1)).astype(float64)
    v.shape = (n,n,n)
    print('Test 5 point gradient', file=log)

    print('Numpy 5 point gradient on {:,}^3 grid  '.format(n), file=log)
    grad = []
    for i in range(3):
        # five-point stencil for gradient
        s = (-roll(v,-2,axis=i) + 8*roll(v,-1,axis=i) - 8*roll(v,1,axis=i) + roll(v,2,axis=i)) * (1.0/12.0)
        grad.append(s)

    print('Loading lizard library',file=log)
    # load library
    t = gradient_5pt_3d(v[:3,:3,:3])
    print('lizard 5 point gradient for {:,}^3 grid'.format(n),file=log)
    grad2 = gradient_5pt_3d(v)
    print('done',file=log)
    err_max = max(abs(grad[0] - grad2[:,:,:,0]).max(),
                  abs(grad[1]- grad2[:,:,:,1]).max(),
                  abs(grad[2]- grad2[:,:,:,2]).max())
    

    print('Maximum error', err_max,file=log)
    assert(err_max<1e-4)
    log.close()

def _test_interp_speed():
    # test the speed of 3d interp
    import numpy as np
    from time import time
    from lizard.log import VerboseTimingLog
    from lizard.lizard_c import map_coords, get_cells

    log = VerboseTimingLog()

    gw = 192
    npts = 128**3
    rs = RandomState(seed=123)
    pts = np.reshape(rs.rand(3*npts)*gw, (npts,3))
    grid = np.reshape(np.arange(gw**3*3),(gw,gw,gw,3)) % 15

    for do_sort in [False, True]:
        
        if do_sort:
            print('Sorted', file=log)
            idx = np.argsort(get_cells(pts/gw, gw))
            pts = pts[idx].copy()
        else:
            print('Unsorted (lots of cache misses)',file=log)

        ans0 = empty(pts.shape)
        t0 = time()
        print('  3x1d interpolation of {:,} points'.format(npts), file=log)
        for axis in range(3):
            ans0[:,axis] = map_coords(grid[:,:,:,axis], pts.T)
        t1 = time()
        print('  = {:,} pts/sec'.format(int(npts/(t1-t0))),file=log)

        t0 = time()
        print('  3d interpolation of {:,} points'.format(npts), file=log)
        ans1 = interp_vec3(grid,pts)
        t1 = time()
        print('  = {:,} pts/sec'.format(int(npts/(t1-t0))),file=log)

        err = ans0 - ans1
        max_err = abs(err).max()
        print('  Max diff', max_err,file=log)

def _test_k_func():
    from numpy import sin, cos, argmax, flatnonzero
    from lizard.log import VerboseTimingLog

    n = 192

    L = 100.0
    k0_val = 0.0 
    func = lambda x : (cos(x) - sin(x)/x)/(4*pi)

    # load lizard
    res0 = kfunc_on_grid(L, 1, func, k0_val) # n=1 easy

    log = VerboseTimingLog()
    print('Evaluating a function by building k directly', file=log)
    print('  making k', file=log)
    k, kmag, inv_k2 = make_k_values(L, n)
    print('  calling func', file=log)
    res1 = empty(kmag.size, dtype=float64)
    res1[1:] = func(kmag.ravel()[1:])
    res1[0] = k0_val
    res1.shape = kmag.shape
    print('Evaluating only on packed points', file=log)
    res0 = kfunc_on_grid(L, n, func, k0_val)
    print('Checking errors', file=log)



    # comparison
    err = abs(res1-res0).max()
    idx = argmax(abs(res1-res0))

    print('error', err, 'Maximum at', ((idx//(n*n))%n, (idx//n)%n, idx%n), file=log)
    if err>1e-13:

        first_err = flatnonzero(abs(res1.ravel()-res0.ravel()))[0]
        idx = first_err
        print('first error', ((idx//(n*n))%n, (idx//n)%n, idx%n), file=log)
        print('res0',res0,file=log)
        print('res1',res1)
    assert(err<1e-13)


# you'll want to run "python -m lizard.grid" to hit this
if __name__=='__main__':
#    _test_cic_speed()
    _test_k_func()
#    _test_5pt()
#    _test_interp_speed()
