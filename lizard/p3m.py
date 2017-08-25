"""
Particle-Particle Particle-Mesh (P^3M) module

Module for calculating inter-particle forces via the use of direct summation
for close neighbours + far contribution from weighting onto a grid (cloud-in-
cell) and using an FFT.

Peter Creasey  - Dec 2015
"""
from __future__ import print_function, absolute_import, division, unicode_literals
from numpy import zeros_like, array, floor, square, sqrt, reshape, sin, cos, \
    pi, empty, searchsorted, arange, where, float64, logical_not, exp, isscalar
from numpy.fft import fftn, ifftn
from scipy.special import erf
from lizard.ngb_kernel import radial_kernel_evaluate, periodic_kernel, setup_newton
from lizard.grid import kfunc_on_grid, get_cic, gradient_5pt_3d, interp_vec3
from lizard.log import MarkUp as MU, null_log

def _ta_cubic(a):
    """
    Make a force-split with cubic short range kernel, i.e. within |x|<r_core
    we use the force GM(1/r^2 - 4r + 3r^2) rather than GM/r^2.
    """
    return -12* (sin(a) / a  + 2 * cos(a)/(a*a) - 2.0/(a*a)) / (a*a)

def _ta_quartic(a):
    """
    Make a force-split with quartic short range kernel, i.e. within |x|<r_core
    we use the force GM(1/r^2 - 5r/2 + 3r^3/2) rather than GM/r^2, and the 
    long-range potential corresponds to
    15/8 - 5r^2 / 4 + 3 x^4 /8 
    """
    return where(a>1e-2, 15*((3-a*a)*sin(a) - 3*a*cos(a))/(a*a*a*a*a),
                 1.0 - (3/56.0)*(a*a)) # Taylor for low a, since cancellation is terrible

def _ta_erf(a, rsmult=4.5):
    """
    Make a force-split with error-function kernel, i.e. within |x|<r_core
    we use the force GM/r * erfc(r/2r_s) rather than GM/r^2, where 
    r_s = r_core/4.5 (as for Gadget)
    """
    return exp(-square(a/rsmult))

def get_force_split(r_split, mode='cubic', deconvolve_cic=False):

    class PeriodicForceSplit:
        def __init__(self, rsplit, trans_func, kernel_func, deconvolve_cic):
            self._rsplit = float(rsplit)
            # Lazy building
            self._fft_wts = None
            self._trans_func = trans_func
            self._kernel_func = kernel_func
            self._deconvolve_cic = bool(deconvolve_cic)
        def rsplit(self):
            return self._rsplit

        def get_kernel(self, r_soft, kernel_pts):
            return self._kernel_func(self._rsplit, r_soft, kernel_pts)

        def get_fft_wts(self, log):
            if self._fft_wts is None:
                fft_size, wanted = _suggest_fft_size(self._rsplit)
                print('Wanted fft of size', wanted, 'using', fft_size, 
                      file=log)
                print('Building weights for {:,}^3 FFT'.format(fft_size),
                      file=log)
                self._fft_wts = _build_fft_wts(fft_size, self._rsplit, 
                                               log, trans_func=self._trans_func, 
                                               deconvolve_cic=self._deconvolve_cic)
            return self._fft_wts

    if mode=='cubic':
        return PeriodicForceSplit(r_split, trans_func=_ta_cubic, 
                                  kernel_func=_newton_soft_cubic_kernel, 
                                  deconvolve_cic=deconvolve_cic)
    elif mode=='quartic':
        return PeriodicForceSplit(r_split, trans_func=_ta_quartic, 
                                  kernel_func=_newton_soft_quartic_kernel,
                                  deconvolve_cic=deconvolve_cic)
    elif mode=='erf':
        return PeriodicForceSplit(r_split, trans_func=_ta_erf, 
                                  kernel_func=_newton_soft_erf_kernel,
                                  deconvolve_cic=deconvolve_cic)
    else:
        raise Exception('Unknown force split mode '+str(mode))

def pp_accel(force_split, wts, pos, r_soft, kernel_pts=500, log=null_log):
    """ 
    Calculate the short range particle-particle accelerations in a periodic box
    Set r_soft to None if you want Newtonian
    """
    r_split = force_split.rsplit()
    kernel = force_split.get_kernel(r_soft, kernel_pts)
    return periodic_kernel(r_split, kernel, pos, wts, log)

class PMAccel:
    """
    Particle-mesh acceleration (via FFT)
    """
    def __init__(self, force_split):
        self._fs = force_split
        # Lazy building
        self._fft_wts = None

    def accel(self, wts, pos, log=null_log):
        """ find the long-range force """

        if self._fft_wts is None:
            self._fft_wts = self._fs.get_fft_wts(log)
        return _grid_accel(pos, wts, self._fft_wts, log)

class IncrementalPMAccel:
    """
    Incremental version of the particle-mesh acceleration, i.e. it re-uses the
    FFT and the time derivative to extrapolate the acceleration
    """
    def __init__(self, force_split):
        self._fs = force_split
        # Lazy building
        self._fft_wts = None
        self._accel_patch = None # will contain (t0,t1,acc, da/dt) grids

    def accel(self, wts, pos, vel, t, log):
        """ find the long-range force """
        if self._fft_wts is None:
            self._fft_wts = self._fs.get_fft_wts(log)

        acc, patch = _patch_extrapolate_or_replace(wts, pos, vel, t, 
                                                   self._fft_wts,
                                                   self._accel_patch,log)
        self._accel_patch = patch
        return acc


def _patch_extrapolate_or_replace(wts, pos, vel, t, fft_wts, old_patch, log, err_tol=1.5):
    """
    Compare times and see if we should linearly extrapolate an old
    acceleration grid, or build a new one.
    """
    fft_size = fft_wts.shape[0]    
    if old_patch is not None:

        patch_num, t0, t1, acc, da_dt = old_patch

        # Can we extrapolate?
        if t<t1: 
            print('Linear extrapolation of {:,}^3 patch ({:,})'.format(fft_size, patch_num),
                  file=log)
            lin_acc = acc + (t-t0)*da_dt # for comparison even if we don't use
            
            print('Interpolating {:,} points'.format(len(pos)),file=log)

            accel_long = interp_vec3(lin_acc, pos*fft_size) 
            return accel_long, old_patch
    else:
        patch_num = 0


    # Make a new patch
    patch_num = patch_num + 1

    print('Creating acceleration patch number', patch_num, file=log)
    dt_max, accel0, daccel_dt = _grid_accel_deriv(pos, wts, vel, fft_wts, log=log)


    if patch_num>1:
        # Compare old and linearly estimated
        print('Linear extrapolation of patch', patch_num-1, file=log)
        lin_acc = acc + (t-t0)*da_dt # for comparison even if we don't use
        rms_acc = (3*square(accel0).mean())**0.5
        rms_err = (3*square(accel0-lin_acc).mean())**0.5
        dt_nonlin = (t1-t0)/err_tol
        err = rms_err / rms_acc
        print('Error on acc %.3f%%'%(100*err), 'after', (t-t0)/dt_nonlin, 'linear timescales',file=log)
        if err>0.1:
            raise Exception('zOMG long-range scales non-linear, what happened to the universe?', file=log)

    dt_max *= err_tol # only use 10% for accuracy
    print('Patch expected to last for dt=', dt_max, file=log)
    new_patch = (patch_num, t, t+dt_max, accel0, daccel_dt)
    
    print('interpolating {:,} points'.format(len(pos)),file=log)
    accel_long = interp_vec3(accel0, pos*fft_size)
    return accel_long, new_patch

def _newton_soft_erf_kernel(rmax, r_soft, npts=500, rsmult=4.5):
    """
    Newtonian force modified with the error function to match at r=rmax, i.e.
    d/dr [ -erfc(r/2r_s) / r ] term rather than the Newtonian 1/r^2.

    d/dr -> erfc(r/2r_s)/r^2 + exp(-(r/2r_s)^2) / sqrt(pi) r r_s

    Optional short-range softening using the Monaghan kernel
    """
    r = (arange(npts)+1) * (float(rmax)/npts) # vals rmax/npts,..., rmax

    rs = rmax/rsmult

    # Erf multiplier for short-range potential
    kernel = (erf(r/(2*rs))-1)/(r*r*r) - exp(-square(r/(2*rs))) / (sqrt(pi) * r * r * rs) # grad/r, multipliers for dx
        
    if r_soft is None:
        return kernel

    h = 2.8 * r_soft
    if h>rmax:
        raise Exception('Softening (2.8r_soft=%.4f) is greater than cutoff (%.4f)'%(h,rmax))

    a = r/h
    # Monaghan gravity, extra 1/r to make a multiplier for dx
    kernel_soft = where(a>1, -1.0/(r*r*r),
                        -(1.0/(h*h*h))*where(a<0.5,
                                             (5-18*a**2+15*a**3)*32/15.0,
                                             (320-720*a+576*(a**2)-160*(a**3) - 1/(a*a*a))/15))

    # Add the Erf modifier
    return kernel_soft + kernel

def _newton_soft_cubic_kernel(rmax, r_soft, npts=500):
    """
    Newtonian force modified with a cubic (potential) to match at r=rmax, i.e.
    1/r^2 - 4r + 3r^2 term rather than Newtonian 1/r^2.

    Optional softening at short range force using the Monaghan kernel 
    (h=2.8 r_soft)

    rmax   - maximum range of force (cubic splitting scale)
    r_soft - softening length (Plummer equivalent), for use with Monaghan kernel
             (h=2.8 r_soft)

    """
    r = (arange(npts)+1) * (float(rmax)/npts) # vals rmax/npts,..., rmax

    if r_soft is None:
        kernel = -1/(r*r*r) + (4*rmax-3*r)/(rmax**4) # grad/r, multipliers for dx
        return kernel

    h = 2.8 * r_soft
    if h>rmax:
        raise Exception('Softening (2.8r_soft=%.4f) is greater than cutoff (%.4f)'%(h,rmax))

    a = r/h
    # Monaghan gravity, extra 1/r to make a multiplier for dx
    kernel_soft = where(a>1, -1.0/(r*r*r),
                        -(1.0/(h*h*h))*where(a<0.5,
                                             (5-18*a**2+15*a**3)*32/15.0,
                                             (320-720*a+576*(a**2)-160*(a**3) - 1/(a*a*a))/15))
    
    kernel = kernel_soft + (4*rmax-3*r)/(rmax**4) # add the cubic modifier
    return kernel
    

def _build_fft_wts(ngrid, r_core, log, trans_func=_ta_cubic, deconvolve_cic=False):
    """
    Build the k-modes for the long-range force with the cubic/whatever core,
    i.e. find k-modes phi_k s.t. ifftn(phi_k).real gives the 
    -4 pi * Green's function of the Laplacian at r>r_core (i.e. asymptotic to 
    1/r if r<<1) and has a cubic core that is continuous with continuous 1st 
    derivative at r=r_core.
    """
    def t_func(kmag):
        # potential as a function of k (transfer function mult by 1/k^2), ->1 as k->0
        # Transfer function to make a 1/r core into a cubic that matches at r=r_core
        t_k = trans_func(kmag * r_core)

        if deconvolve_cic:
            # Deconvolve for two CICs (mass and acceleration interpolation)
            t_k *= _deconvolve_cic(kmag/ngrid)
            
        # Multiply the usual gravity modes by the transfer function
        return (4*pi*ngrid*ngrid*ngrid) * t_k /(kmag*kmag)

    phi_k = kfunc_on_grid(1.0, ngrid, t_func, k0_val=0.0, log=log)
    return phi_k


def _grid_accel(pos, wts, fft_wts, log):
    """ Acceleration on PM grid using the given FFT weights """

    fft_size = fft_wts.shape[0]
    npos = array(pos) * fft_size
    print('Doing {:,}^3 CIC'.format(fft_size),file=log)
    # extra fft_size in wts to make finite difference the grad (dx = 1/fft_size)
    wts = array(wts) * fft_size
    cic = get_cic(npos, fft_size, wts) 
    print('Total mass on grid %3f. Doing FFT of weights.'%cic.sum(),file=log)
    modes = fftn(cic)
    print(MU.OKBLUE+'Inverse {:,}^3 FFT'.format(fft_size)+MU.ENDC,file=log)
    npot = ifftn(modes*fft_wts).real # potential * fft_size
    print('Gradient via finite-differences',file=log)
    grad = gradient_5pt_3d(npot) 
    print('interpolating {:,} points'.format(len(pos)),file=log)
    accel_long = interp_vec3(grad, npos) 
    return accel_long

def _grid_accel_deriv(pos, wts, vel, fft_wts, log):
    """ Accel and rate of change on the grid using the given FFT weights """
    pos = array(pos)
    fft_size = fft_wts.shape[0]
    # extra fft_size so that finite difference is grad (dx = 1/fft_size)
    nwts = array(wts) * fft_size

    print('Doing {:,}^3 CIC'.format(fft_size),file=log)
    # momenta in grid coords 
    if isscalar(nwts):
        mom = vel * nwts * fft_size 
    else:
        mom = vel * reshape(nwts, (len(nwts),1)) * fft_size 

    cic = get_cic(pos*fft_size, fft_size, mass=nwts, mom=mom)
    print('Total mass on grid %3e. Doing FFT of weights.'%cic.sum().real,file=log)
    modes = fftn(cic)
    print(MU.OKBLUE+'Inverse {:,}^3 FFT'.format(fft_size)+MU.ENDC,file=log)
    pot_times_n = ifftn(modes*fft_wts) # imaginary part contains dpot/dt
    print('Gradient via finite-differences of accel',file=log)
    acc = gradient_5pt_3d(pot_times_n.real) # dx = 1/ngrid
    print('Gradient for da/dt',file=log)
    da_dt = gradient_5pt_3d(pot_times_n.imag) # dx = 1/ngrid
    print('RMS and maximum', file=log)
    rms_acc = (3*square(acc).mean())**0.5
    max_da_dt = square(da_dt).sum(3).max()**0.5
    if max_da_dt<=0:
        raise Exception('Zero acceleration rate => zero vel or floating point problem?')
    dt_est = rms_acc/max_da_dt
    return dt_est, acc, da_dt


def _suggest_fft_size(rsplit):
    """
    suggest an fft size for a given split scale (on the unit box).
    Want FFT to have 6+ cells per over the split scale, but also be a multiple
    of 2s and 3s.
    """
    if rsplit>0.18:
        raise Exception('rsplit {:,}>0.18, compact kernel becomes aspherical'.format(rsplit))

    fft_sizes = (36,48,54,64,72,96,108,128,144,162,192,216,225,243,256,288,324,384,432, 486,512,576) # too much mem for me!
    desired_size = 6.0/rsplit
    # find best size bigger than this
    idx = searchsorted(fft_sizes, desired_size)
    if idx==len(fft_sizes):
        raise Exception('rsplit {:,} wants FFT of {:,}> largest allowed ({:,})'.format(rsplit, desired_size,fft_sizes[-1]))
    
    return fft_sizes[idx], desired_size


def subcube_periodic_ghosts(pts, cen, hw, ghost_rad):
    """
    Mark the points within a sub-cube in the periodic unit box, and all their 
    ghosts, and the ghosts of those outside.

    hw        - half width of sub-box
    cen       - centre of sub-box
    ghost_rad - radius with which to consider ghosts

    returns idx_in, idx_in_nonghost, idx_out, idx_out_nonghost

    """
    if hw>=0.5:
        raise Exception('Sub box with half width {:,} is as large or larger than the whole box'.format(hw))

    import numpy as np
    # Centre the points on zero, and with coords in [-0.5, 0.5)
    pos = (array(pts) - cen) + 0.5
    pos -=  0.5 + floor(pos)
    # mark those within the box
    pos = np.abs(pos) # |x|,|y|,|z|
    max_norm = pos.max(axis=1)

    in_box = max_norm<hw # if all |x|,|y|,|z|< hw then definitely in box
    out_box = logical_not(in_box) # or outside
    in_plus_ghost = square(np.maximum(pos-hw,0)).sum(axis=1)<(ghost_rad**2)
    idx_in = np.flatnonzero(in_plus_ghost)
    idx_in_nonghost = np.flatnonzero(in_box[idx_in])
    idx_out = np.flatnonzero(max_norm>(hw-ghost_rad))
    idx_out_nonghost = np.flatnonzero(logical_not(in_box[idx_out]))
    
    return idx_in, idx_in_nonghost, idx_out, idx_out_nonghost


def _inter_cubic_fft_wts(ngrid, r_coarse, r_fine):
    """
    Build the k-modes for the intermediate-range force with the cubic core, 
    i.e. find k-modes phi_k s.t. ifftn(phi_k).real gives the -4 pi * Green's function 
    of the Laplacian at r>r_core (i.e. asymptotic to 1/r if r<<1) and has a 
    cubic core that is continuous with continuous 1st derivative at r=r_core.
    """
    def t_func(kmag):

        a1 = kmag * r_coarse
        a2 = kmag * r_fine

        # Transfer function to make a 1/r core into a cubic that matches at r=r_core
        t_k = 12* ((sin(a1) / a1  + 2 * cos(a1)/(a1*a1) - 2.0/(a1*a1)) / (a1*a1) - 
                   (sin(a2) / a2  + 2 * cos(a2)/(a2*a2) - 2.0/(a2*a2)) / (a2*a2))

        # Multiply the usual gravity modes by the transfer function
        return (4*pi*ngrid*ngrid*ngrid) * t_k /(kmag*kmag)

    phi_k = kfunc_on_grid(1.0, ngrid, t_func, k0_val=0.0)
    return phi_k

class IntermediateGrid:
    """ Make an extra (non-periodic) grid """
    def __init__(self, r_coarse, ngrid, centre, hw_min, ncell_kern=6):
        """
        [ncell_kern=6] - Minimum number of grid cells per kernel
        """
        self.r_coarse = float(r_coarse)
        assert(len(centre)==3)
        self.centre = array(centre, dtype=float64)

        self.ngrid = int(ngrid)
        hw, dx, sub_hw = _get_inter_grid(hw_min, r_coarse, self.ngrid)
        self.hw = hw
        self.r_fine = dx * ncell_kern
        self._ncell_coarse = self.r_coarse / dx
        self.sub_hw = sub_hw # half width of sub-box
        self._fft_wts = None # lazy building
        
    def split_in_vs_out(self, pos):
        
        idx_in, idx_in_nonghost, idx_out, idx_out_nonghost = subcube_periodic_ghosts(pos, self.centre, self.sub_hw, self.r_coarse)
        return idx_in, idx_in_nonghost, idx_out, idx_out_nonghost

    def topleft(self):
        return self.centre - self.hw

    def pm_accel(self, pos_in, wts_in, idx_in_nonghost, log=null_log):
        topleft = self.topleft()
        width = self.hw*2

        if self._fft_wts is None:
            r_max = self.r_coarse / width
            r_min = self.r_fine / width
            print('Building {:,}^3 FFT weights'.format(self.ngrid), file=log)
            self._fft_wts = _inter_cubic_fft_wts(self.ngrid, r_max, r_min)
            print('Rcoarse in grid cells', self._ncell_coarse , 'Rfine', r_min*self.ngrid, file=log)

        accel = _inter_pm_accel(pos_in, wts_in, topleft, width, self._fft_wts, idx_in_nonghost, self._ncell_coarse, log)
        return accel

    def pp_accel(self, pos_in, wts_in, idx_in_nonghost, r_soft=None, kernel_pts=150,log=null_log):
        accel = _inter_pp_accel(self, pos_in, wts_in, r_soft, kernel_pts,log=log)
        return accel[idx_in_nonghost]

def _get_inter_grid(hw_isol, r_coarse, ngrid, ncell_grad=2):
    """ 
    Find the half-width and r_fine for an intermediate grid that guarantees
    that particles within the central hw_isol box will *not* be ghosts of the
    particles outside the box (i.e. they are completely excluded from that
    calculation)

    hw_isol        - half-width from centre which is excluded from outer region
    r_coarse       - force-splitting scale of the coarse grid
    ngrid          - number of grid cells (per dimension)
    [ncell_grad=2] - 5 point gradient uses 2 cells either side, so we need this 
                     buffer to avoid periodicity

    returns box_hw, dx, r_B
    box_hw   - half width of box
    dx       - size of single cell
    r_B      - hw in which force is calculated via this grid (outside is ghosts)

    Illustration of one corner of the intermediate force grid:
    
    -->| hw_isol
    ---------->| hw_isol + r_coarse
    ------------------>| hw_isol + 2*r_coarse
    -------------------->| box_hw := hw_isol + 2*r_coarse + ncell_grad * dx / 2
    A A B B B B C C C C .
    A A B B B B C C C C .
    B B B B B B C C C C .
    B B B B B B C C C C .
    B B B B B B C C C C .
    B B B B B B C C C C .
    C C C C C C C C C C .
    C C C C C C C C C . .
    C C C C C C C C . . .
    C C C C C C C . . . .
    . . . . . . . . . . .

    where dx is given by box_hw / (0.5 * ngrid)
    

    Zone . - empty cells just there as a buffer (for 5-point gradient)
    Zone C - contains 'ghost' particles for correct force on B, but whose force calculated elsewhere
    Zone B - Forces calculated here, but are also the ghosts for particles in C
    Zone A - Forces calculated here and only here
    """

    ncell_grad = 2 


    r_A = hw_isol # particles which will *only* have force calculations from short+med+long
    r_B = r_A + r_coarse # Zone B - these particles have short+med+long forces, but they are also the ghosts of particles in C, so they will be used outside too
    r_C = r_B + r_coarse # Zone C - these particles are just here as 'ghosts' to get the force on B correct
    hw = ngrid * r_C  / (ngrid - ncell_grad) # need an extra buffer to account for periodicity when doing displacement
    dx = 2 * hw / ngrid
    
    if hw>=0.5:
        raise Exception('Isolated box %.3f times bigger than periodic box (should be smaller)'%(hw*2))
    return hw, dx, r_B
    
def _inter_pm_accel(pos, wts, topleft, width, fft_wts, idx_nonghosts, ncell_ghost, log, ncell_grad=2):
    """ Force on the grid using the given FFT weights """

    fft_size = fft_wts.shape[0]
    grid_pos = array(pos) - topleft
    grid_pos = (fft_size/width) * (grid_pos - floor(grid_pos))
    print('Isolated PM force on {:,} grid'.format(fft_size), file=log)
    # Check ghosts are far enough from boundary to avoid periodic effects
    ghost_range = ncell_grad//2 
    if grid_pos.min()<ghost_range  or grid_pos.max()>fft_size-ghost_range:
        print('Grid pos in', grid_pos.min(axis=0), grid_pos.max(axis=0), file=log)

        raise Exception('Ghosts too close to boundary to be isolated')

    print('Doing {:,}^3 CIC'.format(fft_size),file=log)    
    cic = get_cic(grid_pos, fft_size, wts)
    print('Total mass on grid %3f. Doing FFT of weights.'%cic.sum(),file=log)
    modes = fftn(cic)
    print('Inverse {:,}^3 FFT'.format(fft_size),file=log)
    pot = ifftn(modes*fft_wts).real
    print('Gradient via finite-differences',file=log)

    clip = int(ncell_ghost) - ncell_grad # Clip off boundaries that were only used for ghosts & gradients
    scale = fft_size / (width * width) # dx = 1/ngrid, then because we scaled all the positions, we diluted the r^-2 force, need to rescale back
    pot = pot[clip:-clip,clip:-clip,clip:-clip] * scale 
    accel_grid = gradient_5pt_3d(pot) 

    pos_ng = grid_pos[idx_nonghosts] - clip
    print('interpolating {:,} points'.format(len(pos_ng)),file=log)
    if pos_ng.min()<0.0 or pos_ng.max()>fft_size-2*clip:
        print('Grid pos non-ghost in', pos_ng.min(axis=0), pos_ng.max(axis=0), file=log)    
        raise Exception('Real particles outside central region of isolated grid')

    return interp_vec3(accel_grid, pos_ng)

def _inter_pp_accel(ig, pos, wts, r_soft, kernel_pts, log=null_log):
    
    topleft = ig.topleft()
    width = ig.hw * 2.0
    scaling = 1.0 / width

    kernel = _newton_soft_cubic_kernel(ig.r_fine, r_soft, kernel_pts)
    
    scaled_pos = pos - topleft
    scaled_pos = (scaled_pos - floor(scaled_pos)) * scaling
    
    print('Scaled pos in', scaled_pos.min(axis=0), 'to', scaled_pos.max(axis=0), file=log)
    # Since we 'widen' the positions the dx increases (scaling the acceleration), to compensate
    kernel *= width 

    # Don't need to worry about periodicity since this is an isolated box
    pairs, accel = radial_kernel_evaluate(ig.r_fine*scaling, kernel, scaled_pos, wts, log)
    return accel

def _newton_soft_quartic_kernel(rmax, r_soft, npts=500):
    """
    Newtonian force modified with a quartic (potential) to match at r=rmax, i.e.
    1/r^2 - 5r/2 + 3r^3/2 term rather than Newtonian 1/r^2.

    Optional softening at short range force using the Monaghan kernel 
    (h=2.8 r_soft)

    rmax   - maximum range of force (splitting scale)
    r_soft - softening length (Plummer equivalent), for use with Monaghan kernel
             (h=2.8 r_soft)

    """
    r = (arange(npts)+1) * (float(rmax)/npts) # vals rmax/npts,..., rmax

    if r_soft is None:
        kernel = -1/(r*r*r) + (2.5*rmax*rmax-1.5*r*r)/(rmax**5) # grad/r, multipliers for dx
        return kernel

    h = 2.8 * r_soft
    if h>rmax:
        raise Exception('Softening (2.8r_soft=%.4f) is greater than cutoff (%.4f)'%(h,rmax))

    a = r/h
    # Monaghan gravity, extra 1/r to make a multiplier for dx
    kernel_soft = where(a>1, -1.0/(r*r*r),
                        -(1.0/(h*h*h))*where(a<0.5,
                                             (5-18*a**2+15*a**3)*32/15.0,
                                             (320-720*a+576*(a**2)-160*(a**3) - 1/(a*a*a))/15))
    
    kernel = kernel_soft + (2.5*rmax*rmax -1.5*r*r)/(rmax**5) # add the quartic modifier
    return kernel

def _deconvolve_cic(k):
    """
    k - magnitude of a k mode for the N^3 box (if you want a unit box pass in
        k = w/N where N is the number of grid points)

    Deconvolution (i.e. sharpening) that approximately subtracts the effect of 
    both the CIC for the mass and then the CIC for the force interpolation: 

    T(k) = 1 + k^2 / 6

    which can be found as the O(k^2) Taylor expansion (i.e. depends only on the
    magnitude of k) to the Springel (2005) deconvolution

    T(k) = [sinc(k_x/2) sinc(k_y/2) sinc(k_z/2)]^-4

    which is the filter that you would expect if one models each of the two CICs
    as the convolution with the triangular function:

    C(x) :=      0, |x| >= 1, or
             1-|x|, |x| <= 1.

    in each of x,y,z. 

    Note that in reality the effect of the CIC depends on how close you are to
    the corner of a grid cell, i.e. the effect of CIC is not that of 'true' 
    convolution (since it depends on absolution position). For this reason you
    should only use this on a kernel that is already suppressed at large-k 
    (e.g. the long-range gravity kernel), where the behaviour of T(k) for* 
    k -> pi is unimportant (excluding it diverging, which would still cause 
    problems).**
    
    * pi is the largest wavenumber for the DFT on at integer x-points from 
      0 to N-1

    ** this may be the longest doc-string for a 1-line function ever written.
    """
#    return 1.0 + (1/6.0) * square(k) 
    return 1.0 + (1/6.0)*square(k) + (11.0/45.0)*square(square(k*0.5))

def test_inter():
    
    ngrid = 96
    r_fine = 6.0/ngrid
    r_coarse = 40.0/ngrid
    fft_wts = _inter_cubic_fft_wts(ngrid,r_coarse, r_fine)
    
    kernel = ifftn(fft_wts).real
    acc = sqrt(square(gradient_5pt_3d(kernel)*ngrid).sum(3).ravel())


    import pylab as pl
#    pl.imshow(kernel[0])

    from lizard.grid import make_k_values
    r = make_k_values(1.0, ngrid)[1]
    r *= 1.0/(ngrid*2*pi)
#    pl.semilogy(r.ravel(),abs(kernel.ravel()), 'k,')
    pl.semilogy(r.ravel(),acc, 'k,')
    pl.axvline(r_fine, c='k', ls=':')
    pl.axvline(r_coarse, c='k', ls=':')
    
    acc_from_pp = _newton_soft_cubic_kernel(r_fine, None, 200) # these are multipliers that you multiply by r
    r = (arange(len(acc_from_pp))+1) * r_fine / len(acc_from_pp)
    acc_from_pp = abs(acc_from_pp * r + 1/(r*r))
    pl.semilogy(r, acc_from_pp)
    import numpy as np
    r = np.linspace(r_fine, r_coarse, 200)
    acc_cubic_mid = -1.0/(r*r) + (4*r*r_coarse - 3*r*r)/(r_coarse**4)
    pl.semilogy(r, abs(acc_cubic_mid))
    pl.show()

def test_kernel():
    """
    Compare the transfer functions for the cubic, quartic and gadget kernels

    1) Shows how the truncation errors on the polynomial function are smaller
    (even though as you go to ->infty the exponential decays quicker), because
    we truncate relatively early.

    2) Shows how this results in interpolation errors. Since in principle a 
    second order (in x^2,y^2,z^2) function can be differented and interpolated 
    on a grid perfectly, the leading source of error is the third derivative.
   
    """
    import numpy as np
    import pylab as pl

    latexParams = {'figure.dpi':150,
               'figure.figsize':[6.64,4.98],
               'text.usetex':True,
               'text.fontsize':8,
               'font.size':8,
               'axes.labelsize':8,
               'figure.subplot.top':0.95,
               'figure.subplot.right':0.95,
               'figure.subplot.bottom':0.15,
               'figure.subplot.left':0.15,
               'lines.linewidth':0.5,
               'lines.markersize':3,
               'savefig.dpi':600,
               'legend.fontsize':8}
    pl.figure(dpi=100)
    max_a = 5.625 * sqrt(0.75)  # 5.625 cells, where truncation begins
    a = np.linspace(1e-3, int(max_a)+1,10000)
    rmax = 2.0
    rs = rmax / 4.5 # from Gadget-2 paper

    pl.subplot(221)
    pl.plot(a, _ta_quartic(a), label='quartic')
    pl.plot(a, _ta_cubic(a), label='cubic')
    pl.plot(a, _ta_erf(a), label='erf (gadget)')

    pl.legend(frameon=False, loc=3)
    pl.axvline(max_a, ls=':', color='k')
    pl.ylabel('t(k/ksplit)')
    pl.xlabel('k/ksplit')
    ax=pl.subplot(222)
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    npts = 1000

    delta_r = float(rmax)/npts
    r = (np.arange(npts)+1) * delta_r # vals rmax/npts,..., rmax
    k_cubic = _newton_soft_cubic_kernel(rmax, None, npts)
    k_quartic = _newton_soft_quartic_kernel(rmax, None, npts)
#    pl.plot(r, (r**-2)+r*k_cubic)
#    pl.plot(r,(r**-2) +r*k_quartic)
# potential
    pl.plot(r/rmax, 2-2*(r/rmax)**2+(r/rmax)**3, label='cubic')
    pl.plot(r/rmax, 15/8.0 - 1.25*(r/rmax)**2 +0.375 *(r/rmax)**4, label='quartic')


    gad_pot = rmax*erf(r/(2*rs))/r
    gad_pot_pp = np.gradient(np.gradient(gad_pot))*((rmax/delta_r)**2)
    gad_pot_ppp = np.gradient(gad_pot_pp)*rmax/delta_r
    pl.plot(r/rmax, gad_pot, label='erf (gadget)')

    pl.legend(frameon=False, loc=3)

    pl.ylabel(r'$\phi_{\rm long}(r/r_{\rm split})$')
# 2nd deriv
    ax=pl.subplot(223)
    pl.plot(r/rmax, -4+6*(r/rmax))
    pl.plot(r/rmax,  -2.5 +3*4*0.375 *(r/rmax)**2)
    pl.plot(r/rmax, gad_pot_pp)
# 3rd deriv
    ax=pl.subplot(224)
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()

    pl.plot(r/rmax, 6+r*0, label='cubic')
    pl.plot(r/rmax,   +2*3*4*0.375 *(r/rmax), label='quartic')
   
    pl.plot(r/rmax, gad_pot_ppp, label='erf (gadget)')
    pl.ylabel(r'$\phi^{\prime\prime\prime}_{\rm long}(r/r_{\rm split})$')
    pl.xlabel(r'$r/r_{\rm split}$')

    pl.ylim(-10,30)

    pl.show()
def test_grad(mode):
    """
    Plot acceleration split into long and short range split
    """
    import sys
    ngrid = 64
    r_fine = 6.0/ngrid

    fs = get_force_split(r_fine, mode=mode) # 'cubic', 'quartic'

    
    fft_wts = fs.get_fft_wts(sys.stdout)
    kernel = ifftn(fft_wts).real
    acc = sqrt(square(gradient_5pt_3d(kernel)*ngrid).sum(3).ravel())
    
    import numpy as np

    import pylab as pl
    from lizard.grid import make_k_values
    r = make_k_values(1.0, ngrid)[1]
    r *= 1.0/(ngrid*2*pi)
#    pl.semilogy(r.ravel(),abs(kernel.ravel()), 'k,')
    pl.semilogy(r.ravel(),acc, 'k,')
    pl.axvline(r_fine, c='k', ls=':')

    
    acc_from_pp = fs.get_kernel(1e-3, 200) # these are multipliers that you multiply by r
    r = (arange(len(acc_from_pp))+1) * r_fine / len(acc_from_pp)
    acc_from_pp = 1/(r*r) + (acc_from_pp * r) #- 1/(r*r)
    pl.semilogy(r, acc_from_pp, 'b')

    r = np.linspace(r_fine, 0.75**0.5, 200)
    acc_cubic_mid = -1.0/(r*r)# + (4*r*r_coarse - 3*r*r)/(r_coarse**4)
    pl.semilogy(r, abs(acc_cubic_mid),'g')
    pl.show()

if __name__=='__main__':
#    test_inter()
#    test_kernel()
    test_grad('cubic')
