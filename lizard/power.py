"""
Cosmological power spectrum (linear growth etc.)
"""

from __future__ import print_function
from numpy import sin, cos, power, square, pi, sqrt, loadtxt, log, array, exp, interp
from scipy.integrate import quad
from .log import null_log
from iccpy.cgs import yr, mpc

def hubble_closure(H0, OmegaM0, OmegaLam0):
    """
    Closure (returns a function) that gives the hubble rate per expansion 
    factor a. OmegaK is assumed to be 1-omegaM - omegaLam

    H0        - Hubble rate at a=1 
    omegaM    - matter density at a=1
    omegaLam0 - Dark energy density at a=1

    returns 
    hubble     - function where hubble(a) is da/dt / a, with units of as H0
    """
    def hubble(a):
        return H0*sqrt((OmegaM0 + a*(1 - OmegaM0 - OmegaLam0)) / (a*a*a) + OmegaLam0)
    return hubble

def cosmological_perturbation_growth(a, OmegaM, OmegaLam):
    """ 
    Growth factor of perturbations up to expansion factor a, assuming LCDM 
    a - Desired expansion factor for cosmological perturbations
    OmegaM   - Omega Matter 
    OmegaLam - Lambda

    It is assumed that OmegaK = 1 - OmegaM - OmegaLam
    """

    hubble = hubble_closure(1.0, OmegaM, OmegaLam)
    growth_func = lambda s: power(s / (OmegaM + s * (1.0 - OmegaM - OmegaLam) + OmegaLam * s * s * s), 1.5)

    D1 = hubble(1.0) * quad(growth_func, 0.0, 1.0)[0]
    Da = hubble(a) * quad(growth_func, 0.0, a)[0]

    return Da / D1

    
def tophat_variance(r, power_spec):
    """ 
    Calculate the variance in a tophat of radius r,
    useful to normalise a power spectrum by sigma_8
    (see also Simon White's Les Houches lectures)

    Returns: variance
    """

    def sigma2(k):
        kr = r * k

        if kr < 1e-8:
            return 0

        w = 3 * (sin(kr) / kr - cos(kr)) / (kr*kr);
        x = 4 * pi * k * k * w * w * power_spec(k);

        return x
        
    # integrate from k = 0-500/R
    sigma_int2, abs_err = quad(sigma2, 1e-8, 500.0 / r)

    print('Absolute error on integration', abs_err)
    return sigma_int2

def efstathiou(k):
    """ Return the Efstathiou spectrum evaluated at k (in inverse Mpc/h) """

    ShapeGamma = 0.21 # Shape of power spec
    # constants
    AA = 6.4 / ShapeGamma # Mpc/h
    BB = 3.0 / ShapeGamma # Mpc/h
    CC = 1.7 / ShapeGamma # Mpc/h
    nu = 1.13
       
    res = k / power(1.0 + power(AA * k + power(BB * k, 1.5) + CC * CC * k * k, nu), 2 / nu)

    return res


def tabulated_pspec(kvals, pvals, ns=0.961):
    """
    Make a continuous power spectrum via linear interpolation from the table
    
    kvals - list of k (in h^-1 Mpc)
    pvals - P(k) (in Mpc^3 h^-3)

    [ns=0.961] - spectral index tilt, used in the extrapolation of the spectrum
                 which is assumed to be a power law with index k^ns below kmin
                 and k^(ns-4) above kmax.

    returns a function pspec(k) 
    """
    log_kvals = log(kvals)
    log_pvals = log(pvals)
    # go for k^1 and k^-3 at low and high end
    low_efolds = 10
    high_efolds = 10
    log_kvals = array([log_kvals[0]-low_efolds]+list(log_kvals) + [log_kvals[-1]+high_efolds])
    log_pvals = array([log_pvals[0]-low_efolds*ns]+list(log_pvals) + [log_pvals[-1]+high_efolds*(ns-4)])
    def pspec(k):
        # linear interpolation in log space
        res = exp(interp(log(k), log_kvals, log_pvals))
        return res
    return pspec

def readspec(name):
    """ Utility function """
    k, Pk = loadtxt(name).T
    return tabulated_pspec(k,Pk)


def pspec_normalised_by_sigma8_expansion(pspec, sigma8, a, omegaM, omegaL):
    """ Return a normalised power spectrum
    pspec  - (function) input power spectrum function (of k in h/Mpc)
    sigma8 - (float) desired sigma_8 (std. dev. at 8 Mpc/h)
    a      - (float) desired expansion factor
    omegaM - (float) Omega Matter (for LCDM pert. growth)
    omegaL - (float) Omega Lambda (for LCDM pert. growth)

    Returns: new_pspec 
    """

    eight_mpc_h = 8 

    old_sq_sigma8 = tophat_variance(eight_mpc_h, pspec)
    norm = sigma8 * sigma8 / old_sq_sigma8

    print('Normalisation for Sigma_8 = %f is %f'%(sigma8, norm))

    D = cosmological_perturbation_growth(a, omegaM, omegaL)
    f = lambda k : pspec(k) * (norm * D * D)

    return f


def velocity_from_displacement(a, omegaM, omegaL, H0):
    """
    In the Zel'dovich approximation the velocities are a multiple of the
    displacements, dependent only on the scale factor and the cosmology.
    Note omegaK is assumed to be 1 - omegaM - omegaL (i.e. radiation negligable)

    a      - the scale factor
    omegaM - matter fraction (e.g. 0.272)
    omegaL - cosmological constant (e.g. 0.728)

    Returns the multiplication factor for displacements to peculiar velocities, i.e.
    v = vel_fac * (x-x0)un
    where (x-x0) is in Mpc
    and v in km/s

    """

    hubble = hubble_closure(H0, omegaM, omegaL)
    omegaMa = omegaM / (a*a*a)
    
    vel_fac = hubble(a) * ((omegaMa * H0 * H0 / (hubble(a)*hubble(a))) ** (3.0 / 5.0))
    return vel_fac


def test_power():

    from numpy import linspace
    k = power(10.0, linspace(-3.5, 2.5, 200))
    z = 49
    a = 1.0 / (1.0+ z)
    pspec = pspec_normalised_by_sigma8_expansion(efstathiou, sigma8=0.9, a=a, omegaM=0.3, omegaL=0.7)
    Delta2 = k*k*k * pspec(k) # dimensionless variance

    import pylab as pl
    from iccpy.figures import latexParams
    pl.rcParams.update(latexParams)

    pl.loglog(k, pspec(k), 'k')
    pl.xlabel(r'$k \; \rm h/Mpc$')
    pl.xlim(0.003, 300)
    pl.ylim(2e-9,3e1)

    pl.ylabel(r'$ P(k)$')
    pl.show()

def test_vel_fac():
    omegaM = 0.3
    omegaL = 0.7
    a = 0.03
    H0 = 75.0 # km / s / Mpc
    eps = 1e-6

    v0 = cosmological_perturbation_growth(a, omegaM, omegaL)
    print('Perturbation size at a=%f, %f'%(a, v0))
    v1 = cosmological_perturbation_growth(a+eps, omegaM, omegaL)
    print('Perturbation size at a=%f, %f'%(a+eps, v1))


    gr = (v1-v0)/(eps*v0)
    print('Growth rate (per a)', gr)
    
    hubble = hubble_closure(H0, omegaM, omegaL)
    
    da_dt = a * hubble(a) 
    gr = gr * da_dt
    print('Growth rate', gr, 'km/s/Mpc')
#    from iccpy.cgs import mpc
    print('Growth rate a^0.5 * ', a * gr / (H0/100) / sqrt(a), 'km/s/h^-1 cMpc (i.e. gadget units)')
    
    print('Calc', velocity_from_displacement(a, omegaM, omegaL, H0))

class CosmologicalParticleModel:
    """
    Wrap a regular particle model to include the cosmological expansion
    terms.
    """
    def __init__(self, part_model, pos, wts, vel, r_soft, a0, omegaM, omegaL, H0, 
                 hub_crit=0.05, r_soft_maxphys=None, log=null_log):
        
        self._part_model = part_model
        self._hub_crit = float(hub_crit)
        self._a = float(a0)
        self._log = log
        # keep this scaling out to avoid dCIC/dt problems
        self._H0 = float(H0*1e5/mpc) 
        inv_H0 = 1.0 / self._H0
        # choose the softening function
        if r_soft_maxphys is None:
            r_soft_func = lambda a : float(r_soft)
        else:
            r_soft_func = lambda a : min(r_soft, r_soft_maxphys/a)

        self._r_soft_func = r_soft_func

        # hubble function, in units of sec^-1
        self._hubble_func  = hubble_closure(1.0, omegaM, omegaL) 
    
        hub = self._hubble_func(a0)

        part_model.set_time_pos_wts_vel(0.0, pos, wts*(inv_H0*inv_H0), vel*inv_H0)

    def get_time(self):
        """ 
        return integration parameter (in this case cosmological expansion a)
        """
        return float(self._a)

    def update(self, kicker):
        # da/dt / a in units of sec^-1
        a = self._a
        hub = self._hubble_func(a)    
        da_dt = hub * a
        dt_da = 1.0 / da_dt

        # Wrap the kicker before passing it to the underlying particle
        # distribution such that the cosmological expansion terms 
        # (hubble drag and change of time unit) are taken into account.
        def cosmo_kicker(dts, vel, acc):
            # scale the acceleration and apply Hubble drag
            accel = acc * (a**-3) - 2 * hub * vel

            # time step limits need to be scaled like accel
            da_seq = tuple((dt * (a**1.5) * da_dt, txt) for dt, txt in dts)
            # Expansion step limiter
            da_hub = self._hub_crit * a 
            da_seq = da_seq + ((da_hub, 'Hubble expansion'),)

            # convert to rate w.r.t a
            vel_a = vel * dt_da # i.e. dx/da
            acc_a = accel * (dt_da * dt_da) # Expansion limiter keeps 2nd order small

            # kick w.r.t a
            da, dpos, dvel_a = kicker(da_seq, vel_a, acc_a)
            
            # update my 'time'
            print('Universe expanded by %.3f%% from a=%.5f-%.5f'%(100*da/self._a, self._a, self._a+da), file=self._log)
            self._a = self._a + da

            # scale back
            dt = da * dt_da
            dvel = dvel_a * da_dt
            return dt, dpos, dvel

        r_soft = self._r_soft_func(a)
        return self._part_model.update(cosmo_kicker, r_soft)


    def get_pos_wts_vel(self):
        """ convert back out our scaling """
        pos,wts,vel = self._part_model.get_particles()
        H0 = self._H0
        return pos, wts*H0*H0, vel*H0

    def write_steps(self, steps):
        # Prettify with unit conversion
        da_dt = self._H0 * self._hubble_func(self._a) * self._a
        for da, key in steps:
            print('  %.3f Myr (%s)'%(da/(da_dt*1e6*yr),key), file=self._log)

    
if __name__=='__main__':
#    test_power()
    test_vel_fac()
