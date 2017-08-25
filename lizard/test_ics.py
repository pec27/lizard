from grid import *
from power import pspec_normalised_by_sigma8_expansion, efstathiou, cosmological_perturbation_growth, readspec
from uniform import load_glass
from numpy import *
from os import path

def pspec_on_grid(ngrid, boxsize, pspec):

    k1, kmag, inv_k2 = make_k_values(boxsize, ngrid)
    
    kmag[0,0,0] = kmag[0,0,1] # avoid calculating the power spectrum on k=0
    pspec_grid = pspec(kmag.ravel())
    pspec_grid[0] = 0 # Always 0 pawer on k=0 mode

    kmin, kmax, kbins, kvol = powerspec_bins(ngrid, boxsize)
    kmid_bins = 0.5 * (kmin + kmax).ravel()
    kbins = kbins.ravel()
    pspec = bincount(kbins, weights=pspec_grid) / bincount(kbins)
    
    return kmid_bins, pspec


    
def test_disp_fft():
    """ Test the FFT that produces the displacement field """
    import pylab as pl
    from iccpy.figures import latexParams
    from numpy.fft import ifftn
    a = 1.0 / (1.0+51.0)
    pspec = pspec_normalised_by_sigma8_expansion(efstathiou, sigma8=0.8, a=a, omegaM=0.279, omegaL=0.721)

    ngrid = 128
    boxsize = 1000.0 # Mpc / h
    dx = float(boxsize)/ngrid
    print 'Building displacement field'
    disp_grid = build_displacement(boxsize=boxsize, ngrid=ngrid, power_func=pspec, seed=12345)

    print 'Poor mans divergence'
    
    dvg = (disp_grid[0] - roll(disp_grid[0], 1, 0)) + (disp_grid[1] - roll(disp_grid[1], 1, 1)) + (disp_grid[2] - roll(disp_grid[2], 1, 2))
    dvg *= 1.0/dx
    delta = -dvg # The overdensity field

    print 'FFT back to spectra'
    modes = ifftn(delta)
    print 'Making graphs'

    pl.rcParams.update(latexParams)
    pl.figure(figsize=(3.32, 3.32), dpi=150)
    pl.subplot(111)
    
    kmin, kmax,kvol, kmid_bins, powerspec, p_err = modes_to_pspec(modes, boxsize)
    pl.errorbar(kmid_bins, powerspec, yerr=p_err, fmt='o', label=r'$\rm CIC$')

    kvals, pspec_bins = pspec_on_grid(ngrid, boxsize, pspec)
    pl.loglog(kvals, pspec_bins, 'k', ls='--', label=r'$\rm Efstathiou$')    

    pl.legend().draw_frame(False)

    pl.xlabel(r'$k \; \rm h^{-1} Mpc$')
    pl.ylabel(r'$P(k)$')

    pl.show()
    
def test_disp_field(redshift=50, boxsize=500.0):

    boxsize = float(boxsize) # Mpc / h    
    import pylab as pl
#    from iccpy.figures import latexParams
    a = 1.0 / (1.0+redshift)
    pspec = pspec_normalised_by_sigma8_expansion(efstathiou, sigma8=0.8, a=a, omegaM=0.279, omegaL=0.721)

    ngrid = 64
    ngrid_modes = 64 # for the binning of points

    glass = path.join(path.dirname(path.abspath(__file__)), '../example_scripts/glass64.dat')
    uni_pts, uni_sizes = load_glass(glass, boxsize, repeat=1)
    uni_pts = uni_pts.T # go from (N,3) to (3,N) array
    print 'pts in', uni_pts.min(), uni_pts.max()
    print 'Building displacement field'
    disp_grid = build_displacement(boxsize=boxsize, ngrid=ngrid, power_func=pspec)

    print 'Interpolating the displacement field'

    disps = interpolate_displacement_grid(uni_pts, disp_grid, boxsize, order=1) # displacement of all the points

    pos = uni_pts.copy()
    pos[0] += disps[:,0]
    pos[1] += disps[:,1]
    pos[2] += disps[:,2]

    # make periodic
    pos = remainder(pos, boxsize)
#    pl.rcParams.update(latexParams)
    pl.rcParams.update({'figure.subplot.left':0.1})
    pl.figure(figsize=(6.64, 3.32), dpi=150)
    pl.subplot(121)
    pl.plot(pos[0].ravel(), pos[1].ravel(), 'b,')
    pl.ylabel(r'$y \; \rm Mpc/h$')
    pl.xlabel(r'$x \; \rm Mpc/h$')
    pl.subplot(122)

    print 'Making CIC'
#    modes = cic_modes(reshape(pos, (3, pos.shape[1])), boxsize, ngrid_modes)
    modes = cic_modes(pos, boxsize, ngrid_modes)


    print 'Making power spectrum bins'
    kmin,kmax,kvol,kmid_bins, powerspec, p_err = modes_to_pspec(modes, boxsize)

    pl.errorbar(kmid_bins, powerspec, yerr=p_err, fmt='o', label=r'$\rm CIC$')


    kvals, pspec_bins = pspec_on_grid(ngrid_modes, boxsize, pspec)
    pl.loglog(kvals, pspec_bins, 'k', ls='--', label=r'$\rm Efstathiou binned$')    

    pl.legend()#frameon=False)
    pl.xlabel(r'$k \; \rm h^{-1} Mpc$')
    pl.ylabel(r'$P(k)$')



    pl.show()

def test_smooth():

    import pylab as pl
    from iccpy.figures import latexParams
    a = 1.0 / (1.0+127.0)
    pspec = pspec_normalised_by_sigma8_expansion(efstathiou, sigma8=0.9, a=a, omegaM=0.3, omegaL=0.7)

    ngrid = 64
    boxsize = 10.0 # Mpc / h
    z,y,x = mgrid[:ngrid,:ngrid,:ngrid] * (boxsize/float(ngrid))
    
    disp = build_displacement(boxsize=boxsize, ngrid=ngrid, power_func=pspec)
    pl.plot(disp[0,0,0])
    pl.show()




def test_readspec(name, omegaM, redshift, sigma8):
    """ read power spectrum from file, compare with matter spec """
    from numpy import loadtxt
    import pylab as pl
#    from iccpy.figures import latexParams
#    pl.rcParams.update(latexParams)
    data = loadtxt(name)
    kmin, kmax,kvol, powerspec, ps_err = data.transpose()

    kmid_bins = 0.5 * (kmin+kmax)

    pl.errorbar(kmid_bins, powerspec, yerr=ps_err, fmt='o', label=r'$\rm CIC$')
#    pl.loglog(kmid_bins, powerspec, 'o', label=r'$\rm CIC$')

    a = 1.0 / (1.0+redshift)

    pspec = pspec_normalised_by_sigma8_expansion(efstathiou, sigma8=sigma8, a=a, omegaM=omegaM, omegaL=1-omegaM)

    pl.loglog(kmid_bins, pspec(kmid_bins), 'k', ls='--', label=r'$\rm Efstathiou$')

    planck_file = path.join(path.dirname(path.abspath(__file__)), '../example_scripts/planck_2013.txt')
    
    planck = readspec(planck_file)
    planck = pspec_normalised_by_sigma8_expansion(planck, sigma8=sigma8, a=a, omegaM=omegaM, omegaL=1-omegaM)    
    pl.loglog(kmid_bins, planck(kmid_bins), 'r', ls='-', label=r'$\rm Planck$')

    #pl.legend(frameon=False)
    pl.legend().draw_frame(False)
    pl.xlabel(r'$k \; \rm h\, Mpc^{-1}$')
    pl.ylabel(r'$P(k)$')

    #pl.imshow(modes[0].real)

#    pl.savefig('snap_000_pspec.pdf')
    pl.show()

    return
    
    
if __name__=='__main__':
    test_disp_field()
#    test_disp_fft()
                          
    #test_smooth()
#    test_readspec('test/LG_3spheres_spec512.dat', omegaM=0.279, redshift=50, sigma8=0.8)
#    test_readspec('test/LG_3spheres_spec256.dat', omegaM=0.279, redshift=50, sigma8=0.8)

#    test_readspec('test/ICspec.dat', omegaM=0.279, redshift=50, sigma8=0.8)
#    test_readspec('test/spec2.dat', omegaM=0.279, redshift=33, sigma8=0.8)
