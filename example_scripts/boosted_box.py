"""
Example script showing how to add the boosted modes.

This is still a bit rough (and uses multiple pspec drawing that
is not included in the committed repo). 
"""

from lizard import *
import lizard as liz
from lizard.log import VerboseTimingLog
from numpy import load, save, loadtxt
from lizard.uniform import make_glass_uniform, uniform_box
from os import path, mkdir
from lizard.power import tabulated_pspec
from lizard.test_ics import test_readspec
from lizard.gadget import make_spec_from_ICs

log = VerboseTimingLog(filename='box_boost.log', also_stdout=True, insert_timings=True)

drt = 'boosted_box/'
dmonly = True


seed = 123
sigma8 = 0.8
planck_name = path.join(liz.__path__[0], '../example_scripts/planck_2013.txt')
k, Pk = loadtxt(planck_name).T
powerfunc = tabulated_pspec(k, Pk)

base_grid_size = 64*2 # highest resolution displacement grid 
boost_grid_size, boost_grid_repeats = 128//2, 4
a = 1.0/128.0 # redshift 127
omegaM = 0.279
omegaL = 0.721
omegab = 0.05
H0     = 70.0 # km/s/Mpc
boxsize = 500.0 # Mpc/h

grid_drt = path.join(drt, 'grids') # directory for grids
gadget_drt = path.join(drt, 'gadget') # directory for gadget files


###################################################################
#### The following steps write to disk, so comment them out
#### once you have done each one if you want to go step by step.
###################################################################
gridsize = base_grid_size*2

disp_file = path.join(grid_drt, 'displ_grid_%d.npy'%base_grid_size)
particle_file = path.join(drt, 'particles_'+{True:'dmonly_',False:''}[dmonly]+'%d.dat'%gridsize) 
particle_boost_file = path.join(drt, 'particles_boost_'+{True:'dmonly_',False:''}[dmonly]+'%d.dat'%gridsize) 
gadget_file = path.join(gadget_drt, ('IC_%d'%gridsize)+{True:'_dmonly',False:''}[dmonly]+'.dat') 
gadget_boost_file = path.join(gadget_drt, ('IC_boost_%d'%gridsize)+{True:'_dmonly',False:''}[dmonly]+'.dat') 
spec_file = path.join(drt, 'spec%d.dat'%gridsize)
spec_boost_file = path.join(drt, 'spec_boost_%d.dat'%gridsize)
zoom_spec_file = path.join(drt, 'zoom_spec%d.dat'%gridsize)
if True:
    mkdir(drt)
    mkdir(grid_drt)
    mkdir(gadget_drt)

    # boost (high freq) file
    boost_file = pspec_to_displacement_boost(seed+1, powerfunc, grid_drt, boxsize, base_grid_size, boost_grid_size, boost_grid_repeats, sigma8, omegab, omegaM, omegaL, H0, a,log)

    # Convert the power spectrum to a grid of displacements
    pspec_to_displacements(seed, powerfunc, grid_drt, boxsize, base_grid_size, sigma8, 
                           omegab, omegaM, omegaL, H0,a,log)

   

    # Discretise the volume into cells and displace these into a particle distribution
    pts, sizes = uniform_box(boxsize, gridsize)
    displace(a, omegaM, omegaL, omegab, H0, boxsize, disp_file, pts.T, sizes, particle_file, dm_only=dmonly, log=log)

    displace(a, omegaM, omegaL, omegab, H0, boxsize, disp_file, pts.T, sizes, particle_boost_file, dm_only=dmonly, log=log,
             boost_file=boost_file, boost_grid_repeats=boost_grid_repeats)


    # Format the particle file as a gadget file
    gadget(a,H0,boxsize,omegaM,omegaL,particle_file, gadget_file, use_double=True, dm_only=dmonly, log=log)
    gadget(a,H0,boxsize,omegaM,omegaL,particle_boost_file, gadget_boost_file, use_double=True, dm_only=dmonly, log=log)

import numpy as np
zoom_topleft = np.array([0.1*boxsize]*3)
zoom_size = boxsize*0.4
zoom_ngrid = base_grid_size#//2
#make_spec_from_ICs(gadget_file, spec_file, ngrid=base_grid_size//2, zoom_name=zoom_spec_file, zoom_topleft=zoom_topleft, zoom_ngrid=zoom_ngrid, zoom_size=zoom_size, log=log)
make_spec_from_ICs(gadget_file, spec_file, ngrid=base_grid_size*2, log=log)
make_spec_from_ICs(gadget_boost_file, spec_boost_file, ngrid=base_grid_size*2, log=log)

# Draws the graph of the spectrum to compare.
import pylab as pl
test_readspec((spec_file, spec_boost_file), omegaM, redshift=(1.0/a-1), sigma8=sigma8)#, zoom_spec_file=zoom_spec_file)
pl.show()
