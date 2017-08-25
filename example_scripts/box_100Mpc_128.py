"""
Example script to make a gadget file for cosmological simulation with matter and 
velocity power spectrum given by the Zel'dovich approximation.

To make the example 256^3 grid you will need about 1.3 GB ram on your computer.
"""

from lizard import *
from lizard.log import VerboseTimingLog
from numpy import load, save
from lizard.uniform import make_glass_uniform, uniform_box
from os import path, mkdir
from lizard.power import efstathiou
from lizard.test_ics import test_readspec
from lizard.gadget import make_spec_from_ICs

log = VerboseTimingLog(filename='box100.log', also_stdout=True, insert_timings=True)

drt = 'box100/'
dmonly = True # Only DM


seed = 123
sigma8 = 0.8
powerfunc = efstathiou
base_grid_size = 256 # highest resolution displacement grid 

a = 1.0/51.0 # redshift 50
omegaM = 0.279
omegaL = 0.721
omegab = 0.05
H0     = 70.0 # km/s/Mpc
boxsize = 100.0 # Mpc/h

grid_drt = path.join(drt, 'grids') # directory for grids
gadget_drt = path.join(drt, 'gadget') # directory for gadget files


###################################################################
#### The following steps write to disk, so comment them out
#### once you have done each one if you want to go step by step.
###################################################################
mkdir(drt)
mkdir(grid_drt)
mkdir(gadget_drt)

# Convert the power spectrum to a grid of displacements
pspec_to_displacements(seed, powerfunc, grid_drt, boxsize, base_grid_size, sigma8, omegab, omegaM, omegaL, H0,a,log)


for gridsize in [256, 128]: # resolution wanted for particles

    disp_file = path.join(grid_drt, 'displ_grid_%d.npy'%gridsize)
    particle_file = path.join(drt, 'particles_'+{True:'dmonly_',False:''}[dmonly]+'%d.dat'%gridsize) 
    gadget_file = path.join(gadget_drt, ('IC_%d_'%gridsize)+{True:'dmonly',False:''}[dmonly]+'.dat') 
    spec_file = path.join(drt, 'spec%d.dat'%gridsize)

    # Discretise the volume into cells and displace these into a particle distribution
    # (see also make_glass.py for making a glass file
    # pts, sizes = make_glass_uniform('glass64.dat', gridsize, boxsize) # Some glass files have too much long-range power, be careful
    pts, sizes = uniform_box(boxsize, gridsize)
    displace(a, omegaM, omegaL, omegab, H0, boxsize, disp_file, pts.T, sizes, particle_file, dm_only=dmonly, log=log)

    # Format the particle file as a gadget file
    gadget(a,H0,boxsize,omegaM,omegaL,particle_file, gadget_file, use_double=False, dm_only=dmonly, log=log)

    make_spec_from_ICs(gadget_file, spec_file, ngrid=gridsize, log=log)

# Draws the graph of the spectrum to compare.
test_readspec(spec_file, omegaM, redshift=(1.0/a-1), sigma8=sigma8)
