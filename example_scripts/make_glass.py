"""
Script to use the P^3M solver to make a glass file
"""
from __future__ import print_function
from lizard.p3m import CubicPeriodicForceSplit
import numpy as np
from lizard.ngb_kernel import *
from numpy.random import RandomState
from scipy.spatial import cKDTree
from lizard.grid import *
from lizard.log import VerboseTimingLog
import cPickle as pickle # to save the file


glass_size = 64

lattice_end_frac = 0.85 # Stop when the minimum distance has reached this fraction of the lattice spacing
max_iterations = 1000
seed = 123


out = 'glass%d.dat'%glass_size
log = VerboseTimingLog(filename='glass.log', also_stdout=False, insert_timings=True)

npts = glass_size**3
rs = RandomState(seed=seed)
pos = (np.reshape(rs.rand(3*npts), (3,npts)) + np.reshape(np.mgrid[:glass_size,:glass_size,:glass_size], (3,npts))).T * (1.0/glass_size)

wts = np.ones(npts)
rcrit = 3.0/ glass_size # good guess for splitting scale
r_soft = 0.3/glass_size # 0.3 x interparticle spacing
steps_per_min_dist = 5 # Don't calculate the minimum distance between points every step

min_dist = 1.0
fs = CubicPeriodicForceSplit(rcrit, 500, r_soft=r_soft, log=log)

for i in range(max_iterations):
    pairs, accel_short = fs.short_force(wts, pos)
    accel_long = fs.long_force(wts, pos) 
    
    accel = accel_short + accel_long
    
    max_accel = sqrt(square(accel).sum(1).max())
    rms_accel = sqrt(square(accel).sum(1).mean())
    rms_short = sqrt(square(accel_short).sum(1).mean())
    rms_long = sqrt(square(accel_long).sum(1).mean())
    
    dt0 = 1.5e-3 / sqrt(glass_size)
    dt = dt0*(max_accel**(-0.6))
    
    print('Timestep', i, 'Maximum acceleration', max_accel, 'timestep', dt, 'RMS',rms_accel, 'RMS short', rms_short, 'RMS long', rms_long, file=log)
    print('Timestep', i, 'Maximum acceleration', max_accel, 'timestep', dt, 'RMS',rms_accel, 'RMS short', rms_short, 'RMS long', rms_long)
    
    
    vel = -accel # run gravity 'backwards' to make glass, hubble drag
    pos = pos + vel*dt
    pos = pos - floor(pos) # restrict to [0,1)
    
    if i%steps_per_min_dist==0:
        # find the minimum distance between points (ignore repeats)
        tree = cKDTree(pos)
        dist, idx = tree.query(pos,k=2)
        min_dist = dist[:,1].min() * glass_size # minimum distance as a fraction of lattice spacing
        print('Nearest pair, fraction=', min_dist, 'of lattice spacing', file=log)
        print('Nearest pair, fraction=', min_dist, 'of lattice spacing')
        if min_dist>lattice_end_frac:
            break

if min_dist<lattice_end_frac:
    raise Exception('Failed to converge after max=%d iterations'%max_iterations);

# Save the file
f = open(out, 'wb')
pickle.dump(pos, f, pickle.HIGHEST_PROTOCOL)
f.close()

#### Plot the top slice 1/glass_size
import pylab as pl
idx = np.flatnonzero(pos[:,2]<1.0/glass_size)
pl.plot(pos[idx,0], pos[idx,1], 'b.')
pl.show()



