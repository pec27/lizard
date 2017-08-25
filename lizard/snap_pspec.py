""" Make the power spectrum of a snapshot """

from iccpy.gadget import load_snapshot, load_ICsnapshot
from iccpy.gadget.labels import cecilia_labels
from numpy import float32, float64, empty, cumsum, sqrt, square, bincount
from grid import cic_modes, make_k_values, powerspec_bins

def snap_to_modes(snap, grid_n):
    """

    Calculate the fourier modes of the (normalised) density field from the given snapshot
    snap - gadget snapshot
    grid_n - n modes (e.g. 32 for a 32^3 FFT)
    
    returns
    modes -  cubic array of modes (complex)
    box_size - the width of the box

    """
    print 'snap', snap
    print snap.header
    boxsize = float(snap.header.boxsize)
    redshift = float(snap.header.redshift)

    a = float(snap.header.time)
    print 'Expansion factor', a
    print 'Box size (kpc/h)', boxsize
    print 'Redshift', redshift

    num_parts = snap.header.num_particles_total
    total_parts = sum(num_parts)

    print 'Number of particles', total_parts, num_parts

    prec = float64
    
    pos = empty((3,total_parts), dtype=prec)
    mass = empty((total_parts,), dtype=prec)
    idx = cumsum(num_parts)
    idx0 = idx - num_parts

    for ptype in range(6):
        print 'Reading', num_parts[ptype], 'particles of type', ptype
        


        smass = snap['MASS'][ptype]
        mass[idx0[ptype]:idx[ptype]] = smass
        spos = snap['POS '][ptype]
        for i in range(3):
            pos[i][idx0[ptype]:idx[ptype]] = spos[:,i]

        if num_parts[ptype]>1:
            vel = snap['VEL '][ptype]
            rms_vel = sqrt(square(vel).sum(1).mean(dtype=float64))
            print 'Ptype', ptype, 'has RMS vel', rms_vel, 'km/s at redshift', redshift
    print 'Pos in', pos.min(axis=0), pos.max(axis=0)
    print 'Mass in', mass.min(), mass.max()

    print 'Building CIC modes'
    modes = cic_modes(pos, boxsize, grid_n, mass)

    # convert box size to Mpc/h
    boxsize = boxsize * 1e-3
    return modes, boxsize

def snap_to_modes_CLUES(snap, grid_n):
    """

    As snap_to_modes, but for the CLUES ICs, which are badly behaved in that
    the total number of particles in the snapshot is not the same number as
    stored in num_particles_total.

    """
    print 'snap', snap
    print snap.header
    boxsize = float(snap.header.boxsize)
    redshift = float(snap.header.redshift)
    pmasses = snap.header.mass
    a = float(snap.header.time)
    print 'Expansion factor', a
    print 'Box size (kpc/h)', boxsize
    print 'Redshift', redshift

#    num_parts = snap.header.num_particles_total
    num_parts = [ 6141952, 6141952, 365184, 51864, 0, 16753024] # <- Yes, its a hack, but blame the person who put the wrong values in the CLUES headers...
    
    total_parts = sum(num_parts)

    print 'Number of particles', total_parts, num_parts

    prec = float64
    
    pos = empty((3,total_parts), dtype=prec)
    mass = empty((total_parts,), dtype=prec)
    idx = cumsum(num_parts)
    idx0 = idx - num_parts

    for ptype in range(6):
        print 'Reading', num_parts[ptype], 'particles of type', ptype
        smass = snap.mass[ptype]
        mass[idx0[ptype]:idx[ptype]] = smass
        spos = snap.pos[ptype]
        print 'spos shape', spos.shape
        print spos
        for i in range(3):
            pos[i][idx0[ptype]:idx[ptype]] = spos[:,i]
        
        if num_parts[ptype]>1:
            vel = snap.vel[ptype]
            rms_vel = sqrt(square(vel).sum(1).mean(dtype=float64))
            print 'Ptype', ptype, 'has RMS vel', rms_vel, 'km/s at redshift', redshift
        

    print 'Pos in', pos.min(axis=0), pos.max(axis=0)
    print 'Mass in', mass.min(), mass.max()

    print 'Building CIC modes'
    modes = cic_modes(pos, boxsize, grid_n, mass)

    # convert box size to Mpc/h
    boxsize = boxsize * 1e-3
    return modes, boxsize

def test(ngrid = 32):
    from numpy import *




    drt1 ="/store/stellcomp/group/pcreasey/runs/2Mpc_LG_3spheres/run_03_pmgrid_lowsoft/outputs"
    drt1 ="/store/stellcomp/cecilia/sims/LG/new_7Mpc_2048/outputs"
    snapnum = 0
    snap = load_snapshot(directory=drt1, snapnum=snapnum, label_table=cecilia_labels)
    modes, boxsize = snap_to_modes(snap, ngrid)

    print 'Making power spectrum bins'
    kmin, kmax, kbins, kvol = powerspec_bins(ngrid, boxsize)

    wts = square(modes.ravel().real) + square(modes.ravel().imag)

    print 'Summing power spectrum'
    v1 = bincount(kbins, weights=wts) 
    powerspec = v1 / kvol

    # work out error on power spectrum
    v2 = bincount(kbins, weights=square(wts))
    v0 = bincount(kbins)
    p_err = sqrt((v2*v0 - v1*v1)/v0) / kvol 
    
    print '#k_bin_min k_bin_max k_bin_vol Power power_error'
    for l,h,vol,p,er in zip(kmin, kmax, kvol, powerspec, p_err):
        print l,h,vol,p,er


def make_spec_from_ICs(InitCondFile, out_name, ngrid=32):

    
    snap = load_ICsnapshot(InitCondFile)


    modes, boxsize = snap_to_modes_CLUES(snap, ngrid)

    print 'Making power spectrum bins'
    kmin, kmax, kbins, kvol = powerspec_bins(ngrid, boxsize)

    wts = square(modes.ravel().real) + square(modes.ravel().imag)

    print 'Summing power spectrum'
    v1 = bincount(kbins, weights=wts) 
    powerspec = v1 / kvol

    # work out error on power spectrum
    v2 = bincount(kbins, weights=square(wts))
    v0 = bincount(kbins)
    p_err = sqrt((v2*v0 - v1*v1)/v0) / kvol 

    print 'Writing power spectrum to', out_name
    f = open(out_name, 'w')
    f.write('#ICs = %s\n'%(InitCondFile,))
    f.write('#Values in Mpc/h\n')
    f.write('#k_bin_min k_bin_max k_bin_vol Power power_error\n')
    for l,h,vol,p,er in zip(kmin, kmax, kvol, powerspec, p_err):
        f.write('%8.5f %8.5f %6.5e %6.5e %6.5e\n'%(l,h,vol,p,er))
    f.close()

    print 'done'


if __name__=='__main__':
    test(32)

#    make_spec_from_ICs('/store/stellcomp/group/pcreasey/runs/2Mpc_LG_3spheres/ICs/LG_3spheres_2048/IC.dat', 'test/LG_3spheres_spec512.dat', ngrid=512)

#    make_spec_from_ICs('out/IC.dat', 'test/ICspec.dat', ngrid=32)
