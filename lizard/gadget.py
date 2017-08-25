"""
Module for making output to gadget, spectra from a gadget file
"""
from __future__ import print_function, absolute_import
from numpy import vstack, float64, arange, uint32, array, zeros, load, cumsum, hstack, float32, empty
import pickle
from iccpy.gadget import binary_snapshot_io, load_ICsnapshot
from .log import null_log
from . import grid

MAX_REC_BYTES = 2**31 - 1 # dont write ICs with too many bytes in the record

def gadget(a,H0,boxsize,omegaM,omegaL,particle_file,out_name='out/IC.dat',use_double=True, dm_only=False, log=null_log):
    """ Create a gadget initial conditions file """
    
    out_dtype = {True:float64,False:float32}[use_double]

    print('Reading', particle_file, file=log)
    f = open(particle_file, 'rb')
    dm_nums = load(f)
    print('%d different masses of DM particles found'%len(dm_nums), file=log)
    if len(dm_nums)>4:
        print('Cannot make gadget files with more than 4 dm masses, we need',file=log)
        print('2 for gas+stars, and only 6 available slots!', file=log)
        raise Exception('Too many dm masses')

                              
    print('Reading masses', file=log)
    if not dm_only:
        gas_mass = float(load(f))
    dm_mass = load(f)

    print('Reading positions', file=log)
    if dm_only:
        dm_pos = load(f)
        print('Scaling positions to kpc/h', file=log)
        pos = (dm_pos * 1e3).astype(out_dtype) # Convert to kpc/h
        print('pos', pos.shape, 'in', pos.min(), pos.max(), file=log)
        num_particles = [0] + list(dm_nums)

        mass_header = [0] + [mass*1e-10 for mass in dm_mass]
    else:
        gas_pos = load(f)
        dm_pos = load(f)

        print('Scaling positions to kpc/h', file=log)
        pos = (vstack((gas_pos, dm_pos)) * 1e3).astype(out_dtype) # Convert to kpc/h
        print('pos', pos.shape, 'in', pos.min(), pos.max(), file=log)
        num_particles = [gas_pos.shape[0]] + list(dm_nums)
        mass_header = [gas_mass*1e-10] + [mass*1e-10 for mass in dm_mass]
    
    ids = arange(1,pos.shape[0]+1).astype(uint32) # TODO: One day we will have more than 4 billion particles...
    print('Reading velocities', file=log)
    if dm_only:
        dm_vel = load(f)
        vel = dm_vel.astype(out_dtype)
    else:
        gas_vel = load(f)
        dm_vel = load(f)
        vel = vstack((gas_vel, dm_vel)).astype(out_dtype)

    f.close()

    if len(mass_header)==5:
        # Account for stars  (type 4)
        mass_header = array(mass_header[:4] + [0] + mass_header[4:5])
        num_particles = array(num_particles[:4] + [0] + num_particles[4:5])
    else:
        # need 6 particle types for gadget
        pad_to_6 = [0]*(6-len(mass_header))
        mass_header = array(mass_header+pad_to_6)
        num_particles = array(num_particles + pad_to_6)

    print('Particle masses', mass_header, file=log)
    print('Number of particles', num_particles, file=log)

    gas_temp = zeros(num_particles[0], dtype=out_dtype)

    flag_double = {True:1,False:0}[use_double]

    # Largest arrays are the coordinates (and velocities). Check that they meet array bounds (64 bit floats)
    if pos.size * 4 *(1+flag_double) > MAX_REC_BYTES:
        num_files = int((pos.size * 4 * (1+flag_double)) / MAX_REC_BYTES) + 1
        print('Too many particles for a single gadget file, splitting into', num_files, 'files', file=log)

        for i,(num_particles_thisfile, segments) in enumerate(split_particles(num_particles, num_files)):
            # Particle data just for this file
            pos_i = vstack([pos[i0:i1] for i0,i1 in segments])
            vel_i = vstack([vel[i0:i1] for i0,i1 in segments])
            ids_i = hstack([ids[i0:i1] for i0,i1 in segments])
            if num_particles_thisfile[0]>0:
                # Have gas
                gas_temp_thisfile = zeros(num_particles_thisfile[0], dtype=out_dtype)
                extra_data = [gas_temp_thisfile]
            else:
                extra_data = []
            print('Number of particles in this file', num_particles_thisfile, file=log)
            
            mass_i = array(mass_header).copy()
            for ptype,np in enumerate(num_particles_thisfile):
                if np==0:
                    mass_i[ptype]=0

            header = dict((('num_particles', num_particles_thisfile),
                           ('mass', mass_i), ('time',float(a)), ('redshift',float(1.0/a - 1)), 
                           ('flag_sfr',0) , ('flag_feedback',0), 
                           ('num_particles_total', num_particles), 
                           ('flag_cooling',0), ('num_files',num_files), ('boxsize',float(boxsize*1000)), 
                           ('omega0', float(omegaM)), ('omegaLambda', float(omegaL)), ('hubble0', float(H0/100.0)), ('flag_stellarage',0), 
                           ('buffer', [0]*56), ('flag_metals', 0), ('npartTotalHighWord', [0,0,0,0,0,0]), 
                           ('flag_entropy_instead_u', 0), ('flag_doubleprecision', flag_double)))
        
            out_i = out_name + '.%d'%i
            print('Writing gadget snapshot', out_i, file=log)
            binary_snapshot_io.write_snapshot_file(out_i, header, pos_i, vel_i, ids_i, None, extra_data)

    else:
        
        header = dict((('num_particles', num_particles),
                       ('mass', mass_header), ('time',float(a)), ('redshift',float(1.0/a - 1)), 
                       ('flag_sfr',0) , ('flag_feedback',0), 
                       ('num_particles_total', num_particles), 
                       ('flag_cooling',0), ('num_files',1), ('boxsize',float(boxsize*1000)), 
                       ('omega0', float(omegaM)), ('omegaLambda', float(omegaL)), ('hubble0', float(H0/100.0)), ('flag_stellarage',0), 
                       ('buffer', [0]*56), ('flag_metals', 0), ('npartTotalHighWord', [0,0,0,0,0,0]), 
                       ('flag_entropy_instead_u', 0), ('flag_doubleprecision', flag_double)))
        
        print('Writing gadget snapshot', out_name, file=log)
        binary_snapshot_io.write_snapshot_file(out_name, header, pos, vel, ids, None, [gas_temp])
        

def split_particles(num_particles, num_files):
    """
    num_particles - number of particles in each type
    num_files -

    returns iterable of (numparts_thisfile, segments) for each file
    """
    npi = []
    parts_per_file = [p/num_files + 1 for p in num_particles]
    

    for i in range(num_files):
        segments = []
        np_thisfile = []
        for ptype, np_ptype in enumerate(num_particles):
            if np_ptype==0:
                np_thisfile.append(0)
                continue
            i0 = i * parts_per_file[ptype]
            i1 = min((i+1) * parts_per_file[ptype], np_ptype)
            np_thisfile.append(i1-i0)
            istart = i0 + sum(num_particles[:ptype])
            iend = i1 +  sum(num_particles[:ptype])
            segments.append((istart,iend))

        
        yield tuple(np_thisfile), segments
        continue

                
                
def _IC_matterspec(InitCondFile, grid_n, log):
    """

    Find the Fourier modes of the matter distribution of a Gadget format 1
    IC snapshot (e.g. produced by the gadget function above).

    InitCondFile - Name of the format 1 IC file
    grid_n       - n modes (e.g. 32 for a 32^3 FFT)
    
    returns
    modes -  cubic array of modes (complex)
    box_size - the width of the box

    """
    snap = load_ICsnapshot(InitCondFile)

    boxsize = float(snap.header.boxsize)
    redshift = float(snap.header.redshift)

    a = float(snap.header.time)
    print('Expansion factor', a, file=log)
    print('Box size (kpc/h)', boxsize, file=log)
    print('Redshift', redshift, file=log)

    num_parts = snap.header.num_particles_total
    total_parts = sum(num_parts)

    print('Number of particles', total_parts, num_parts, file=log)

    prec = float64
    
    pos = empty((3,total_parts), dtype=prec)
    mass = empty((total_parts,), dtype=prec)
    idx = cumsum(num_parts)
    idx0 = idx - num_parts

    for ptype in range(6):
        print('Reading', num_parts[ptype], 'particles of type', ptype, file=log)
        


        smass = snap.mass[ptype]

        mass[idx0[ptype]:idx[ptype]] = smass
        spos = snap.pos[ptype]
        for i in range(3):
            pos[i][idx0[ptype]:idx[ptype]] = spos[:,i]

    print('Pos in', pos.min(axis=0), pos.max(axis=0), file=log)
    print('Mass in', mass.min(), mass.max(), file=log)

    print('Building CIC modes', file=log)
    modes = grid.cic_modes(pos, boxsize, grid_n, mass)

    # convert box size to Mpc/h
    boxsize = boxsize * 1e-3
    return modes, boxsize, a

                
def make_spec_from_ICs(InitCondFile, spec_name, ngrid=64, log=null_log):
    """ Find the matter power spectrum for the given IC file """
    print('Reading IC file', file=log)
    modes, boxsize, expansion = _IC_matterspec(InitCondFile, ngrid, log)
    print('Making power spectrum bins for box size', boxsize, file=log)

    kmin, kmax, kvol, kmid_bins, powerspec, p_err = grid.modes_to_pspec(modes, boxsize)

    print('Writing power spectrum to', spec_name, file=log)
    f = open(spec_name, 'w')
    f.write('#ICs = %s\n'%(InitCondFile,))
    f.write('#Expansion factor = %.5f\n'%expansion)
    f.write('#\n#k values in h Mpc^-1, P(k) in Mpc^3 h^-1 (although note factor (2pi)^-3 compared to other power spectrum definitions)\n')
    f.write('#k_bin_min k_bin_max k_bin_vol Power power_error\n')
    for l,h,vol,p,er in zip(kmin, kmax, kvol, powerspec, p_err):
        f.write('%8.5f %8.5f %6.5e %6.5e %6.5e\n'%(l,h,vol,p,er))
    f.close()

    print('Finished writing spectrum to', spec_name, file=log)
            
        

