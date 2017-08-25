#!/usr/bin/python
""" 
Lagrangian Initialisation of Zeldovich Amplitudes for Resimulatins of Displacements (LIZARD)
Peter Creasey - July 2013
"""
from __future__ import print_function,absolute_import
import sys
from sys import argv
from optparse import OptionParser
from .grid import build_displacement, displacement_sampler_factory, displacement_boost
import pickle
from .gadget import gadget
from numpy import unique, float32, array, float64, save, load
from .power import pspec_normalised_by_sigma8_expansion
from .uniform import load_glass, mass_field
from os import path

hline = '-'*60
displ_grid_name = 'displ_grid_%d.npy' # Name for the displacement grids
displ_boost_name = 'dipl_boost_%d_%d.npy' # name for boost
def power(args):

    print('Loading parameters from powerparams.py')
    from powerparams import seed, powerfunc, boxsize, gridsize, disp_file, sigma8, omegab, omegaM, omegaL, H0, a
    pspec_to_displacements(seed, powerfunc, boxsize, gridsize, disp_file, sigma8, omegab, omegaM, omegaL, H0,a)


def pspec_to_displacements(seed, powerfunc, output_drt, boxsize, base_grid_size, sigma8, omegab, omegaM, omegaL, H0,a,log=sys.stdout):
    """ Make a displacement field from  the power spectrum """
    if not path.isdir(output_drt):
        raise Exception('Directory '+str(output_drt)+' does not exist, please create')

    print(hline, file=log)
    print('\nlizard power\nGeneration of a Gaussian random field for the displacement of particles.', file=log)
    print(hline, file=log)
    print('Expansion factor   %f'%a, file=log)
    print('Omega Matter       %f'%omegaM, file=log)
    print('Omega Lambda       %f'%omegaL, file=log)
    print('Omega baryons      %f'%omegab, file=log)
    print('sigma_8            %f'%sigma8, file=log)
    print('Hubble0 (km/s/Mpc) %f'%H0, file=log)

    print('Seed %d\nBase grid size %d^3\nBoxsize %4.1f Mpc/h'%(seed, base_grid_size, boxsize), file=log)


    # grid sizes, subsample until we get to 64 or an odd size
    grid_sizes = [base_grid_size]
    while grid_sizes[-1]%2==0 and grid_sizes[-1]/2 >= 64: 
        grid_sizes.append(grid_sizes[-1]/2) 
        
    # Build the names
    disp_files = [path.join(output_drt, displ_grid_name%grid_size) for grid_size in grid_sizes]


    if any([path.exists(disp_file) for disp_file in disp_files]):
        raise Exception('One or more of '+str(disp_files)+' already exists, please remove and start again')        

    print('Output file(s): %s\n'%disp_files, file=log)
    print(hline, file=log)

    print('Normalising power spectrum to sigma8', file=log)
    power_at_sigma8 = pspec_normalised_by_sigma8_expansion(pspec=powerfunc, sigma8=sigma8, a=a, omegaM=omegaM, omegaL=omegaL)
    print('Building fourier modes', file=log)
    disp_grid = build_displacement(boxsize=boxsize, ngrid=base_grid_size, power_func=power_at_sigma8, seed=seed, log=log).astype(float32)
    disp_file = disp_files.pop(0)
    grid_size = grid_sizes.pop(0)
    print('- Writing %d^3 grid to file'%grid_size, disp_file, file=log)
    save(disp_file, disp_grid)
    for disp_file, grid_size in zip(disp_files, grid_sizes):
        print('- Subsampling to {:,}^3'.format(grid_size), file=log)
        disp_grid = disp_grid[:,::2,::2,::2] 
        print('- Writing to', disp_file, file=log)
        save(disp_file, disp_grid)
        print('Done', file=log)

def pspec_to_displacement_boost(seed, powerfunc, output_drt, boxsize, 
                                base_grid_size, boost_grid_size, 
                                boost_grid_repeats, sigma8, omegab, omegaM, 
                                omegaL, H0, a,log=sys.stdout):                             
    """ 
    Like pspec_to_displacement, but makes a boost grid, which contains
    only the high-frequencies not contained in a uniform grid of size 
    base_grid_size.

    """
    if not path.isdir(output_drt):
        raise Exception('Directory '+str(output_drt)+' does not exist, please create')

    boost_file = path.join(output_drt, displ_boost_name%(boost_grid_repeats, boost_grid_size))

    if path.exists(boost_file):
        raise Exception('File '+str(boost_file)+' already exists, please remove and start again')        
    
    print(hline, file=log)
    print('\nBoost mode generation for higher res ICs in', boost_file, file=log)
    print(hline, file=log)
    print('Expansion factor   %f\n'%a, 
          'Omega Matter       %f\n'%omegaM, 
          'Omega Lambda       %f\n'%omegaL, 
          'Omega baryons      %f\n'%omegab, 
          'sigma_8            %f\n'%sigma8,
          'Hubble0 (km/s/Mpc) %f\n'%H0, 
          'Seed %d\nBase grid size %d^3\nBoxsize %4.1f Mpc/h\n'%(seed, base_grid_size, boxsize), 
          hline, file=log)

    print('Normalising power spectrum to sigma8', file=log)
    power_at_sigma8 = pspec_normalised_by_sigma8_expansion(pspec=powerfunc, sigma8=sigma8, a=a, omegaM=omegaM, omegaL=omegaL)

    print('Building {:,}^3 boost grid'.format(boost_grid_size), file=log)
    boost_grid = displacement_boost(orig_boxsize=boxsize, orig_ngrid=base_grid_size, boost_ngrid=boost_grid_size, 
                                    nrepeat=boost_grid_repeats, power_func=power_at_sigma8, seed=seed, log=log)
    print('- Writing to', boost_file, file=log)
    save(boost_file, boost_grid)
    print('Done', file=log)
    return boost_file

    
def displace(a, omegaM, omegaL, omegab, H0, boxsize, disp_file, uni_pts, uni_sizes, out_file='out/particles.dat', 
             dm_only=False, log=sys.stdout, boost_file=None, boost_grid_repeats=None):
    """ Displace a uniform distribution of particles """


    print(hline, file=log)
    print('lizard displace\nDisplaces a uniform distribution of particles', file=log)
    print(hline, file=log)
    print('Displacement field           : %s'%disp_file, file=log)
    print(hline, file=log)
    print('Reading displacement field', file=log)
    

    disp_grid = load(disp_file)
    gridsize = disp_grid.shape[1]

    print('Gridsize %d^3\nBoxsize %4.1f Mpc/h'%(gridsize, boxsize), file=log)
    print(hline, file=log)
    if boost_file is None:
        disp_sampler = displacement_sampler_factory(disp_grid, boxsize, log=log)
    else:
        print('Reading boost file', boost_file, file=log)
        boost_grid = load(boost_file)
        boost_grid_size = boost_grid.shape[0]
        print('Boost grid size %d^3, tiled %d times'%(boost_grid_size, boost_grid_repeats), file=log)
        disp_sampler = displacement_sampler_factory(disp_grid, boxsize, boost_grid=boost_grid, 
                                                    boost_repeats=boost_grid_repeats, log=log)

    if dm_only:
        print('Making DM only ICs', file=log)
        dm_pos, dm_mass, dm_vel, dm_nums = mass_field(boxsize, uni_pts, uni_sizes, disp_sampler, a, omegaM, omegaL, H0, log=log)
        print('DM particle mass ', dm_mass, 'Msun/h', file=log)
        print('Output file : %s'%out_file, file=log)
        # Note that since I gave up on .npz files (IOError with large arrays) we need to keep the order ok here
        f = open(out_file, 'wb')
        print('Numbers of different DM particle types', file=log)
        save(f, dm_nums)
        print('Saving masses', file=log)
        save(f, dm_mass)
        print('Saving positions', file=log)
        save(f, dm_pos)
        print('Saving velocities', file=log)
        save(f, dm_vel)

        f.close()

    else:
        print('Making ICs with gas+DM', file=log)
        gas_pos, gas_mass, gas_vel, dm_pos, dm_mass, dm_vel, dm_nums = mass_field(boxsize, uni_pts, uni_sizes, disp_sampler, a, omegaM, omegaL, H0, omegab=omegab, log=log)

        print('Gas particle mass %.4e'%gas_mass, 'Msun/h', file=log)
        print('DM particle mass ', dm_mass, 'Msun/h', file=log)

        print('Output file : %s'%out_file, file=log)
        # Note that since I gave up on .npz files (IOError with large arrays) we need to keep the order ok here
        f = open(out_file, 'wb')
        print('Numbers of different DM particle types', file=log)
        save(f, dm_nums)
        print('Saving masses', file=log)
        save(f, gas_mass)
        save(f, dm_mass)
        print('Saving positions', file=log)
        save(f, gas_pos)
        save(f, dm_pos)
        print('Saving velocities', file=log)
        save(f, gas_vel)
        save(f, dm_vel)

        f.close()

    print('Done', file=log)

def make_gadget(out_file):
    from powerparams import a,H0, boxsize, omegaM, omegaL
    gadget(a,H0,boxsize,omegaM,omegaL,'out/particles.dat',out_name='out/IC.dat')

    
def graph(name):
    """ graph one of the output files """
    from graph import draw_disp_grid
    from graph import draw_pts

    f = open(name, 'rb')
    obj  = pickle.load(f)
    f.close()


    name = obj[0]
    if name=='displacements':
        print('Drawing displacement grid')
        name, a, omegaM, omegaL, omegab, sigma8, H0, boxsize, seed, disp_grid = obj
        draw_disp_grid(boxsize, disp_grid)
    elif name=='uni_points':
        print('Drawing point distribution')
        name, boxsize, pts, sizes = obj
        draw_pts(boxsize, pts, sizes)

    else:
        print('%s not understood'%name)

    
if __name__ == '__main__':
    if len(argv)==1:
        print('Usage: lizard.py [help] <command>')
        print('\nwhere command is one of power, displace, gadget, graph')
        print('Type lizard.py help <command> for help on each')
        exit(0)

    if argv[1]=='help':
        print('Sorry, no help yet')
    
    if argv[1]=='power':
        power(argv)
    elif argv[1]=='displace':
        if len(argv)!=5:
            print('Usage: lizard displace <displacement_field.dat> <glass32.dat> <num_repeats>')
            exit(0)

        disp_file = argv[2]
        glass_file = argv[3]
        num_repeats = int(argv[4])
        displace(disp_file, glass_file, num_repeats)
    elif argv[1]=='graph':
        if len(argv)!=3:
            print('Usage: lizard graph <x.dat>')
            exit(1)

        name = argv[2]
        graph(name)
    elif argv[1]=='glass':
        if len(argv)!=4:
            print('Usage: lizard glass <nside> <glass.dat>')
            print('e.g. lizard glass 32 glass32.dat')
            exit(-1)
        nside = argv[2]
        name = argv[3]
        glass(nside, name)
    elif argv[1]=='gadget':
        gadget(argv[2])
    else:
        print(argv)
        exit(1)
    exit(0)

    # give some help for command line usage 
    parser = OptionParser("lizard.py power/displace/gadget [options]")
    parser.add_option("-o", action="store", type="string", dest="movie_directory", default=None)
    parser.add_option("-c", "--colour_table", action="store", 
                      type="string", dest="colour_table", default="default", 
                      help="[default: default, but try also %s]"%','.join(colour_scales.keys()))
    options, args = parser.parse_args(argv)
    print('Options', options, 'args', args)

    data_file = args[1]
    # check some of these paths exist...
    assert(path.exists(data_file))
    if options.movie_directory is not None:
        assert(path.exists(options.movie_directory))
    colour_func = colour_scales[options.colour_table]
    
    test(data_file, colour_func, options.movie_directory)


