"""
Utilities for making multi-mass particle distributions that approximate 
a uniform field (i.e. for displacement with Zel'dovich)
"""
from __future__ import print_function, division, unicode_literals, absolute_import
from .power import velocity_from_displacement
from numpy import *
import pickle
from .lizard_c import make_BH_tree, leaf_depths_centres, make_BH_ellipsoid_tree
from .log import null_log
from numpy.linalg import eigvalsh

def load_glass(name, boxsize, repeat=2):
    """
    Reads a glass file and repeats it.
    Returns 
    uni_pts   - points distributed over the box
    uni_sizes - sizes s.t. sum(sizes**3) = boxsize**3
    """
    print('Loading pos from file', name)
    f = open(name, 'rb')
    pts = pickle.load(f)
    f.close()

    
    scale = boxsize / float(repeat)
    res = empty((repeat,repeat, repeat) + pts.shape, dtype=float64)
    for i in range(repeat):
        for j in range(repeat):
            for k in range(repeat):
                res[i,j,k] = pts + (i,j,k)

    res.shape = (repeat*repeat*repeat*pts.shape[0], 3)
    res *= scale

    uni_sizes = empty(res.shape[0], dtype=float64)
    uni_sizes[:] = power(res.shape[0], -1.0/3.0) * boxsize
    return res, uni_sizes

def uniform_box(boxsize, ngrid):
    """ make an (ngrid^3,3) array of point positions """
    dx = boxsize / ngrid
    pts = mgrid[:ngrid,:ngrid,:ngrid] * dx
    pts = pts.reshape((3,ngrid*ngrid*ngrid)).astype(float32)
    
    sizes = empty(pts.shape[1], dtype=float64)
    sizes[:] = dx
    return pts, sizes


def highres_sphere(boxsize, centre, radius, ngrid_low, ngrid_high):
    
    pts_l, size_l = uniform_box(boxsize, ngrid_low)
    pts_h, size_h = uniform_box(radius*2.0, ngrid_high)
    pts_h -= radius

    keep_h = flatnonzero(square(pts_h).sum(1)<radius*radius)
    pts_h, size_h = pts_h[keep_h], size_h[keep_h]

    keep_l = flatnonzero(square(pts_l-centre).sum(1)>radius*radius)
    pts_l, size_l = pts_l[keep_l], size_l[keep_l]

    n_h, n_l = pts_h.shape[0], pts_l.shape[0]
    pts = empty((n_h+n_l, 3), dtype=float64)
    sizes = empty((n_h+n_l), dtype=float64)
    pts[:n_h] = pts_h + centre
    pts[n_h:] = pts_l
    sizes[:n_h] = size_h
    sizes[n_h:] = size_l
    return pts, sizes

def mass_field(boxsize, uni_pts, uni_sizes, disp_sampler, a, omegaM, omegaL, H0, omegab=None, log=null_log):
    """ 
    displace particle distribution and calculate masses

    [omegab=None, (redshift zero baryon fraction) otherwise then the highest 
    resolution dark-matter particles are split into DM+gas]

    """
    # G = 6.6725985000000e-08 # big G, cm^3 s^-2 g^-1
    # msun =  1.9889225000000e33 # solar mass, g  
    # pc = 3.0856775807000e18 # 1 parsec, cm
    # =>
    G = 4.3009e-9 # km^2 Mpc s^-2 Msun^-1
    h = H0 / 100.0 # h (to convert to physical)

    check_vol = (square(uni_sizes)*uni_sizes).sum()/boxsize**3
    if check_vol<0.99999 or check_vol>1.00001:
        raise Exception('Particles dont fill volume: Sum(sizes^3) != boxsize^3. Check parameters?')

    # find the velocity factor to convert displacements (Mpc) to velocities (km/s)
    vel_disp = velocity_from_displacement(a, omegaM, omegaL, H0) # km / s / Mpc
    vel_disp = vel_disp * sqrt(a) / h # convert to Gadget units km / s / (cMpc/h), with extra factor root(a)

    print('Velocity multiplication factor', vel_disp, 'km/s /(cMpc/h) with extra factor sqrt(a)', file=log)

    discrete_sizes = sorted(unique(uni_sizes))
    num_size_particles = len(discrete_sizes)
    print('Boxsize: %4.1f Mpc/h'%boxsize, file=log)
    print('{:,} particles, {:,} different sizes'.format(uni_pts.shape[0], num_size_particles), file=log)
    indices = [flatnonzero(uni_sizes==size) for size in discrete_sizes]
    
    if omegab is not None:
        # first do the gas particles 
        highres = indices[0]     # only the highest resolution
        num_gas = len(highres)
        print('Number of gas (high resolution) particles {:,}'.format(num_gas), file=log)
        # Calculate shifts (in z-direction) for high-res particles
        # such that they are 0.5*cell_size apart, but CoM is in the same place
        dm_displace  = (0.5 * discrete_sizes[0]) * omegab/omegaM
        gas_displace = dm_displace - 0.5 * discrete_sizes[0]

        print('Gas particles are being displaced by %6.3f kpc/h in z-direction'%float(1e3*gas_displace), file=log)
        pos = uni_pts[highres].copy()
        pos[:,2] = remainder(pos[:,2] + gas_displace, boxsize) # stay in [0,boxsize)

        gas_vel = empty((num_gas, 3), dtype=float32)
        disps = disp_sampler(pos.T)
        for axis in range(3):
            gas_vel[:,axis] = disps[:,axis] * vel_disp
            gas_pos = remainder(disps + pos, boxsize) # stay in [0,boxsize)
            
        print('Gas positions in', gas_pos.min(axis=0), 'to', gas_pos.max(axis=0), file=log)
        gas_size = discrete_sizes[0]

    # Now the DM particles (all resolutions)
    dm_pos = []
    dm_vel = []
    dm_nums = array([len(idx) for idx in indices])

    for idx, size in zip(indices, discrete_sizes):
        print('DM of size', size, file=log)
        pos = uni_pts[idx]
        if omegab is not None and size==discrete_sizes[0]:
            print('Highest res DM particles are being displaced by %6.3f kpc/h in z-direction'%float(1e3*dm_displace), file=log)
            # Need to stay in [0, boxsize)
            pos = remainder(pos + array((0,0,dm_displace),dtype=pos.dtype), boxsize) 

        vel = empty(pos.shape, dtype=float32)
        disps = disp_sampler(pos.T)
        vel[:] = disps * vel_disp

        disps += pos
        dm_pos.append(disps)
        dm_vel.append(vel)

    dm_sizes = discrete_sizes
        
    part_vol = array(discrete_sizes) * square(discrete_sizes) / (h*h*h) # in Mpc^3
    # Critical density of the universe
    rho_c = (3 * H0 * H0) / (8 * pi * G) # in Msun/Mpc^3
    
    if omegab is None:
        dm_density = omegaM * rho_c
    else:
        baryon_density = omegab * rho_c
        dm_density = (omegaM - omegab) * rho_c
        
        gas_mass = baryon_density * part_vol[0] * h # in Msun/h

    dm_mass = [dm_density * part_vol[0] * h] # highres
    for vol in part_vol[1:]:
        dm_mass.append(omegaM * rho_c * vol * h) # Density of gas+dm for low res particles, in Msun/h
    
    dm_pos = vstack(dm_pos)
    dm_mass = array(dm_mass)
    dm_vel = vstack(dm_vel)
    remainder(dm_pos, boxsize, dm_pos) # wrap to between [0,boxsize)
    print('DM positions in', dm_pos.min(axis=0), 'to', dm_pos.max(axis=0), file=log)

    if omegab is None:
        # Only DM
        return dm_pos, dm_mass, dm_vel, dm_nums    
    return gas_pos, gas_mass, gas_vel, dm_pos, dm_mass, dm_vel, dm_nums    

def save(name, boxsize, pts, sizes):
    
    f = open(name, 'wb')
    data = ('uni_points', boxsize, pts, sizes)
    pickle.dump(data, f, 2)
    f.close()

def make_glass_uniform(glass_name='glass64.dat', gridsize=1024, boxsize=50.0):
    """
    Tile the glass file to make a uniform distribution
    """
    repeats = gridsize/64
    print('number of repeats', repeats)
    if repeats*64 != gridsize:
        raise Exception('grid size was not a multiple of glass size!')
    uni_pts, uni_sizes = load_glass(glass_name, boxsize, repeats)
    print('uni points', uni_pts.shape)
    print('uni_sizes', uni_sizes.shape)

    return uni_pts, uni_sizes

def make_zoom_uniform(ctr=(25.0,25.0,25.0), rmin=5.0, gridsize=1024, boxsize=50.0, log=null_log):
    """
    Make a discretisation of a periodic box with a high resolution region 
    at a given centre at the resolution given by gridsize, and 3 other levels of
    refinement outside such that the Barnes-Hut criteria is met (opening angle
    < 0.1 radians). Returns the positions and sizes

    ctr      - 3 cpt centre of the zoom region (in [0,boxsize])
    rmin     - radius around centre to resolve (will wrap)
    gridsize - grid resolution for highest res region (e.g. 1024 for 1024^3) 
    boxsize  - size of the box

    Returns
   
    uni_pts   - points distributed over the box (n,3), in [0,boxsize]
    uni_sizes - sizes such that sum(sizes^3)== boxsize^3
    """
    nref = 0
    while gridsize>2**nref:
        nref = nref + 1

    if gridsize!=2**nref:
        raise Exception(str(gridsize)+' was not a power of 2.')


    levels = [nref-4,nref-2,nref-1,nref]
    if nref<4:
        levels = list(range(nref+1))

    print('Making points in levels', levels, '(maximum refinement level %d)'%nref, file=log)
    max_nodes = max((int(4*((2**nref) * rmin/boxsize)**3),1000000))
    print('Guess for max_nodes {:,}'.format(max_nodes), file=log)
  
    zoom_ctr = array(ctr, dtype=float64)/boxsize
    print('Building tree',file=log)
    tree = make_BH_tree(zoom_ctr, rmin/boxsize, bh_crit=0.1, max_nodes=max_nodes, allowed_levels=levels)
    print('Number of nodes {:,}, number of leaves {:,}'.format(len(tree), tree.max()+1), file=log)

    print('Finding centres', file=log)
    depth, uni_pts = leaf_depths_centres(tree)



    l_depths = sorted(unique(depth))
    num_at_depth = [len(flatnonzero(depth==d)) for d in l_depths]


    for d,n in zip(l_depths, num_at_depth):
        print('Number of leaves of depth %2d:'%d,"{:15,}".format(n), file=log)


    print('Scaling to box of', boxsize, 'on a side', file=log)
    uni_pts *= boxsize
    uni_sizes = boxsize * power(2.0,-depth) # (cell) sizes, s.t. sum(sizes**3)=boxsize**3
    
    
    return uni_pts, uni_sizes

def make_zoom_uniform_ellipsoid(ctr=(25.0,25.0,25.0), 
                                ellipsoid=((0.04,0,0),(0,0.04,0),(0,0,0.04)),
                                gridsize=1024, boxsize=50.0, bh_crit=0.15):
    """
    Like make_zoom_uniform but for an ellipsoidal region

    ctr      - 3 cpt centre of the zoom region (in [0,boxsize])
    ellipsoid- Matrix A of maximum refinement surface, x.A.x = 1
    gridsize - grid resolution for highest res region (e.g. 1024 for 1024^3) 
    boxsize  - size of the box

    Returns
   
    uni_pts   - points distributed over the box (n,3), in [0,boxsize]
    uni_sizes - sizes such that sum(sizes^3)== boxsize^3
    """
    nref = 0
    while gridsize>2**nref:
        nref = nref + 1

    if gridsize!=2**nref:
        raise Exception(str(gridsize)+' was not a power of 2.')


    levels = [nref-4,nref-2,nref-1,nref]
    if nref<4:
        levels = list(range(nref+1))

    # scale the matrix for the unit box
    A = array(ellipsoid, dtype=float64)*(boxsize*boxsize)
    node_est = 1.0/sqrt(cumprod(eigvalsh(A))[-1]) * (8**nref) * 4
    print('Making points in levels', levels, '(maximum refinement level %d)'%nref)
    max_nodes = max((int(node_est),1000000))
    print('Guess for max_nodes {:,}'.format(max_nodes))
  
    zoom_ctr = array(ctr, dtype=float64)/boxsize
    print('Building tree')
    tree = make_BH_ellipsoid_tree(zoom_ctr, A, bh_crit=bh_crit, max_nodes=max_nodes, allowed_levels=levels)
    print('Number of nodes {:,}, number of leaves {:,}'.format(len(tree), tree.max()+1))

    print('Finding centres')
    depth, uni_pts = leaf_depths_centres(tree)



    l_depths = sorted(unique(depth))
    num_at_depth = [len(flatnonzero(depth==d)) for d in l_depths]


    for d,n in zip(l_depths, num_at_depth):
        print('Number of leaves of depth %2d:'%d,"{:15,}".format(n))


    print('Scaling to box')
    uni_pts *= boxsize
    uni_sizes = boxsize * power(2.0,-depth) # (cell) sizes, s.t. sum(sizes**3)=boxsize**3
    
    
    return uni_pts, uni_sizes


def read_uniform_distrib(name):

    print('Reading particles from', name)
    f = open(name, 'rb')
    test, boxsize, uni_pts, uni_sizes = pickle.load(f)
    f.close()
    print('Checking')
    levels = unique(uni_sizes)
    print('Number of unique size particles', len(levels))
    return uni_pts, uni_sizes


