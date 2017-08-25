"""
Module for interfacing to the C-library (liblizard.so)
"""
from __future__ import print_function, division, absolute_import
# Note *dont* import unicode_literals since we require the ndpointer flags to 
# be the default string types in both python 2 and 3 respectively
from numpy.ctypeslib import ndpointer
from numpy.linalg import eigvalsh
import ctypes
from numpy import float64, empty, array, int32, zeros, float32, require, int64, uint32, complex128
from numpy import roll, diff, flatnonzero, uint64, cumsum, square
from os import path
from .log import MarkUp as MU, null_log

_liblizard = None
_hash_kernel_set = False

c_contig = 'C_CONTIGUOUS'

def _initlib(log):
    """ Init the library (if not already loaded) """
    global _liblizard

    if _liblizard is not None:
        return _liblizard


    name = path.join(path.dirname(path.abspath(__file__)), '../build/liblizard.so')
    if not path.exists(name):
        raise Exception('Library '+str(name)+' does not exist. Maybe you forgot to make it?')

    print(MU.OKBLUE+'Loading liblizard - C functions for lizard calculations', name+MU.ENDC, file=log)
    _liblizard = ctypes.cdll.LoadLibrary(name)

    # Coordinate interpolation
    # C declaration is below
    # int interpolate_periodic(const int grid_width, const unsigned long long npts, const double *gridvals, const double *coordinates, double *out_interp)

    func = _liblizard.interpolate_periodic
    func.restype = ctypes.c_int
    func.argtypes = [ctypes.c_int, ctypes.c_ulonglong, ndpointer(float64, flags=c_contig), 
                     ndpointer(float64, flags=c_contig), ndpointer(float64, flags=c_contig)]

    # Interpolation of grid of vectors
    # void interp_vec3(const int grid_n, const unsigned long long npts, 
    #		 const double *grid, const double *pts, double *out)

    func = _liblizard.interp_vec3
    func.restype = None
    func.argtypes = [ctypes.c_int, ctypes.c_ulonglong, ndpointer(float64, flags=c_contig), 
                     ndpointer(float64, flags=c_contig), ndpointer(float64, flags=c_contig)]


    # Make the Barnes-Hut tree
    # int make_BH_refinement(int *tree, const int MAX_NODES,   
    # const int max_refine, const double rmin, const float bh_crit, const double *pos)

    func = _liblizard.make_BH_refinement
    func.restype = ctypes.c_int
    func.argtypes = [ndpointer(int32), ctypes.c_int, 
                     ctypes.c_int, ctypes.c_double, ctypes.c_float, ndpointer(float64)]

    # Make the Barnes-Hut tree about an ellipsoidal refinement region
    # int make_BH_ellipsoid_refinement(int *tree, const int MAX_NODES, 
    #				 const int max_refine, const double *A, const double k,
    #				 const float bh_crit, const double *pos)

    func = _liblizard.make_BH_ellipsoid_refinement
    func.restype = ctypes.c_int
    func.argtypes = [ndpointer(int32), ctypes.c_int, 
                     ctypes.c_int, ndpointer(float64), ctypes.c_double, ctypes.c_float, ndpointer(float64)]


    # Force a given set of refinement levels
    # int force_refinement_levels(int *tree, const int total_nodes, const int MAX_NODES, const int *levels)

    func = _liblizard.force_refinement_levels
    func.restype = ctypes.c_int
    func.argtypes = [ndpointer(int32), ctypes.c_int, ctypes.c_int,ndpointer(int32)]

    # Find the depths and centres of each leaf
    #  leaf_depths_centres(const int *tree, int *out_depths, double *out_centres)

    func = _liblizard.leaf_depths_centres
    func.restype = None
    func.argtypes = [ndpointer(int32), ndpointer(int32), ndpointer(float64)]

    # count number of leaves in each node
    # int count_leaves(const int *tree, const int node, const int n_leaves, int *out)

    func = _liblizard.count_leaves
    func.restype = ctypes.c_int
    func.argtypes = [ndpointer(int32), ctypes.c_int, ctypes.c_int, ndpointer(int32)]    

    # Find the particles around a given node
    # int leaves_in_box_ex_node(const int *tree, const int idx, const int excl_node, const int n_leaves, 
    #			  const double x0, const double y0, const double z0, 
    #			  const float chw, const float box_w, const int max_ngbs, int *out)

    func = _liblizard.leaves_in_box_ex_node
    func.restype = ctypes.c_int
    func.argtypes = [ndpointer(int32), ctypes.c_int, ctypes.c_int, ctypes.c_int, 
                     ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_float, ctypes.c_float, 
                     ctypes.c_int, ndpointer(int32)]

    # Build octree
    # int octree_from_pos(int *tree,const int max_nodes, const double *pos, const int n_pos)
    func = _liblizard.octree_from_pos
    func.restype = ctypes.c_int
    func.argtypes = [ndpointer(int32), ctypes.c_int, ndpointer(float64), ctypes.c_int]


    # Build gravity oct-tree
    # int build_gravity_octtree(int *tree, const int max_nodes, double *mass_com, const int n_parts)

    #    func = _liblizard.build_gravity_octtree
    #func.restype = ctypes.c_int
    #func.argtypes = [ndpointer(int32), ctypes.c_int, ndpointer(float64), ctypes.c_int]


    # Build Mass, CoM for each node in octree
    # void octree_mcom(const int *tree, const int n_pos, const double *p_mcom, double *out)
    func = _liblizard.octree_mcom
    func.restype = None
    func.argtypes = [ndpointer(int32), ctypes.c_int, ndpointer(float64), ndpointer(float64)]

    # Initialise kernel
    # int init_kernel(const int num_pts, const float *pts, const float rad_min, const float rad_max)
    func = _liblizard.init_kernel
    func.restype = ctypes.c_int
    func.argtypes = [ctypes.c_int, ndpointer(ctypes.c_float, flags=c_contig), ctypes.c_float, ctypes.c_float]

    # Evaluate a kernel for a series of points
    # int kernel_evaluate(const double *pos, const int num_pos, const int *tree, const int n_leaves, const double *tree_mcom, double *out, float bh_crit)
    func = _liblizard.kernel_evaluate
    func.restype = ctypes.c_int
    func.argtypes = [ndpointer(float64), ctypes.c_int, ndpointer(int32), ctypes.c_int, ndpointer(float64), ndpointer(float64), ctypes.c_float]

    # Find all the leaves of a given node, in order
    # int write_leaves(int *tree,const int n_leaves, const int node, int *out)
    func = _liblizard.write_leaves
    func.restype = ctypes.c_int
    func.argtypes = [ndpointer(int32), ctypes.c_int, ctypes.c_int,ndpointer(int32)]

    # Find the cell for each point
    # void find_lattice(const double *pos, const int num_pos, const int nx, int *out)
    func = _liblizard.find_lattice
    func.restype = None
    func.argtypes = [ndpointer(float64, flags=c_contig), ctypes.c_int, ctypes.c_int, ndpointer(int32)]

    # Build octrees in a single sort+sweep
    # int build_octree_iterator(const double *pos, const int num_pos, const int nx, const int bucket_size,
    # 				int *restrict sort_idx, int *restrict out, const int buf_size)
    func = _liblizard.build_octree_iterator
    func.restype = ctypes.c_int
    func.argtypes = [ndpointer(float64), ctypes.c_int, ctypes.c_int, ctypes.c_int, 
                     ndpointer(int32), ndpointer(float64), ctypes.c_int]

    # void fill_treewalk_xyzw(const int num_trees, double *restrict twn_ptr, const int32_t *restrict tree_sizes, 
    #		       const double *restrict xyzw)
    func = _liblizard.fill_treewalk_xyzw
    func.restype = ctypes.c_int
    func.argtypes = [ctypes.c_int, ndpointer(float64), ndpointer(int32), ndpointer(float64)]

    func =_liblizard.get_tree_iterator_size
    func.restype = ctypes.c_int
    func.argtypes = []


    # Walk the tree-walk-nodes for BH gravity
    # long BHTreeWalk(const int *restrict root_sizes, const int num_roots, const int max_depth,
    #           const int *restrict cells,
    #		const int ngrid, const double *restrict tree_walk_nodes,
    #		const double theta, const double *restrict xyzw, double *restrict acc)
    func = _liblizard.BHTreeWalk
    func.restype = ctypes.c_long
    func.argtypes = [ndpointer(int32), ctypes.c_int, ctypes.c_int, ndpointer(int32), 
                     ctypes.c_int, ndpointer(float64), 
                     ctypes.c_double, ndpointer(float64), ndpointer(float64)]

    # Set-up kernel for neighbour summation
    # int setup_hash_kernel(const double rcut, const int num_wts, const double *kernel_wts)
    func = _liblizard.setup_hash_kernel
    func.restype = ctypes.c_int
    func.argtypes = [ctypes.c_double, ctypes.c_int, ndpointer(float64)]

    # Find neighbours
    # long radial_kernel_evaluate(const double *xyzw, const int num_cells, 
    #        const int* cells, const int ngrid,	double *accel)
    func = _liblizard.radial_kernel_evaluate
    func.restype = ctypes.c_long
    func.argtypes = [ndpointer(float64), ctypes.c_int, ndpointer(int32), 
                     ctypes.c_int, ndpointer(float64)]
    # long radial_kernel_cellmean(const double *xyzw, const int num_cells, const int* cells, const int ngrid,
    #   		      const int stencil, double *accel)
    func = _liblizard.radial_kernel_cellmean
    func.restype = ctypes.c_long
    func.argtypes = [ndpointer(float64), ctypes.c_int, ndpointer(int32), 
                     ctypes.c_int, ctypes.c_int, ndpointer(float64)]

    # Find neighbours
    # long find_ngbs5x5(const double *mpos, const int num_cells, const int* cells, 
    # const double rcrit, const int ngrid, double *accel)
    func = _liblizard.find_ngbs5x5
    func.restype = ctypes.c_long
    func.argtypes = [ndpointer(float64), ctypes.c_int, ndpointer(int32), 
                     ctypes.c_double, ctypes.c_int, ndpointer(float64)]


    # Cloud-in-cell interpolation of points
    # int cloud_in_cell_3d(const int num_pts, const int ngrid, const double *mcom, double *out) 
    func = _liblizard.cloud_in_cell_3d
    func.restype = ctypes.c_int
    func.argtypes = [ctypes.c_int,ctypes.c_int,ndpointer(float64, ndim=2),ndpointer(float64)]
    
    # int cloud_in_cell_3d_vel(const int num_pts, const int ngrid, const double * mcom, const double * mom, double complex *out)
    func = _liblizard.cloud_in_cell_3d_vel
    func.restype = ctypes.c_int
    func.argtypes = [ctypes.c_int,ctypes.c_int,ndpointer(float64),ndpointer(float64),ndpointer(complex128)]

    # Five-point gradient for derivatives
    # void gradient_5pt_3d(const int ngrid, const double *vals, double *restrict out)
    func = _liblizard.gradient_5pt_3d
    func.restype = None
    func.argtypes = [ctypes.c_int,ndpointer(float64),ndpointer(float64)]

    # unpack vals in 0,1,... n/2 into 3d grid
    # void unpack_kgrid(const int n, const double *packed_vals, double *unpacked_vals)
    func = _liblizard.unpack_kgrid
    func.restype = None
    func.argtypes = [ctypes.c_int,ndpointer(float64),ndpointer(float64)]

    # Find masks for cell and its neighbours
    # void ngb_cell_masks_3x3x3(const int num_pts, const double rcrit, const int *cells,
    #                           const uint64_t *ngb_masks, const double *pts, uint64_t *out)
    func = _liblizard.ngb_cell_masks_3x3x3
    func.restype = None
    func.argtypes = [ctypes.c_int,ctypes.c_double,ndpointer(int32),
                     ndpointer(uint64),ndpointer(float64), 
                     ndpointer(uint64)]

    # Greedy region-growing
    # int region_grow_domains3d(const int ngrid, const long max_sum, int *restrict grid,
    # 	    		       int *restrict bdrybuf, const int bufsize)

    func = _liblizard.region_grow_domains3d
    func.restype = ctypes.c_int
    func.argtypes = [ctypes.c_int,ctypes.c_long, ndpointer(int32),ndpointer(int32), ctypes.c_int]

    return _liblizard

def gradient_5pt_3d(grid, log=null_log):
    """ 
    Find the derivative of a periodic 3d lattice using the 5-point stencil
    Returns an (n,n,n,3) array of the derivatives (in x,y,z).
    """
    g = require(grid, dtype=float64, requirements=['C']) 
    
    n = g.shape[0]
    assert(g.shape==(n,n,n))

    lib = _initlib(log)

    out = empty((n*n*n*3),dtype=float64)
    lib.gradient_5pt_3d(n,g,out)
    out.shape = (n,n,n,3)
    return out
    
def get_cic(pos, ngrid, mass=None, mom=None,log=null_log):
    """ 
    pos   - centre-of-mass for the particles (centres should be in [0, ngrid) )
    ngrid - number of cells (width) of the box
    [mass]- masses of the particles (optional, default 1)
    [mom] - momenta of the particles - if this is set we return a complex value
            for each cell containing the CIC in the real part, and the time
            derivative in the imaginary.

    """
    npts = pos.shape[0]
    assert(pos.shape==(npts, 3))
    mcom = empty((npts, 4), dtype=float64)
    mcom[:,1:] = pos
    if mcom[:,1:].min()<0:
        raise Exception("All positions should be >=0 (try making periodic first?)")
    if mcom[:,1:].max()>=ngrid:
        raise Exception("All positions should be <ngrid (try making periodic first?)")        

    
    if mass is None:
        mcom[:,0] = 1
    else:
        mcom[:,0] = mass
    
    lib = _initlib(log)
    if mom is None:
        # No momenta so just the standard cic
        # Initialise to zero since we just accumulate
        res = zeros(ngrid*ngrid*ngrid, dtype=float64)
        lib.cloud_in_cell_3d(npts, ngrid, mcom, res)
        res.shape = (ngrid,ngrid,ngrid)
        return res
    
    # Have momentum too, so call with vel
    assert(mom.shape==(npts,3))
    r_mom = require(mom, dtype=float64, requirements=['C']) 
    # Initialise to zero since we just accumulate
    res = zeros(ngrid*ngrid*ngrid, dtype=complex128)
    lib.cloud_in_cell_3d_vel(npts, ngrid, mcom, r_mom,res)
    res.shape = (ngrid,ngrid,ngrid)
    return res
    

def get_cells(pts, ncell, log=null_log):
    """
    For an (N,3) array of points in [0,1), find lattice index for 
    (ncell**3,) array
    """
    lib = _initlib(log)
    p = require(pts, dtype=float64, requirements=['C']) 
    npts = p.shape[0]
    assert(p.shape ==(npts,3))
    out = empty(npts, dtype=int32)

    res = lib.find_lattice(p, npts, ncell, out)
    return out

def build_octrees(pos, bucket_size, ngrid, wts, log=null_log, buf_size=None):
    """ Build the grid of octrees in a single sweep with fancy position decorating """

    lib = _initlib(log)

    sizeof_twn = lib.get_tree_iterator_size()
    guess_nodes = buf_size is None
    npos = len(pos)
    pts = require(pos, dtype=float64, requirements=['C'])
    if guess_nodes:
        max_nodes = int(npos*2.1)+1
        buf_size = max_nodes * (sizeof_twn//8)
        print('Guessed number of nodes {:,}'.format(max_nodes),file=log)
    buf = empty(buf_size, dtype=float64)
    sort_idx = empty(npos, dtype=int32)

    num_roots = lib.build_octree_iterator(pts, npos, ngrid, bucket_size, 
                                          sort_idx, buf, buf_size)
    if num_roots==-1:
        raise Exception('Out of memory')
    if num_roots==-2:
        raise Exception('>bucket_size points have indistinguishable double representations')

    class tree:
        root_counts = buf.view(dtype=int32)[:num_roots]
        num_nodes = root_counts.sum()
        root_indices = empty(num_roots, dtype=int32)
        root_cells =  buf.view(dtype=int32)[num_roots:num_roots*2]

        it = buf[num_roots:num_roots + (sizeof_twn*num_nodes)//8]

        sof32 = sizeof_twn//4 # size of TWN in 32 bit ints
        # use intimate knowledge of structure layout
        n = it.view(dtype=int32)[sof32-2::sof32]
        depth_next = it.view(dtype=int32)[sof32-3::sof32]
        breadth_next = it.view(dtype=int32)[sof32-4::sof32]
        depths = it.view(dtype=int32)[sof32-1::sof32]
        


    tree.root_indices[1:] = cumsum(tree.root_counts[:-1])
    tree.root_indices[0] = 0

    tree.fill = tree.n[tree.root_indices]

    print('{:,} filled cells(trees),'.format(num_roots),
          '{:,}-{:,} nodes per tree,'.format(tree.root_counts.min(), tree.root_counts.max()),
          'av. %.2f'%tree.root_counts.mean(dtype=float64), file=log)

    print('{:,}-{:,} points per tree'.format(tree.fill.min(),tree.fill.max()), 
          '(av. %.2f),'%tree.fill.mean(dtype=float64), 
          'av. point in a tree of {:,} points'.format(square(tree.fill.astype(int64)).sum()//npos),
          file=log)

          
    leaf_counts = tree.n[flatnonzero(tree.n<=bucket_size)]

    av_leaf_size_per_pt = square(leaf_counts).sum()/float(npos)
    print('%d-%d points per leaf (leaf size %d), average %.2f, average point is in a leaf of %.2f pts'%(leaf_counts.min(), leaf_counts.max(), bucket_size, leaf_counts.mean(dtype=float64), av_leaf_size_per_pt), file=log)


    print('Actual number of nodes used {:,}, total memory {:,} bytes'.format(tree.num_nodes, tree.num_nodes*sizeof_twn),file=log)
    
    print('Indexing {:,} points for octree-ordered xyzw'.format(npos),file=log)
    xyzw = empty((npos+tree.num_nodes, 4), dtype=float64)
    xyzw[:npos,:3] = pts[sort_idx]
    
    if sum(array(wts).shape)<=1:
        xyzw[:npos,3] = wts
    else:
        xyzw[:npos,3] = wts[sort_idx]


    print('Building xyzw for {:,} nodes'.format(tree.num_nodes), file=log)

    tree.max_depth = lib.fill_treewalk_xyzw(num_roots, tree.it, tree.root_counts, xyzw)
    tree.xyzw = xyzw

    print('Max leaf depth %d'%tree.max_depth, file=log)
    return tree, sort_idx


def bh_tree_walk(tree, ngrid, theta, xyzw,log=null_log):
    """ Kernel summation over BH tree """

    lib = _initlib(log)

    if not _hash_kernel_set:
        raise Exception('Please set-up the kernel before trying a neighbour-summation')

    num_trees = len(tree.root_cells)

    rt_sizes = require(tree.root_counts, dtype=int32, requirements=['C'])
    cells = require(tree.root_cells, dtype=int32, requirements=['C'])
    twn = require(tree.it, dtype=float64, requirements=['C'])
    xyzw = require(tree.xyzw, dtype=float64, requirements=['C'])

    npts = len(xyzw) - tree.num_nodes

    out = zeros(npts*3, dtype=float64)
    num_kernels = lib.BHTreeWalk(rt_sizes, num_trees, tree.max_depth, cells, ngrid, twn, theta, xyzw, out)
    if num_kernels<0:
        raise Exception('Hash table too big for kernel summation')
    acc = out
    acc.shape = (npts, 3)

    return num_kernels, acc

def lattice_setup_kernel(rad_max, weights, log=null_log):
    """
    setup the kernel interpolation weights with
    0 -> 0, dx -> wts[0], 2dx -> wts[1],..., rad_max -> wts[-1]
    where dx := rad_max / len(wts)
    """
    global _hash_kernel_set

    lib = _initlib(log)
    # C-ordering of double array
    wts = require(weights, dtype=float64, requirements=['C'])
    res = lib.setup_hash_kernel(rad_max, len(wts), wts)
    if res<0:
        raise Exception('You can only set up to MAX_KERNEL_WTS-1=%d weights in the interpolation table, use less or recompile'%(-(res+1)))
    _hash_kernel_set = True
    return
    
def lattice_kernel(pts, lattice_data, ngrid, log=null_log, masses=None, stencil=None):

    if not _hash_kernel_set:
        raise Exception('Please set-up the kernel before trying a neighbour-summation')

    lib = _initlib(log)
    ncells = lattice_data.shape[0]
    npts = len(pts)
    xyzw = empty((npts, 4), dtype=float64)
    xyzw[:,:3] = pts
    if masses is None:
        xyzw[:,3] = 1
    else:
        xyzw[:,3] = masses

    # lattice pos, start, end for each cell
    cells = require(lattice_data, dtype=int32, requirements=['C'])
    accel = zeros(xyzw[:,1:].shape, dtype=float64)
    if stencil is None:
        res = lib.radial_kernel_evaluate(xyzw, ncells, cells, ngrid, accel)
    else:
        res = lib.radial_kernel_cellmean(xyzw, ncells, cells, ngrid, stencil, accel)
        if res==-2:
            raise Exception('Monopole stencil must be 5 or 7')            
    if res==-1:
        raise Exception('Hash table not big enough to store data')

    pairs = res
    return pairs, accel



def lattice_kernel5x5(pts, lattice_data, rcrit, ngrid):
    lib = _initlib()
    ncells = lattice_data.shape[0]
    npts = len(pts)
    mpos = empty((npts, 4), dtype=float64)
    mpos[:,1:] = pts
    mpos[:,0] = 1

    # lattice pos, start, end for each cell
    cells = require(lattice_data, dtype=int32, requirements=['C'])
    accel = zeros(mpos[:,1:].shape, dtype=float64)
    res = lib.find_ngbs5x5(mpos, ncells, cells, rcrit, ngrid, accel)
    return res

def build_hash3d(pts, nlevel, hash_prime, bucket_bits, log=null_log):

    lib = _initlib()
    npts = pts.shape[0]
    if nlevel*3+bucket_bits>27:
        raise Exception('Number of grid elements %d is too many')

    ngrid = 2**nlevel
    bucket_depth = 2**bucket_bits

    tot_buckets = 8**nlevel
    out = zeros((ngrid,ngrid,ngrid,bucket_depth), dtype=int64)
    res = lib.build_hash3d(pts, npts, nlevel, hash_prime, bucket_bits, out)
    print('Maximum shift {:,} or {:,} buckets'.format(res, res//bucket_depth), file=log)
    return out
def kernel_hash3d(pts, hgrid, hash_prime, rcrit):

    lib = _initlib()
    ngrid = hgrid.shape[0]
    bucket_depth = hgrid.shape[3]
    npts = len(pts)
    out = zeros(pts.shape, dtype=float64)
    evals = lib.ngb_sum_hash3d(pts, npts, ngrid, hash_prime, bucket_depth, hgrid, rcrit, out)
    
    return out, evals

def leaves_for_node(tree, n_leaves, node, log=null_log):
    """ return all leaves of the given node """
    lib = _initlib(log)
    out = empty(n_leaves, dtype=int32)
    leaves_found=lib.write_leaves(tree, n_leaves,  node, out)
    print('leaves found', leaves_found, file=log)
    out = out[:leaves_found]
    return out
def kernel_evaluate(pos, tree, tree_mcom, rmin, rmax, kvals):
    """
    
    kvals - logarithmically spaced values between rmin and rmax
    """
    lib = _initlib(null_log)

    n_kernel = len(kvals)
    res = lib.init_kernel(n_kernel, array(kvals, dtype=float32), rmin, rmax)
    if res!=0:
        raise Exception('Failed to initialise. Too many points for kernel?')
    
    n_leaves = tree_mcom.shape[0] - tree.shape[0]
    out = empty(pos.shape, dtype=float64)
    num_evals = lib.kernel_evaluate(pos, pos.shape[0], tree, n_leaves, tree_mcom, out, 0.1)
    if num_evals<0:
        raise Exception('Neighbour seach for kernel ran out of stack space')

    return num_evals, out

def octree_mcom(tree, mass, pos):
    """
    Find mass and centre-of-mass for each node in the tree
    """
    lib = _initlib(null_log)

    num_pos, dim = pos.shape
    assert(dim==3)
    n_nodes = tree.shape[0]

    part_mcom = empty((num_pos+n_nodes,4), dtype=float64) # expected dtype
    part_mcom[:num_pos,0] = mass # particle mass
    part_mcom[:num_pos,1:] = pos
    
    lib.octree_mcom(tree, num_pos, part_mcom, part_mcom[num_pos:])
    
    inv_mass = 1.0/part_mcom[num_pos:,0] # TODO could be zero if we have massless particles

    for i in range(3):
        part_mcom[num_pos:,i+1] *= inv_mass

    return part_mcom

    
    
def neighbours_for_node(tree, num_leaves, node, xyz0, box_w, max_leaves=1000000):
    assert(len(xyz0)==3)
    assert(node!=0)
    lib = _initlib(null_log)
    
    out = empty(max_leaves, dtype=int32)
    found = lib.leaves_in_box_ex_node(tree, 0, node, num_leaves, xyz0[0], xyz0[1], xyz0[2], 0.5, box_w, max_leaves, out)
    if found==-1:
        raise Exception('Out of space, try to raise max_leaves=%d?'%max_leaves)

    return out[:found]

def count_leaves(tree, num_leaves):
    """
    For the given octree find the number for each leaf
    """
    num_nodes, octants = tree.shape
    assert(octants==8) # should be 8 octants for each node!
    assert(tree.dtype==int32)
    _initlib()    
    out = empty(num_nodes, dtype=int32)
    
    check = _liblizard.count_leaves(tree, 0, num_leaves, out)
    assert(check==num_leaves) # Total number of leaves should equal number in the tree!
    return out

def make_BH_tree(pos, rmin, max_refinement=None,max_nodes=None, bh_crit=0.1, allowed_levels=None, log=null_log):
    """ 
    Make the Barnes-Hut refinement tree 
    pos     - position of the central refinement region
    rmin    - distance out to which we want maximum refinement
    bh_crit - Barnes-Hut opening angle criteria for refinement
    [allowed_levels] - Allowed levels for the tree
    """
    if allowed_levels is not None:
        if max_refinement is not None:
            print('WARNING: max_refinement ignored as allowed_levels has been set',file=log)
        max_refinement = max(allowed_levels)
    else:
        if max_refinement is None:
            raise Exception('One of max_refinement or allowed_levels must be set!')

    if max_refinement==0:
        # Nothing to do, we will never refine!
        return array([], dtype=int32)

    _initlib(log)
    if max_nodes is None:
        # Make a guess for how many nodes we need
        max_nodes = 1000000

        
    tree = empty(max_nodes*8, dtype=int32)
    ppos = array(pos, dtype=float64)
    nodes_used = _liblizard.make_BH_refinement(tree, max_nodes,
                                  max_refinement, rmin, bh_crit, ppos)

    if nodes_used==-1:
        raise Exception('max_nodes=%d was not enough. Try to increase?'%max_nodes)



    if allowed_levels is None:
        tree.shape = (max_nodes,8)
        return tree[:nodes_used]
    
    # Otherwise make us have good levels
    levels = array(sorted(allowed_levels), dtype=int32)
    
    nodes_used = _liblizard.force_refinement_levels(tree, nodes_used, max_nodes, levels)

    if nodes_used==-1:
        raise Exception('max_nodes=%d was not enough. Try to increase?'%max_nodes)
    tree.shape = (max_nodes,8)
    return tree[:nodes_used]

def make_BH_ellipsoid_tree(pos, A, max_refinement=None,max_nodes=None, bh_crit=0.1, allowed_levels=None,log=null_log):
    """ 
    Like make_BH_tree, but for an ellipsoidal region.
    pos     - position of the central refinement region
    A       - Matrix for ellipse, such that the surface within which we want 
              maximum refinement is x.A.x = 1
    bh_crit - Barnes-Hut opening angle criteria for refinement
    [allowed_levels] - Allowed levels for the tree
    """
    if allowed_levels is not None:
        if max_refinement is not None:
            print('WARNING: max_refinement ignored as allowed_levels has been set',file=log)
        max_refinement = max(allowed_levels)
    else:
        if max_refinement is None:
            raise Exception('One of max_refinement or allowed_levels must be set!')

    if max_refinement==0:
        # Nothing to do, we will never refine!
        return array([], dtype=int32)

    lib = _initlib(log)
    if max_nodes is None:
        # Make a guess for how many nodes we need
        max_nodes = 1000000

    
    # Matrix for the ellipsoid
    Aconv = empty((3,3), dtype=float64)
    k = 1.0/eigvalsh(A).max()
    Aconv[:] = A * k
    
    tree = empty(max_nodes*8, dtype=int32)
    ppos = array(pos, dtype=float64)
    nodes_used = lib.make_BH_ellipsoid_refinement(tree, max_nodes,
                                  max_refinement, Aconv,k, bh_crit, ppos)

    if nodes_used==-1:
        raise Exception('max_nodes=%d was not enough. Try to increase?'%max_nodes)


    if allowed_levels is None:
        tree.shape = (max_nodes,8)
        return tree[:nodes_used]
    
    # Otherwise make us have good levels
    levels = array(sorted(allowed_levels), dtype=int32)
    
    nodes_used = _liblizard.force_refinement_levels(tree, nodes_used, max_nodes, levels)

    if nodes_used==-1:
        raise Exception('max_nodes={:,} was not enough. Try to increase?'.format(max_nodes))
    tree.shape = (max_nodes,8)
    return tree[:nodes_used]

def build_octree(pos, max_nodes=None):
    npos, dim = pos.shape

    assert(dim==3) # should be 3 dimensional!

    if max_nodes is None:
        max_nodes = npos//2 + 1

    lib = _initlib(null_log)

    tree = empty((max_nodes,8), dtype=int32)
    
    nodes_used = lib.octree_from_pos(tree, max_nodes, pos, npos)

    if nodes_used==-1:
        raise Exception('max_nodes=%d was not enough. Try to increase?'%max_nodes)
    return tree[:nodes_used]


    
def build_gravity_octtree(pos,masses, max_nodes=None):
    """ 
    Make the gravity octtree for the particles
    """
    nparts = len(masses)

    if max_nodes is None:
        max_nodes = nparts // 2


    _initlib()

    # Store mass,com data of the particles
    mcom = empty((nparts+max_nodes, 4), dtype=float64)
    mcom[:nparts,0] = masses
    mcom[:nparts,1:] = pos

    tree = empty((max_nodes,8), dtype=int32)
    
    nodes_used = _liblizard.build_gravity_octtree(tree, max_nodes, mcom, nparts)

    if nodes_used==-1:
        raise Exception('max_nodes=%d was not enough. Try to increase?'%max_nodes)
    return tree[:nodes_used], mcom[:(nparts+nodes_used)]


def leaf_depths_centres(tree):
    """ 
    Get every leaf depth and centre for the given tree.
    Centres are in 0-1
    """
    if len(tree)==0:
        # Whole tree is one leaf!
        return array([0]), array([[0.5,0.5,0.5]], dtype=float64)

    assert(tree.dtype==int32)
    num_leaves = tree.max()+1
    out_depths = empty(num_leaves, dtype=int32)
    out_centres = empty((num_leaves,3), dtype=float64)
    
    lib = _initlib(null_log)

    lib.leaf_depths_centres(tree, out_depths, out_centres)
    return out_depths, out_centres

def map_coords(vals,coords,log=null_log,order=1):
    """
    do the interpolation from the 3d grid (periodically wrapped)
    """

    if order!=1:
        raise Exception('Can only do linear interpolation at this point!')
    _initlib(log)

    assert(len(vals.shape)==3)
    grid_width = vals.shape[0]
    assert(vals.shape[1]==grid_width)
    assert(vals.shape[2]==grid_width)

    num_pts = coords.shape[1]
    
    assert(coords.shape==(3,num_pts))
    # C-contiguous double arrays (makes a copy if necc)
    pcoords = require(coords, dtype=float64, requirements=['C']) 
    gvals = require(vals, dtype=float64, requirements=['C'])
    out = empty(num_pts, dtype=float64)
    
    res = _liblizard.interpolate_periodic(grid_width, num_pts, gvals, pcoords, out)
    if res != 0:
        print('An error occurred', res,file=log)
    return out

def interp_vec3(grid,coords,log=null_log):
    """
    Interpolation from the 3d grid of 3d vectors (periodically wrapped)
    Like map_coords but speed-up for vectors

    grid   - (N,N,N,3) array
    coords - (M,3) array in right-open interval [0,N)
    [log]

    returns (M,3) array of 1d interpolated values
    """

    lib = _initlib(log)

    # C-contiguous double arrays (makes a copy if necc)
    gvals = require(grid, dtype=float64, requirements=['C'])
    grid_width = gvals.shape[0]    
    assert(gvals.shape==(grid_width,grid_width,grid_width,3))

    pcoords = require(coords, dtype=float64, requirements=['C']) 
    num_pts = pcoords.shape[0]    
    assert(pcoords.shape==(num_pts,3))
    if pcoords.max()>=grid_width or pcoords.min()<0:
        print('Max pos', pcoords.max(axis=0), 'min', pcoords.min(axis=0), file=log)
        raise Exception('All coordinates must be in the right-open interval [0,grid_width)')

    out = empty(num_pts*3, dtype=float64) # make sure this is big enough...
    
    res = lib.interp_vec3(grid_width, num_pts, gvals, pcoords, out)

    out.shape = (num_pts, 3)
    return out

def unpack_kgrid(n, vals, log=null_log):
    """
    Unpack the 'pyramid' of values u>=v>=w into the (n,n,n) k-grid.

    n    - the size of the grid
    vals - m(m+1)(m+2)/6 values in the pyramid

    returns out - (n,n,n) float64 array.
    """
    lib = _initlib(log)
    
    v = require(vals, dtype=float64, requirements=['C'])
    m = 1+n//2
    assert(len(vals)==(m*(m+1)*(m+2))//6)
    out = empty(n*n*n, dtype=float64)
    
    lib.unpack_kgrid(n, v, out)
    out.shape = (n,n,n)

    return out


def adjacent_cell_masks(pts, cell_masks, rcrit, log=null_log):
    """                                                 
    For each point create a bitmask that that is the logical-or of all masks                                      
    for all the domains that it is within rcrit of.     
    """

    cell_masks = array(cell_masks)
    ngrid = cell_masks.shape[0]

    r = float(rcrit) * ngrid
    if r>1.0:
        raise Exception('rcrit>1.0, need to look outside 3x3x3 block of neighbours')

    assert(cell_masks.shape==(ngrid,ngrid,ngrid))
    num_pts = len(pts)
    pos = array(pts)*ngrid
    pos = require(pos, dtype=float64, requirements=['C'])
    assert(pos.shape==(num_pts, 3))
    
    # initialise space for masks
    ngb_masks = empty((ngrid, ngrid, ngrid, 27), dtype=uint64)
    # Fill in the masks of all the neighbouring cells
    ngb = 0
    inc = [1,0,-1] # roll increments for left, middle, right
    for i in inc:
        ri = roll(cell_masks, i, axis=0)
        for j in inc:
            rj = roll(ri, j, axis=1)
            for k in inc:
                ngb_masks[:,:,:,ngb] = roll(rj, k, axis=2)
                ngb += 1

    cells = get_cells(pts, ngrid, log)
    res = empty(num_pts, dtype=uint64)
    lib = _initlib(log)

    lib.ngb_cell_masks_3x3x3(num_pts, r*1.001, cells, ngb_masks, pos, res)
    return res


def domain_regiongrow(weights, min_sum, log=null_log, buf=None):
    """
    weights - (n,n,n) grid of >=0 integer weights
    min_sum - minimum sum in each domain

    returns - (n,n,n) grid of nearly contiguous domains
    """
    grid = -array(weights, dtype=int32, order='C')
    n = grid.shape[0]
    assert(grid.shape==(n,n,n))

    if buf is None:
        bdry_buffer = empty(min(1000000, 4*grid.size), dtype=int32) # 4 bytes
    else:
        bdry_buffer = require(buf, dtype=int32, order='C')

    lib = _initlib(log)

    res = lib.region_grow_domains3d(n, -min_sum, grid, bdry_buffer, 4*bdry_buffer.size)

    if res==-1:
        raise Exception('Ran out of memory on floodfill')

    return grid
