from __future__ import print_function, division
from lizard.zoom import *
from time import time

def test_memory_overflow():
    # Make a big tree with little memory

    try:
        tree = make_BH_tree((0.5,0.4,0.4), 0.2, max_nodes=100,max_refinement=11, bh_crit=0.1)
    except Exception:
        print('Caught the momory exception, good.')
    else:
        raise Exception('Why no memory error?')



def test_large():
    nref = 10 # used to be 11 but too much for an automated test
    print('Building tree')
    boxsize = 50
    zval = 20.1
    rmin = 7.0
    max_nodes = 100000000
    levels = [nref-4,nref-2,nref-1,nref]
    print('Making tree')
    zoom_centre = array((25,20,20.1))

    tree = make_BH_tree(zoom_centre/boxsize, rmin/boxsize, bh_crit=0.1, max_nodes=max_nodes,allowed_levels=levels)
    print('Number of nodes {:,}'.format(len(tree)))
    assert(len(tree)==3095461)
    print('Finding centres')
#    depth, centre = leaf_props(tree, boxsize)
    depth, centre = leaf_depths_centres(tree)
    print('done')
#    centre *= boxsize
    print('Total leaves', "{:,}".format(len(depth)))
    assert(len(depth)==21668228)
    l_depths = sorted(unique(depth))
    num_at_depth = [len(flatnonzero(depth==d)) for d in l_depths]

    for d,n in zip(l_depths, num_at_depth):
        print('Number of leaves of depth %2d:'%d,"{:15,}".format(n))

    print('Finding low-res particles')
    idx = flatnonzero(depth<nref)
    print('Calculating distance from zoom centre')
    r2 = square(centre[idx]*boxsize - zoom_centre).sum(1)
    print('Minimum dist', sqrt(r2.min()))

def test_grav_tree():
    from numpy.random import RandomState
    r = RandomState(123)

    print('Building tree')
    boxsize = 50.0
    zval = 20.1
    rmin = 7
    nref = 9
    zoom_centre = array((25,20,zval)) / boxsize
    levels = [nref-4,nref-2,nref-1,nref]
    max_nodes = 100000000
    print('Making tree')

    tree = make_BH_tree(zoom_centre, rmin/boxsize, max_nodes=max_nodes,bh_crit=0.1, allowed_levels=levels)
    print('Number of nodes {:,}'.format(len(tree)))
    assert(len(tree)==660115)
    print('Finding centres')
#    depth, centre = leaf_props(tree, boxsize)
    depth, centre = leaf_depths_centres(tree)
    print('done')
    print("Number of high-resolution particles {:,}".format(len(flatnonzero(depth==nref))))


    n=262144//16
    pos = r.rand(3*n)
    pos.shape=(n,3)
    centre = pos
    print('Pos in', pos.min(), pos.max())
    n_pos = len(centre)
    print('Number of positions {:,}'.format(n_pos), 'mean interparticle spacing %6.4f'%(n_pos**(-1.0/3.0)))
#    masses = pos[:,0]*0+1
    print('Building octree from these')    
    octree = build_octree(centre)
    print('Tree shape', octree.shape)
    
    mass = ones(centre.shape[0])
    print('Building mass, CoM for nodes')
    mcom = octree_mcom(octree, mass, centre)
    print('Sum of masses', mass.sum(dtype=float64), 'for node 0', mcom[n_pos,0])
    print('Mean position', centre.mean(axis=0, dtype=float64), 'for node 0', mcom[n_pos,1:])
    print('Finding some particles in a small region')

    wanted = 100000
    box_w = (float(wanted)/n_pos)**0.33333
    print('Box width', box_w)
    idx = neighbours_for_node(octree, n_pos, n_pos*10, (0.4,0.4,0.4), box_w)
    print('idx in', idx.min(), idx.max())
    print('Centres in', centre[idx].min(axis=0), centre[idx].max(axis=0))
    print(len(idx), 'found')
    print('Building kernel')

    # 1/r2 kernel
    kshort_rmin, kshort_rmax = 7e-05,0.052
    kshort_vals = kshort_rmin**(-2) * exp(-2*linspace(0, log(kshort_rmax/kshort_rmin), 150)) 
    print('Short range kernel from', kshort_rmin, 'to', kshort_rmax)
    print('Performing kernel evaluations')
    
    t = time()
    evals, res = kernel_evaluate(centre[idx].copy(), octree, mcom, kshort_rmin, kshort_rmax, kshort_vals)
    rate =  int(evals/ (time()-t))
    eval_per_p = evals / len(idx)
    print('{:,} kernel evaluations, or {:,}/s, {:,}/particle'.format(evals, rate, eval_per_p))
#    chunk_tree(octree, n_pos, n_pos/10)
    

