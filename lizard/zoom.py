"""
Create a 'zoom' region (subdivide the unit cube until some volume at a given
depth).
"""

from .lizard_c import make_BH_tree, leaf_depths_centres, build_gravity_octtree, count_leaves, neighbours_for_node, build_octree, octree_mcom, kernel_evaluate
from numpy import *

def decompose_tree(tree, node, xyz0, cell_hw, n_leaves, leaves_per_node, max_leaves):
    """
    Recursively decompose the node until we reach max_leaves per node
    Return a tuple of nodes which satisfies this


    """
    
    if leaves_per_node[node]<=max_leaves:
        # found!
        return ((node, xyz0, cell_hw),)

    found = []


    for ix,x in enumerate([xyz0[0],xyz0[0]+cell_hw]):
        for iy,y in enumerate([xyz0[1], xyz0[1]+cell_hw]):
            for iz,z in enumerate([xyz0[2],xyz0[2]+cell_hw]):
                octant = ix*4+iy*2+iz
        
                child=tree[node, octant]
                if child==n_leaves:
                    continue # empty
        
                if child>=0:
                    # direct child, we have to return this
                    return ((node, xyz0, cell_hw),)
                # otherwise a node
                found += decompose_tree(tree, -child, (x,y,z), cell_hw*0.5, n_leaves, leaves_per_node, max_leaves)

    return tuple(found)


        

    
def chunk_tree(octree, n_leaves, leaves_wanted):
    """
    Finds nodes of the tree such that each has <= the given number of elements
    """
    print('Finding number of leaves for each nodes')
    leaves_per_node = count_leaves(octree, n_leaves)
    
    print('Looking for', leaves_wanted, 'from each node')
    
    nodes = decompose_tree(octree, 0, (0,0,0), 0.5, n_leaves, leaves_per_node, leaves_wanted)
    print('Nodes found', len(nodes))
    nodes = sorted(nodes, key=lambda x: -leaves_per_node[x[0]])

    bad_node = n_leaves * 10
    dist = 1.25*4.5/512
    print('Rcut', dist)
    for n, xyz, chw in nodes:
        max_leaves = leaves_per_node[n]*2
#        leaves_in_box = neighbours_for_node(octree, n_leaves, bad_node, xyz, float(chw*2), max_leaves=max_leaves)
#        assert(leaves_per_node[n]==len(leaves_in_box))
        box0 = [c-dist for c in xyz]
        edge_parts = neighbours_for_node(octree, n_leaves, n, box0, dist*2+chw*2, max_leaves=max_leaves)
        n_edge = len(edge_parts)

        print('{:,} particles, edge {:,}, total {:,}'.format(leaves_per_node[n], n_edge,leaves_per_node[n]+n_edge), 'size', (chw+dist)*2)
        
#    num_children = get_num_children(
#    for octant in range(8):
        
    
def draw_square(centre, hw):
    x0 = centre[0] - hw
    x1 = centre[0] + hw
    y0 = centre[1] - hw
    y1 = centre[1] + hw
    return (x0,x1,x1,x0,x0), (y0,y0,y1,y1,y0)
    

def _test_draw():
    """ make a tree and draw the cells """
    import pylab as pl    
    print('Building tree')
    boxsize = 50
    zval = 20.1
    rmin = 1.0
    zoom_centre = array((25,20,zval), dtype=float64)
    print('Making tree')
    tree = make_BH_tree(zoom_centre/boxsize, rmin/boxsize, bh_crit=0.9, allowed_levels=[3,5,6,7])
    print('Number of nodes', len(tree))

    print('Finding centres')
#    depth, centre = leaf_props(tree, boxsize)
    depth, centre = leaf_depths_centres(tree)
    print('done')

    centre *= boxsize
    hw = 0.5*boxsize * power(2.0,-depth) # half widths
    print('Number of leaves', len(depth))
    l_depths = sorted(unique(depth))
    num_at_depth = [len(flatnonzero(depth==d)) for d in l_depths]


    for d,n in zip(l_depths, num_at_depth):
        print('Number of leaves of depth %2d:'%d,"{:15,}".format(n))
    assert(num_at_depth==[476,2244,358,976])

    in_plane = flatnonzero(abs(centre[:,2]-zval)<hw)
    print('Number of squares to draw', len(in_plane))
    for i in in_plane:
        xpts, ypts = draw_square(centre[i], hw[i])
        pl.plot(xpts, ypts, 'r')
    pl.show()



if __name__=='__main__':
    _test_draw()

