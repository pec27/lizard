from numpy import *
from numpy.random import random
from scipy.spatial import cKDTree
import cPickle as pickle
def neighbour_vecs(pos, neighbours, boxsize):
    ngb = neighbours.shape[1]
    dx = empty(neighbours.shape+(pos.shape[1],), dtype=pos.dtype)
    for i in range(ngb):
        dx[:,i] = remainder(pos[neighbours[:,i]] - pos + boxsize/2, boxsize) - boxsize/2

    return dx

def make_glass(n=32, ndims=2):

    if ndims==2:
        h = 0.75
        ngb = 30
        f0 = 0.03
    if ndims==3:
        h = 0.4
        ngb = 40
        f0 = 0.05
    
    if ndims==2:
        pts = mgrid[:n,:n]
    elif ndims==3:
        pts = mgrid[:n,:n,:n]

    pos = reshape(pts, (ndims,n**ndims)).T.astype(float32)   
    # offset each index a little bit
    pos += (random(pos.shape) - 0.5) #* 0.5
    print 'Pos', pos.shape
    # Find the periodic neighbours, ignoring self

    shift_pos = empty_like(pos)

    ngb_indices = empty((pos.shape[0], ngb), dtype=uint32)
    for corner in range(2**ndims):
        sides = tuple((corner>>axis)&1 for axis in range(ndims))
        print 'corner', corner, 'sides', sides
        for axis, side in enumerate(sides):
            shift_pos[:,axis] = remainder(pos[:,axis]+n/4+side*n/2, n)

        # Find the neighbours for this sub_box        
        # For all the indices we look at, we should be in the 'center' of the box
        print 'Building tree'
        tree = cKDTree(shift_pos)
        indices = arange(n/2) + sides[0]*n/2
        for i, side in enumerate(sides[1:]):
            indices = add.outer(indices*n, arange(n/2) + side*n/2).ravel()

        ### WARNING - do not remove the 'copy', otherwise the cKDTree will fail (doesn't cope with indexed arrays)
        subbox_pos = shift_pos[indices].copy()
        dist, ngb_idx = tree.query(subbox_pos, ngb+1) # Distance and neighbour index
        ngb_indices[indices] = ngb_idx[:,1:] # Ignore self

    
    # Do 100 iterations of 'anti-gravity'
    rmax = 3.0
    pos0 = pos.copy()
    min_dist_last = 0.0
    inv_h2 = 1.0 / (h*h)
    for i in range(5000):
        
        dx = neighbour_vecs(pos, ngb_indices, n)
        dx2 = square(dx).sum(2)        
        min_dist = sqrt(dx2.min())
        max_dist = sqrt(dx2.max())
        f = dx2 * inv_h2
        mult = where(f<3*3, -f0*exp(-0.5*f), 0.0) 
        mult.shape = dx2.shape+(1,)
        vel = (dx * mult).sum(1) # particle velocity
        vel_max = sqrt(square(vel).sum(1).max())
        print 'Step %d maximum vel %5.4f minimum distance %5.4f max dist %5.4f'%(i, vel_max, min_dist, max_dist), pos.dtype
        pos += vel
        pos = remainder(pos, n)
        if min_dist_last>min_dist:
            break
        min_dist_last = min_dist
#        print 'dx', dx[:,indices]

    pos *= 1.0 / n
    return pos 

def test():
    pos = make_glass(16, 3)
    
    import pylab as pl
    pl.plot(pos[:,0], pos[:,1], 'b,')
#    pl.plot(pos0[:,0], pos0[:,1], 'r,')
    pl.show()
    
def glass_file(n=32):
    pos = make_glass(n, 3)
    name = 'out/glass%d.dat'%n
    print 'Writing to file', name
    f = open(name, 'wb')
    pickle.dump(pos, f, 2)
    f.close()
    print 'done'
if __name__=='__main__':

    glass_file(64)
