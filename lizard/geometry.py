"""
Utilities for bounding the volume for high-resolution discretisation in a zoom
simulation

Peter Creasey Mar 2015

Notes: 
The basic approach for cosmological simulations is to conduct a low resolution
dark-matter only simulation, identify particles that fall in the volume of
interest and then trace these back to their origins at high redshift. The
volume contained by the convex hull that encloses these points is then 
generally a good bound for the volume that should be discretised at high
resolution.

In order to simplify things my current approach is to bound this volume again
with its own ellipsoid. Numerically speaking this was probably not the best
approach, as finding the distance from a point to an ellipsoid (i.e. the 
limiting computation when deciding whether to refine boundary particles/cells)
is not straighforward (unlike it is for spheres), and in terms of computational
time it would be quicker to keep the boundary facets and search those (perhaps
in some kind of hash-table). Nevertheless, the development time for this has
not been worth it so far.

Usage:
With an (N,3) array of points, call bounding_ellipsoid(points) to get the 
equation of the ellipsoid and the points on the convex hull.

"""

from numpy import pi,sin,cos, dot, diag, outer, sqrt, cumprod, float64, empty, ones,argmax
from numpy.linalg import inv, eigh, norm
from scipy.spatial import ConvexHull
from time import time

def bounding_ellipsoid(points, tol=0.001):
    """
    Find the bounding ellipsoid of an (N,dim) array of points, first by finding
    the points that define the convex hull, and then using Khachiyans algorithm
    to determine the minimum-volume ellipsoid that bounds this.

    points   - (N,2) or (N,3) array of points
    [tol]    - optional tolerance of ellipsoid finding algorithm

    Returns: A, k, convex_p
    
    A,c      - 3x3 array, (3,) vector s.t. (x-c).A.(x-c) = 1 defines the surface 
               of the ellipsoid
    convex_p - points on the convex hull

    Notes:
    The number of points making up the convex hull of N points is O(log^(d-1) N)
    (see Dwyer 1987). In my tests this is about 5 log^2 N in 3d, i.e. 

    1,000,000 points => ~1,000 boundary points
    10^20 points     => ~10,000 boundary points
    so it is really worth it.
    """
    num_points, dim = points.shape


    print 'Finding the convex hull containing the {:,} points'.format(num_points)
    hull = ConvexHull(points)
    sub_points = points[hull.vertices].copy()
    print 'Convex hull contains {:,} points. Calculating the minimum volume ellipsoid'.format(len(sub_points))
    A, c = mvee(sub_points)
    evals, evecs = eigh(A)
    ax_lens = 1.0/sqrt(evals) # Semi-axes
    print 'Axes of ellipsoid have (semi)lengths',ax_lens
    prod = cumprod(ax_lens)[-1] # Product of the semi-axes
    if len(ax_lens)==2:
        print 'Area of ellipse is %e, or equivalent to a circle of radius %e'%(prod * pi, sqrt(prod))
    elif len(ax_lens)==3:
        print 'Volume of ellipsoid is %f, or equivalent to a sphere of radius %f'%(4.0*pi * prod/3.0, prod**(1.0/3.0))
    return A,c,sub_points
    
def mvee(points, tol = 0.001):
    """
    Iterative algorithm to find the minimum volume enclosing ellipsoid
    
    points - (N, d) array where d is the dimension
    [tol]  - optional tolerance

    returns A,c
    Finds the ellipse equation in "center form"
    (x-c).T * A * (x-c) = 1

    Note that if you include internal points it is still correct, but just slower.

    """
    N, d = points.shape
    Q = empty((d+1, N), dtype=points.dtype)
    Q[:d] = points.T
    Q[d] = 1

    err = tol+1.0
    u = ones(N)/N
    step = 0
    while err > tol:
        step = step + 1

        # assert u.sum() == 1 # invariant
        X = dot(Q*u, Q.T)
        M = diag(dot(dot(Q.T, inv(X)), Q))
        jdx = argmax(M)
        step_size = (M[jdx]-d-1.0)/((d+1)*(M[jdx]-1.0))
        new_u = (1-step_size)*u
        new_u[jdx] += step_size
        err = norm(new_u-u)
        u = new_u
    print 'Total steps', step
    c = dot(u,points)        
    A = inv(dot(points.T * u, points) - outer(c,c))/d
    return A, c


def test_large():
    from numpy.random import RandomState
    from numpy import linspace
    rs = RandomState(seed=123)
    n = 1000000
    xy = rs.rand(n*2)
    xy.shape = (n,2)
    xy[:,0] = 1.8 * xy[:,0] + 0.2

    t0 = time()
    A,c,edge_points = bounding_ellipsoid(xy)
    t1 =time()

    print 'A', A
    print 'Centre', c

    evals, evecs = eigh(A)
    ax_lens = 1.0/sqrt(evals)
    ax0 = evecs[:,0]
    ax1 = evecs[:,1]
    theta = linspace(0,2*pi, 1000)
    x_e = c[0]+ ax_lens[0] * cos(theta) * ax0[0] + ax_lens[1] * sin(theta)*ax1[0]
    y_e = c[1]+ax_lens[0] * cos(theta) * ax0[1] + ax_lens[1] * sin(theta)*ax1[1]
    print 'axis lengths', ax_lens

    

    print 'in', t1-t0, 'seconds'
    import matplotlib.pyplot as pl
    pl.plot(edge_points[:,0], edge_points[:,1], 'bx')
    pl.plot(x_e,y_e, 'r')
    pl.show()

def test3d():
    from numpy.random import RandomState
    from numpy import linspace
    rs = RandomState(seed=123)
    n = 100000
    xy = rs.rand(n*3)
    xy.shape = (n,3)
    xy[:,0] = 1.8 * xy[:,0] + 0.2
    xy[:,2] = xy[:,2]*0.5

    t0 = time()
    A,c,edge_points = bounding_ellipsoid(xy)
    t1 =time()

    print 'A', A
    print 'Centre', c

    evals, evecs = eigh(A)
    ax_lens = 1.0/sqrt(evals)
    ax0 = evecs[:,0]
    ax1 = evecs[:,1]
    ax2 = evecs[:,2]
    # ignore the axis closest to z
    ig = argmax(abs(evecs[2]))
    print 'ignoring axis',ig
    if ig==0:
        ax0=ax2
    elif ig==1:
        ax1=ax2

    theta = linspace(0,2*pi, 1000)
    x_e = c[0]+ ax_lens[0] * cos(theta) * ax0[0] + ax_lens[1] * sin(theta)*ax1[0]
    y_e = c[1]+ax_lens[0] * cos(theta) * ax0[1] + ax_lens[1] * sin(theta)*ax1[1]
    print 'axis lengths', ax_lens

    

    print 'in', t1-t0, 'seconds'
    import matplotlib.pyplot as pl
    pl.plot(edge_points[:,0], edge_points[:,1], 'bx')
    pl.plot(x_e,y_e, 'r')
    pl.show()

if __name__=='__main__':
    test_large()
    test3d()

