from __future__ import print_function
from lizard.uniform import *

def test_uniform_box():

    boxsize = 64.0
    np = 32
    pts, sizes = uniform_box(boxsize, np) # 64 Mpc/h box, 64^3 particles
    assert(pts.shape==(3,np**3))
    assert(sizes.shape==(np**3,))

#    save('out/uniform.dat', boxsize, pts, sizes)

def test_zoom():
    """ 
    A test for making a zoom-region (spherical), uncomment the lines at the
    end if you want to save it
    """
    print('Building tree')
    boxsize = 50
    rmin = 5.0
    grid = 1024
    uni_pts, uni_sizes = make_zoom_uniform((25.0,25.0,25.0), rmin, grid, boxsize)
    #    name='/store/stellcomp/group/pcreasey/lizard_ics/data/zoom_5Mpc_1024.dat', grid=1024)
    # name='/store/stellcomp/group/pcreasey/lizard_ics/data/zoom_5Mpc_2048.dat', grid=2048)
    #    save(name, boxsize, uni_pts, uni_sizes)

def test_zoom_ellipsoid():
    """ 
    A test for making an ellipsoidal zoom-region, uncomment the lines at the
    end if you want to save it
    """
    boxsize = 50
    grid = 2048/2
    A = array([[6.85659873e-02, 2.31305793e-02,-4.64576308e-02],
               [  2.31305793e-02,  4.74549973e-02, -1.91083511e-02],
               [ -4.64576308e-02,  -1.91083511e-02,  9.55905177e-02]])
    ctr = array([28.8443,  25.5187,  19.6828])
    print('Building tree')
    uni_pts, uni_sizes = make_zoom_uniform_ellipsoid(ctr, A, grid, boxsize)
#    name='/store/stellcomp/group/pcreasey/lizard_ics/data/zoom_5Mpc_2048_ellipse.dat'
#    save(name, boxsize, uni_pts, uni_sizes)
    
if __name__=='__main__':
    test_uniform_box()
    test_zoom()
    test_zoom_ellipsoid()
    
    
