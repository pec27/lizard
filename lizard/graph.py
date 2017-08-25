"""
Utilities to graph the displacement fields
"""
from iccpy.figures import latexParams
import pylab as pl
from numpy import arange, unique, flatnonzero

def draw_disp_grid(boxsize, grid):
    """ draw a slice of the displacement grid """
    pl.rcParams.update(latexParams)
    pl.figure(figsize=(6.64,6.64), dpi=100)

    limit = max(-grid.min(), grid.max())
    norm = pl.Normalize(vmin=-limit, vmax=limit)
    
    slice = grid[:,0]
    pl.subplot(221)
    n = grid.shape[1]
    x = arange(n)*(boxsize/n)
    pl.imshow(slice[0], extent=(0,boxsize, 0, boxsize), norm=norm)
    pl.colorbar().set_label(r'$\rm Mpc/h$')
    pl.ylabel(r'$y \; \rm Mpc/h$')
    pl.title(r'$\Psi_x$')

    pl.subplot(222)
    pl.imshow(slice[1], extent=(0,boxsize, 0, boxsize), norm=norm)

    pl.title(r'$\Psi_y$')
    pl.colorbar().set_label(r'$\rm Mpc/h$')

    pl.subplot(223)

    pl.imshow(slice[2], extent=(0,boxsize, 0, boxsize), norm=norm)
    pl.colorbar().set_label(r'$\rm Mpc/h$')
    #pl.ylabel(r'$y \; \rm Mpc/h$')
    pl.title(r'$\Psi_z$')

    pl.show()

def draw_pts(boxsize, pts, sizes):
    pl.rcParams.update(latexParams)

    part_sizes = sorted(unique(sizes))
    syms = (',', '.', 'o')

    for i,size in enumerate(part_sizes):
        sym = syms[i%len(syms)]
        pts_i = pts[size==sizes]
        print 'Drawing', pts_i.shape[0], 'points'

        pl.plot(pts_i[:,0], pts_i[:,1], sym, c='k')
    pl.xlabel(r'$x \; \rm Mpc/h$')
    pl.show()
