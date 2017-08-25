/*
  Module for doing a fixed-radius nearest neighbour kernel summation, i.e.
  for every pair of points with distance |p1-p0|<R, summing the kernel
  (for point 0) of

  (p0-p1) * mass1 * kernel(|p1-p0|)

  i.e. the kernel is a compact function that is zero outside R.

  Methodology for this is to put particles in a lattice whose grid size is
  1R or 2R. The lattice cell is then found for each particle (see find_lattice
  in grid.c), the particles sorted by their lattice cell, then these cells
  put into a hash-table. Neighbouring particles can then be found just by
  looking up adjacent cells in the hash-table.

  An approximation is also given for highly clustered points where each (non-
  empty) cell has a tree, and the Barnes-Hut (opening angle) criterion is used
  to decide whether to apply the effect of the kernel from a tree-node en masse
  or break into its children (and ultimately individual points).

  Peter Creasey January 2015
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "tree.h" // For the tree-walk nodes

// Magic number that specifies how full the hash-table should be to be most efficient
// Hash tables perform very badly beyond about 0.7
#define DESIRED_LOAD 0.6

// Hash table primes from planetmath.org
#define MAX_TAB_NO 15 // number of table sizes
static const int HASHTABLE_SIZES[MAX_TAB_NO] = {1024,2048,4096,8192,16384,32768,65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216};
static const int HASH_PRIMES[MAX_TAB_NO] = {769,1543,3079,6151,12289, 24593,49157, 98317, 196613, 393241, 786433, 1572869, 3145739,6291469,12582917};

// kernel for force calc
#define MAX_KERNEL_WTS 5001 
static double kernel[MAX_KERNEL_WTS-1], diff_kernel[MAX_KERNEL_WTS-1], kernel_rcut, kernel_rcut2, kernel_r_to_idx;
static int kernel_num_wts;

int setup_hash_kernel(const double rcut, const int num_wts, const double *kernel_wts)
{
  /*
    Fill in the interpolation table for the kernel function used in find_ngb.
    Note that the r=0 point is always included, so you can only request up to 
    (MAX_KERNEL_WTS-1) points.
    
    On failure return -MAX_KERNEL_WTS
   */
  if (num_wts>=MAX_KERNEL_WTS)
    return -MAX_KERNEL_WTS; 
  kernel_num_wts = num_wts;
  kernel_rcut = rcut;
  kernel_rcut2 = kernel_rcut*kernel_rcut;
  kernel_r_to_idx = (double)kernel_num_wts/kernel_rcut;

  double kernel_prev = 0.0;
  for (int i=0;i<kernel_num_wts;i++)
    {
      diff_kernel[i] = kernel_wts[i]-kernel_prev;
      kernel[i] = kernel_prev - i*diff_kernel[i]; // Relative to y-intercept
      kernel_prev = kernel_wts[i];
    }
  return 0;
}

static inline int trim3x3x3(const double x, const double y, const double z, const double r2)
{
  /*
    Given a point (x,y,z) and a square distance r^2<=1, find the cells on the
    integer lattice that could contain points within r^2 only including 'left'
    cells (i.e. those whose key X*ngrid^2+Y*ngrid+Z is <= our own).

    In terms of the surrounding 3x3x3 block this test is
    
    ? ? ?     ? A ?     . . .
    ? A ?     A M .     . . .
    ? ? ?     . . .     . . .

    Key
    ? - test to see if this is within r^2
    A - Almost certainly adjacent, dont bother to test
    M - My cell!
    . - 'right' cell, ignore

    Returns an integer mask, where the bits correspond to the 'cell_shift' in find_ngbs
    
   */
  const double d1=x - floor(x), d2=y - floor(y), d3=z - floor(z);
  // Square distances to left and right cells
  const double d1L = d1*d1, d2L=d2*d2,d2R=(1-d2)*(1-d2), d3L=d3*d3,d3R=(1-d3)*(1-d3);
  // Test the adjacent cells to see if they are further than rcrit
  return (d1L+d2L+d3L<=r2) |  (d1L+d2L<=r2)<<1  | (d1L+d2L+d3R<=r2)<<2 |
    (d1L+d3L<=r2)<<3 | (d1L+d3R<=r2)<<5 |
    (d1L+d2R+d3L<=r2)<<6 |     (d1L+d2R<=r2)<<7 | (d1L+d2R+d3R<=r2)<<8 |
    (d2L+d3L<=r2)<<9 | (d2L+d3R<=r2)<<11 | 0x3410;

}

long radial_kernel_evaluate(const double *xyzw, const int num_cells, const int* cells, const int ngrid,
	       double *restrict accel)
{
  /* 
     Find the neighbours using the lattice based approach. Each of the given cells contains at least 1 pt
     cells - (num_cells x 3) array of cell lattice coord, start (in pos) and end (in pos)

     xyzw - (n*4) array of x,y,z,weight for each particle
     returns the number of pairs found
   */
  const long n2 = (long)ngrid * ngrid;
  const double rncrit2 = kernel_rcut2*n2;
  /* Relative indices in the grid of the bins of the adjacent cells */
  const int cell_shift[13] = {-n2-ngrid-1,-n2-ngrid,-n2-ngrid+1,
			      -n2-1,-n2,-n2+1,
			      -n2+ngrid-1,-n2+ngrid,-n2+ngrid+1,			      
			      -ngrid-1,-ngrid,-ngrid+1,
			      -1};

  long pair_count=0; // Counter the number of pairs (within kernel_rcut) we find.

  /*
    First build the hashtable of all the start and end indices
   */

  // Find the next power of 2 large enough to hold table at desired load
  int tab_size=0;
  while (tab_size<MAX_TAB_NO && HASHTABLE_SIZES[tab_size]*DESIRED_LOAD<num_cells) tab_size++;
  if (tab_size==MAX_TAB_NO)
    return -1; // Table too big

  const int hsize = HASHTABLE_SIZES[tab_size], hprime = HASH_PRIMES[tab_size];
  const int hmask = hsize-1;

  // Zero-ed hashtable (index + start + end for each cell)
  int *htable = calloc(hsize * 3, sizeof(int));
  


  for (int i=0;i<num_cells;i++)
    {
      // indices of start (i0) to end (i1) of 3x3x3=27 neighbours of each central cell
      int ngb_cell[14]; 
      int adj_mask=0x2000; // Neighbour list always contains me.

      // Find each of the neighbouring cells (if present) in the hashtable

      for (int adj=0;adj<13;adj++)
	{
	  const int wanted_cell = cells[i*3] + cell_shift[adj];
	  // Search for cell or 0
	  
	  for (int j=(wanted_cell*hprime)&hmask;htable[j*3];j=(j+1)&hmask)
	    {
	      if (~(htable[j*3]^wanted_cell))
		  continue;

	      adj_mask |= 1<<adj;
	      ngb_cell[adj] = j*3+1;
	      break;
	    }
	}

      // Insert my cell into hash table at the next free spot
      int ins;
      for (ins=(cells[i*3]*hprime)&hmask; htable[ins*3]; ins=(ins+1)&hmask);

      htable[ins*3] = ~cells[i*3]; // Bit twiddle to make sure never 0
      ngb_cell[13]  = ins*3+1;
      htable[ngb_cell[13]+1] = htable[ngb_cell[13]] = cells[i*3+1]; // set cursor to cell start

      // Go through every particle in this (non-empty) cell
      do {
	const int j = htable[ngb_cell[13]+1]; // first point in cell
	double acc_j[3] = {accel[j*3],accel[j*3+1],accel[j*3+2]};

	// Mask those cells in the block which are out-of-range of this particle
	int adj_mask_j = adj_mask & trim3x3x3(xyzw[j<<2]*ngrid, xyzw[j<<2|1]*ngrid, xyzw[j<<2|2]*ngrid, rncrit2);

	// Go over adjacent cells
	for (int adj=0;adj_mask_j;adj++, adj_mask_j>>=1)
	  {
	    if (!(adj_mask_j&1))
	      continue; // Cell out-of-range or empty
	    
	    for (int k=htable[ngb_cell[adj]]; k<htable[ngb_cell[adj]+1]; k++)
	      {
		// PERFORMANCE:
		// Perhaps surprisingly, although we only need float precision once we go to relative
		// coords, it is currently better to stay with doubles as the casts are so expensive.
		// Good to keep an eye on this though.
		
		const double dx = xyzw[j<<2] - xyzw[k<<2],
		  dy = xyzw[j<<2|1] - xyzw[k<<2|1],
		  dz = xyzw[j<<2|2] - xyzw[k<<2|2];
		const double r2 = dx*dx+dy*dy+dz*dz;
		if (r2>=kernel_rcut2) // Outside square radius, ignore
		  continue; 
		pair_count++;
		/* position in interpolation table */
		// PERFORMANCE: For 100 neighbours, the interpolation is taking 
		// about 15% of the time (compared to say 1/(sqrt(r2)*r2)), 
		// independent of table size. float makes it worse.
		
		const double f_interp = kernel_r_to_idx * sqrt(r2);
		const int i0 = f_interp; 
		const double acc_mult = kernel[i0] + f_interp*diff_kernel[i0];
		const double wt_k=xyzw[k<<2|3]*acc_mult, wt_j=xyzw[j<<2|3]*acc_mult;

		acc_j[0] += wt_k * dx;
		acc_j[1] += wt_k * dy;
		acc_j[2] += wt_k * dz;
		
		accel[k*3]   -= wt_j * dx;
		accel[k*3+1] -= wt_j * dy;
		accel[k*3+2] -= wt_j * dz;
	      }
	  }
	// Store acceleration j
	accel[j*3]   = acc_j[0];
	accel[j*3+1] = acc_j[1];
	accel[j*3+2] = acc_j[2];

      } while ((++htable[ngb_cell[13]+1])<cells[i*3+2]);
    }
  free(htable);
  return pair_count;
}
long radial_kernel_cellmean(const double *xyzw, const int num_cells, const int* cells, const int ngrid,
			      const int stencil, double *accel)
{
  /*
    Very similiar to radial_kernel_evaluate, but for non-adjacent cells uses the
    approximation that the contribution of those particles is close to that of
    the kernel evaluated at their mass weighted centre - i.e. if the kernel were
    Newtonian (1/r^2) this would be equivalent to taking the monopole 
    approximation.

    Arguments as for radial_kernel_evaluate, but with the addition of:
    
    monopole_stencil - Either 5 or 7 (for 5^3 or 7^3 stencils). 

    If 5, then kernel_rcut should be at most 2/n, (i.e. we look 2 grid cells 
    either side), if 7 then 3/n (i.e. 3 either side).

    Tests indicate that the turnover for using 5^3 is an average of 140 
    neighbours per particle, and for 7^3 is around 400, i.e. if your expected
    number of neighbours is is 0-140 use radial_kernel_evaluate, 140-400 use
    stencil=5 and >400 use stencil=7.


      Sum_i (x-x_i) * wt_i * kernel(|x-x_i|) ~= (x - CoM) * W * kernel(|x - CoM|)
    
    where 
      W   := Sum_i wt_i 
      CoM := Sum_i wt_i * x_i / W
     
    returns force_pairs - the number of force-pairs calculated (note this will
                          be significantly smaller than the number of pairs 
			  within r_crit due to the monopoles).

   */
  long pair_count=0; // Number of force pairs evaluated

  // A few constants (not for performance, just brevity!)
  const int n2 = ngrid*ngrid;
  const int I=-n2,II=-2*n2, III=-3*n2, JJJ=-3*ngrid, JJ=-2*ngrid, J=-ngrid;

  /*
    3x3x3 neighbours for the direct forces

    D D D     D D D     . . .
    D D D     D I .     . . .
    D D D     . . .     . . .

    I - internal cell (me, counted separately)
    D - Direct summation cell
    . - Index>me, counted later.

   */
  const int cell_shift_direct[13] = 
    {I+J-1, I+J, I+J+1,
     I-1  , I  , I+1,
     I-J-1, I-J, I-J+1,			      

       J-1,   J, J+1,
     -1};
  
  /*
    Neighbours to sum over with the monopole approximation.
    
    . . 7 7 7 . .    . 7 7 7 7 7 .    7 7 7 7 7 7 7    7 7 7 7 7 7 7    
    . 7 7 7 7 7 .    7 5 5 5 5 5 7    7 5 5 5 5 5 7    7 5 5 5 5 5 7    
    7 7 7 7 7 7 7    7 5 5 5 5 5 7    7 5 D D D 5 7    7 5 D D D 5 7    
    7 7 7 7 7 7 7    7 5 5 5 5 5 7    7 5 D D D 5 7    7 5 D I . . .    
    7 7 7 7 7 7 7    7 5 5 5 5 5 7    7 5 D D D 5 7    . . . . . . .
    . 7 7 7 7 7 .    7 5 5 5 5 5 7    7 5 5 5 5 5 7    . . . . . . .
    . . 7 7 7 . .    . 7 7 7 7 7 .    7 7 7 7 7 7 7    . . . . . . .


    I - internal cell (me), direct summation
    D - Direct summation cell (adjacent, counted before)
    5 - Monopole approximation in the 5x5x5 stencil (49)
    7 - Monopole approximation in the 7x7x7 stencil (142-49)
    . - Either index>me (counted later) or outside kernel

  */
  const int monopole_cells = (stencil==5) ? 49 : 142;

  if (stencil!=5 && stencil!=7)
    return -2; // Stencil size must be 5 or 7!



  const int cell_shift_monopole[142] =
  // 5x5x5 cells (see diagram above)
    {II+JJ-2, II+JJ-1, II+JJ, II+JJ+1, II+JJ+2,
     II +J-2, II +J-1, II +J, II +J+1, II +J+2,
     II   -2, II   -1, II   , II   +1, II   +2,
     II -J-2, II -J-1, II -J, II -J+1, II -J+2,
     II-JJ-2, II-JJ-1, II-JJ, II-JJ+1, II-JJ+2,

      I+JJ-2,  I+JJ-1,  I+JJ,  I+JJ+1, I+JJ+2,
      I +J-2,                          I +J+2,
      I   -2,                          I   +2,
      I -J-2,                          I -J+2,
      I-JJ-2,  I-JJ-1,  I-JJ,  I-JJ+1, I-JJ+2,

        JJ-2,    JJ-1,    JJ,    JJ+1,   JJ+2,
         J-2,                             J+2, 
          -2,

     // 7x7x7 (note some cells are now outside the sphere)
                        III+JJJ-1, III+JJJ, III+JJJ+1,
              III+JJ-2, III +JJ-1, III +JJ, III +JJ+1, III+JJ+2,
     III+J-3, III +J-2, III  +J-1, III  +J, III  +J+1, III +J+2, III+J+3,
     III  -3, III-   2, III    -1, III    , III    +1, III+   2, III  +3,
     III-J-3, III -J-2, III  -J-1, III  -J, III  -J+1, III -J+2, III-J+3,
              III-JJ-2, III -JJ-1, III -JJ, III -JJ+1, III-JJ+2,
                        III-JJJ-1, III-JJJ, III-JJJ+1,

     // next slab
              II+JJJ-2,  II+JJJ-1,  II+JJJ,  II+JJJ+1, II+JJJ+2,
     II+JJ-3,                                                    II+JJ+3,
     II +J-3,                                                    II +J+3,
     II   -3,                                                    II   +3,
     II -J-3,                                                    II -J+3,
     II-JJ-3,                                                    II-JJ+3,
                II-JJJ-2, II-JJJ-1, II-JJJ,  II-JJJ+1, II-JJJ+2,

     // only 2 two go...
      I+JJJ-3,   I+JJJ-2,  I+JJJ-1,  I+JJJ,   I+JJJ+1,  I+JJJ+2, I+JJJ+3,
      I +JJ-3,                                                   I +JJ+3,
      I  +J-3,                                                   I  +J+3,
      I    -3,                                                   I    +3,
      I  -J-3,                                                   I  -J+3,
      I -JJ-3,                                                   I -JJ+3,
      I-JJJ-3,   I-JJJ-2,  I-JJJ-1,  I-JJJ,   I-JJJ+1,  I-JJJ+2, I-JJJ+3,

     // you're almost there...
        JJJ-3,     JJJ-2,    JJJ-1,    JJJ,     JJJ+1,    JJJ+2,   JJJ+3,
         JJ-3,                                                      JJ+3,
          J-3,                                                       J+3,
           -3}; // Done! Total 49+37+20+24+12=142

  // Data for hash cells
  struct Hash_cell {
    int cell_fill; // bit twiddle of cell (never zero)
    int cell_start, cell_end; // Indices of where particles are in xyzw
    double xyzw[4]; // Centre-of-mass and mass

  };
  
  // Find the next power of 2 large enough to hold table at desired load
  int tab_size=0;
  while (tab_size<MAX_TAB_NO && HASHTABLE_SIZES[tab_size]*DESIRED_LOAD<num_cells) tab_size++;
  if (tab_size==MAX_TAB_NO)
    return -1; // Table too big

  const int hsize = HASHTABLE_SIZES[tab_size], hprime = HASH_PRIMES[tab_size];
  const int hmask = hsize-1;

  // Zero-ed hashtable for each cell
  struct Hash_cell *htable = calloc(hsize, sizeof(struct Hash_cell));

  /*
    Loop over each cell, doing the following
    1: Build mass and CoM for my cell
    2: Do direct sum for all particles in my cell
    3: Do direct sum of adjacent (3x3x3) cells
    4: Do monopole moment of 5x5x5 block
    5: Add cell to hash table
   */
    
  for (int i=0;i<num_cells;i++)
    {
      struct Hash_cell cur; // Current cell

      cur.cell_fill = ~cells[i*3]; // Bit twiddle to make sure never 0
      cur.cell_start = cells[i*3+1];
      cur.cell_end = cells[i*3+2];
      
      // Part 1: Build the mass and centre-of-mass for this cell
      if (cur.cell_start+1==cur.cell_end)
	{
	  // Only 1 particle in cell
	  cur.xyzw[0] = xyzw[cur.cell_start<<2];
	  cur.xyzw[1] = xyzw[cur.cell_start<<2|1];
	  cur.xyzw[2] = xyzw[cur.cell_start<<2|2];
	  cur.xyzw[3] = xyzw[cur.cell_start<<2|3];
	}
      else
	{
	  // Loop over particles
	  cur.xyzw[0] = 0;
	  cur.xyzw[1] = 0;
	  cur.xyzw[2] = 0;
	  cur.xyzw[3] = 0;

	  for (int j=cur.cell_start;j<cur.cell_end;j++)
	    {
	      cur.xyzw[0] += xyzw[j<<2|3]*xyzw[j<<2];
	      cur.xyzw[1] += xyzw[j<<2|3]*xyzw[j<<2|1];
	      cur.xyzw[2] += xyzw[j<<2|3]*xyzw[j<<2|2];
	      cur.xyzw[3] += xyzw[j<<2|3];
	    }
	  // Mass weighted centre
	  const double inv_wt = 1.0/cur.xyzw[3];  
	  cur.xyzw[0] *= inv_wt;
	  cur.xyzw[1] *= inv_wt;
	  cur.xyzw[2] *= inv_wt;

	  // Part 2: Do direct sum of internal
	  for (int j=cur.cell_start+1;j<cur.cell_end;j++)
	    {
	      for (int k=cur.cell_start;k<j;k++)
		{
		  const double dx = xyzw[j<<2] - xyzw[k<<2],
		    dy = xyzw[j<<2|1] - xyzw[k<<2|1],
		    dz = xyzw[j<<2|2] - xyzw[k<<2|2];
		  const double r2 = dx*dx+dy*dy+dz*dz;
		  if (r2>=kernel_rcut2) // Outside square radius, ignore
		    continue; 
		  pair_count++;
		  /* position in interpolation table */
		  const double f_interp = kernel_r_to_idx * sqrt(r2);
		  const int i0 = f_interp; 
		  const double acc_mult = kernel[i0] + f_interp*diff_kernel[i0];
		  const double wt_k=xyzw[k<<2|3]*acc_mult, wt_j=xyzw[j<<2|3]*acc_mult;
		  
		  accel[j*3]   += wt_k * dx;
		  accel[j*3+1] += wt_k * dy;
		  accel[j*3+2] += wt_k * dz;
		  
		  accel[k*3]   -= wt_j * dx;
		  accel[k*3+1] -= wt_j * dy;
		  accel[k*3+2] -= wt_j * dz;
		}
	    }
	}

      // Part 3: Direct sum of neighbouring 13 cells
      const struct Hash_cell *ngb; 

      for (int adj=0;adj<13;adj++)
	{
	  const int wanted_cell = cells[i*3] + cell_shift_direct[adj];
	  // Search for cell or 0
	  for (int look=(wanted_cell*hprime)&hmask; htable[look].cell_fill; look=(look+1)&hmask)
	    {
	      if (~(htable[look].cell_fill^wanted_cell))
		continue; // Not the cell I was looking for
	      // Found the neighbour
	      ngb = &htable[look];
	      goto ngb_direct;
	    }
	  // Cell not in table
	  continue;
	ngb_direct:
	  // Direct sum:
	  for (int j=cur.cell_start;j<cur.cell_end;j++)
	    {
	      for (int k=ngb->cell_start;k<ngb->cell_end;k++)
		{
		  const double dx = xyzw[j<<2] - xyzw[k<<2],
		    dy = xyzw[j<<2|1] - xyzw[k<<2|1],
		    dz = xyzw[j<<2|2] - xyzw[k<<2|2];
		  const double r2 = dx*dx+dy*dy+dz*dz;
		  if (r2>=kernel_rcut2) // Outside square radius, ignore
		    continue; 
		  pair_count++;
		  /* position in interpolation table */
		  const double f_interp = kernel_r_to_idx * sqrt(r2);
		  const int i0 = f_interp; 
		  const double acc_mult = kernel[i0] + f_interp*diff_kernel[i0];
		  const double wt_k=xyzw[k<<2|3]*acc_mult, wt_j=xyzw[j<<2|3]*acc_mult;
		  
		  accel[j*3]   += wt_k * dx;
		  accel[j*3+1] += wt_k * dy;
		  accel[j*3+2] += wt_k * dz;
		  
		  accel[k*3]   -= wt_j * dx;
		  accel[k*3+1] -= wt_j * dy;
		  accel[k*3+2] -= wt_j * dz;
		}
	    }
	}

      // Part 4: Monopole sum for non-adjacent cells
      for (int adj=0;adj<monopole_cells;adj++)
	{
	  const int wanted_cell = cells[i*3] + cell_shift_monopole[adj];
	  
	  // Search for cell or 0
	  for (int look=(wanted_cell*hprime)&hmask; htable[look].cell_fill; look=(look+1)&hmask)
	    {
	      if (~(htable[look].cell_fill^wanted_cell))
		continue; // Not the cell I was looking for
	      // Found the neighbour
	      ngb = &htable[look];
	      goto ngb_monopole; 
	    }
	  // Cell didnt exist in table (=>empty)
	  continue;
	ngb_monopole:
	  // Apply the grav from neighbouring CoM to my particles
	  for (int j=cur.cell_start;j<cur.cell_end;j++)
	    {
	      const double dx = xyzw[j<<2] - ngb->xyzw[0],
		dy = xyzw[j<<2|1] - ngb->xyzw[1],
		dz = xyzw[j<<2|2] - ngb->xyzw[2];
	      const double r2 = dx*dx+dy*dy+dz*dz;
	      if (r2>=kernel_rcut2) // Outside square radius, ignore
		continue; 
	      pair_count++;
	      // position in interpolation table 
	      const double f_interp = kernel_r_to_idx * sqrt(r2);
	      const int i0 = f_interp; 
	      const double acc_mult = kernel[i0] + f_interp*diff_kernel[i0];
	      const double wt_ngb=ngb->xyzw[3]*acc_mult;
	      
	      accel[j*3]   += wt_ngb * dx;
	      accel[j*3+1] += wt_ngb * dy;
	      accel[j*3+2] += wt_ngb * dz;
	    }
	  // ... apply the grav from my CoM to neighbours
	  for (int j=ngb->cell_start;j<ngb->cell_end;j++)
	    {
	      const double dx = xyzw[j<<2] - cur.xyzw[0],
		dy = xyzw[j<<2|1] - cur.xyzw[1],
		dz = xyzw[j<<2|2] - cur.xyzw[2];
	      const double r2 = dx*dx+dy*dy+dz*dz;
	      if (r2>=kernel_rcut2) // Outside square radius, ignore
		continue; 
	      pair_count++;
	      // position in interpolation table 
	      const double f_interp = kernel_r_to_idx * sqrt(r2);
	      const int i0 = f_interp; 
	      const double acc_mult = kernel[i0] + f_interp*diff_kernel[i0];
	      const double wt_ngb=cur.xyzw[3]*acc_mult;
	      
	      accel[j*3]   += wt_ngb * dx;
	      accel[j*3+1] += wt_ngb * dy;
	      accel[j*3+2] += wt_ngb * dz;
	    }
	  // Done with this neighbour, continue
	}

      // Part 5: Add my cell to the hash table
      int j = (cells[i*3] * hprime) & hmask;
      // Find next free spot
      while (htable[j].cell_fill) 
	j = (j+1)&hmask;
      htable[j] = cur;

    }

  /*
    Done
   */ 
  
  free(htable);
  return pair_count;
}

static inline long trim5x5x5(const double x, const double y, const double z, const float r2)
{
  /*
    As for trim3x3x3 except r^2<=2, and thus we neighbours could be in *2*
    cells either side, i.e. we need look in the 5x5x5 block:
    
    ? ? ? ? ?    ? A A A ?    ? A A A ?    . . . . .    . . . . .
    ? A A A ?    A A A A A    A A A A A    . . . . .    . . . . .
    ? A A A ?    A A A A A    A A M . .    . . . . .    . . . . .
    ? A A A ?    A A A A A    . . . . .    . . . . .    . . . . .
    ? ? ? ? ?    ? A A A ?    . . . . .    . . . . .    . . . . .

    Key
    ? - test to see if this is within r^2
    A - Almost certainly adjacent, dont bother to test
    M - My cell!
    . - 'right' cell, ignore

    Returns an integer mask, where the bits correspond to the 'cell_shift' in find_ngbs

   */

  // Fractional part of x,y,z
  const double d1=x - floor(x), d2=y - floor(y), d3=z - floor(z);
  // Square distance to left and right planes of integer lattice (LL refers to 2nd
  // left etc.)
  const double d1L = d1*d1, d2L=d2*d2,d2R=(1-d2)*(1-d2), d3L=d3*d3,d3R=(1-d3)*(1-d3),
    d1LL = (1+d1)*(1+d1), d2LL=(1+d2)*(1+d2),d2RR=(2-d2)*(2-d2), d3LL=(1+d3)*(1+d3),d3RR=(2-d3)*(2-d3);  
  // Currently just do the obvious ones...
  return 0xdc0739c0 | (d1LL+d2LL+d3LL<=r2) | (d1LL+d2LL+d3L<=r2)<<1 | 
    (d1LL+d2LL<=r2)<<2 | (d1LL+d2LL+d3R<=r2)<<3 | (d1LL+d2LL+d3RR<=r2)<<4 | 
    (d1LL+d2L+d3LL<=r2)<<5 | (d1LL+d2L+d3RR<=r2)<<9 |  (d1LL+d3LL<=r2)<<10 |
    (d1LL+d3RR<=r2)<<14 | (d1LL+d2R+d3LL<=r2)<<15 | (d1LL+d2R+d3RR<=r2)<<19 |
    (d1LL+d2RR+d3LL<=r2)<<20 | (d1LL+d2RR+d3L<=r2)<<21 | (d1LL+d2RR<=r2)<<22 | 
    (d1LL+d2RR+d3R<=r2)<<23 | (d1LL+d2RR+d3RR<=r2)<<24 | (d1L+d2LL+d3LL<=r2)<<25 | 
    (d1L+d2LL+d3RR<=r2)<<29 |
    (long)(0x7fb9dfff | (d1L+d2RR+d3LL<=r2)<<13 | (d1L+d2RR+d3RR<=r2)<<17 |
	   (d2LL+d3LL<=r2)<<18 | (d2LL+d3RR<=r2)<<22 )<<32;
}

long find_ngbs5x5(const double *mpos, const int num_cells, const int* cells, const double rcut, const int ngrid,
	       double *accel)
{
  /* 
     As for find_ngbs but this time using 2 nearest cells (hence 5x5x5 grid), and rcut*ngrid <=2
   */
  const float rcut2 = rcut*rcut;
  const long n2 = (long)ngrid * ngrid;
  const double rncrit2 = rcut2*n2;
  /* Relative indices in the grid of the bins of the adjacent cells */

  const int cell_shift[63] = {-2*n2-2*ngrid-2, -2*n2-2*ngrid-1, -2*n2-2*ngrid, -2*n2-2*ngrid+1, -2*n2-2*ngrid+2, 
			      -2*n2-ngrid-2, -2*n2-ngrid-1, -2*n2-ngrid, -2*n2-ngrid+1, -2*n2-ngrid+2, 
			      -2*n2-2, -2*n2-1, -2*n2, -2*n2+1, -2*n2+2, 
			      -2*n2+ngrid-2, -2*n2+ngrid-1, -2*n2+ngrid, -2*n2+ngrid+1, -2*n2+ngrid+2, 
			      -2*n2+2*ngrid-2, -2*n2+2*ngrid-1, -2*n2+2*ngrid, -2*n2+2*ngrid+1, -2*n2+2*ngrid+2, 
			      // next 5x5
			      -n2-2*ngrid-2, -n2-2*ngrid-1, -n2-2*ngrid, -n2-2*ngrid+1, -n2-2*ngrid+2, 
			      -n2-ngrid-2, -n2-ngrid-1, -n2-ngrid, -n2-ngrid+1, -n2-ngrid+2, 
			      -n2-2, -n2-1, -n2, -n2+1, -n2+2, 
			      -n2+ngrid-2, -n2+ngrid-1, -n2+ngrid, -n2+ngrid+1, -n2+ngrid+2, 
			      -n2+2*ngrid-2, -n2+2*ngrid-1, -n2+2*ngrid, -n2+2*ngrid+1, -n2+2*ngrid+2, 
			      // Tetris brick of remainder
			      -2*ngrid-2, -2*ngrid-1, -2*ngrid, -2*ngrid+1, -2*ngrid+2, 
			      -ngrid-2, -ngrid-1, -ngrid, -ngrid+1, -ngrid+2, 
			      -2, -1, 0};

  int adj_i0_i1[2*63]; // Start and end indices of particles in 63 shifted cells
  long ngb =0; // Counter for the number of neighbours we find

  
  // First build the hashtable of all the start and end indices

  // Find the next power of 2 large enough to hold table at desired load
  int tab_size=0;
  while (tab_size<MAX_TAB_NO && HASHTABLE_SIZES[tab_size]*DESIRED_LOAD<num_cells) tab_size++;
  if (tab_size==MAX_TAB_NO)
    return -1; // Table too big

  const int hsize = HASHTABLE_SIZES[tab_size], hprime = HASH_PRIMES[tab_size];
  const int hmask = hsize-1;

  printf("Wanted table to fit %d cells, used size %d\n",num_cells, (int)hsize);
  // Zero-ed hashtable (index + start + end for each cell)
  int *htable = calloc(hsize * 3, sizeof(int));
  // Insert the cells
  for (int i=0;i<num_cells;i++)
    {
      int j = (cells[i*3] * hprime) & hmask;
      // Find next free spot
      while (htable[j*3]) j = (j+1)&hmask;
      htable[j*3] = ~cells[i*3]; // Bit twiddle to make sure never 0
      htable[j*3+1] = cells[i*3+1];
      htable[j*3+2] = cells[i*3+2];
    }


  // Done building hashtable.

  for (int i=0;i<num_cells;i++)
    {
      long adj_mask=1L<<62; // Current cell is always found

      // Find each of the neighbouring cells in the hashtable
      for (int adj=0;adj<62;adj++)
	{
	  const int wanted_cell = cells[i*3] + cell_shift[adj];
	  // Search for cell or 0
	  for (int j= (wanted_cell*hprime)&hmask;htable[j*3];j=(j+1)&hmask)
	    {
	      if (~(htable[j*3]^wanted_cell))
		continue;

	      adj_mask |= 1L<<adj;
	      adj_i0_i1[adj*2] = htable[j*3+1];
	      adj_i0_i1[adj*2+1] = htable[j*3+2];
	      break;

	    }
	}
      int j=adj_i0_i1[2*62] = cells[i*3+1]; // neighbour list for current cell (max always set to j)


      // Go through particles in this cell
      do {
	adj_i0_i1[125]=j; // Don't exceed current particle
	// Mask those cells in the block which are out-of-range
	long adj_mask_j = adj_mask & trim5x5x5(mpos[j<<2|1]*ngrid, mpos[j<<2|2]*ngrid, mpos[j<<2|3]*ngrid, rncrit2);

	// Go over cells
	for (int adj=0;adj_mask_j;adj+=2, adj_mask_j>>=1)
	  {
	    if (!(adj_mask_j&1))
	      continue; // Cell out-of-range or empty
	    
	    for (int k=adj_i0_i1[adj]; k<adj_i0_i1[adj+1]; k++)
	      {
		// Perhaps surprisingly, although we only need float precision once we go to relative
		// coords, it is currently better to stay with doubles the casts are so expensive.
		// Good to keep an eye on this though.
		
		const double dx = mpos[j<<2|1] - mpos[k<<2|1],
		  dy = mpos[j<<2|2] - mpos[k<<2|2],
		  dz = mpos[j<<2|3] - mpos[k<<2|3];
		
		const double r2 = dx*dx+dy*dy+dz*dz;
		if (r2>=rcut2) // Outside square radius, ignore
		  continue; 
		
		ngb+=2;
		// TODO add the part for the kernel summation
		const double acc_mult = 1.0/sqrt(r2);
		const double mass_k=mpos[k<<2], mass_j=mpos[j<<2];
		
		accel[j*3+0] += mass_k * (acc_mult*dx);
		accel[j*3+1] += mass_k * (acc_mult*dy);
		accel[j*3+2] += mass_k * (acc_mult*dz);
		
		accel[k*3]   -= mass_j * (acc_mult*dx);
		accel[k*3+1] -= mass_j * (acc_mult*dy);
		accel[k*3+2] -= mass_j * (acc_mult*dz);
	      }
	  }
      } while ((++j)<cells[i*3+2]);
    }
  free(htable);

  return ngb;
}

void ngb_cell_masks_3x3x3(const int num_pts, const double rcrit, const int *cells,
			  const uint64_t *ngb_masks, const double *pts, uint64_t *out)
{
  /*
    Construct a bitmask for each point defined as the mask of it's cell OR-ed
    with the masks of those cells it is within rcrit of, in the 3x3x3-1
    neighbouring cube.

    num_pts    - N points
    rcrit      - Radius within which we mark cells
    *cells     - N indices into ngb_masks
    *ngb_masks - X*27 array of 3x3x3 neighbours (C-order) indexed via *cells
    *pts       - N*3 array of positions in [0,1)
    *out       - store the result

    The primary purpose of this routine is to identify the 'ghost' particles 
    which neighbour a domain - i.e. you split the volume into coarse cells, each
    of which is identified with a bit, (e.g. 1<<5) to indicate which process
    it needs to be sent to (maximum 32 processes on a single node). The masks
    are then logical-OR-ed together. 

    Performance tips:
    The premise is n_pts >> n_cells and so it was worth pre-computing those 
    27 masks for each cell. Usually you will do much better when the positions 
    are sorted to get better behaviour of the cache, branches etc. 
    
    See also radial_kernel_evaluate() for point-point interactions using
    locality-sensitive hashing.
   */

  const double r2 = rcrit*rcrit;
  for (int i=0;i<num_pts;i++) {
    
    const double x=pts[i*3],y=pts[i*3+1],z=pts[i*3+2]; // Position data
    const double d1=x - floor(x), d2=y - floor(y), d3=z - floor(z); // Relative to cell corner
    // Square distances to left and right cells
    const double d1L = d1*d1, d1R=(1-d1)*(1-d1),
      d2L=d2*d2,d2R=(1-d2)*(1-d2), 
      d3L=d3*d3,d3R=(1-d3)*(1-d3);
    
    const uint64_t *ngb = &(ngb_masks[cells[i]*27]); // Index of 0th neighbour
    /*
      Test the adjacent cells to see if they are further than rcrit.
      When point is near adjacent cell (square distance < r^2) include the bitmask with logical or.
      Looks a bit voodoo code, but really just looping over z,y,x. Central mask (13) is always included.
     */
    out[i] = 
      ((d1L+d2L+d3L<r2)*ngb[0])   | ((d1L+d2L<r2)*ngb[1])  | ((d1L+d2L+d3R<r2)*ngb[2]) | 
      ((d1L+    d3L<r2)*ngb[3])   | ((d1L    <r2)*ngb[4])  | ((d1L+    d3R<r2)*ngb[5]) | 
      ((d1L+d2R+d3L<r2)*ngb[6])   | ((d1L+d2R<r2)*ngb[7])  | ((d1L+d2R+d3R<r2)*ngb[8]) | 

      ((d2L+    d3L<r2)*ngb[9])   | ((d2L    <r2)*ngb[10]) | ((    d2L+d3R<r2)*ngb[11]) | 
      ((d3L        <r2)*ngb[12])  | ngb[13]                | ((        d3R<r2)*ngb[14]) | 
      ((d2R+    d3L<r2)*ngb[15])  | ((d2R    <r2)*ngb[16]) | ((    d2R+d3R<r2)*ngb[17]) | 

      ((d1R+d2L+d3L<r2)* ngb[18]) | ((d1R+d2L<r2)*ngb[19]) | ((d1R+d2L+d3R<r2)*ngb[20]) | 
      ((d1R+    d3L<r2)*ngb[21])  | ((d1R    <r2)*ngb[22]) | ((d1R+    d3R<r2)*ngb[23]) | 
      ((d1R+d2R+d3L<r2)*ngb[24])  | ((d1R+d2R<r2)*ngb[25]) | ((d1R+d2R+d3R<r2)*ngb[26]);
  }
}


static int BHtreewalkLR(const int left_rootnode, const int right_rootnode,
			 const TreeIterator *restrict nodes,
			 double *restrict acc, const double *restrict xyzw, 
			 const double *restrict area_ov_th2)
{
  /* 
     Recurse through two trees (left and right), to apply the kernels from the
     right points to the left, unless the Barnes-Hut criteria says you 
     can keep the node closed.

     Returns number of kernels
   */

  int num_kernels = 0; 

  const double rcut2 = kernel_rcut2; // optimisation for gcc

  // Loop over left tree
  const TreeIterator *node_i = nodes + left_rootnode;

  do {
    
    while (node_i->depth_next) // Is a branch, recurse depth-first
      node_i += node_i->depth_next; 

    // This node's particle loop indices
    const int i_start = node_i->istart;
    const int i_end = i_start + node_i->n;

    if (left_rootnode==right_rootnode) 
      { 
	// Apply for all pairs of particles *within* my leaf(bucket), using 
	// symmetry to avoid double-counting.
	const double *xyzw_j_stop = xyzw + i_end*4-4;
	double *restrict acc_j = acc + i_start*3;
	for (const double *xyzw_j = xyzw + i_start*4;xyzw_j<xyzw_j_stop;acc_j+=3, xyzw_j+=4)
	  {
	    
	    double *restrict acc_k = acc_j;
	    for (const double *xyzw_k = xyzw_j+4;
		 xyzw_k<=xyzw_j_stop;
		 xyzw_k+=4)
	      {
		acc_k += 3;
		const double dx=xyzw_j[0]-xyzw_k[0], 
		  dy = xyzw_j[1]-xyzw_k[1], 
		  dz = xyzw_j[2]-xyzw_k[2];
		
		const double r2 = dx*dx + dy*dy + dz*dz;
		if (r2>=rcut2) // outside kernel radius
		  continue;
		num_kernels++;
		
		const double f_interp = kernel_r_to_idx * sqrt(r2);
		const int i0 = f_interp; 
		const double acc_mult = kernel[i0] + f_interp*diff_kernel[i0];
		const double wt_j=xyzw_j[3]*acc_mult, wt_k=xyzw_k[3]*acc_mult;
		
		acc_j[0] += wt_k * dx;
		acc_j[1] += wt_k * dy;
		acc_j[2] += wt_k * dz;
		
		acc_k[0] -= wt_j * dx;
		acc_k[1] -= wt_j * dy;
		acc_k[2] -= wt_j * dz;
	      }
	  }
      }

    // For every particle in left node, loop over right
    for (int i=i_start;i<i_end;++i) // This turns out to be best way to cycle (not in groups)
      {
	// Load-update-store acceleration for particle i
	double ai_x = acc[i*3], ai_y=acc[i*3+1], ai_z=acc[i*3+2];
	// Point i position
	const double x_i = xyzw[i<<2], y_i=xyzw[i<<2|1],z_i=xyzw[i<<2|2];


	const TreeIterator *node_j = nodes + right_rootnode;
	do
	  {
	    // check Barnes-Hut


	    const double *xyzw_j = xyzw + node_j->com_idx; // idx for CoM of node j
	    // Distance of node j from point i
	    const double dx_ij = xyzw_j[0] - x_i,
	      dy_ij = xyzw_j[1] - y_i, 
	      dz_ij = xyzw_j[2] - z_i;


	    const double dr2 = dx_ij*dx_ij + dy_ij*dy_ij + dz_ij*dz_ij;
	    
	    if ((dr2 >= area_ov_th2[node_j->depth]))
	      {
		// Dont open: just add the mean kernel of node_j to me
		if (dr2<rcut2) // inside kernel radius
		  {
		    num_kernels++;
		    
		    const double f_interp = kernel_r_to_idx * sqrt(dr2);
		    const int i0 = f_interp; 
		    const double wt_j = (kernel[i0] + f_interp*diff_kernel[i0])*
		      xyzw_j[3];
		    
		    ai_x -= wt_j * dx_ij;
		    ai_y -= wt_j * dy_ij;
		    ai_z -= wt_j * dz_ij;
		  }
	      } // Else open node:
	    else if (node_j->depth_next) // Is a branch, recurse
	      {
		node_j += node_j->depth_next; // Depth first
		continue;
	      }
	    else // or a leaf
	      {
		// Apply every point in leaf -> i
		if (node_i!=node_j)
		  {
		    const int j_end = node_j->istart + node_j->n;
		    
		    for (xyzw_j = xyzw + node_j->istart*4;
			 xyzw_j<xyzw+j_end*4
			   ;xyzw_j+=4)
		      {
			const double dx=xyzw_j[0] - x_i, 
			  dy = xyzw_j[1] - y_i, 
			  dz = xyzw_j[2] - z_i;
			
			const double r2 = dx*dx + dy*dy + dz*dz;

			if (r2>=rcut2) // outside kernel radius
			  continue;

			num_kernels++;

			// Kernel interpolation
			const double f_interp = kernel_r_to_idx * sqrt(r2);
			const int i0 = f_interp; 
			const double wt_j = (kernel[i0] + f_interp*diff_kernel[i0])*
			  xyzw_j[3];
			
			ai_x -= wt_j * dx;
			ai_y -= wt_j * dy;
			ai_z -= wt_j * dz;
		      }
		  }

	      }
	    
	    // Breadth-recurse 
	    node_j = nodes + node_j->breadth_next;
	  }
	while (node_j!=nodes);

	acc[i*3]   = ai_x;
	acc[i*3+1] = ai_y;
	acc[i*3+2] = ai_z;
      }
  } while ((node_i = nodes + node_i->breadth_next)!=nodes); // iterate in breadth

  return num_kernels;
}

long BHTreeWalk(const int *restrict root_sizes, const int num_roots, 
		const int max_depth, const int *restrict cells,
		const int ngrid, double *restrict tree_iterator,
		const double theta, const double *restrict xyzw, double *restrict acc)
{
  /*
    Kernel summation over the tree using the Barnes-Hut (opening angle) 
    criterion for a hash-grid of trees.

    root_sizes- nodes per trees
    num_roots - number of roots (and filled cells)
    max_depth - maximum depth of any leaf
    cells     - indices of particle data start for each (filled) cell
    ngrid     - split unit cube into ngrid^3
    tree_iterator - tree (cast)
    theta     - opening angle crit.
    xyzw      - (N,4) particle data, xyzw[cells[i]:cells[i+1]] is the data for 
                cell i
    acc       - (N,3) kernel output data
    

    returns num_kernels - number of kernel evaluations, or -1 if hash table was 
                          too big.
   */

  const long n2 = (long)ngrid * ngrid;
  /* Relative indices in the grid of the bins of the adjacent cells (only left) */
  const int cell_shift[13] = {-n2-ngrid-1,-n2-ngrid,-n2-ngrid+1,
			      -n2-1,-n2,-n2+1,
			      -n2+ngrid-1,-n2+ngrid,-n2+ngrid+1,			      
			      -ngrid-1,-ngrid,-ngrid+1,
			      -1};

  const TreeIterator *nodes = (TreeIterator*)tree_iterator;

  double *area_ov_th2; // Node areas / theta^2 (precomputed for BH comparison)
  // Find the next power of 2 large enough to hold table at desired load
  int tab_size=0;
  while (tab_size<MAX_TAB_NO && HASHTABLE_SIZES[tab_size]*DESIRED_LOAD<num_roots) tab_size++;
  if (tab_size==MAX_TAB_NO)
    return -1; // Table too big

  const int hsize = HASHTABLE_SIZES[tab_size], hprime = HASH_PRIMES[tab_size];
  const int hmask = hsize-1;

  // Zero-ed hashtable (index + start for each cell)
  int *htable = calloc(hsize * 2, sizeof(int));
  if (!htable)
    return -1; // not enough mem.

  if (!(area_ov_th2=malloc((max_depth+1)*sizeof(double))))
    return -1; // not enough mem. for areas 

  long num_kernels=0;
  if (theta>0.0)
    {
      double area0 = 1.0/(ngrid*ngrid*theta*theta);
      for (int i=0;i<=max_depth; i++)
	area_ov_th2[i] = area0 / (1<<(2*i)); // TODO - this is only true of octrees
    }
  else
    {
      for (int i=0;i<=max_depth; i++)
	area_ov_th2[i] = 4.0; // Larger than any distance^2 in unit cube
    }

  // For each tree, add to the hash-table and calculate kernels
  for (int t=0, cell_root=0;
       t<num_roots;
       cell_root+=root_sizes[t++]) // Index in trees where my tree starts
    {
      const int cell_t = cells[t];

      // Search for adjacent trees in the hash table
      for (int adj=0;adj<13;adj++)
	{
	  const int wanted_cell = cell_t + cell_shift[adj];

	  // Search for cell or 0 (no points there)
	  for (int j=(wanted_cell*hprime)&hmask;htable[j*2];j=(j+1)&hmask)
	    {
	      if (~(htable[j*2]^wanted_cell))
		continue;
	      const int adj_root = htable[j*2+1];

	      // Found adjacent tree, lets do Barnes-Hut treewalk to apply
	      // their kernels to *me*
	      num_kernels += BHtreewalkLR(cell_root, adj_root,  // me<-adjacent
					  nodes, acc, xyzw, area_ov_th2);

	      // Ok, now do backwards to apply my kernels to *them* 
	      num_kernels += BHtreewalkLR(adj_root, cell_root, // me->adjacent
					 nodes, acc, xyzw, area_ov_th2);
					
	      break;
	    }
	}
      // Do the BH on myself
      num_kernels += BHtreewalkLR(cell_root, cell_root, // me->me
				 nodes, acc, xyzw, area_ov_th2);
      
      // Insert my cell into hash-table at the next free spot
      int ins;
      for (ins=(cell_t*hprime)&hmask; htable[ins*2]; ins=(ins+1)&hmask);

      htable[ins*2] = ~cell_t; // Bit twiddle to make sure never 0
      htable[ins*2+1] = cell_root;
    }
  
  free(area_ov_th2); // clean up cached areas
  free(htable); // cleanup hash table
  return num_kernels;
}
