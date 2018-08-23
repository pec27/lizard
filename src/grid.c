/*

  Functions for bilinear interpolation of a 3d grid, cloud in cell, 
  finding the gradient of a 3d lattice using 5-point stencils.

  Peter Creasey 2015

 */
#include <math.h>
#include <stddef.h>
#include <string.h> // memcpy
#include <complex.h>

void find_lattice(const double *pos, const int num_pos, const int nx, int *out)
{
  /*
    Find the bucket index ((int)(p[0]*nx)*nx + (int)(p[1]*nx))*nx + (int)p[2]*nx 
    for every point
   */

  for (int i=0;i<num_pos;i++)
    out[i] = ((int)(pos[i*3]*nx)*nx + (int)(pos[i*3+1]*nx))*nx + pos[i*3+2]*nx;

}

int interpolate_periodic(const int grid_width, const unsigned long long npts, const double *gridvals, const double *coordinates, double *out_interp)
{
  /* 
     1d interpolation of 3d gridded data 
     I wrote this function because scipys map_coordinates (which I 
     usually use for interpolation of 3d grids) doesnt seem
     to handle 4GB+ arrays properly.
  */
  double cd[3], frac[3]; // Coords to be interpolated
  int axis, i;
  size_t coord_idx, i0[3], i1[3], gw2; // Left and right indices 
  gw2 = (size_t)grid_width*grid_width;
  for (coord_idx=0; coord_idx<npts; coord_idx++)
    {
      for (axis=0;axis<3;axis++)
	{
	  cd[axis] = coordinates[npts*(size_t)axis+coord_idx];
	  frac[axis] = cd[axis] - floor(cd[axis]);
	  i = (int)cd[axis]%grid_width;// Wrap correctly
	  i0[axis] = i<0 ? i + grid_width : i;
	  i1[axis] = (i0[axis]+1)%grid_width;
	}
      // Convert to indices
      i0[0] *= gw2;
      i1[0] *= gw2;
      i0[1] *= grid_width;
      i1[1] *= grid_width;
      
      out_interp[coord_idx] = (1.0 - frac[0]) * (1.0 - frac[1]) * (1.0 - frac[2]) * gridvals[i0[0]+i0[1]+i0[2]] +
	(1.0 - frac[0]) * (1.0 - frac[1]) * frac[2] * gridvals[i0[0]+i0[1]+i1[2]] +
	(1.0 - frac[0]) * frac[1] * (1.0 - frac[2]) * gridvals[i0[0]+i1[1]+i0[2]] +
	(1.0 - frac[0]) * frac[1] * frac[2] * gridvals[i0[0]+i1[1]+i1[2]] +
	frac[0] * (1.0 - frac[1]) * (1.0 - frac[2]) * gridvals[i1[0]+i0[1]+i0[2]] +
	frac[0] * (1.0 - frac[1]) * frac[2] * gridvals[i1[0]+i0[1]+i1[2]] +
	frac[0] * frac[1] * (1.0 - frac[2]) * gridvals[i1[0]+i1[1]+i0[2]] +
	frac[0] * frac[1] * frac[2] * gridvals[i1[0]+i1[1]+i1[2]];
    }

  return 0;
}

void interp_vec3(const int grid_n, const unsigned long long npts, 
		 const double *grid, const double *pts, double *out)
{
  /* 
     Linearly interpolate a grid of 3d vectors at the given positions (last grid
     point rolls).

     grid_n - N for N*N*N*3 grid
     npts   - number of points to interpolate
     *grid  - array of N*N*N*3 values
     *pts   - positions to interpolate, MUST be in right-open [0,N)
     *out   - array for N*3 outputs

     Like interpolate_periodic but with vec3 data to interpolate. By 
     interpolating all at the same time we essentially reduce cache-misses
     by factor 3 and depending on the compiler and proc get a few SSE wins.
  */
  const int max_index = grid_n - 1; 
  const size_t fwd_i = (size_t)grid_n*grid_n*3,
    back_i = -max_index*(size_t)grid_n*grid_n*3,
    fwd_j = grid_n*3,
    back_j = -max_index*(size_t)grid_n*3,
    fwd_k = 3,
    back_k = -max_index*3;

  for (size_t pt=0; pt<npts*3;)
    {
      /*
	WARNING: Not exactly write-only-code, but relatively heavily tested
	for performance in both cache-poor (unsorted) and cache-rich (sorted)
	cases for point sets.
       */

      /* Index of the point */
      const int i = pts[pt], j=pts[pt+1], k=pts[pt+2];
      const size_t idx_ijk = fwd_i*i + fwd_j*j + fwd_k*k;

      /* When interpolating 3 components (vec3) it is worth pre-computing
	 weight multiples for 4 i,j components */
      const double wRi = pts[pt]-i,wRj= pts[pt+1]-j;
      const double wLi = 1-wRi,wLj= 1-wRj;
      const double w_ij[4]={wLi*wLj, wLi*wRj, wRi*wLj, wRi*wRj};

      /* Roll next point? */
      const size_t inc_k = k<max_index ? fwd_k : back_k,
	inc_j = j<max_index ? fwd_j : back_j,
	inc_i = i<max_index ? fwd_i : back_i;
	
      /* Indices of first 4 corners (k) */
      const size_t cnrL[4] = {idx_ijk, inc_j + idx_ijk, inc_i + idx_ijk,
			     inc_j + inc_i + idx_ijk};

      /* Indices of second 4 (next k) */
      const size_t cnrR[4] = {cnrL[0] + inc_k, cnrL[1] + inc_k, 
			      cnrL[2] + inc_k, cnrL[3] + inc_k};

      /* Interpolate in i,j */
      const double vL[3] = {w_ij[0]*grid[cnrL[0]]   + w_ij[1]*grid[cnrL[1]] + 
			    w_ij[2]*grid[cnrL[2]]   + w_ij[3]*grid[cnrL[3]],
			    w_ij[0]*grid[cnrL[0]+1] + w_ij[1]*grid[cnrL[1]+1] + 
			    w_ij[2]*grid[cnrL[2]+1] + w_ij[3]*grid[cnrL[3]+1],
			    w_ij[0]*grid[cnrL[0]+2] + w_ij[1]*grid[cnrL[1]+2] + 
			    w_ij[2]*grid[cnrL[2]+2] + w_ij[3]*grid[cnrL[3]+2]};

      const double vR[3] = {w_ij[0]*grid[cnrR[0]]   + w_ij[1]*grid[cnrR[1]] +
			    w_ij[2]*grid[cnrR[2]]   + w_ij[3]*grid[cnrR[3]],
			    w_ij[0]*grid[cnrR[0]+1] + w_ij[1]*grid[cnrR[1]+1] + 
			    w_ij[2]*grid[cnrR[2]+1] + w_ij[3]*grid[cnrR[3]+1],
			    w_ij[0]*grid[cnrR[0]+2] + w_ij[1]*grid[cnrR[1]+2] +
			    w_ij[2]*grid[cnrR[2]+2] + w_ij[3]*grid[cnrR[3]+2]};
      /* Interpolate in k */
      const double wRk=pts[pt+2]-k;
      const double v[3] = {vL[0] + wRk*(vR[0]-vL[0]),
			   vL[1] + wRk*(vR[1]-vL[1]),
			   vL[2] + wRk*(vR[2]-vL[2])};

      out[pt++] = v[0];
      out[pt++] = v[1];
      out[pt++] = v[2];
    }
}

int cloud_in_cell_3d(const int num_pts, const int ngrid, const double * restrict mcom, double *restrict out)
{
  /*
    Cloud-In-Cell in 3 dimensions: 
    Take a given set of masses and positions on a cubic lattice and apply them to 
    the eight corners by the bilinear interpolation weights.
    
    num_pts - number of points
    ngrid   - width of cubic grid, i.e. centres-of-mass lie in [0,ngrid) along x,y,z
    mcom    - (double) num_pts*4 of mass and centre-of-mass (mass,x,y,z) s.t. x,y,z in [0,ngrid)
    out     - array of size ngrid*ngrid*ngrid to hold outputs

    Notes:
    We are usually dominated by cache-misses, i.e. on big grids the 8 corners of
    the next point are only sometimes in the cache. If the average number
    of particles per *filled* cell is larger than about 2, then you are probably
    wise to order the particles by cell first (see find_lattice) or 
    Peano-Hilbert and the run through those. 

   */

  const int ngrid2 = ngrid * ngrid;

  for (int pt=0;pt<((size_t)num_pts)<<2;pt+=4)
    {
      const double mass = mcom[pt];
      const double p[3] = {mcom[pt | 1],mcom[pt | 2],mcom[pt | 3]};
      const int ip[3] = {p[0],p[1],p[2]}; // Integer part of postion (left corner in 3d)

      // Left weights
      const double wL[3] = {(ip[0]+1)-p[0],(ip[1]+1)-p[1],(ip[2]+1)-p[2]}; 
      // Right weights
      const double wR[3] = {1-wL[0],1-wL[1],1-wL[2]};
      // Indices into 3d array (wrap right point if equal to ngrid)
      const int iL[3] = {ip[0]*ngrid2,ip[1]*ngrid,ip[2]};
      const int iR[3] = {ip[0]==ngrid-1 ? 0 : (ip[0]+1)*ngrid2,
			 ip[1]==ngrid-1 ? 0 : (ip[1]+1)*ngrid,
			 ip[2]==ngrid-1 ? 0 : ip[2] + 1};

      // bilinear interpolation onto the eight corners
      const double w0 = mass * wL[0] * wL[1] * wL[2], 
	w1 = mass * wL[0] * wL[1] * wR[2],
	w2 = mass * wL[0] * wR[1] * wL[2],
	w3 = mass * wL[0] * wR[1] * wR[2],
	w4 = mass * wR[0] * wL[1] * wL[2],
	w5 = mass * wR[0] * wL[1] * wR[2],
	w6 = mass * wR[0] * wR[1] * wL[2],
	w7 = mass * wR[0] * wR[1] * wR[2];
      

      out[iL[0]+iL[1]+iL[2]] += w0;
      out[iL[0]+iL[1]+iR[2]] += w1;
      out[iL[0]+iR[1]+iL[2]] += w2;
      out[iL[0]+iR[1]+iR[2]] += w3;
      out[iR[0]+iL[1]+iL[2]] += w4;
      out[iR[0]+iL[1]+iR[2]] += w5;
      out[iR[0]+iR[1]+iL[2]] += w6;
      out[iR[0]+iR[1]+iR[2]] += w7;
      
    }
  return 0;
}

int cloud_in_cell_3d_vel(const int num_pts, const int ngrid, 
			 const double *mcom, const double* mom, double complex *out)
{
  /*
    As for cloud_in_cell_3d but also using the *velocity* of each particle in 
    order to calculate the time derivative of the CIC values. This is useful 
    for, for example, calculating the linear time extrapolation of the 
    potential. 

    num_pts - number of points
    ngrid   - width of cubic grid, i.e. centres-of-mass lie in [0,ngrid) along x,y,z
    mcom    - (double) num_pts*4 of mass and centre-of-mass (mass,x,y,z)
    mom     - (double) num_pts*3 of momentum (px,py,pz) per particle
    out     - array of size ngrid*ngrid*ngrid*2 to hold outputs of 
              (cic, dcic/dt) for each cell.

    Notes:
    It is tempting to think that you can combine particles by their mass-
    weighted centres to find the derivative in CIC, however this is not
    true because the weights are not linear. One way to see this is to
    consider the motion of two particles in a 2-d grid as follows:

    X---      ---
    |   |    |X  |
    |   | to |   |  
    |   |    |  X|
     ---X     ---

    with weights
    1 0  to 0.625 0.375
    0 1     0.375 0.625
    at the corners, whilst clearly the CoM stays in the same position (the 
    centre).

   */

  const int ngrid2 = ngrid * ngrid;
  for (int pt=0;pt<num_pts;pt++)
    {
      const double mass = mcom[pt<<2];
      // Centre of mass of this particle
      const double x[3] = {mcom[pt<<2 | 1],mcom[pt<<2 | 2],mcom[pt<<2 | 3]};
      // Integer part of position (left corner in 3d)
      const int ix[3] = {x[0],x[1],x[2]}; 

      // Left weights (linear interpolation)
      const double wL[3] = {(ix[0]+1)-x[0],(ix[1]+1)-x[1],(ix[2]+1)-x[2]}; 
      // Right weights
      const double wR[3] = {1-wL[0],1-wL[1],1-wL[2]};
      // Indices into 3d array (wrap right point if equal to ngrid)
      const int iL[3] = {ix[0]*ngrid2,ix[1]*ngrid,ix[2]};
      const int iR[3] = {ix[0]==ngrid-1 ? 0 : (ix[0]+1)*ngrid2,
			 ix[1]==ngrid-1 ? 0 : (ix[1]+1)*ngrid,
			 ix[2]==ngrid-1 ? 0 : ix[2] + 1};

      // bilinear interpolation onto the eight corners
      const double w0 = mass * wL[0] * wL[1] * wL[2], 
	w1 = mass * wL[0] * wL[1] * wR[2],
	w2 = mass * wL[0] * wR[1] * wL[2],
	w3 = mass * wL[0] * wR[1] * wR[2],
	w4 = mass * wR[0] * wL[1] * wL[2],
	w5 = mass * wR[0] * wL[1] * wR[2],
	w6 = mass * wR[0] * wR[1] * wL[2],
	w7 = mass * wR[0] * wR[1] * wR[2];

      const double *p = &mom[pt*3];
      // time derivative of these weights 
      const double d0 = -p[0]*wL[1]*wL[2] - p[1]*wL[0]*wL[2] - p[2]*wL[0]*wL[1],
	d1 = -p[0]*wL[1]*wR[2] - p[1]*wL[0]*wR[2] + p[2]*wL[0]*wL[1],
	d2 = -p[0]*wR[1]*wL[2] + p[1]*wL[0]*wL[2] - p[2]*wL[0]*wR[1],
	d3 = -p[0]*wR[1]*wR[2] + p[1]*wL[0]*wR[2] + p[2]*wL[0]*wR[1],
	d4 = p[0]*wL[1]*wL[2] - p[1]*wR[0]*wL[2] - p[2]*wR[0]*wL[1],
	d5 = p[0]*wL[1]*wR[2] - p[1]*wR[0]*wR[2] + p[2]*wR[0]*wL[1],
	d6 = p[0]*wR[1]*wL[2] + p[1]*wR[0]*wL[2] - p[2]*wR[0]*wR[1],
	d7 = p[0]*wR[1]*wR[2] + p[1]*wR[0]*wR[2] + p[2]*wR[0]*wR[1];

      // Store the CIC in the real part and the time derivative in the imaginary
      out[iL[0]+iL[1]+iL[2]] += w0 + I*d0;
      out[iL[0]+iL[1]+iR[2]] += w1 + I*d1;
      out[iL[0]+iR[1]+iL[2]] += w2 + I*d2;
      out[iL[0]+iR[1]+iR[2]] += w3 + I*d3;
      out[iR[0]+iL[1]+iL[2]] += w4 + I*d4;
      out[iR[0]+iL[1]+iR[2]] += w5 + I*d5;
      out[iR[0]+iR[1]+iL[2]] += w6 + I*d6;
      out[iR[0]+iR[1]+iR[2]] += w7 + I*d7;
      
    }
  return 0;
}

void gradient_5pt_3d(const int ngrid, const double *restrict vals, double *restrict out)
{
  /*
    Calculate the gradient of a periodic 3-d lattice with finite differencing
    using the 5-point stencil. The parameters are

    ngrid  - integer side of lattice (i.e. ngrid^3 points)
    vals   - double array (ngrid^3 in C ordering) of lattice values
    out    - float array of outputs (ngrid^3 * 3 array, last dimension
             contains the 3 derivatives)

    The 5 point differencing formula (technically only 4 points since the 
    centre has zero weight) is

    f'(x) = -f(x+2)/12 + 2*f(x+1)/3 - 2*f(x-1)/3 + f(x-2)/12

    This is about 5-10 times faster than the Numpy equivalent using roll, 
    presumably because we get a bit better use of the cache. I already tried
    making a complex128 version, but it provided no benefit (perhaps too many
    casts?).
   */

  const size_t n = ngrid; // size_t version of the same pointer (reduce casting a bit)
  const size_t n2 = n * n;
  const double w0 = 1.0/12.0, w1 = -2.0/3.0; // Weights for 5 point stencil

  if (ngrid<3)
    {
      /* Algorithm fails for 1^3 or 2^3 grids, but these are easy because 
	 they have zero gradient by symmetry */
      for (int i=0;i<ngrid*ngrid*ngrid*3;i++)
	out[i] = 0.0;
      return;
    }

  // Now do gradient for 2nd and 3rd dimensions
  out += 1;
  for (int i=0;i<ngrid;i++)
    {
      // Values for gradient over 2nd axis
      const double *vj0 = vals + (n-2)*n,
	*vj1 = vals + (n-1)*n,
	*vj2 = vals + n,
	*vj3 = vals + 2*n;

      const double *v4 = vals+2;

      vals += n2; // used for wrap comparison

      for (int j=ngrid;j--; v4+=n)
      	{
	  // Values for gradient over 3rd axis 
	  double v0 = v4[ngrid-4],
	    v1 = v4[ngrid-3],
	    v2 = *(v4-2), v3=*(v4-1);

	  for (int col=ngrid; col--; out+=3)
	    {
	      // Find the gradients in each of three directions
	      const double g1 = w0*((*vj0++)-(*vj3++)) + w1*((*vj1++) - (*vj2++)),
		g2 = w0*(v0-(*v4)) + w1*(v1 - v3);
	      
	      out[0] = g1; out[1] = g2;

	      v0 = v1; v1=v2; v2=v3; v3=(*v4++); 
	      if (col==2) // wrap
		v4 -= n;
	      
	    }

	  vj0 = vj1 - n;
	  vj2 = vj3 - n;
	  if (vj1==vals)
	    vj1 -= n2;
	  // cant be else if since ngrid could be 3
	  if (vj3==vals)
	    vj3 -= n2;
	} 
    }

  // Back to start
  out -= 3*n*n2+1;
  vals -= n*n2;

  // In order to reduce pressure on cache, when doing the gradient in 1st (of 3)
  // dimensions, loop over 2nd, 1st, 3rd 

  const int out_jump = (n2-n)*3;

  for (int j=0;j<ngrid;j++,vals+=n)
    {
      // Indices i-2, i-1,i+1,i+2 modulo ngrid
      const double *vi0 = vals + (n-2)*n2,
	*vi1 = vals + (n-1) * n2,
	*vi2 = vals + n2,
	*vi3 = vals + 2*n2;

      for (int i=ngrid;i; out+=out_jump)
	{

	  // Find the gradients in the i-direction
	  for (int k=0;k<ngrid; ++k, out+=3)
	    out[0] = w0*(vi0[k] - vi3[k]) + w1*(vi1[k] - vi2[k]);


	  // Iterate the pointers to next val (in i)
	  vi0 = vi1; vi2 = vi3;
	  vi1 += n2; vi3 += n2;
	  // Wrap 
	  if (i--==ngrid)
	    vi1 = vals;
	  if (i==2) // cant be else if since ngrid could be 3
	    vi3 = vals;
	}
      out -= 3*n*(n2-1);
    }
  out -= n2*3; // back to start
  vals -= n2;

}
void unpack_kgrid(const int n, const double *packed_vals, double *unpacked_vals)
{
  /*
    Unpack the fft-like values g[i,j,k] where i,j,k run from 0,...,n-1 and g 
    has the symmetries that 
    
       g[i,j,k] = g[q_i, q_j, q_k] 
    where q_i := i for i<M
                 n-i for i>=M
    where M:= int(n/2) + 1
	       
    and so the values have been stored (for efficient packing or evaluation
    purposes) in the '3d-triangle':

      q[u(u+1)(u+2)/6 + v(v+1)/2 + w], where u>=v>=w>=0.

    which is an almost 48-fold reduction (6 cyclic permutations and 8 
    reflections, minus a few repeats).

    Arguments:
    
    n             - size of the 3d grid
    packed_vals   - M(M+1)(M+2)/6 (double) inputs
    unpacked_vals - place to store n^3 (double) output values

   */
  
  // i,j,k run from 0,1,...,mid-1
  // if i,j,k in 1,2,..., bmid-1, then the are repeated, i.e. at n-1, n-2,..., n+1-bmid
  const int mid = 1 + n/2; 
  const int bmid = n - n/2;


  /*
    Nearly write-only code: this follows the increments in the packed-indices
    (p_idx) as we step through the unpacked idx (u_idx :=i*n*n+j*n+k). These
    steps change depending on whether i>j>k.

    p_step_j contains the steps in p_idx. As k runs from 
       0...min(i,j)-1        p_idx increments by 1
       min(i,j)...max(i,j)-1 p_idx increments by k+1
       max(i,j)...M-1        p_idx increments by (k+1)(k+2)/2
       
    p_start_j contains the k=0 index. As j runs from
       0...i-1               p_start_j increments by j+1 
       i...M-1               p_start_j increments by (j+1)(j+2)/2

    Thank goodness for unit-tests.
   */
  int p_step_j[mid]; // steps
  
  for (int i=0, p_start=0;i<mid;i++) // first packed index
    {
      const int sum_i = ((i+1)*(i+2))/2; // Sum 1...i+1

      // Set the steps to be j+1 for 0...i, sum_j for i...M-1
      for (int j=0;j<i;j++)
	p_step_j[j] = j+1;
      for (int j=i,sum_j=sum_i; j<mid; sum_j += (++j)+1)
	p_step_j[j] = sum_j;

      // Initialise the packing and unpacking indices, loop over j
      size_t u_idx = n*(size_t)n*i;
      for (int p_idx=p_start,k,j=0;j<mid;j++)
	{

	  // p_idx incremented by 1, k+1, or (k+1)(k+2)/2
	  for (k=0; k<mid; p_idx+=p_step_j[k++]) 
	      unpacked_vals[u_idx++] = packed_vals[p_idx];
	  
	  // 1 or 2 steps back (i.e. from k-> -k), depending on whether n is odd
	  while (k>=bmid)
	    p_idx -= p_step_j[--k];	    

	  // Reverse direction [1-bmid, 2-bmid, ... -1]
	  for (; k>0; p_idx -= p_step_j[--k])
	    unpacked_vals[u_idx++] = packed_vals[p_idx];

	  // Iterate j
	  p_idx += p_step_j[j]; // j+1 or (j+1)(j+2)/2
	  // Critical k for increments changes
	  p_step_j[j] = (j>=i)*j+1;  

	  // Reflection j -> n-j
	  if ((j>0) & (j<bmid))
	    memcpy(unpacked_vals+n*(size_t)(n*i+n-j), unpacked_vals+n*(size_t)(n*i+j), n*sizeof(double));
	}	
      // Reflection i -> n-i
      if ((i>0) & (i<bmid))
	memcpy(unpacked_vals+n*(size_t)n*(n-i), unpacked_vals+n*(size_t)n*i, n*(size_t)n*sizeof(double));	

      // Iterate i, therefore j-crossover changes
      p_start += sum_i;
    }
}
