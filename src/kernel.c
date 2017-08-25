/*
  A module for evaluating at a set of points x_i a vector field which is defined
  as the gradient of the convolution of a radial function with a set of 'masses'
  at points y_j. This is useful, for example, for calculating the radial 
  component of a softened gravitational force in a non-periodic domain.

  For simplicity, the gradient of the radial kernel is given as
  y * k(|y|) and thus the convolution is 

  v(x_i) = Sum_j (x_i - y_j) * m_j * k(|x_i - y_j|)

  In practice we only wish to know this to some given accuracy (my reference
  value is 0.2%), and so we tabulate the kernel logarithmically in radius.
  To find neighbours within the kernel we ask that the values are in a tree, 
  and to further approximate our calculation (and importantly reduce our 
  cache-misses), we occasionally allow a tree-node to be treated as a single 
  particle using the Barnes-Hut criterion. This is effectively a criterion
  on the smoothness of the kernel, i.e. that the kernel evaluated at the 
  radius of the centre of mass is a good approximation to the average of the
  kernels individual particles, when they are limited to being within some 
  multiple of the radius of the centre (e.g. ~0.05 for BH=0.1).
  
  See lizard.c for the tree construction.

 */



//#define DEBUG
#include <math.h>
#ifdef DEBUG
  #include <stdio.h>

#endif
#define MAX_PTS 200
static float kernel_vals[MAX_PTS];
static int kernel_num_pts;
static float kernel_rad_max2, kernel_idx_mult, kernel_rad_min2, kernel_rad_max; // Radius within which to evaluate kernel
static int node_evals; // Number of evaluations of nodes
#ifdef DEBUG
static int max_stack=1;
#endif

typedef struct
{

  float chw; // cell half width
  float p[3]; // top left corner
  int node;
} Treenode;

int init_kernel(const int num_pts, const float *pts, const float rad_min, const float rad_max)
{
  if (num_pts>MAX_PTS)
    return 1; // Too many points

  kernel_num_pts = num_pts;
  for (int i=0;i<num_pts;i++)
    kernel_vals[i] = pts[i];
  kernel_rad_max = rad_max;
  kernel_rad_max2 = rad_max*rad_max;
  kernel_rad_min2 = rad_min*rad_min;
  kernel_idx_mult = 0.5*(num_pts-1) / log(rad_max/rad_min);
  return 0;
}

static int kernel_walk_tree(const int *tree, 
			    const float bh_crit2, const int n_leaves, const double *pos, const double *mcom, double *out)
{
  /*
    Walk the tree node for the given kernel, but without recursion.

    tree     - connectivity of the octree (n_nodes * 8 array)
    bh_crit2 - square of the Barnes-Hut opening angle criterion (normally 0.01)
    n_leaves - number of leaves in the tree (denotes an empty cell)
    
    *pos     - 3d position of x (evaluation position)
    *mcom    - masses and centres of masses of all the leaves, followed by all the
               nodes (i.e. (n_leaves+n_nodes)*4 array)
    *out     - where we add the force

    Returns the number of kernel evaluations performed.
   */

  #define STACK_SIZE 50
  Treenode nodes[STACK_SIZE];
  Treenode c;
  int stack = 1;
  int mcom_idx;
  int num_kernels=0;

  nodes[0].node = 0;
  nodes[0].chw = 0.5;
  /*  nodes[0].x0 = -pos[0];
  nodes[0].y0 = -pos[1];
  nodes[0].z0 = -pos[2];*/
  nodes[0].p[0] = -pos[0];
  nodes[0].p[1] = -pos[1];
  nodes[0].p[2] = -pos[2];

  
  while (stack)
    {
      // Pop a node from the stack
      const Treenode cur = nodes[--stack];

      // Mask for non-empty octants
      int keep_octants = (tree[cur.node<<3]!=n_leaves) | 
	((tree[cur.node<<3|1]!=n_leaves) | 
	 ((tree[cur.node<<3|2]!=n_leaves) | 
	  ((tree[cur.node<<3|3]!=n_leaves) | 
	   ((tree[cur.node<<3|4]!=n_leaves) | 
	    ((tree[cur.node<<3|5]!=n_leaves) | 
	     ((tree[cur.node<<3|6]!=n_leaves) | 
	      (tree[cur.node<<3|7]!=n_leaves)<<1)<<1)<<1)<<1)<<1)<<1)<<1;

      // Half width of the child nodes
      c.chw = cur.chw*0.5;

      /* This piece of c-voodoo masks the half-spaces which are more than
	 kernel_rad_max away */
      const float left_crit = cur.chw+kernel_rad_max;
      const float right_crit = kernel_rad_max - cur.chw;

      keep_octants &= ~((cur.p[0]+left_crit<0)*0xF |
			(cur.p[1]+left_crit<0)*0x33 |
			(cur.p[2]+left_crit<0)*0x55 |	
			(right_crit-cur.p[0]<0)*0xF0 | 
			(right_crit-cur.p[1]<0)*0xCC | 
			(right_crit-cur.p[2]<0)*0xAA);
		
      /* Go through eight octants of the octree-node */
      for (int octant=0;keep_octants;octant++,keep_octants>>=1)
	{
	  if (!(keep_octants&1)) 
	    continue; // Either empty or too far
	
	  const int rx = (octant>>2) & 1, ry=(octant>>1) & 1, rz = octant & 1;

	  // Find the topleft of this child node
	  c.p[0] =  cur.p[0]+rx*cur.chw; c.p[1] = cur.p[1]+ry*cur.chw; c.p[2] = cur.p[2]+rz*cur.chw;
	  
	  float f[3] = {fabs(c.p[0]+c.chw)-c.chw, fabs(c.p[1] + c.chw)-c.chw, fabs(c.p[2] + c.chw)-c.chw};
	  if (f[0]<0) f[0]=0;
	  if (f[1]<0) f[1]=0;
	  if (f[2]<0) f[2]=0;
	  const float r2=f[0]*f[0]+f[1]*f[1]+f[2]*f[2];
	  
	   if (r2>kernel_rad_max2)
	     continue;

	   mcom_idx = tree[cur.node<<3|octant]; 

	   if (mcom_idx<0)
	    {
	      if (r2*bh_crit2<cur.chw*cur.chw) // A node, do we open?
		{
		  if (stack==STACK_SIZE)
		    return -1;

		  // Push this node to the stack
		  c.node = -mcom_idx;
		  nodes[stack++] = c;
#ifdef DEBUG
		  if (stack>max_stack) max_stack=stack;
#endif
		  continue;
		}
	      /* otherwise do the kernel sum for the node */
	      mcom_idx = n_leaves - mcom_idx;
	    }



	  // BOOM! The next line is usually the slowest, because there are too many particles to fit in the cache,
	  // so we have to drag their position and mass out of main memory.
	  // Displacement from doubles (needs to be accurate)
	  const float mass = mcom[mcom_idx<<2],dx = mcom[mcom_idx<<2 |1] - pos[0], dy = mcom[mcom_idx<<2 |2] - pos[1], dz = mcom[mcom_idx<<2 |3] - pos[2];


	  const float dist2 = dx*dx+dy*dy+dz*dz; /* Square distance to particle (or node) centre */
#ifdef DEBUG
	  if (dist2<r2) //NB - you can occasionally hit this just because floats are insufficiently accurate
	    printf("WARNING: Minimum box pos was %6.5f, yet particle at %6.5f (float inaccuracy?)\n", (float)sqrt(r2), (float)sqrt(dist2));
#endif
	  // Kernel linear interpolation 
	  if ((dist2<kernel_rad_min2) || (dist2>kernel_rad_max2))
	    continue; // Actually the particle was not in the kernel

	  // In tests the logarithms takes ~10% of the calculation time. This is
	  // relatively acceptable since it gives a good point distribution.
      
	  const float idx = kernel_idx_mult * log(dist2/kernel_rad_min2);
	  const int i0 = (int)idx;
#ifdef DEBUG
	  if ((idx<0) || (idx>=kernel_num_pts-1))
	    printf("ERROR: Found an idx (%6.4f) outside the range (0-%d).\n", idx,kernel_num_pts-1);
#endif
	  const int i1 = idx==kernel_num_pts-1 ? idx : idx+1;
	  const float wt = mass * ((1+i0-idx)*kernel_vals[i0] + (idx-i0)*kernel_vals[i1]);

	  out[0] += wt*dx;
	  out[1] += wt*dy;
	  out[2] += wt*dz;

	  num_kernels++;
	}
    }

  return num_kernels;
}


int kernel_evaluate(const double *pos, const int num_pos, const int *tree, const int n_leaves, const double *tree_mcom, double *out, float bh_crit)
{
  /*
    Evaluate the kernel at the given positions
  */
  int num_kernels = 0, kernels;
  node_evals = 0;
#ifdef DEBUG
  printf("Evaluating %d points for kernel from r in %f to %f\n", num_pos, (float)sqrt(kernel_rad_min2), (float)sqrt(kernel_rad_max2));
#endif

  for (int i=0;i<num_pos;i++)
    {
      out[i*3] = out[i*3+1] = out[i*3+2] = 0.0; // Zero the results
      kernels = kernel_walk_tree(tree, bh_crit*bh_crit, n_leaves, &pos[i*3], tree_mcom, &out[i*3]);
      num_kernels += kernels;
      if (kernels<0)
	{
#ifdef DEBUG
	  printf("OUT OF STACK SPACE!");
#endif
	  return -1;
	}
    }
#ifdef DEBUG
  printf("Number of node evaluations %d, max stack used %d\n", node_evals, max_stack);
#endif
  return num_kernels;
}
