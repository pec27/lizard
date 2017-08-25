/*

  Functions for bilinear interpolation of a 3d grid, making trees.

  Peter Creasey 2014

 */
#include <math.h>
#include <stddef.h>



static inline float smallest_interval_val(const float x0, const float x1)
{
  /* 
     Find the nearest point to zero in the set of periodically repeated
     intervals
     S := union(..., [x0-1,x1-1], [x0,x1], [x0+1,x1+1], [x0+2,x1+2],...)
     given -1 < x0 < x1 < +1

     This is useful for finding the closest point on a box.

     Returns the absolute distance
  */
  if (x0<0)
    {
      if (x1>0) return 0; // covers 0
      if (x0+1<-x1) return x0+1;
      return x1;
    }
  if (x1-1 > -x0) return x1-1;
  return x0;
}

static inline int disjoint_interval(const double x1, const float w0, const float w1)
{
  /*
    Is the set [0,w0] disjoing from set [x1,x1+w1], periodically repeated?
    with x1 in reals
    w0,w1 in [0,1]

   */
  const double x = x1 - floor(x1);
  return (x+w1<=1) && (x>=w0);
}
static int refine_BH(int *tree, int *n_nodes,const int MAX_NODES, int *n_leaves,
		     const int max_refine, const double rmin, const float bh_crit,
		     const double x0, const double y0, const double z0, const double cw)
{
  /*
    Recurse through the octtree until BH opening angle criteria is met for all
    cells within r_min of (0,0,0), splitting up to max_refine times.

    *tree      - node indices (+ve for leaves)
    *n_nodes   - number of nodes created so far
    *n_leaves  - number of leaves created so far
    box_size   - periodicity
    max_refine - number of times we can split (if zero has to be a leaf)
    cw         - cell width
    rmin       - minimum radius around (0,0,0) from which we want full refinement
    bh_crit    - Barnes-Hut opening angle criteria (e.g. 0.1)
    x0,y0,z0   - top left (minimum coords) of cell


    Returns the node or leaf of this cell, to be put in the tree.
   */
  if ((*n_nodes==-1) ||  (max_refine==0)) 
    {
      // Either we ran out of nodes, or we 
      // reached the refinement limit. Either way a leaf
      return (*n_leaves)++;
    }
  double r2, dx;

  // Find the square of the distance to the closest point in cell
  dx = smallest_interval_val(x0,x0+cw);
  r2 = dx*dx;
  dx = smallest_interval_val(y0,y0+cw);
  r2 += dx*dx;
  dx = smallest_interval_val(z0,z0+cw);
  r2 += dx*dx;
  dx = cw + bh_crit * rmin;
  dx *= dx;
  // Test if (sqrt(r2)-rmin)*bh_crit >= cw
  if (r2*bh_crit*bh_crit >= dx)
    return (*n_leaves)++; // Dont need to split, Just make a new leaf

  // Larger than opening angle criteria, refine
  if (*n_nodes==MAX_NODES)
    {
      *n_nodes = -1; 
      return -1; // out-of-nodes
    }

  // Make a new node
  // make a copy as it gets changed here and in the child calls
  const int new_node = (*n_nodes)++; 
  int *tnode = &tree[(size_t)new_node<<3]; // current tree node
  const double hw = cw * 0.5; // Half the cell width (i.e. width of next cell)
  
  // find the new nodes for each of the children
  *tnode++ = refine_BH(tree, n_nodes,MAX_NODES, n_leaves, 
		       max_refine-1, rmin, bh_crit, x0, y0,z0,hw);
  *tnode++ = refine_BH(tree, n_nodes,MAX_NODES, n_leaves, 
		       max_refine-1, rmin, bh_crit, x0, y0,z0+hw,hw);
  *tnode++ = refine_BH(tree, n_nodes,MAX_NODES, n_leaves, 
		       max_refine-1, rmin, bh_crit, x0, y0+hw,z0,hw);
  *tnode++ = refine_BH(tree, n_nodes,MAX_NODES, n_leaves, 
		       max_refine-1, rmin, bh_crit, x0, y0+hw,z0+hw,hw);
  *tnode++ = refine_BH(tree, n_nodes,MAX_NODES, n_leaves, 
		       max_refine-1, rmin, bh_crit, x0+hw, y0,z0,hw);
  *tnode++ = refine_BH(tree, n_nodes,MAX_NODES, n_leaves, 
		       max_refine-1, rmin, bh_crit, x0+hw, y0,z0+hw,hw);
  *tnode++ = refine_BH(tree, n_nodes,MAX_NODES, n_leaves, 
		       max_refine-1, rmin, bh_crit, x0+hw, y0+hw,z0,hw);
  *tnode   = refine_BH(tree, n_nodes,MAX_NODES, n_leaves, 
		       max_refine-1, rmin, bh_crit, x0+hw, y0+hw,z0+hw,hw);

  // return (negative of) this new node
  return -new_node;
}
static int refine_BH_ellipsoid(int *tree, int *n_nodes,const int MAX_NODES, int *n_leaves,
			       const int max_refine, const double *M, const float bh_crit,
		     const double x0, const double y0, const double z0, const double cw)
{
  /*
    Recurse through the octtree until BH opening angle criteria is met for all
    cells within r_min of (0,0,0), splitting up to max_refine times.

    *tree      - node indices (+ve for leaves)
    *n_nodes   - number of nodes created so far
    *n_leaves  - number of leaves created so far
    box_size   - periodicity
    max_refine - number of times we can split (if zero has to be a leaf)
    cw         - cell width
    *M         - 7 values, Upper half of symmetric 3x3 matrix (A) and constant k which 
                 defines the surface of the ellipsoid as x.A.x=k
		 k is chosen s.t. the largest eigenvalue of M is 1.
    bh_crit    - Barnes-Hut opening angle criteria (e.g. 0.1)
    x0,y0,z0   - top left (minimum coords) of cell


    Returns the node or leaf of this cell, to be put in the tree.
   */
  if ((*n_nodes==-1) ||  !max_refine) 
    return (*n_leaves)++; /* Either we ran out of nodes, or we reached the
			     refinement limit. Either way a leaf */

  /* 
     Find the distance from the cell to the closest point on the ellipsoid.
     TODO - currently this just finds x.M.x, but the full test is probably
     too expensive to do for every cell.
  */

  const double dx = smallest_interval_val(x0,x0+cw), 
    dy = smallest_interval_val(y0,y0+cw),
    dz = smallest_interval_val(z0,z0+cw);

  const double r2 = dx*(M[0]*dx + 2*M[1]*dy + 2*M[2]*dz) +
    dy*(M[3]*dy + 2*M[4]*dz) + dz*M[5]*dz - M[6];

  // Test if bh_crit^2 * (xMx - lam) >= cw^2
  if (r2*bh_crit*bh_crit >= cw*cw)
    return (*n_leaves)++; // Dont need to split, just return this as a leaf

  // Larger than opening angle criteria, refine
  if (*n_nodes==MAX_NODES)
    return (*n_nodes = -1); // out-of-nodes


  // Make a new node
  // make a copy as it gets changed here and in the child calls
  const int new_node = (*n_nodes)++; 
  int *tnode = &tree[(size_t)new_node<<3]; // current tree node
  const double hw = cw * 0.5; // Half the cell width (i.e. width of next cell)
  
  // find the new nodes for each of the children
  *tnode++ = refine_BH_ellipsoid(tree, n_nodes,MAX_NODES, n_leaves, 
		       max_refine-1, M, bh_crit, x0, y0,z0,hw);
  *tnode++ = refine_BH_ellipsoid(tree, n_nodes,MAX_NODES, n_leaves, 
		       max_refine-1, M, bh_crit, x0, y0,z0+hw,hw);
  *tnode++ = refine_BH_ellipsoid(tree, n_nodes,MAX_NODES, n_leaves, 
		       max_refine-1, M, bh_crit, x0, y0+hw,z0,hw);
  *tnode++ = refine_BH_ellipsoid(tree, n_nodes,MAX_NODES, n_leaves, 
		       max_refine-1, M, bh_crit, x0, y0+hw,z0+hw,hw);
  *tnode++ = refine_BH_ellipsoid(tree, n_nodes,MAX_NODES, n_leaves, 
		       max_refine-1, M, bh_crit, x0+hw, y0,z0,hw);
  *tnode++ = refine_BH_ellipsoid(tree, n_nodes,MAX_NODES, n_leaves, 
		       max_refine-1, M, bh_crit, x0+hw, y0,z0+hw,hw);
  *tnode++ = refine_BH_ellipsoid(tree, n_nodes,MAX_NODES, n_leaves, 
		       max_refine-1, M, bh_crit, x0+hw, y0+hw,z0,hw);
  *tnode   = refine_BH_ellipsoid(tree, n_nodes,MAX_NODES, n_leaves, 
		       max_refine-1, M, bh_crit, x0+hw, y0+hw,z0+hw,hw);

  // return (negative of) this new node
  return -new_node;
}


static int refine_leaves_to_levels(int *tree, const size_t index,int *n_nodes,const int depth,const int MAX_NODES, const int *levels, int *n_leaves)
{
  /*
    Force a refinement up to given levels
    *tree    - the tree
    index    - index we are looking at
    *n_nodes - number of nodes created so far
    depth    - current depth of node
    MAX_NODES- maximum number of nodes
    levels   - levels we are allowed to refine to (e.g. 0,5,10)
    n_leaves - number of leaves created so far

    return 0 or -1 on out-of-nodes
  */
 
  if (depth>*levels) 
    levels++; // go to next higher level in list

  if (tree[index]<0)
    {
      // We found a node, recurse through the children
      size_t node_idx = (size_t)(-tree[index])<<3;

      for (int octant=0;octant<8;octant++)
	if (refine_leaves_to_levels(tree, node_idx+octant,n_nodes,depth+1,MAX_NODES, levels, n_leaves))
	  return -1;

      // done and all ok!
      return 0;
    }
  // We have found a leaf, check if the depth ok
  // Recurse through the list
  if (depth==*levels)
    return 0; // Depth was in the list, so leaf is ok
      
  // Need to refine more...
  const int new_node = *n_nodes; // make a copy, that one gets changed
  if (new_node==MAX_NODES)
    {
      // ran out of nodes
      *n_nodes = -1;
      return -1;
    }
  // else make a new node (with this leaf)
  *n_nodes = new_node + 1;
  const int leaf = tree[index];
  tree[index] = -new_node;
  
  size_t tidx = (size_t)new_node<<3; 
  
  tree[tidx] = leaf; // the old leaf is the first leaf of the new node
  // other 7 are newly minted...
  for (int i=0;i<7;i++)
    tree[tidx+i+1] = (*n_leaves)++;
  
  // Now check all of these for further refinement
  for (int i=0;i<8;i++)
      if (refine_leaves_to_levels(tree, tidx+i,n_nodes,depth+1,MAX_NODES, levels, n_leaves))
	return -1;

  // done, and all was ok!
  return 0;
  
}

int make_BH_refinement(int *tree, const int MAX_NODES, 
		       const int max_refine, const double rmin, const float bh_crit, const double *pos)
{

  /* Make the BH-tree up to the given refinement. Returns either
   the total nodes used, or -1 if we ran out of nodes. */
  int total_leaves = 0;
  int total_nodes = 0;
  

  refine_BH(tree, &total_nodes, MAX_NODES, &total_leaves,
	    max_refine, rmin, bh_crit,
	    -pos[0], -pos[1], -pos[2], 1.0);
  
  return total_nodes;

}
int make_BH_ellipsoid_refinement(int *tree, const int MAX_NODES, 
				 const int max_refine, const double *A, const double k,
				 const float bh_crit, const double *pos)
{

  /* 
     Refine the tree around the given ellipsoid.
     
     A,k - 3x3 matrix (symmetric) and constant such that the ellipsoid surface
           is defined by x.A.x = k, and the largest eigenvalue of A is 1.

     Returns either
     the total nodes used, or -1 if we ran out of nodes. 
  */
  // Matrix to hold the ellipse data (6 values of upper triangular matrix and constant)
  const double ellipsoid[7] = {A[0], A[1], A[2], A[4], A[5], A[8], k}; 
  int total_leaves = 0, total_nodes = 0;
  refine_BH_ellipsoid(tree, &total_nodes, MAX_NODES, &total_leaves,
		      max_refine, ellipsoid, bh_crit, -pos[0], -pos[1], -pos[2], 1.0);
  
  return total_nodes;

}

int force_refinement_levels(int *tree, const int total_nodes, const int MAX_NODES, const int *levels)
{
  /* 
     Force the tree to have a given refinement level
     
     *tree       - the tree (needs to be non-empty)
     total_nodes - number of nodes in the tree
     MAX_NODES   - maximum number of nodes we are allowed
     levels      - the allowed refinement levels (e.g. [4, 8,9,10]). Must be sorted, and the
                   largest>= the deepest leaf depth.
  */

  int total_leaves = 0;
  
  int new_total_nodes = total_nodes;
  // Count the number of leaves
  if (total_nodes==0)
    {
      // The whole tree is one leaf
      // is this ok?
      if (levels[0]==0)
	return 0; // Yes!

      // Otherwise make a root node and check it
      if (MAX_NODES==0)
	return -1; // Who asks to refine a tree with no memory for it?

      new_total_nodes = 1;
      for (int i=0;i<8;i++)
	{
	  tree[i] = total_leaves++; // make a new leaf
	  if (refine_leaves_to_levels(tree, i,&new_total_nodes,1,MAX_NODES, levels, &total_leaves))
	    return -1;
	}
      return new_total_nodes;

    }

  // There is a root node, lets count the leaves...
  for (size_t i=0; i<(size_t)total_nodes<<3;i++)
    if (tree[i]>=0) total_leaves++;
  
  
  // Force the refinement for each child of the root node, if we run out
  // of nodes return.
  for (int i=0;i<8;i++)
    if (refine_leaves_to_levels(tree, i,&new_total_nodes,1,MAX_NODES, levels, &total_leaves))
      return -1;
  
  return new_total_nodes;

}
static void leaves_centres_depths(const int idx,const int *tree, const double cw, const int depth, 
			    const double x0, const double y0, const double z0, int *depths, 
			    double *centres)
{
  /*
    recursively find the depths and centres of all the leaves
    idx      - index to examine (-ve => node, +ve => leaf)
    *tree    - the tree
    cw       - cell width
    depth    - current depth
    x0,y0,z0 - minima of cell (i.e. top left corner)
    depths   - store the leaf depths
    centres  - store the half-width and centre of each cell (4*n array)
   */
  const double hw = cw*0.5; // cell half width

  if (idx>=0)
    {
      // Leaf found!
      depths[idx] = depth;
      double *ctr = centres + (const size_t)idx*3;
      *ctr++ = x0+hw;
      *ctr++ = y0+hw;
      *ctr   = z0+hw;
      return;
    }
  const int *cur = &tree[(const size_t)(-idx)<<3]; // node
  // Recurse through 8 children
  const int d = depth+1;
  leaves_centres_depths(*cur++, tree, hw,d, x0, y0, z0, depths, centres);
  leaves_centres_depths(*cur++, tree, hw,d, x0, y0, z0+hw, depths, centres);
  leaves_centres_depths(*cur++, tree, hw,d, x0, y0+hw, z0, depths, centres);
  leaves_centres_depths(*cur++, tree, hw,d, x0, y0+hw, z0+hw, depths, centres);
  leaves_centres_depths(*cur++, tree, hw,d, x0+hw, y0, z0, depths, centres);
  leaves_centres_depths(*cur++, tree, hw,d, x0+hw, y0, z0+hw, depths, centres);
  leaves_centres_depths(*cur++, tree, hw,d, x0+hw, y0+hw, z0, depths, centres);
  leaves_centres_depths(*cur, tree, hw,d, x0+hw, y0+hw, z0+hw, depths, centres);
}
void leaf_depths_centres(const int *tree, int *out_depths, double *out_centres)
{
  /*
    Find the depths and centres of every leaf. Must be at least 1 node.
    Centres are for the unit box (0-1), i.e. you have to scale.

    *tree   - tree
    *out_depths  - write the depths
    *out_centres - write the centres (n*3 array)
   */
 
  const int *tptr = tree; // Pointer to the first node

  // First node exists, follow the eight children
  leaves_centres_depths(*tptr++, tree, 0.5,1, 0,0,0, out_depths, out_centres);
  leaves_centres_depths(*tptr++, tree, 0.5,1, 0,0,0.5, out_depths, out_centres);
  leaves_centres_depths(*tptr++, tree, 0.5,1, 0,0.5,0, out_depths, out_centres);
  leaves_centres_depths(*tptr++, tree, 0.5,1, 0,0.5,0.5, out_depths, out_centres);
  leaves_centres_depths(*tptr++, tree, 0.5,1, 0.5,0,0, out_depths, out_centres);
  leaves_centres_depths(*tptr++, tree, 0.5,1, 0.5,0,0.5, out_depths, out_centres);
  leaves_centres_depths(*tptr++, tree, 0.5,1, 0.5,0.5,0, out_depths, out_centres);
  leaves_centres_depths(*tptr,   tree, 0.5,1, 0.5,0.5,0.5, out_depths, out_centres);
}




int leaves_in_box_ex_node(const int *tree, const int idx, const int excl_node, const int n_leaves, 
			  const double x0, const double y0, const double z0, 
			  const float chw, const float box_w, const int max_ngbs, int *out)
{
  /*
    Recursively find all the particles within the box 
    [x0,x0+box_w],[y0,y0+box_w],[z0,z0+box_w], *excluding* the given node.
    This is useful for finding the neighbour particles of those selected from a node.

    x0,y0,z0 - Relative position of the top left corner of the node to that of the cell
    chw      - cell half width
    box_w    - box in which we want all the particles (excluding those in the node)

   */ 

  // Ok, check through the sub-cells
  int leaves_found = 0, child_ngbs;
  for (int rx=0;rx<2;rx++) // right x
    {						
      const double x1 = x0 - rx*chw;

      if (disjoint_interval(x1, chw, box_w)) // Overlap in the x-coord?
	continue;
      for (int ry=0;ry<2;ry++)
	{
	  const double y1 = y0 - ry*chw; // Overlap in the y-coord?
	  if (disjoint_interval(y1, chw, box_w))
	    continue;
	  for (int rz=0;rz<2;rz++)
	    {
	      const double z1 = z0 - rz*chw;
	      if (disjoint_interval(z1, chw, box_w))
		continue;

	      int child = tree[((idx<<1 |rx)<<1 | ry)<<1 | rz];

	      if (child==n_leaves)
		continue; // Empty
	      if (child>=0)
		{
		  // Is a leaf, add
		  if (leaves_found==max_ngbs)
		    return -1; // Out of space!
		  out[leaves_found++] = child;
		  continue;
		}
	      if (child==-excl_node)
		continue; // Ignore self

	      // Otherwise a node
	      child_ngbs = leaves_in_box_ex_node(tree, -child, excl_node, n_leaves, 
						 x1,y1,z1, chw*0.5, box_w, 
						 max_ngbs-leaves_found, out+leaves_found);
	      if (child_ngbs<0)
		return -1; // out of space!
	      leaves_found += child_ngbs;
	    }
	}
    }
  return leaves_found;
}


int count_leaves(const int *tree, const int node, const int n_leaves, int *out)
{
  /*
    Count the number of leaves for every node in the octree, put them in *out.

    tree     - octree data (8 ints per node)
    node     - node to count for
    n_leaves - total number of leaves
    out      - for sizes (needs to have space for 1 int per node)
    
    Return total number of leaves
   */
  int leaves_found=0;
  for (int octant=0;octant<8;octant++)
    {
      const int child = tree[node<<3 | octant];
      if (child==n_leaves)
	continue; // empty
      if (child>=0)
	leaves_found++; // a leaf
      else // a node
	leaves_found += count_leaves(tree, -child, n_leaves, out);
    }
  out[node] = leaves_found;
  return leaves_found;
}

static int add_p_to_octree(int *tree, const size_t idx, const int n_leaves, 
			    const double x0, const double y0, const double z0,
			    const float chw, const int max_nodes, const double *leaf_centres,
			    const int leaf, int n_nodes)
{
  
  /*
    Add add a particle with mass and centre-of-mass to the octtree

    x0,y0,z0  - cell minima (top left corner)
    chw       - cell half width
    returns -1 on out-of-nodes
   */

  // Ok, which octant does this go in?
  const int rx = leaf_centres[leaf*3] > x0 + chw ? 1 : 0;
  const int ry = leaf_centres[leaf*3+1] > y0 + chw ? 1 : 0;
  const int rz = leaf_centres[leaf*3+2] > z0 + chw ? 1 : 0;
  const int octant = (rx<<1 | ry)<<1 | rz;

  const int child = tree[(size_t)idx<<3 | octant];
  if (child==n_leaves)
    {
      // Great, its empty, just add this particle
      tree[idx<<3 | octant] = leaf;
      return n_nodes;
    }
  // Ok we will need the cell data of this octant

  // Upper left corner of the octant
  const double x1 = x0 + chw*rx, y1 = y0+chw*ry, z1=z0+chw*rz;

  if (child<0) // This is a node, just recurse
      return add_p_to_octree(tree, -child, n_leaves, 
			     x1, y1, z1,chw*0.5,
			     max_nodes, leaf_centres, leaf, n_nodes);

  // This is a leaf, we need to split the octant and add both leaves
  if (n_nodes==max_nodes)
    return -1; // out of nodes
  
  const size_t new_node = n_nodes++; // make a copy and update

  for (int i=0;i<8;i++)
    tree[new_node<<3 | i] = n_leaves; // No children

  tree[idx<<3 | octant] = -new_node;

  // Add the old child
  n_nodes = add_p_to_octree(tree, new_node, n_leaves, x1, y1, z1,chw*0.5,
			    max_nodes, leaf_centres, child, n_nodes);
  if (n_nodes==-1)
    return -1; // check for out-of-nodes

  // Add the original leaf
  return add_p_to_octree(tree, new_node, n_leaves, x1, y1, z1,chw*0.5,
			 max_nodes, leaf_centres,leaf, n_nodes);
}


int octree_from_pos(int *tree,const int max_nodes, const double *pos, const int n_pos)
{
  /*
    Build an octtree based on the particle data

    *tree      - where we write tree connectivity, i.e. (max_nodes * 8) array
    max_nodes  - maximum number of nodes we can add to the tree
    *pos       - position (x,y,z) of each particle, used to make leaves
    n_leaves   - number of particles for the tree

    returns number of nodes used or -1 if we ran out of nodes
   */

  if (n_pos==0)
    return 0; // No leaves to add. That was easy

  if (max_nodes<1)
    return -1; // Leaves to add, but no tree to add them to!
  
  int n_nodes=1;
  for (int i=0;i<8;i++)
    tree[i] = n_pos; // Set octants of 1st node to n_pos to indicate empty

  // Now add every particle to root node (0)
  for (int i=0;i<n_pos; i++)
    {
      n_nodes = add_p_to_octree(tree, 0, n_pos, 0.0, 0.0,0.0, 0.5, max_nodes, pos, i, n_nodes);
      if (n_nodes==-1)
	return -1;
    }
  return n_nodes;
}


void octree_mcom(const int *tree, const int n_pos, const double *p_mcom, double *out)
{
  /*
    For the given tree, add the mass and mass*centre for each node

    tree   -  Connectivity of the tree
    n_pos  - number of positions to add
    p_mcom - (n_pos * 4) array of mass,x,y,z for each particle
    out    - where we add the mass and mass*centre. 
             WARNING: This is only added, so you need to zero it first.

    Returns - void
  */
  if (n_pos==0)
    return; // No positions to add, that was easy.

  
  // Loop over particles
  for (int p=0;p<n_pos;p++)
    {
      int node = 0;
      float chw = 0.5; // Cell half-width
      
      double mass=p_mcom[p<<2],x=p_mcom[p<<2|1], y=p_mcom[p<<2|2], z=p_mcom[p<<2|3];
      const double mx=mass*x, my=mass*y,mz=mass*z; // Mass * centre

      if (mass==0)
	continue; // Massless particles can be skipped

      out[0] += mass; // Update the mass and mass*centre
      out[1] += mx;
      out[2] += my;
      out[3] += mz;
      
      int rx = x>chw, ry=y>chw,rz=z>chw;
      while ((node=-tree[((node<<1 | rx)<<1 | ry)<<1 | rz])>0) // Check the next is a node
	{
	  // Recurse into node
	  x -= rx * chw;
	  y -= ry * chw;
	  z -= rz * chw;
	  chw *= 0.5;
	  
	  out[node<<2]   += mass; // Update the mass and mass*centre
	  out[node<<2|1] += mx;
	  out[node<<2|2] += my;
	  out[node<<2|3] += mz;

	  // Find the octant for the next cell
	  rx = x>chw;
	  ry = y>chw;
	  rz = z>chw;
	}
    }


}
int write_leaves(int *tree,const int n_leaves, const int node, int *out)
{
  /* 
     Write all the leaves for the given node in out.
     (Use count_leaves to make sure out is big enough to hold all leaves)
   */
  int n_found = 0;
  for (int octant=0;octant<8;octant++)
    {
      const int child = tree[node<<3 | octant];
      if (child==n_leaves)
	continue; // Empty
      if (child>=0)
	out[n_found++] = child; // Leaf
      else // Node
	n_found += write_leaves(tree, n_leaves, -child, &out[n_found]);
    }
  return n_found;
}
