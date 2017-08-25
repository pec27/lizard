/*
  Module for octree calculations, i.e. find where to put points into a 
  bucketed octree, find the weights and center-of-mass (CoM) of the nodes and 
  then construct a TreeIterator so that this can be walked efficiently. 

  TODO - could still make the 2-pass tree construction more efficient
  in memory by putting octree indices at end of temp and filling from start.
  Would be more speed efficient to do the breadth sort per root also. 

  Peter Creasey - November 2016
 */
#include <math.h>
#include <stdlib.h> // for malloc
#include "tree.h"
#include <string.h> // for memcpy, TODO get rid of with better memory management

// (maximum) number of bits in the fractional part  of a double (assuming not
// subnormal). This effectively determines how deep an octree could ever be.
#define DBL_BITS 52


int get_tree_iterator_size(void)
{
  /* For array allocation */
  return sizeof(TreeIterator); 
}


static int build_tree_xyzw(const int root, const TreeIterator *restrict tree,
			    const int num_nodes, double *restrict xyzw)
{
  /*
    Build xyzw (CoM and M) for every node in the tree
    
    return the maximum depth of any leaf
   */
  const int root_pt = tree[root].istart;
  const int pt_end = root_pt + tree[root].n;

  for (int pt_idx=root_pt, cur_node=root;pt_idx<pt_end;)
    {
      while (tree[cur_node].depth_next) // Recurse to leaves
	cur_node += tree[cur_node].depth_next;

      const int com_idx = tree[cur_node].com_idx;

      // Start with position 0 in the bucket
      int i = tree[cur_node].istart;
      xyzw[com_idx]   = xyzw[i<<2]   * xyzw[i<<2|3];
      xyzw[com_idx+1] = xyzw[i<<2|1] * xyzw[i<<2|3];
      xyzw[com_idx+2] = xyzw[i<<2|2] * xyzw[i<<2|3];
      xyzw[com_idx+3] = xyzw[i<<2|3];

      const int i_end = tree[cur_node].istart + tree[cur_node].n;

      for (i++;i<i_end;++i)
	{
	  xyzw[com_idx]   += xyzw[i<<2]   * xyzw[i<<2|3];
	  xyzw[com_idx+1] += xyzw[i<<2|1] * xyzw[i<<2|3];
	  xyzw[com_idx+2] += xyzw[i<<2|2] * xyzw[i<<2|3];
	  xyzw[com_idx+3] += xyzw[i<<2|3];
	}
      cur_node = tree[cur_node].breadth_next;
      pt_idx = i_end;
    }

  // Fill M*CoM, M for all *branches*, using backward sweep
  for (int i=root+num_nodes-1;i>=root;--i)
    {
      if (!tree[i].depth_next)
	continue; // Leaf, continue

      // Branch, add up CoM*M of my children
      int child = i + tree[i].depth_next;
      for (int j=0;j<4;j++)
	xyzw[tree[i].com_idx+j] = xyzw[tree[child].com_idx+j];

      // Cycle through my children
      for (int pts = tree[child].n;pts<tree[i].n;pts+=tree[child].n) 
	{
	  child = tree[child].breadth_next;
	  for (int j=0;j<4;j++)
	    xyzw[tree[i].com_idx+j] += xyzw[tree[child].com_idx+j];
	  
	}
    }
  int max_depth=0;

  // Normalize CoM*M -> CoM
  for (int i=root;i<root+num_nodes;i++)
    {
      const double inv_wt = 1.0/xyzw[tree[i].com_idx+3];

      for (int j=0;j<3;j++)
	xyzw[tree[i].com_idx+j] *= inv_wt;

      if (max_depth<tree[i].depth)
	max_depth = tree[i].depth;
    }
  return max_depth;
}
/* 
   Temporary structure where we decorate the positions with their integer 
   representations so we can use integer ops to sort them
*/
typedef struct {
  int32_t root, idx; // Grid cell and initial index
  uint64_t ifrac[3]; // integer representations of fractional part of posn
} tree_idx;

static inline int _cmp_octree_idx(const void *a, const void *b)
{
  /* 
     Comparator function for octree elements, essentially just compare in order
     1 - Which octree
     2 - Index within the octree (i.e. bucket)
     3 - initial order
  */
  const tree_idx *ta = a, *tb = b;

  // Diff root node
  if (ta->root!=tb->root)
    return (ta->root < tb->root) ? -1 : 1;
  
  // XOR the integer representations, then look for leading bit.
  const uint64_t cmp[3] = {ta->ifrac[0]^tb->ifrac[0],
			   ta->ifrac[1]^tb->ifrac[1],
			   ta->ifrac[2]^tb->ifrac[2]};

  // Bah voodoo code. 
  // If differences in the first (x) coordinate are leading, return 1/-1 on 
  // whether for a_x > b_x, if second a_y>b_y etc. Finally check the original index

  int idx_cmp = 0;


  if ((cmp[1]>cmp[0]) && ((cmp[1]^cmp[0])>cmp[0])) // check leading bit is higher
    idx_cmp=1;
  if ((cmp[2]>cmp[idx_cmp]) && ((cmp[2]^cmp[idx_cmp])>cmp[idx_cmp]))
    idx_cmp=2;
  if (cmp[idx_cmp])
    return (ta->ifrac[idx_cmp]>tb->ifrac[idx_cmp])*2-1;
  else
    return (ta->idx > tb->idx)*2-1;
}

static int inline compare_to_depth(const uint64_t *restrict ia, const uint64_t *restrict ib,
				   const int depth)
				
{
  /*
    Compare the tree indices down to level 'depth'.

    Return the depth difference, i.e. if key_a and key_b are same for the first 
    15 octants, but we asked to compare to depth 20, then then result would
    be 5 (=20-15). If all requested are the same then return 0.

   */

  int num_diff_octants=0;

  for (uint64_t diff = ((ia[0]^ib[0]) | 
			(ia[1]^ib[1]) |
			(ia[2]^ib[2]))>>(DBL_BITS - depth);
       diff; ++num_diff_octants)
    diff>>=1;

  return num_diff_octants;
}

int build_octree_iterator(const double *pos, const int num_pos, const int nx, const int bucket_size,
				int *restrict sort_idx, double *restrict out, const int buf_size)
{
  /* 
     Function to sort the positions in their octree order, and build a set
     of tree nodes to cover them (down to bucket_size)

     buf_size - size of buffer (in doubles)
     
     returns num_roots - the number of roots (filled grid cells), or -1 on out-of-mem

     out - Array where first num_roots elements are number of nodes in that root, followed by the tree iterators
  */
 
  const int stack_size = DBL_BITS+1;
  int node_stack[stack_size];


  if (!num_pos)
    return 0; // (later assume at least one root)

  tree_idx *octree_idx = (tree_idx*)malloc(num_pos* sizeof(tree_idx));

  
  if (!octree_idx)
    return -1; // out of memory
  // Fill in the indices
  const double dbl_to_int = 1L<<DBL_BITS;


  for (int i=0;i<num_pos;++i)
    {
      const double pnx=pos[i*3]*nx, pny = pos[i*3+1]*nx, pnz = pos[i*3+2]*nx;
	
      // First integer is the grid position
      octree_idx[i].root = ((int)(pnx)*nx + (int)pny)*nx + pnz;
      octree_idx[i].idx = i; // initial idx
      // Next 3 are the ifracs, 52 bit representations of doubles
      octree_idx[i].ifrac[0] = (pnx - floor(pnx)) * dbl_to_int;
      octree_idx[i].ifrac[1] = (pny - floor(pny)) * dbl_to_int;
      octree_idx[i].ifrac[2] = (pnz - floor(pnz)) * dbl_to_int;
    }


  // Sort by grid pos (root), then where it would appear in octree, finally by
  // initial idx
  qsort(octree_idx, num_pos, sizeof(tree_idx), _cmp_octree_idx);
  
  // Count the distinct roots, copy over the sort indices
  int num_roots=1;
  sort_idx[0] = octree_idx[0].idx;
  for (int i=1;i<num_pos;++i)
    {
      sort_idx[i] = octree_idx[i].idx;
      if (octree_idx[i].root!=octree_idx[i-1].root)
	num_roots++;
    }

  const int max_nodes = ((buf_size - num_roots) * sizeof(double))/sizeof(TreeIterator);

  
  // Sweep over tree-ordered points, put into buckets
  if (max_nodes<=0)
    return -1; // out of memory

  int32_t *root_counts = (int32_t *)out;
  int32_t *root_cells = &root_counts[num_roots]; // Start this buffer at end


  int cur_depth=0, num_nodes=1, levels_done;

  /* Temporary nodes created during the initial sweep */
  typedef struct {
    int n, depth; // num points and depth of node
  } bare_node;

  bare_node *nodes = (bare_node *)(&root_cells[num_roots]);
  nodes[0].n = 0; // initial index
  nodes[0].depth = 0;
  node_stack[0] = 0;
  int old_nodes = 0;
  for (int i=1,i0=0, iroot=0;i<num_pos;++i)
    {
      // Different at this depth? Then start a new bucket
      if (octree_idx[i].root != octree_idx[i0].root)
	{
	  root_cells[iroot] = octree_idx[i0].root;
	  root_counts[iroot++] = num_nodes - old_nodes;
	  old_nodes = num_nodes;
	  levels_done = cur_depth+1; // Different roots... close all levels
	}
      else if (!(levels_done=compare_to_depth(octree_idx[i].ifrac, 
					      octree_idx[i0].ifrac, cur_depth)))
	{      
	  // Not different at this depth... want to put point in this bucket
	  
	  if (i-i0<bucket_size)
	    continue; // Add to bucket
	  
	  // Reached maximum bucket size, must split
	  if ((++cur_depth)==stack_size)
	    return -2; // Too many points to separate into leaves


	  if (num_nodes==max_nodes)
	    return -1; // out of memory
	  
	  i = nodes[num_nodes].n = i0; // Copy parent start
	  nodes[num_nodes].depth = cur_depth;
	  node_stack[cur_depth] = num_nodes++;
	  continue; // Retry with this point
	}
      
      // Close levels_done levels
      do { 
	const int pop = node_stack[cur_depth--];
	i0 = nodes[pop].n;
	nodes[pop].n = i - i0; // Now contains number of points
      } while (--levels_done);
      
      // Make a new child
      if (num_nodes==max_nodes)
	return -1; // out of memory
      
      node_stack[++cur_depth] = num_nodes;
      nodes[num_nodes].depth = cur_depth;
      i0 = nodes[num_nodes++].n = i; // Start with zero points

    }
  root_counts[num_roots-1] = num_nodes - old_nodes;
  root_cells[num_roots-1] = octree_idx[num_pos-1].root;


  // Every node in the stack needs to be closed, incl. root
  do {
    const int pop = node_stack[cur_depth];
    nodes[pop].n = num_pos - nodes[pop].n; // Now contains number of points in branch
  } while (cur_depth--);

  // No longer need the octree index, can re-use it as a buffer
  memcpy(octree_idx, nodes, num_nodes*sizeof(bare_node));


  // Re-sort by breadth into a tree iterator
  TreeIterator *tree_it = (TreeIterator *)nodes; // Swap buffers
  nodes = (bare_node *)octree_idx;

  int depth_counts[stack_size]; // Number of nodes at each depth, used when we want to re-sort by breadth
  int start_idx[stack_size];
  int prev_max_depth = stack_size; // to clear for next node
  int pt_cur=0; // current point

  for (int iroot=0, node=0;iroot<num_roots;++iroot)
    {
      // Nodes for this root

      // reset counts at each depth
      for (int i=0;i<prev_max_depth;++i)
	depth_counts[i] = 0;

      const int next_root = node+root_counts[iroot];

      // Count nodes per level, excl. root
      for (int i=node+1;i<next_root;++i)
	depth_counts[nodes[i].depth-1]++;
      // Build the start indices when re-sorted by breadth
      start_idx[1] = node+1; // Always one root
      for (prev_max_depth=1;start_idx[prev_max_depth]+depth_counts[prev_max_depth-1]<next_root;++prev_max_depth)
	start_idx[prev_max_depth+1] = start_idx[prev_max_depth] + depth_counts[prev_max_depth-1];

      // Add root to iterator
      if ((tree_it[node].n = nodes[node].n)<=bucket_size)
	tree_it[node].depth_next = 0;

      tree_it[node].depth = 0;
      tree_it[node].breadth_next = 0; 
      tree_it[node].breadth_prev = 0;
      tree_it[node].depth_prev = 0;
      start_idx[0] = ++node;

      // Add all subsequent nodes in the tree
      int cur_depth=0; 
      for (;node<next_root;++node)
	{
	  // Each node contains its number of points
	  const int j = start_idx[nodes[node].depth];
	  
	  tree_it[j].depth = nodes[node].depth;
	  tree_it[j].depth_prev = 0; // I have no children (yet)

	  const int parent = start_idx[tree_it[j].depth-1]-1; 
	  // My parents depth-prev is always the *last* child, so
	  tree_it[parent].depth_prev = j - parent;

	      
	  // Am I 1st child?
	  if (cur_depth<tree_it[j].depth)
	    {
	      // Prev node has me as depth_next (shift to me)
	      tree_it[parent].depth_next = tree_it[parent].depth_prev;
	      // My breadth-prev is same as my parent
	      tree_it[j].breadth_prev = tree_it[parent].breadth_prev;
	    }
	  else
	    {
	      // Breadth-prev is the last node at the same level (my sibling)
	      tree_it[j].breadth_prev = j-1;

	      // Every node >=my depth has me as its breadth-next
	      while (cur_depth>=tree_it[j].depth)
		tree_it[start_idx[cur_depth--]-1].breadth_next = j;
	    }
	  

	  cur_depth = tree_it[j].depth;
	  start_idx[cur_depth]++; // couldnt do earlier, because might point to sibling
	  
	  // Copy point count, mark as leaf
	  if ((tree_it[j].n = nodes[node].n)<=bucket_size)
	    tree_it[j].depth_next = 0;
	}

      // All remaining nodes have no breadth-next
      while (cur_depth)
	tree_it[start_idx[cur_depth--]-1].breadth_next = 0;

      // Fill in the particle starts (from n)
      int i=next_root-root_counts[iroot];
      do {
	tree_it[i].istart = pt_cur;
	tree_it[i].com_idx = (i + num_pos)<<2;
	
	if (tree_it[i].depth_next)
	  i += tree_it[i].depth_next;
	else
	  {
	    pt_cur += tree_it[i].n;
	    i = tree_it[i].breadth_next;
	  }
      }	while (i);
      
    }

  // clean up
  free(octree_idx);
  return num_roots;
}

int fill_treewalk_xyzw(const int num_trees, const double *restrict twn_ptr, const int32_t *restrict tree_sizes, 
		       double *restrict xyzw)
{
  /*

    Fill in the xyzw (mean position and total weight) over the tree iterator

    TODO should prob put this with build_tree_xyzw all in one function.

   */
  const TreeIterator *twn = (TreeIterator*)twn_ptr;
  int max_depth=0;

  for (int tree=0,root_idx=0;tree<num_trees;root_idx+=tree_sizes[tree++])
    {
      // Find the CoM and M of each node
      const int tree_depth = build_tree_xyzw(root_idx, twn, tree_sizes[tree], xyzw);
      if (max_depth<tree_depth)
	max_depth = tree_depth;
    }
  return max_depth;
}
