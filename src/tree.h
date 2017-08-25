#ifndef TREE_H_
#define TREE_H_

#include <stdint.h> // for 32 bit integer


/*
  Bi-directional iterator for walking a tree (currently octree), has a 
  depth-next for recursing down to leaves, and a breadth-next for the next
  sibling (/aunt/great-aunt/great-great-aunt etc. as available).
 */
 
typedef struct {
  uint32_t com_idx; // idx of the CoM & M (in xyzw) of this node
  uint32_t istart; // Start index (in xyzw array) of the particles
  uint32_t breadth_prev; // Prev node in breadth-walk;
  int32_t depth_prev; //Prev node in depth-walk

  uint32_t breadth_next; // Next node in a breadth-first walk (sibling, aunt or 0 for done).
  int32_t depth_next; // *shift* to first child, 0 if leaf.
  uint32_t n; // Number of points in this branch. <= bucket_size indicates leaf
  uint32_t depth; // Number of levels deep (0 if root)
} TreeIterator;

#endif
