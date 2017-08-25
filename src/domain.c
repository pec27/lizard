/*
  Domain code (region-growing voxels)
 */


static int greedy_region_grow_mask3d(const int flood_mask, int *grid, const int i0,const int j0, const int k0, const int ngrid, long *region_ctr, 
				     void *restrict buf, const int bufsize)
{
  /* 
     Region-growing algorithm for a 3d periodic grid, where the highest adjacent
     value is consumed first, and 'adjacent' refers the 6 faces of cubic cells.

     Starting from the given cell, grow the region by adding the highest non-
     masked neighbour until we have reached max_sum, or there are no more non-
     masked cells on the boundary.


     flood_mask - positive integer to insert into grid to mark cells
     ngrid      - integer side length of grid
     grid       - N^3 integer array of values indicating:
                    <=0 unused and available to be added to grid
		    >0 another mask
     i0,j0,k0   - coordinates of starting point (grid[i0,j0,k0]<=0)

     *region_ctr- Negative sum remaining to find (updated)
     
     Returns either 

     A) No more contiguous unmasked cells, return unexplained sum (max_sum - my_sum)
     B) Reached max_sum or below - return 0

*/
  
  /* It might seem counter-intuitive that we store the adjacent values in a 
     sorted linked-list as lookups are essentially linear in time. However one
     should recall that the data set we are searching is likely nearly 
     *continuous* and so the adjacent cells to the current minima are likely 
     nearly minima themselves, i.e. they will be inserted very near the head of
     the list.
  */
  const long N = ngrid;
  const int nm1 = ngrid-1;
  // Structure for linked list of cells neighbouring region
  struct Ngb_cell {
    int ijk[3];
    int val;
  };

  struct Ngb_cell *bdry = (struct Ngb_cell *)buf;
  const int bdry_size = bufsize/sizeof(struct Ngb_cell);

  if (bdry_size<1) return -1; // Buffer too small for 1 cell!

  bdry[0].ijk[0] = i0; bdry[0].ijk[1] = j0; bdry[0].ijk[2] = k0;
  bdry[0].val = grid[(i0*N+j0)*N+k0];
  grid[(i0*N+j0)*N+k0] = flood_mask; // mark as used

  int list_size=1;
  int cells_added=0;

  while ((*region_ctr)<0 && list_size)
    {
      /* Pop largest cell from list */
      const struct Ngb_cell cur_cell=bdry[--list_size];      
      *region_ctr -= cur_cell.val;
      cells_added++;


      //      printf("Adding cell at %d %d %d\n", cur_cell.ijk[0], cur_cell.ijk[1], cur_cell.ijk[2]);
      /* Add neighbours of current cell over each face*/      
      struct Ngb_cell adj_cell=cur_cell;

      for (int face=0;face<6;face++)
	{
	  const int axis = face>>1;
	  const int right_face = face&1; // left (0) or right (1) along axis
	  const int prev_i = cur_cell.ijk[axis];


	  /* voodoo code of i+/-1 wrapped (periodic) in 0...N-1 */
	  //	  adj_cell.ijk[axis] = prev_i-1 +2*right_face + N*((!prev_i) && (!right_face)) - N * ((prev_i==N-1) && right_face);
	  adj_cell.ijk[axis] = prev_i - 1 + 2*right_face + ngrid*(((!prev_i) & (!right_face)) - 
								  ((prev_i==nm1) & right_face));
	  //	  if (adj_cell.ijk[axis]==-1) adj_cell.ijk[axis]=N-1;
	  //	  if (adj_cell.ijk[axis]==N) adj_cell.ijk[axis]=0;

	  
	  //	  adj_cell.ijk[axis] = prev_i+1 - 2*right_face + N*(-((!right_face)|prev_i) +(right_face|(prev_i!=N-1)));

	  const long idx = (adj_cell.ijk[0]*N+adj_cell.ijk[1])*N+adj_cell.ijk[2];


	  if (grid[idx]<=0) /* was not already taken */
	    {
	      // Add to linked list
	      adj_cell.val=grid[idx];
	      // Mark as masked (will be unset if we don't use later)
	      grid[idx] = flood_mask;

	      // Add this cell via insertion sort, i.e. walk back from end until
	      // bigger cell is found
	      int ins_point=list_size++;
	      if (ins_point==bdry_size)
		return -1; // Out of memory

	      for (;ins_point;ins_point--)
		{
		  if (bdry[ins_point-1].val>=adj_cell.val)
		    break;
		  
		  bdry[ins_point] = bdry[ins_point-1]; 
		}
	      bdry[ins_point] = adj_cell;
	    }

	  adj_cell.ijk[axis] = prev_i; // Restore old value
	}
    }

  // Unmask all the unused cells to their old weights
  while (list_size)
    {
      /* Pop largest cell from list */
      const struct Ngb_cell cur_cell=bdry[--list_size];      
      const long idx = (cur_cell.ijk[0]*N+cur_cell.ijk[1])*N+cur_cell.ijk[2];
      grid[idx] = cur_cell.val;
    }



  //  if (region_rmdr>=0)
  //    return cells_added; // Filled enough



  // Else incomplete - ran into mask, need a new starting point
  return cells_added; 

}

int region_grow_domains3d(const int ngrid, const long max_sum, int *restrict grid,
			int *restrict bdrybuf, const int bufsize)
{
  /*
    In-place colour of a series of negative weights (grid[i,j,k]<=0) to have max_sum
    or less in all but the final colour. 

    ngrid   - side length ngrid^3 grid
    max_sum - negative integer indicating sum we want to reach.
    grid    - pointer to weights

    bdrybuf - buffer to hold boundary points
    bufsize - size of buffer (bytes)

    Returns -1 on out-of-memory, 0 otherwise
   */

  const long n=ngrid;
  int domain = 1; // First domain
  long my_sum = max_sum;

  while(1)
    {
      // Find min and its position
      int min=1, i0=0,j0=0,k0=0;

      for (int i=0;i<ngrid;i++)
	for (int j=0;j<ngrid;j++)
	  for (int k=0;k<ngrid;k++)
	    if (grid[(i*n+j)*n+k]<min)
	      {
		i0=i;
		j0=j;
		k0=k;
		min = grid[(i*n+j)*n+k];
	      }

      if (min>0) // All cells are in a domain
	return 0;

      //      partial_greedy_floodfill
      const int cells_added = greedy_region_grow_mask3d(domain, grid, i0,j0,k0, 
							n, &my_sum, bdrybuf, bufsize); 
      if (cells_added==-1)
	return -1; 	  // Out-of-memory

      //      printf("Minimum was %d, trying with domain %d, ", min, domain);
      //printf("cells added %d\n", cells_added);
      if (my_sum>=0)
	{
	  domain++; // Next domain
	  my_sum = max_sum; // reset counter
	}
    }
}
