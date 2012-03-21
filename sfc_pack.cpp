#include <vector>
#include <assert.h>
#include "particles_kernel.cuh"
#include "sfc_pack.h"

//! x walking table for the hilbert curve
static int istep[] = {0, 0, 0, 0, 1, 1, 1, 1};
//! y walking table for the hilbert curve
static int jstep[] = {0, 0, 1, 1, 1, 1, 0, 0};
//! z walking table for the hilbert curve
static int kstep[] = {0, 1, 1, 0, 0, 1, 1, 0};


//! Helper function for recursive hilbert curve generation
/*! \param result Output sequence to be permuted by rule 1
    \param in Input sequence
*/
 static void permute1(unsigned int result[8], const unsigned int in[8])
    {
    result[0] = in[0];
    result[1] = in[3];
    result[2] = in[4];
    result[3] = in[7];
    result[4] = in[6];
    result[5] = in[5];
    result[6] = in[2];
    result[7] = in[1];
    }

//! Helper function for recursive hilbert curve generation
/*! \param result Output sequence to be permuted by rule 2
    \param in Input sequence
*/
 static void permute2(unsigned int result[8], const unsigned int in[8])
    {
    result[0] = in[0];
    result[1] = in[7];
    result[2] = in[6];
    result[3] = in[1];
    result[4] = in[2];
    result[5] = in[5];
    result[6] = in[4];
    result[7] = in[3];
    }

//! Helper function for recursive hilbert curve generation
/*! \param result Output sequence to be permuted by rule 3
    \param in Input sequence
*/
 static void permute3(unsigned int result[8], const unsigned int in[8])
    {
    permute2(result, in);
    }

//! Helper function for recursive hilbert curve generation
/*! \param result Output sequence to be permuted by rule 4
    \param in Input sequence
*/
 static void permute4(unsigned int result[8], const unsigned int in[8])
    {
    result[0] = in[2];
    result[1] = in[3];
    result[2] = in[0];
    result[3] = in[1];
    result[4] = in[6];
    result[5] = in[7];
    result[6] = in[4];
    result[7] = in[5];
    }

//! Helper function for recursive hilbert curve generation
/*! \param result Output sequence to be permuted by rule 5
    \param in Input sequence
*/
 static void permute5(unsigned int result[8], const unsigned int in[8])
    {
    permute4(result, in);
    }

//! Helper function for recursive hilbert curve generation
/*! \param result Output sequence to be permuted by rule 6
    \param in Input sequence
*/
 static void permute6(unsigned int result[8], const unsigned int in[8])
    {
    result[0] = in[4];
    result[1] = in[3];
    result[2] = in[2];
    result[3] = in[5];
    result[4] = in[6];
    result[5] = in[1];
    result[6] = in[0];
    result[7] = in[7];
    }

//! Helper function for recursive hilbert curve generation
/*! \param result Output sequence to be permuted by rule 7
    \param in Input sequence
*/
 static void permute7(unsigned int result[8], const unsigned int in[8])
    {
    permute6(result, in);
    }

//! Helper function for recursive hilbert curve generation
/*! \param result Output sequence to be permuted by rule 8
    \param in Input sequence
*/
 static void permute8(unsigned int result[8], const unsigned int in[8])
    {
    result[0] = in[6];
    result[1] = in[5];
    result[2] = in[2];
    result[3] = in[1];
    result[4] = in[0];
    result[5] = in[3];
    result[6] = in[4];
    result[7] = in[7];
    }

//! Helper function for recursive hilbert curve generation
/*! \param result Output sequence to be permuted by rule \a p-1
    \param in Input sequence
    \param p permutation rule to apply
*/
void permute(unsigned int result[8], const unsigned int in[8], int p)
    {
    switch (p)
        {
        case 0:
            permute1(result, in);
            break;
        case 1:
            permute2(result, in);
            break;
        case 2:
            permute3(result, in);
            break;
        case 3:
            permute4(result, in);
            break;
        case 4:
            permute5(result, in);
            break;
        case 5:
            permute6(result, in);
            break;
        case 6:
            permute7(result, in);
            break;
        case 7:
            permute8(result, in);
            break;
        default:
            assert(false);
        }
    }

//! recursive function for generating hilbert curve traversal order
/*! \param i Current x coordinate in grid
    \param j Current y coordinate in grid
    \param k Current z coordinate in grid
    \param w Number of grid cells wide at the current recursion level
    \param Mx Width of the entire grid (it is cubic, same width in all 3 directions)
    \param cell_order Current permutation order to traverse cells along
    \param traversal_order Traversal order to build up
    \pre \a traversal_order.size() == 0
    \pre Initial call should be with \a i = \a j = \a k = 0, \a w = \a Mx, \a cell_order = (0,1,2,3,4,5,6,7,8)
    \post traversal order contains the grid index (i*Mx*Mx + j*Mx + k) of each grid point
        listed in the order of the hilbert curve
*/
 static void generateTraversalOrder(int i, int j, int k, int w, int Mx, 
		unsigned int cell_order[8], std::vector< unsigned int > &traversal_order)
{
    if (w == 1)
        {
        // handle base case
        traversal_order.push_back(i*Mx*Mx + j*Mx + k);
        }
    else
        {
        // handle arbitrary case, split the box into 8 sub boxes
        w = w / 2;
        
        // we ned to handle each sub box in the order defined by cell order
        for (int m = 0; m < 8; m++)
            {
            unsigned int cur_cell = cell_order[m];
            int ic = i + w * istep[cur_cell];
            int jc = j + w * jstep[cur_cell];
            int kc = k + w * kstep[cur_cell];
            
            unsigned int child_cell_order[8];
            permute(child_cell_order, cell_order, m);
            generateTraversalOrder(ic,jc,kc,w,Mx, child_cell_order, traversal_order);
            }
        }
}


void getSortedOrder3D( unsigned int* m_hCellHash, const SimParams* params)
{
    // start by checking the saneness of some member variables
    assert(params->gridSize.x == params->gridSize.y && params->gridSize.y == params->gridSize.z);
	unsigned int m_grid = params->gridSize.x;
	
	std::vector<unsigned int> reverse_order(m_grid*m_grid*m_grid);
	reverse_order.clear();
	// we need to start the hilbert curve with a seed order 0,1,2,3,4,5,6,7
	unsigned int cell_order[8] = {0,1,2,3,4,5,6,7};
	generateTraversalOrder(0,0,0, m_grid, m_grid, cell_order, reverse_order);
	
	for (unsigned int i = 0; i < m_grid*m_grid*m_grid; i++){
		m_hCellHash[reverse_order[i]] = i;
	}


}

