#ifndef NEW_KERN_CUH
#define NEW_KERN_CUH

#include "vector_types.h"

typedef unsigned int uint;

struct newParams {
	uint N;
	
	uint3 gridSize;
	uint numGridCells;	
	float3 cellSize;
	float3 worldOrigin;

	float3 L;
	float3 Linv;

	float max_ndr_sq;
	float max_fdr_sq;

	uint num_c_neigh;
	uint max_neigh;

	float spring;
	float uf;
	float viscosity;
	float shear;	
};
#endif
