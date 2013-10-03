#ifndef NEW_KERN_CUH
#define NEW_KERN_CUH

#include "vector_types.h"

typedef unsigned int uint;

struct NewParams {
	uint N;
	
	uint3 gridSize;
	uint numCells;	
	float3 cellSize;
	float3 origin;
	uint numAdjCells;

	float3 L;
	float3 Linv;
	
	float max_fdr_sq;

	float spring;
//	float uf;
	float visc;
	float shear;

	float pin_d;
	
	float tanfric;

	float3 extH;
	float Cpol;	
};

uint iDivUp(uint a, uint b);


#endif
