 
 #ifndef PARTICLES_KERNEL_H
#define PARTICLES_KERNEL_H

#ifndef __DEVICE_EMULATION__
#define USE_TEX 0
#endif

#if USE_TEX
#define FETCH(t, i) tex1Dfetch(t##Tex, i)
#else
#define FETCH(t, i) t[i]
#endif

#include "vector_types.h"
typedef unsigned int uint;

// simulation parameters
struct SimParams {

	
    float viscosity;
	float uf;
	float Cpol; //moment = Cpol*H
	
	//set of arrays for initializing the particles
	float pRadius [3];
	float mu_p [3];
	float volfr [3];	
	int nump[3];
	float rstd[3];

	uint3 gridSize;
	uint numCells;
    float3 worldOrigin;
	float3 cellSize;

 	//total number of particles
	uint numBodies;    
	
	float3 externalH;
	int interactionr;
	float shear;

    float globalDamping;
	float cspring;
    float spring;
    float cdamping;
    float boundaryDamping;

	int mutDipIter;

	float colorFmax;

	float flowvel;
	float nd_plug;
	bool flowmode;
};

#endif
