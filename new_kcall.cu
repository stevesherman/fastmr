#include <cstdlib>
#include <cstdio>
#include <cstdlib>
#include <assert.h>

#include "vector_types.h"

#include "cutil_inline.h"
#include "thrust/reduce.h"
#include "thrust/extrema.h"
#include "thrust/device_ptr.h"
#include "thrust/functional.h"
#include "new_kern.cu"

extern "C" 
{

uint iDivUp2(uint a, uint b)
{
	return (a%b==0) ? (a/b) : (a/b + 1);
}

void comp_phash(float* dpos, uint* d_pHash, uint* d_pIndex, uint* d_CellHash, uint numParticles, uint numGridCells)
{
	uint numThreads = 256;
	uint numBlocks = iDivUp2(numParticles, numThreads);
/*	
	uint maxkey = 0;
	thrust::device_ptr<uint> dev_ptr(d_pHash);
	thrust::maximum<uint> mx;
	maxkey = thrust::reduce(dev_ptr, dev_ptr+numParticles, 0, mx);
	printf("omaxkey: %u\n", maxkey);		

	thrust::device_ptr<uint> hashes(d_CellHash);
	maxkey = thrust::reduce(hashes, hashes+numGridCells, 0, mx);
	printf("max hash: %u\n", maxkey);
*/

	comp_phashK<<<numBlocks, numThreads>>> ( (float4*) dpos, d_pHash, d_pIndex, d_CellHash);
	//cudaDeviceSynchronize();
	cutilCheckMsg("in phash computation");	
	/*
	maxkey = thrust::reduce(dev_ptr, dev_ptr+numParticles, 0, mx);
	printf("nmaxkey: %u\n", maxkey);*/
}


void setNParameters(newParams *hostParams){
	cudaMemcpyToSymbol(nparams, hostParams, sizeof(newParams));
}

void find_cellStart(uint* cellStart, uint* cellEnd, uint* phash, uint numParticles, uint numCells)
{
	uint numThreads = 128;
	uint numBlocks = iDivUp2(numParticles, numThreads);
	uint sMemSize = sizeof(uint)*(numThreads+1);

	cudaMemset(cellStart, 0xffffffff, numCells*sizeof(uint));

	findCellStartK<<< numBlocks, numThreads, sMemSize>>>(cellStart, cellEnd, phash);
}

void reorder(uint* d_pSortedIndex, float* dSortedPos, float* dSortedMom, float* oldPos, float* oldMom, uint numParticles)
{
	uint numThreads = 128;
	uint numBlocks = iDivUp2(numParticles, numThreads);

	reorderK<<<numBlocks, numThreads>>>(d_pSortedIndex, (float4*)dSortedPos, (float4*)dSortedMom, (float4*)oldPos, (float4*)oldMom);
}

uint buildNList(uint* nlist, uint* num_neigh, float* dpos, uint* phash, 
		uint* cellStart, uint* cellEnd, uint* cellAdj, uint numParticles, uint max_neigh)
{
	uint numThreads = 64;
	//printf("N %d\n", numParticles);
	uint numBlocks = iDivUp2(numParticles, numThreads);
	
	//printf("nT: %d, nB: %d\n", numThreads, numBlocks);

	buildNListK<<<numBlocks, numThreads>>>(nlist, num_neigh, (float4*) dpos, 
			phash, cellStart, cellEnd, cellAdj);
	cutilCheckMsg("preSync");
	cudaDeviceSynchronize();

	cutilCheckMsg("inNList");
	//find the maximum value using thrust - alternatively atomicMax as in HOOMD
	//if exceeds, realloc and regen
	
	thrust::device_ptr<uint> dev_ptr(num_neigh);
	uint max = 0;
	thrust::maximum<uint> mx;
	max=thrust::reduce(dev_ptr, dev_ptr+numParticles, 0, mx);
	
	return max;
}

		
void magForces(	float* dSortedPos, float* dIntPos, float* newPos, float* dForce, float* dMom, uint* nlist, uint* num_neigh, uint numParticles, float deltaTime)
{
	assert(newPos != dIntPos);
	assert(newPos != dSortedPos);
	uint numThreads = 64;
	uint numBlocks = iDivUp2(numParticles, numThreads);
	//printf("numP: %d numThreads %d, numBlocks %d\n", numParticles, numThreads, numBlocks);	
	magForcesK<<<numBlocks,numThreads>>>( 	(float4*)dSortedPos, (float4*) dMom, (float4*) dIntPos, 
											nlist, num_neigh, (float4*) dForce, (float4*) newPos, deltaTime);
	cutilCheckMsg("hi");
}


}
