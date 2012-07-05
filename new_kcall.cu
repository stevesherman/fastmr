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
//	cudaDeviceSynchronize();
	cutilCheckMsg("in phash computation");	
/*	
	maxkey = thrust::reduce(dev_ptr, dev_ptr+numParticles, 0, mx);
	printf("nmaxkey: %u\n", maxkey);*/
}


void setNParameters(NewParams *hostParams){
	cudaMemcpyToSymbol(nparams, hostParams, sizeof(NewParams));
}

void find_cellStart(uint* cellStart, uint* cellEnd, uint* phash, uint numParticles, uint numCells)
{
	uint numThreads = 128;
	uint numBlocks = iDivUp2(numParticles, numThreads);
	uint sMemSize = sizeof(uint)*(numThreads+1);

	cudaMemset(cellStart, 0xffffffff, numCells*sizeof(uint));

	findCellStartK<<< numBlocks, numThreads, sMemSize>>>(cellStart, cellEnd, phash);
}

void reorder(uint* d_pSortedIndex, float* dSortedPos, float* dSortedMom, float* oldPos, 
		float* oldMom, uint numParticles)
{
	uint numThreads = 128;
	uint numBlocks = iDivUp2(numParticles, numThreads);

	reorderK<<<numBlocks, numThreads>>>(d_pSortedIndex, (float4*)dSortedPos, (float4*)dSortedMom, 
			(float4*)oldPos, (float4*)oldMom);
}

//Note: this func modifies nlist and max_neigh
uint buildNList(uint*& nlist, uint* num_neigh, float* dpos, uint* phash, uint* cellStart, 
		uint* cellEnd, uint* cellAdj, uint numParticles, uint& max_neigh, float max_dist)
{
	uint numThreads = 128;
	uint numBlocks = iDivUp2(numParticles, numThreads);
	cudaFuncSetCacheConfig(buildNListK, cudaFuncCachePreferL1);	

	buildNListK<<<numBlocks, numThreads>>>(nlist, num_neigh, (float4*) dpos, 
			phash, cellStart, cellEnd, cellAdj, max_neigh, max_dist*max_dist);
	
	//cudaDeviceSynchronize();
	cutilCheckMsg("inNList");
	thrust::maximum<uint> mx;
	thrust::device_ptr<uint> numneigh_ptr(num_neigh);
	uint maxn = thrust::reduce(numneigh_ptr, numneigh_ptr+numParticles, 0, mx);
	cutilCheckMsg("max nneigh thrust call");	
	
	if(maxn > max_neigh){
		printf("Extending NList from %u to %u\n", max_neigh, maxn);
		cudaFree(nlist);
		assert(cudaMalloc((void**)&nlist, numParticles*maxn*sizeof(uint)) == cudaSuccess);
		cudaMemset(nlist, 0, numParticles*maxn*sizeof(uint));
		buildNListK<<<numBlocks, numThreads>>>(nlist, num_neigh, (float4*) dpos, 
			phash, cellStart, cellEnd, cellAdj, maxn, max_dist*max_dist);
		cutilCheckMsg("after extension");
		max_neigh = maxn;
	}


return maxn;
}

		
void magForces(	float* dSortedPos, float* dIntPos, float* newPos, float* dForce, float* dMom, 
		uint* nlist, uint* num_neigh, uint numParticles, float deltaTime)
{
	assert(newPos != dIntPos);
	assert(newPos != dSortedPos);
	uint numThreads = 128;
	uint numBlocks = iDivUp2(numParticles, numThreads);
	cudaFuncSetCacheConfig(magForcesK, cudaFuncCachePreferL1);
	
	cudaBindTexture(0, pos_tex, dSortedPos, numParticles*sizeof(float4));
	cudaBindTexture(0, mom_tex, dMom, numParticles*sizeof(float4));

	magForcesK<<<numBlocks,numThreads>>>( 	(float4*)dSortedPos, (float4*) dMom, (float4*) dIntPos, 
											nlist, num_neigh, (float4*) dForce, (float4*) newPos, deltaTime);
	
	cudaUnbindTexture(pos_tex);
	cudaUnbindTexture(mom_tex);

	cutilCheckMsg("Magforces error");
}

void collision_new(	const float* dSortedPos, const float* dOldVel, const uint* nlist, 
		const uint* num_neigh, float* dNewVel, float* dNewPos, uint numParticles, float deltaTime)
{
	uint numThreads = 128;
	uint numBlocks = iDivUp2(numParticles, numThreads);
	cudaFuncSetCacheConfig(collisionK, cudaFuncCachePreferL1);
	
	cudaBindTexture(0, pos_tex, dSortedPos, numParticles*sizeof(float4));
	cudaBindTexture(0, vel_tex, dOldVel, numParticles*sizeof(float4));

	collisionK<<<numBlocks,numThreads>>>( 	(float4*)dSortedPos, (float4*) dOldVel,
											nlist, num_neigh, (float4*) dNewVel, (float4*) dNewPos, deltaTime);
	
	cudaUnbindTexture(pos_tex);
	cudaUnbindTexture(vel_tex);

	cutilCheckMsg("hi");
}
}
