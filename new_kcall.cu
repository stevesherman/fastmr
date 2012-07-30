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


	comp_phashK<<<numBlocks, numThreads>>> ( (float4*) dpos, d_pHash, d_pIndex, d_CellHash);
	cutilCheckMsg("in phash computation");	
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

//uses adjacency definition based on a fixed maximum distace, max_dist
//Note: this func modifies nlist and max_neigh
uint NListFixed(uint*& nlist, uint* num_neigh, float* dpos, uint* phash, uint* cellStart, 
		uint* cellEnd, uint* cellAdj, uint numParticles, uint& max_neigh, float max_dist)
{
	uint numThreads = 128;
	uint numBlocks = iDivUp2(numParticles, numThreads);
	cudaFuncSetCacheConfig(NListFixedK, cudaFuncCachePreferL1);	

	NListFixedK<<<numBlocks, numThreads>>>(nlist, num_neigh, (float4*) dpos, 
			phash, cellStart, cellEnd, cellAdj, max_neigh, max_dist*max_dist);
	
	//cudaDeviceSynchronize();
	cutilCheckMsg("NListFixed");
	thrust::maximum<uint> mx;
	thrust::device_ptr<uint> numneigh_ptr(num_neigh);
	uint maxn = thrust::reduce(numneigh_ptr, numneigh_ptr+numParticles, 0, mx);
	cutilCheckMsg("max nneigh thrust call");	
	
	if(maxn > max_neigh){
		printf("Extending FixNList from %u to %u\n", max_neigh, maxn);
		cudaFree(nlist);
		assert(cudaMalloc((void**)&nlist, numParticles*maxn*sizeof(uint)) == cudaSuccess);
		cudaMemset(nlist, 0, numParticles*maxn*sizeof(uint));
		NListFixedK<<<numBlocks, numThreads>>>(nlist, num_neigh, (float4*) dpos, 
			phash, cellStart, cellEnd, cellAdj, maxn, max_dist*max_dist);
		cutilCheckMsg("after extension");
		max_neigh = maxn;
	}

return maxn;
}

//uses an adjacency definition based on max_dist_m*(rad1 + rad2)
//Note: this func modifies nlist and max_neigh
uint NListVar(uint*& nlist, uint* num_neigh, float* dpos, uint* phash, uint* cellStart, 
		uint* cellEnd, uint* cellAdj, uint numParticles, uint& max_neigh, float max_dist_m)
{
	uint numThreads = 128;
	uint numBlocks = iDivUp2(numParticles, numThreads);
	cudaFuncSetCacheConfig(NListVarK, cudaFuncCachePreferL1);	

	NListVarK<<<numBlocks, numThreads>>>(nlist, num_neigh, (float4*) dpos, 
			phash, cellStart, cellEnd, cellAdj, max_neigh, max_dist_m*max_dist_m);
	
	//cudaDeviceSynchronize();
	cutilCheckMsg("NListVar");
	thrust::maximum<uint> mx;
	thrust::device_ptr<uint> numneigh_ptr(num_neigh);
	uint maxn = thrust::reduce(numneigh_ptr, numneigh_ptr+numParticles, 0, mx);
	cutilCheckMsg("max nneigh thrust call");	
	
	if(maxn > max_neigh){
		printf("Extending VarNList from %u to %u\n", max_neigh, maxn);
		cudaFree(nlist);
		assert(cudaMalloc((void**)&nlist, numParticles*maxn*sizeof(uint)) == cudaSuccess);
		cudaMemset(nlist, 0, numParticles*maxn*sizeof(uint));
		NListVarK<<<numBlocks, numThreads>>>(nlist, num_neigh, (float4*) dpos, 
			phash, cellStart, cellEnd, cellAdj, maxn, max_dist_m*max_dist_m);
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

void mutualMagn(const float* pos, const float* oldMag, float* newMag, const uint* nlist, const uint* numNeigh, uint numParticles)
{
	uint numThreads = 128;
	uint numBlocks = iDivUp2(numParticles, numThreads);
	cudaFuncSetCacheConfig(magForcesK, cudaFuncCachePreferL1);
	
	cudaBindTexture(0, pos_tex, pos, numParticles*sizeof(float4));
	cudaBindTexture(0, mom_tex, oldMag, numParticles*sizeof(float4));

	mutualMagnK<<<numBlocks, numThreads>>>( (float4*) pos, (float4*) oldMag, (float4*) newMag, nlist, numNeigh);

	cudaUnbindTexture(pos_tex);
	cudaUnbindTexture(mom_tex);
	cutilCheckMsg("Mutual Magn error");
}

void integrateRK4(
							const float* oldPos,
							float* PosA,
							const float* PosB,
							const float* PosC,
							const float* PosD,
							float* forceA,
							const float* forceB,
							const float* forceC,
							const float* forceD,
							const float deltaTime,
							const uint numParticles)
{
	uint numThreads = 256; 
	uint numBlocks = iDivUp2(numParticles, numThreads);
	integrateRK4K<<<numBlocks, numThreads>>>(
							 (float4*) oldPos,
							(float4*) PosA,
							 (float4*) PosB,
							 (float4*) PosC,
							 (float4*) PosD,
							(float4*) forceA,
							 (float4*) forceB,
							 (float4*) forceC,
							 (float4*) forceD,
							 deltaTime,
							 numParticles);
}






void relax_new(	const float* dSortedPos, const float* dOldVel, const uint* nlist, 
		const uint* num_neigh, float* dNewVel, float* dNewPos, uint numParticles, float deltaTime)
{
	uint numThreads = 128;
	uint numBlocks = iDivUp2(numParticles, numThreads);
	cudaFuncSetCacheConfig(relaxK, cudaFuncCachePreferL1);
	
	cudaBindTexture(0, pos_tex, dSortedPos, numParticles*sizeof(float4));
	cudaBindTexture(0, vel_tex, dOldVel, numParticles*sizeof(float4));

	relaxK<<<numBlocks,numThreads>>>( 	(float4*)dSortedPos, (float4*) dOldVel,
											nlist, num_neigh, (float4*) dNewVel, (float4*) dNewPos, deltaTime);
	
	cudaUnbindTexture(pos_tex);
	cudaUnbindTexture(vel_tex);

	cutilCheckMsg("hi");
}
}
