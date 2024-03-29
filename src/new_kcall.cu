#include <helper_cuda.h>
#include <helper_cuda_gl.h>
#include <helper_functions.h>

#include <thrust/reduce.h>
#include <thrust/device_ptr.h>

#include "new_kern.cu"

extern "C" 
{


void comp_phash(float* dpos, uint* d_pHash, uint* d_pIndex, uint* d_CellHash, uint numParticles, uint numGridCells)
{
	uint numThreads = 256;
	uint numBlocks = iDivUp(numParticles, numThreads);


	comp_phashK<<<numBlocks, numThreads>>> ( (float4*) dpos, d_pHash, d_pIndex, d_CellHash);
	getLastCudaError("in phash computation");	
}


void setNParameters(NewParams *hostParams){
	cudaMemcpyToSymbol(nparams, hostParams, sizeof(NewParams));
}

void find_cellStart(uint* cellStart, uint* cellEnd, uint* phash, uint numParticles, uint numCells)
{
	uint numThreads = 128;
	uint numBlocks = iDivUp(numParticles, numThreads);
	uint sMemSize = sizeof(uint)*(numThreads+1);

	cudaMemset(cellStart, 0xffffffff, numCells*sizeof(uint));

	findCellStartK<<< numBlocks, numThreads, sMemSize>>>(cellStart, cellEnd, phash);
}

void reorder(uint* d_pSortedIndex, float* dSortedA, float* dSortedB,
		float* oldA, float* oldB, uint numParticles)
{
	uint numThreads = 128;
	uint numBlocks = iDivUp(numParticles, numThreads);

	reorderK<<<numBlocks, numThreads>>>(d_pSortedIndex, (float4*)dSortedA,
			(float4*)dSortedB, (float4*)oldA, (float4*)oldB);
}


uint vertEdge(uint* connections, const uint* nlist, const uint* num_neigh, const float* dPos, 
		float maxth, float maxdist, uint numParticles)
{
	uint numThreads = 128;
	uint numBlocks = iDivUp(numParticles, numThreads);

	vertEdgeK<<<numBlocks,numThreads>>>(nlist, num_neigh,(float4*) dPos, connections, maxth, maxdist*maxdist);

	thrust::device_ptr<uint> conns(connections);
	uint total = thrust::reduce(conns, conns+numParticles, 0,thrust::plus<uint>());

	getLastCudaError("vertical connectivity");
	return total;
}

void magForces(const float* dSortedPos, const float* dIntPos, float* newPos, float* dForce, 
		const float* dMom, const uint* nlist, const uint* num_neigh, uint numParticles, float deltaTime)
{
	assert(newPos != dIntPos);
	assert(newPos != dSortedPos);
	uint numThreads = 128;
	uint numBlocks = iDivUp(numParticles, numThreads);
	cudaFuncSetCacheConfig(magForcesK, cudaFuncCachePreferL1);
	
	cudaBindTexture(0, pos_tex, dSortedPos, numParticles*sizeof(float4));
	cudaBindTexture(0, mom_tex, dMom, numParticles*sizeof(float4));

	magForcesK<<<numBlocks,numThreads>>>( (float4*)dSortedPos, (float4*) dMom, 
			(float4*) dIntPos, nlist, num_neigh, (float4*) dForce, 
			(float4*) newPos, deltaTime);
	
	cudaUnbindTexture(pos_tex);
	cudaUnbindTexture(mom_tex);

	getLastCudaError("Magforces error");
}

void finiteDip(const float* dSortedPos, const float* dIntPos, float* newPos, float* dForce,
		const uint* nlist, const uint* num_neigh, uint numParticles,
		float dipole_d, float F0, float sigma_0, float deltaTime)
{
	assert(newPos != dIntPos);
	assert(newPos != dSortedPos);
	uint numThreads = 128;
	uint numBlocks = iDivUp(numParticles, numThreads);
	cudaFuncSetCacheConfig(finiteDipK, cudaFuncCachePreferL1);

	cudaBindTexture(0, pos_tex, dSortedPos, numParticles*sizeof(float4));

	finiteDipK<<<numBlocks,numThreads>>>( (float4*)dSortedPos, (float4*) dIntPos,
			nlist, num_neigh, (float4*) dForce, (float4*) newPos,
			dipole_d, F0, sigma_0,deltaTime);

	cudaUnbindTexture(pos_tex);

	getLastCudaError("Finite Magforces error");
}

void pointDip(const float* dSortedPos, const float* dIntPos, float* newPos, float* dForce,
		const uint* nlist, const uint* num_neigh, uint numParticles,
		float forceFactor, float deltaTime)
{
	assert(newPos != dIntPos);
	assert(newPos != dSortedPos);
	uint numThreads = 192;
	uint numBlocks = iDivUp(numParticles, numThreads);
	cudaFuncSetCacheConfig(pointDipK, cudaFuncCachePreferL1);

	cudaBindTexture(0, pos_tex, dSortedPos, numParticles*sizeof(float4));

	pointDipK<<<numBlocks,numThreads>>>( (float4*)dSortedPos, (float4*) dIntPos,
			nlist, num_neigh, (float4*) dForce, (float4*) newPos,
			forceFactor,deltaTime);

	cudaUnbindTexture(pos_tex);

	getLastCudaError("Point forces error");
}

void magFricForces(const float* dSortedPos, const float* dIntPos, float* newPos, 
		float* dForceOut, float* dMom, const float* dForceIn, const uint* nlist, 
		const uint* num_neigh, uint numParticles, float static_fric,float deltaTime)
{
	assert(newPos != dIntPos);
	assert(newPos != dSortedPos);
	uint numThreads = 128;
	uint numBlocks = iDivUp(numParticles, numThreads);
	cudaFuncSetCacheConfig(magForcesK, cudaFuncCachePreferL1);
	
	cudaBindTexture(0, pos_tex, dSortedPos, numParticles*sizeof(float4));
	cudaBindTexture(0, mom_tex, dMom, numParticles*sizeof(float4));

	magFricForcesK<<<numBlocks,numThreads>>>((float4*)dSortedPos, (float4*) dMom, 
			(float4*) dForceIn, (float4*) dIntPos, nlist, num_neigh, 
			(float4*) dForceOut,(float4*) newPos,static_fric,deltaTime);
	
	cudaUnbindTexture(pos_tex);
	cudaUnbindTexture(mom_tex);

	getLastCudaError("Magforces error");
}


void mutualMagn(const float* pos, const float* oldMag, float* newMag, 
		const uint* nlist, const uint* numNeigh, uint numParticles)
{
	uint numThreads = 128;
	uint numBlocks = iDivUp(numParticles, numThreads);
	cudaFuncSetCacheConfig(magForcesK, cudaFuncCachePreferL1);
	
	cudaBindTexture(0, pos_tex, pos, numParticles*sizeof(float4));
	cudaBindTexture(0, mom_tex, oldMag, numParticles*sizeof(float4));

	mutualMagnK<<<numBlocks, numThreads>>>( (float4*) pos, (float4*) oldMag, 
			(float4*) newMag, nlist, numNeigh);

	cudaUnbindTexture(pos_tex);
	cudaUnbindTexture(mom_tex);
	getLastCudaError("Mutual Magn error");
}


void integrateRK4(const float* oldPos, float* PosA, const float* PosB,
		const float* PosC, const float* PosD, float* forceA, 
		const float* forceB, const float* forceC, const float* forceD, 
		float deltaTime, uint numParticles)
{
	uint numThreads = 256; 
	uint numBlocks = iDivUp(numParticles, numThreads);
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

void bogacki_ynp1(	const float* d_yn, const float* d_k1, const float* d_k2,
					const float* d_k3, float* d_ynp1, float deltaTime, uint numParticles) {
	uint numThreads = 256;
	uint numBlocks = iDivUp(numParticles, numThreads);
	bogacki_ynp1k<<<numBlocks, numThreads>>>(
								(float4*) d_yn,
								(float4*) d_k1,
								(float4*) d_k2,
								(float4*) d_k3,
								(float4*) d_ynp1,
								deltaTime,
								numParticles);

	getLastCudaError("bogacki_ynp1");

}


void collision_new(	const float* dSortedPos, const float* dOldVel, const uint* nlist, 
		const uint* num_neigh, float* dNewVel, float* dNewPos, uint numParticles, 
		float raxExp, float deltaTime)
{
	uint numThreads = 128;
	uint numBlocks = iDivUp(numParticles, numThreads);
	cudaFuncSetCacheConfig(collisionK, cudaFuncCachePreferL1);
	
	cudaBindTexture(0, pos_tex, dSortedPos, numParticles*sizeof(float4));
	cudaBindTexture(0, vel_tex, dOldVel, numParticles*sizeof(float4));

	collisionK<<<numBlocks,numThreads>>>( 	(float4*)dSortedPos, (float4*) dOldVel, nlist, 
			num_neigh, (float4*) dNewVel, (float4*) dNewPos, raxExp, deltaTime);
	
	cudaUnbindTexture(pos_tex);
	cudaUnbindTexture(vel_tex);

	getLastCudaError("collision");
}


}
