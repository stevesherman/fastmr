#include <cuda_runtime.h>
#include <helper_math.h>
#include <cuda_gl_interop.h>
#include <helper_cuda.h>
#include <helper_cuda_gl.h>
#include <helper_functions.h>


#include <cstdlib>
#include "new_kern.h"

#include "thrust/reduce.h"
#include "thrust/extrema.h"
#include "thrust/device_ptr.h"
#include "thrust/functional.h"

extern __constant__ NewParams nparams;

__global__ void vertNListVarK(uint* nlist,	//	o:neighbor list
							uint* num_neigh,//	o:num neighbors
							const float4* dpos,	// 	i: position
							const float4* dmom, //  i: moments
							const uint* phash,
							const uint* cellStart,
							const uint* cellEnd,
							const uint* cellAdj,
							uint max_neigh,
							float dist_sq,
							float maxcosth)
{
	uint idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx >= nparams.N)
		return;
	float4 pos1 = dpos[idx];
	float3 p1 = make_float3(pos1);
	float rad1 = pos1.w;
	uint hash = phash[idx];
	uint n_neigh = 0;

	for(uint i = 0; i < nparams.numAdjCells; i++)
	{
		//uint nhash = cellAdj[i*nparams.numCells + hash];
		uint nhash = cellAdj[i + hash*nparams.numAdjCells];
		uint cstart = cellStart[nhash];
		if(cstart == 0xffffffff)//if cell empty, skip cell 
			continue;
		uint cend = cellEnd[nhash];
		for(uint idx2 = cstart; idx2 < cend; idx2++){
			if(idx == idx2)//if self interacting, skip
				continue;
			float4 pos2 = dpos[idx2];
			float3 p2 = make_float3(pos2);
			float rad2 = pos2.w;

			float sepdist = rad1+rad2;

			float3 dr = p1 - p2;
			dr.x = dr.x - nparams.L.x*rintf(dr.x*nparams.Linv.x);
			dr.z = dr.z - nparams.L.z*rintf(dr.z*nparams.Linv.z);
			float lsq = dr.x*dr.x + dr.y*dr.y + dr.z*dr.z;
			dr *= rsqrtf(lsq);	

			if(lsq <= dist_sq*sepdist*sepdist && fabs(dr.y) > maxcosth){
				if(n_neigh < max_neigh){
					nlist[nparams.N*n_neigh + idx] = idx2;
				}
				n_neigh++;
			}
		}
	}
	num_neigh[idx] = n_neigh;
}

extern "C" {

uint vertNListVar(uint*& nlist, uint* num_neigh, float* dpos, float* dmom, 
		uint* phash, uint* cellStart, uint* cellEnd, uint* cellAdj, 
		uint numParticles, uint& max_neigh, float max_dist, float maxcosth)
{
	uint numThreads = 128;
	uint numBlocks = (numParticles % numThreads == 0) ? (numParticles/numThreads) : (numParticles/numThreads+1);
	cudaFuncSetCacheConfig(vertNListVarK, cudaFuncCachePreferL1);	

	vertNListVarK<<<numBlocks, numThreads>>>(nlist, num_neigh, (float4*) dpos, (float4*) dmom,
			phash, cellStart, cellEnd, cellAdj, max_neigh, max_dist*max_dist, maxcosth);
	
	//cudaDeviceSynchronize();
	getLastCudaError("NListVar");
	thrust::maximum<uint> mx;
	thrust::device_ptr<uint> numneigh_ptr(num_neigh);
	uint maxn = thrust::reduce(numneigh_ptr, numneigh_ptr+numParticles, 0, mx);
	getLastCudaError("max nneigh thrust call");	
	
	if(maxn > max_neigh){
		printf("Extending VarNList from %u to %u\n", max_neigh, maxn);
		cudaFree(nlist);
		assert(cudaMalloc((void**)&nlist, numParticles*maxn*sizeof(uint)) == cudaSuccess);
		cudaMemset(nlist, 0, numParticles*maxn*sizeof(uint));
		vertNListVarK<<<numBlocks, numThreads>>>(nlist, num_neigh, (float4*) dpos, (float4*) dmom, 
				phash, cellStart, cellEnd, cellAdj, maxn, max_dist*max_dist, maxcosth);
		getLastCudaError("after extension");
		max_neigh = maxn;
	}

	return maxn;
}


}
