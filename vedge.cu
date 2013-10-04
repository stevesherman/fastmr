#include <cuda_runtime.h>
#include <helper_math.h>
#include <cuda_gl_interop.h>
#include <helper_cuda.h>
#include <helper_cuda_gl.h>
#include <helper_functions.h>
#include <cstdlib>

#include "new_kern.h"
#include "vedge.h"

#include "thrust/reduce.h"
#include "thrust/device_ptr.h"
#include "thrust/functional.h"

extern __constant__ NewParams nparams;



//template<class O>
__global__ void funcNListK(uint* nlist,	//	o:neighbor list
							uint* num_neigh,//	o:num neighbors
							const float4* dpos,	// 	i: position
							const uint* phash,
							const uint* cellStart,
							const uint* cellEnd,
							const uint* cellAdj,
							const uint max_neigh,
							VertCond op)
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

			float3 dr = p1 - p2;
			dr.x = dr.x - nparams.L.x*rintf(dr.x*nparams.Linv.x);
			dr.z = dr.z - nparams.L.z*rintf(dr.z*nparams.Linv.z);
			float lsq = dr.x*dr.x + dr.y*dr.y + dr.z*dr.z;
			dr = dr*rsqrtf(lsq);	

			if (op(rad1,rad2,dr,lsq)){
				if(n_neigh < max_neigh){
					nlist[nparams.N*n_neigh + idx] = idx2;
				}
				n_neigh++;
			}
		}
	}
	num_neigh[idx] = n_neigh;
}

//extern "C" {
//uses an adjacency definition based on max_dist_m*(rad1 + rad2)
//Note: this func modifies nlist and max_neigh



//pass in a functor of type NListDistCond
//doesn't use moment data
//template<class O>
uint funcNList(	uint*& nlist, //reference to the nlist pointer
		uint* num_neigh, 
		const float* dpos, 
		const uint* phash, 
		const uint* cellStart, 
		const uint* cellEnd, 
		const uint* cellAdj, 
		const uint numParticles, 
		uint& max_neigh, 
		VertCond op)
{
	uint numThreads = 128;
	uint numBlocks = (numParticles % numThreads == 0) ? (numParticles/numThreads) : (numParticles/numThreads+1);
	cudaFuncSetCacheConfig(funcNListK, cudaFuncCachePreferL1);	

	funcNListK<<<numBlocks, numThreads>>>(nlist, num_neigh, (float4*) dpos, phash, 
			cellStart, cellEnd, cellAdj, max_neigh, op);


	thrust::maximum<uint> mx;
	thrust::device_ptr<uint> numneigh_ptr(num_neigh);
	uint maxn = thrust::reduce(numneigh_ptr, numneigh_ptr+numParticles, 0, mx);
	
	if(maxn > max_neigh) {
		printf("Extending NList from %u to %u\n", max_neigh, maxn);
		cudaFree(nlist);
		assert(cudaMalloc((void**)&nlist, numParticles*maxn*sizeof(uint)) == cudaSuccess);
		cudaMemset(nlist, 0, numParticles*maxn*sizeof(uint));
		funcNListK<<<numBlocks, numThreads>>>(nlist, num_neigh, (float4*) dpos, phash, 
				cellStart, cellEnd, cellAdj, max_neigh, op);
		max_neigh = maxn;
	}

	return maxn;
}

//}
