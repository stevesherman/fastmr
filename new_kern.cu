#ifndef NEW_KERN_CU
#define NEW_KERN_CU
#endif

#define PI 3.141592653589793f

#include <cuda_runtime.h>
#include "cutil_math.h"
#include <cstdlib>

#include "new_kern.cuh"
#include "particles_kernel.cuh"

__constant__ newParams nparams;
__constant__ SimParams params;

__device__ uint3 calcGPos(float3 p)
{
	uint3 gpos;
	gpos.x = floor((p.x - nparams.worldOrigin.x)/nparams.cellSize.x);
	gpos.y = floor((p.y - nparams.worldOrigin.y)/nparams.cellSize.y);
	gpos.z = floor((p.z - nparams.worldOrigin.z)/nparams.cellSize.z);
	gpos.x = (nparams.gridSize.x + gpos.x) % nparams.gridSize.x;
	gpos.y = (nparams.gridSize.y + gpos.y) % nparams.gridSize.y;
	gpos.z = (nparams.gridSize.z + gpos.z) % nparams.gridSize.z;
	
	return gpos;
}

__global__ void comp_phashK(const float4* d_pos, uint* d_pHash, uint* d_pIndex, const uint* d_CellHash)
{
	int idx = blockDim.x*blockIdx.x + threadIdx.x;

	if(idx >= nparams.N) 
		return;

	float4 pos = d_pos[idx];
	float3 p = make_float3(pos);
	uint3 gpos = calcGPos(p);
	uint cell_id = gpos.x + gpos.y*nparams.gridSize.x + 
		gpos.z*nparams.gridSize.y*nparams.gridSize.x;
	
	d_pIndex[idx] = idx;
	d_pHash[idx] = cell_id;
}


__global__ void findCellStartK(uint* cellStart,		//o: cell starts
								uint* cellEnd,			//o: cell ends
								uint* phash)			//i: hashes sorted by hash
{
	extern __shared__ uint sharedHash[]; //size of blockDim+1
	
	uint index = blockIdx.x*blockDim.x + threadIdx.x;
	uint hash;
	if(index < nparams.N )
	{
		hash = phash[index];
		//load all neighboring hashes into memory
		sharedHash[threadIdx.x+1] = hash;
		if(index > 0 && threadIdx.x == 0)
			sharedHash[0] = phash[index-1];
	}
	
	__syncthreads();
	
	if(index < nparams.N)
	{
		//once load complete, compare to hash before and if !=, then write starts/ends
		if(index == 0 || hash != sharedHash[threadIdx.x])
		{
			cellStart[hash] = index;
			if (index > 0)// if not first cell
				cellEnd[sharedHash[threadIdx.x]] = index;
		}

		if (index == nparams.N - 1){//if the last particle, the cell ends here
			cellEnd[hash] = index+1;
		}
	}
}


__global__ void reorderK(uint* dSortedIndex, float4* sortedPos, float4* sortedMom, float4* oldPos, float4* oldMom)
{
	uint idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx >= nparams.N)
		return;

	uint sortedIdx = dSortedIndex[idx];
	sortedPos[idx] = oldPos[sortedIdx];
	sortedMom[idx] = oldMom[sortedIdx];
}


__global__ void magForcesK( float4* dSortedPos,	//i: pos we use to calculate forces
							float4* dMom,		//i: the moment
							float4* integrPos,	//i: pos we use as base to integrate from
							uint* nlist,		//i: the neighbor list
							uint* num_neigh,	//i: the number of inputs
							float4* dForce,		//o: the magnetic force on a particle
							float4* newPos,		//o: the integrated position
							float deltaTime)	//o: the timestep
{
	uint idx = blockDim.x*blockIdx.x + threadIdx.x;
	if(idx >= nparams.N){
		return;
	}
	uint n_neigh = num_neigh[idx];
	float4 pos1 = dSortedPos[idx];
	float3 p1 = make_float3(pos1);
	float radius1 = pos1.w;
	float4 mom1 = dMom[idx];
	float3 m1 = make_float3(mom1);
	float xi1 = mom1.w;
	
	float3 force = make_float3(0,0,0);


	for(uint i = 0; i < n_neigh; i++)
	{
		uint neighbor = nlist[i*nparams.N + idx];
		
		float4 pos2 = dSortedPos[neighbor];
		float3 p2 = make_float3(pos2);
		float radius2 = pos2.w;

		float4 mom2 = dMom[neighbor];
		float3 m2 = make_float3(mom2);
		float xi2 = mom2.w;

		float3 dr = p1 - p2;
		dr.x = dr.x - nparams.L.x*rintf(dr.x*nparams.Linv.x);
		dr.z = dr.z - nparams.L.x*rintf(dr.z*nparams.Linv.z);
		float lsq = dr.x*dr.x + dr.y*dr.y + dr.z*dr.z;
		float3 er = dr*rsqrt(lsq);

		//do a quicky spring	
		if(lsq <= nparams.max_fdr_sq){
			float dm1m2 = dot(m1,m2);
			float dm1er = dot(m1,er);
			float dm2er = dot(m2,er);
			
			
			force += 3.0f*nparams.uf/(4*PI*lsq*lsq) *( dm1m2*er + dm1er*m2
					+ dm2er*m1 - 5.0f*dm1er*dm2er*er);
			
			m1 = (xi1 == 1.0f) ? nparams.mup*nparams.externalH : m1;
			m2 = (xi2 == 1.0f) ? nparams.mup*nparams.externalH : m2;
			dm1m2 = dot(m1,m2);

			
			float sepdist = radius1 + radius2;
			force += 3.0f*nparams.uf*dm1m2/(2.0f*PI*pow(sepdist,4))*
					exp(-nparams.spring*(sqrt(lsq)/sepdist - 1.0f))*er;
			
		}
			
	}
	dForce[idx] = make_float4(force, n_neigh);
	float Cd = 6.0f*PI*radius1*nparams.viscosity;
	float ybot = p1.y - nparams.worldOrigin.y;
	force.x += nparams.shear*ybot*Cd;
	if(ybot < 1.5f*radius1)
		force = make_float3(0,0,0);
	if(ybot - nparams.worldOrigin.y > nparams.L.y - 1.5f*radius1)
		force.x = nparams.shear*nparams.L.y*Cd;

	float3 ipos = make_float3(integrPos[idx]);
	//since newPos can == sortedPos, need to sync to ensure that all threads are recieving the same information
	//__syncthreads();
	newPos[idx] = make_float4(ipos + force/Cd*deltaTime, radius1);

}

__global__ void buildNListK(	uint* nlist,	//	o:neighbor list
							uint* num_neigh,//	o:num neighbors
							float4* dpos,	// 	i: position
							uint* phash,
							uint* cellStart,
							uint* cellEnd,
							uint* cellAdj)
{
	uint idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx >= nparams.N)
		return;
	float4 pos1 = dpos[idx];
	float3 p1 = make_float3(pos1);
	//float rad1 = pos1.w;
	uint hash = phash[idx];
	uint n_neigh = 0;

	for(uint i = 0; i < nparams.num_c_neigh; i++)
	{
		uint nhash = cellAdj[i*nparams.numGridCells + hash];
		uint cstart = cellStart[nhash];
		if(cstart != 0xffffffff) {
			uint cend = cellEnd[nhash];
			for(uint idx2 = cstart; idx2 < cend; idx2++){
				if(idx != idx2){
					float4 pos2 = dpos[idx2];
					float3 p2 = make_float3(pos2);
					//float rad2 = pos2.w;
					float3 dr = p1 - p2;

					dr.x = dr.x - nparams.L.x*rintf(dr.x*nparams.Linv.x);
					dr.z = dr.z - nparams.L.z*rintf(dr.z*nparams.Linv.z);

					float lsq = dr.x*dr.x + dr.y*dr.y + dr.z*dr.z;

					if(lsq <= nparams.max_ndr_sq){
						if(n_neigh < nparams.max_neigh){
							nlist[nparams.N*n_neigh + idx] = idx2;
						}
						n_neigh++;
					}
				}
			}
		}
	}
	num_neigh[idx] = n_neigh;
}






