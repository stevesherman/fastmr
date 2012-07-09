#ifndef NEW_KERN_CU
#define NEW_KERN_CU
#endif

#define PI 3.141592653589793f

#include <cuda_runtime.h>
#include "cutil_math.h"
#include <cstdlib>

#include "new_kern.cuh"
#include "particles_kernel.cuh"

__constant__ NewParams nparams;

texture<float4, cudaTextureType1D, cudaReadModeElementType> pos_tex;
texture<float4, cudaTextureType1D, cudaReadModeElementType> mom_tex;
texture<float4, cudaTextureType1D, cudaReadModeElementType> vel_tex;

__device__ uint3 calcGPos(float3 p)
{
	uint3 gpos;
	gpos.x = floorf((p.x - nparams.origin.x)/nparams.cellSize.x);
	gpos.y = floorf((p.y - nparams.origin.y)/nparams.cellSize.y);
	gpos.z = floorf((p.z - nparams.origin.z)/nparams.cellSize.z);
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
	d_pHash[idx] = d_CellHash[cell_id];
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


__global__ void magForcesK( const float4* dSortedPos,	//i: pos we use to calculate forces
							const float4* dMom,		//i: the moment
							const float4* integrPos,	//i: pos we use as base to integrate from
							const uint* nlist,		//i: the neighbor list
							const uint* num_neigh,	//i: the number of inputs
							float4* dForce,		//o: the magnetic force on a particle
							float4* newPos,		//o: the integrated position
							float deltaTime)	//o: the timestep
{
	uint idx = blockDim.x*blockIdx.x + threadIdx.x;
	if(idx >= nparams.N)
		return;
	uint n_neigh = num_neigh[idx];
	float4 pos1 = dSortedPos[idx];
	//float4 pos1 = tex1Dfetch(pos_tex,idx);
	float3 p1 = make_float3(pos1);
	float radius1 = pos1.w;

	float4 mom1 = dMom[idx];
	//float4 mom1 = tex1Dfetch(mom_tex,idx);
	float3 m1 = make_float3(mom1);
	float xi1 = mom1.w;
	
	float3 force = make_float3(0,0,0);

	uint edges = 0;
	
	for(uint i = 0; i < n_neigh; i++)
	{
		uint neighbor = nlist[i*nparams.N + idx];
		
		float4 pos2 = tex1Dfetch(pos_tex, neighbor);
		float3 p2 = make_float3(pos2);
		float radius2 = pos2.w;
		
		float4 mom2 = tex1Dfetch(mom_tex, neighbor);
		float3 m2 = make_float3(mom2);
		float xi2 = mom2.w;

		float3 er = p1 - p2;//start it out as dr, then modify to get er
		er.x = er.x - nparams.L.x*rintf(er.x*nparams.Linv.x);
		er.z = er.z - nparams.L.x*rintf(er.z*nparams.Linv.z);
		float lsq = er.x*er.x + er.y*er.y + er.z*er.z;
		er = er*rsqrt(lsq);

		if(lsq <= nparams.max_fdr_sq){
			float dm1m2 = dot(m1,m2);
			float dm1er = dot(m1,er);
			float dm2er = dot(m2,er);
			
			force += 3.0f*nparams.uf/(4*PI*lsq*lsq) *( dm1m2*er + dm1er*m2
					+ dm2er*m1 - 5.0f*dm1er*dm2er*er);
			
			//create a false moment for nonmagnetic particles
			//note that here mup gives the wrong volume, so the magnitude of 
			//the repulsion strength is wrong		
			m1 = (xi1 == 1.0f) ? nparams.mup*nparams.extH : m1;
			m2 = (xi2 == 1.0f) ? nparams.mup*nparams.extH : m2;
			dm1m2 = dot(m1,m2);
			
			float sepdist = radius1 + radius2;
			force += 3.0f*nparams.uf*dm1m2/(2.0f*PI*sepdist*sepdist*sepdist*sepdist)*
					exp(-nparams.spring*(sqrt(lsq)/sepdist - 1.0f))*er;
			edges += lsq < nparams.contact_d_sq*sepdist*sepdist ? 1 : 0;
			
		}
			
	}
	dForce[idx] = make_float4(force, (float) edges);
	float Cd = 6.0f*PI*radius1*nparams.visc;
	float ybot = p1.y - nparams.origin.y;
	force.x += nparams.shear*ybot*Cd;
	
	//apply flow BCs
	if(ybot < nparams.pin_d*radius1)
		force = make_float3(0,0,0);
	if(ybot > nparams.L.y - nparams.pin_d*radius1)
		force = make_float3(nparams.shear*nparams.L.y*Cd,0,0);

	float3 ipos = make_float3(integrPos[idx]);
	newPos[idx] = make_float4(ipos + force/Cd*deltaTime, radius1);

}
__global__ void integrateRK4(const float4* oldPos,
							float4* newPos,
							float4* forceA,
							const float4* forceB,
							const float4* forceC,
							const float4* forceD,
							const float deltaTime,
							const uint numParticles)
{
   

	uint index = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;
    if (index >= numParticles) return;          // handle case when no. of particles not multiple of block size

	float4 posData = oldPos[index];
    float3 pos = make_float3(posData.x, posData.y, posData.z);
	float radius = posData.w;
	
	float4 f1 = forceA[index];
    float4 f2 = forceB[index];
	float4 f3 = forceC[index];
	float4 f4 = forceD[index];
	
	float3 force1 = make_float3(f1.x, f1.y, f1.z);
	float3 force2 = make_float3(f2.x, f2.y, f2.z);
	float3 force3 = make_float3(f3.x, f3.y, f3.z);
	float3 force4 = make_float3(f4.x, f4.y, f4.z);
	
	float3 fcomp = (force1 + 2*force2 + 2*force3 + force4)/6;//trapezoid rule	
	forceA[index] = make_float4(fcomp, f1.w);//averaged force
	
	float Cd = 6*PI*nparams.visc*radius;

	float ybot = pos.y - nparams.origin.y;
	fcomp.x += nparams.shear*ybot*Cd;
	
	//apply flow BCs
	if(ybot < nparams.pin_d*radius)
		fcomp = make_float3(0,0,0);
	if(ybot > nparams.L.y - nparams.pin_d*radius)
		fcomp = make_float3(nparams.shear*nparams.L.y*Cd,0,0);

		
	//integrate	
	pos += fcomp*deltaTime/Cd;

	//periodic boundary conditions
   	pos.x -= nparams.L.x*floorf((pos.x - nparams.origin.x)*nparams.Linv.x);
	pos.z -= nparams.L.z*floorf((pos.z - nparams.origin.z)*nparams.Linv.z);
	
	if (pos.y > -1.0f*nparams.origin.y ) { pos.y = -1.0f*nparams.origin.z;}
    if (pos.y < nparams.origin.y ) { pos.y = 1.0f*nparams.origin.z; }

	newPos[index] = make_float4(pos, radius);
}

__global__ void NListFixedK(uint* nlist,	//	o:neighbor list
							uint* num_neigh,//	o:num neighbors
							float4* dpos,	// 	i: position
							uint* phash,
							uint* cellStart,
							uint* cellEnd,
							uint* cellAdj,
							uint max_neigh,
							float max_dist_sq)
{
	uint idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx >= nparams.N)
		return;
	float4 pos1 = dpos[idx];
	float3 p1 = make_float3(pos1);
	//float rad1 = pos1.w;
	uint hash = phash[idx];
	uint n_neigh = 0;

	for(uint i = 0; i < nparams.numAdjCells; i++)
	{
		//uint nhash = cellAdj[i*nparams.numCells + hash];
		uint nhash = cellAdj[i + hash*nparams.numAdjCells];
		uint cstart = cellStart[nhash];
		if(cstart != 0xffffffff) {
			uint cend = cellEnd[nhash];
			for(uint idx2 = cstart; idx2 < cend; idx2++){
				if(idx != idx2){
					float4 pos2 = dpos[idx2];
					//float4 pos2 = tex1Dfetch(pos_tex, idx2);
					float3 p2 = make_float3(pos2);
					//float rad2 = pos2.w;
					float3 dr = p1 - p2;

					dr.x = dr.x - nparams.L.x*rintf(dr.x*nparams.Linv.x);
					dr.z = dr.z - nparams.L.z*rintf(dr.z*nparams.Linv.z);

					float lsq = dr.x*dr.x + dr.y*dr.y + dr.z*dr.z;

					if(lsq <= max_dist_sq){
						if(n_neigh < max_neigh){
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



__global__ void NListVarK(uint* nlist,	//	o:neighbor list
							uint* num_neigh,//	o:num neighbors
							const float4* dpos,	// 	i: position
							const uint* phash,
							const uint* cellStart,
							const uint* cellEnd,
							const uint* cellAdj,
							uint max_neigh,
							float distm_sq)
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
		if(cstart != 0xffffffff) {
			uint cend = cellEnd[nhash];
			for(uint idx2 = cstart; idx2 < cend; idx2++){
				if(idx != idx2){
					float4 pos2 = dpos[idx2];
					//float4 pos2 = tex1Dfetch(pos_tex, idx2);
					float3 p2 = make_float3(pos2);
					float rad2 = pos2.w;
					float sepdist = rad1+rad2;

					float3 dr = p1 - p2;
					dr.x = dr.x - nparams.L.x*rintf(dr.x*nparams.Linv.x);
					dr.z = dr.z - nparams.L.z*rintf(dr.z*nparams.Linv.z);
					float lsq = dr.x*dr.x + dr.y*dr.y + dr.z*dr.z;
					
					if(lsq <= distm_sq*sepdist*sepdist){
						if(n_neigh < max_neigh){
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


__global__ void collisionK( const float4* sortedPos,	//i: pos we use to calculate forces
							const float4* oldVel,
							const uint* nlist,		//i: the neighbor list
							const uint* num_neigh,	//i: the number of inputs
							float4* newVel,		//o: the magnetic force on a particle
							float4* newPos,		//o: the integrated position
							float deltaTime)	//o: the timestep
{
	uint idx = blockDim.x*blockIdx.x + threadIdx.x;
	if(idx >= nparams.N)
		return;
	
	uint n_neigh = num_neigh[idx];
	
	float4 pos1 = sortedPos[idx];
	float3 p1 = make_float3(pos1);
	float radius1 = pos1.w;
	float3 v1 = make_float3(oldVel[idx]);

	float3 force = make_float3(0,0,0);
	
	for(uint i = 0; i < n_neigh; i++)
	{
		uint neighbor = nlist[i*nparams.N + idx];
		
		float4 pos2 = tex1Dfetch(pos_tex, neighbor);
		float3 p2 = make_float3(pos2);
		float radius2 = pos2.w;
		float3 v2 = make_float3(tex1Dfetch(vel_tex,neighbor));

		float3 er = p1 - p2;//start it out as dr, then modify to get er
		er.x = er.x - nparams.L.x*rintf(er.x*nparams.Linv.x);
		er.z = er.z - nparams.L.x*rintf(er.z*nparams.Linv.z);
		float dist = sqrt(er.x*er.x + er.y*er.y + er.z*er.z);
	
		float sepdist = 1.01f*(radius1 + radius2);

		//do a quicky spring	
		if(dist  <= sepdist){
			er = er/dist;
			float3 relVel = v2-v1;  	
			force += -10.0f*(dist - sepdist)*er;
			force += .03f*relVel;
		}
			
	}
	//yes this integration is totally busted, but it works, soooo
	v1 = (v1 + force)*.8f;
	p1 = p1 + v1*deltaTime;

	if(p1.x > -nparams.origin.x ) { p1.x -= nparams.L.x;}
    if(p1.x < nparams.origin.x ) { p1.x += nparams.L.x;}
	
	if(p1.y+radius1 > -nparams.origin.y){ 
		p1.y = -nparams.origin.y - radius1;
		v1.y*= -.03f;	
	}
    if(p1.y-radius1 <  nparams.origin.y){ 
		p1.y = nparams.origin.y + radius1;
		v1.y*= -.03f;	
	}
	
	if(p1.z > -nparams.origin.z ) { p1.z -= nparams.L.z;}
	if(p1.z < nparams.origin.z ) { p1.z += nparams.L.z;}

	newVel[idx] = make_float4(v1);
	newPos[idx]	= make_float4(p1, radius1); 
}



