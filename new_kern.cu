#ifndef NEW_KERN_CU
#define NEW_KERN_CU
#endif

#define PI_F 3.141592653589793f
#define MU_0 4e-7f*PI_F
#ifndef MU_C
#define MU_C 1
#endif

#include <cuda_runtime.h>
#include "cutil_math.h"
#include <cstdlib>

#include "new_kern.h"
#include "particles_kernel.h"

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
	float Cp1 = mom1.w;
	
	float3 force = make_float3(0,0,0);

	for(uint i = 0; i < n_neigh; i++)
	{
		uint neighbor = nlist[i*nparams.N + idx];
		
		float4 pos2 = tex1Dfetch(pos_tex, neighbor);
		float3 p2 = make_float3(pos2);
		float radius2 = pos2.w;
		float sepdist = radius1 + radius2;

		float4 mom2 = tex1Dfetch(mom_tex, neighbor);
		float3 m2 = make_float3(mom2);
		float Cp2 = mom2.w;

		float3 er = p1 - p2;//start it out as dr, then modify to get er
		er.x = er.x - nparams.L.x*rintf(er.x*nparams.Linv.x);
		er.z = er.z - nparams.L.x*rintf(er.z*nparams.Linv.z);
		float lsq = er.x*er.x + er.y*er.y + er.z*er.z;
		er = er*rsqrtf(lsq);

		if(lsq <= 4.0f*4.0f*sepdist*sepdist) { //nparams.max_fdr_sq){
			float dm1m2 = dot(m1,m2);
			float dm1er = dot(m1,er);
			float dm2er = dot(m2,er);
			
			force += 3.0f*MU_0*MU_C/(4*PI_F*lsq*lsq) *( dm1m2*er + dm1er*m2
					+ dm2er*m1 - 5.0f*dm1er*dm2er*er);
			
			//create a false moment for nonmagnetic particles
			//note that here Cp gives the wrong volume, so the magnitude of 
			//the repulsion strength is wrong		
			m1 = (Cp1 == 0.0f) ? nparams.Cpol*nparams.extH : m1;
			m2 = (Cp2 == 0.0f) ? nparams.Cpol*nparams.extH : m2;
			dm1m2 = dot(m1,m2);
			
			force += 3.0f*MU_0*MU_C*dm1m2/(2.0f*PI_F*sepdist*sepdist*sepdist*sepdist)*
					expf(-nparams.spring*(sqrtf(lsq)- sepdist))*er;
		}
			
	}
	dForce[idx] = make_float4(force,0.0f);
	float Cd = 6.0f*PI_F*radius1*nparams.visc;
	float ybot = p1.y - nparams.origin.y;
	force.x += nparams.shear*ybot*Cd;
	
	//apply flow BCs
	if(ybot <= nparams.pin_d*radius1)
		force = make_float3(0,0,0);
	if(ybot >= nparams.L.y - nparams.pin_d*radius1)
		force = make_float3(nparams.shear*nparams.L.y*Cd,0,0);

	float3 ipos = make_float3(integrPos[idx]);
	newPos[idx] = make_float4(ipos + force/Cd*deltaTime, radius1);

}

__global__ void magFricForcesK( const float4* dSortedPos,	//i: pos we use to calculate forces
							const float4* dMom,		//i: the moment
							const float4* dForceIn,  //i: the old force, used to find velocity		
							const float4* integrPos,	//i: pos we use as base to integrate from
							const uint* nlist,		//i: the neighbor list
							const uint* num_neigh,	//i: the number of inputs
							float4* dForceOut,		//o: the magnetic force on a particle
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
	float Cd1 = 6.0f*PI_F*radius1*nparams.visc;

	float4 mom1 = dMom[idx];
	//float4 mom1 = tex1Dfetch(mom_tex,idx);
	float3 m1 = make_float3(mom1);
	float Cp1 = mom1.w;
	
	float3 f1 = make_float3(dForceIn[idx]);

	float3 force = make_float3(0,0,0);

	for(uint i = 0; i < n_neigh; i++)
	{
		uint neighbor = nlist[i*nparams.N + idx];
		
		float4 pos2 = tex1Dfetch(pos_tex, neighbor);
		float3 p2 = make_float3(pos2);
		float radius2 = pos2.w;
		float Cd2 = 6.0f*PI_F*radius1*nparams.visc;
		
		float4 mom2 = tex1Dfetch(mom_tex, neighbor);
		float3 m2 = make_float3(mom2);
		float Cp2 = mom2.w;
		
		float3 f2 = make_float3(dForceIn[idx]);

		float3 er = p1 - p2;//start it out as dr, then modify to get er
		er.x = er.x - nparams.L.x*rintf(er.x*nparams.Linv.x);
		er.z = er.z - nparams.L.x*rintf(er.z*nparams.Linv.z);
		float lsq = er.x*er.x + er.y*er.y + er.z*er.z;
		er = er*rsqrtf(lsq);

		if(lsq <= nparams.max_fdr_sq){
			float dm1m2 = dot(m1,m2);
			float dm1er = dot(m1,er);
			float dm2er = dot(m2,er);
			
			force += 3.0f*MU_0*MU_C/(4*PI_F*lsq*lsq) *( dm1m2*er + dm1er*m2
					+ dm2er*m1 - 5.0f*dm1er*dm2er*er);
			
			//create a false moment for nonmagnetic particles
			//note that here Cp gives the wrong volume, so the magnitude of 
			//the repulsion strength is wrong		
			m1 = (Cp1 == 0.0f) ? nparams.Cpol*nparams.extH : m1;
			m2 = (Cp2 == 0.0f) ? nparams.Cpol*nparams.extH : m2;
			dm1m2 = dot(m1,m2);
			
			float sepdist = radius1 + radius2;
			force += 3.0f*MU_0*MU_C*dm1m2/(2.0f*PI_F*sepdist*sepdist*sepdist*sepdist)*
					expf(-nparams.spring*(sqrtf(lsq)/sepdist - 1.0f))*er;
			if(lsq <= sepdist*sepdist){
				float3 v1 = f1/Cd1 + nparams.shear*p1.y;
				v1 = (p1.y >= nparams.L.y - nparams.pin_d*radius1) ? 
						make_float3(nparams.shear*nparams.L.y,0.0f,0.0f) : v1;
				float3 v2 = f2/Cd2 + nparams.shear*p2.y;
				v2 = (p2.y >= nparams.L.y - nparams.pin_d*radius2) ? 
						make_float3(nparams.shear*nparams.L.y,0.0f,0.0f) : v2;
				float3 relvel = v1 - v2;
				//float3 tanvel = relvel - dot(er,relvel)*er;
				force -= relvel*nparams.tanfric;
			}
		}
			
	}
	dForceOut[idx] = make_float4(force,0.0f);
		float ybot = p1.y - nparams.origin.y;
	force.x += nparams.shear*ybot*Cd1;
	
	//apply flow BCs
	if(ybot <= nparams.pin_d*radius1)
		force = make_float3(0,0,0);
	if(ybot >= nparams.L.y - nparams.pin_d*radius1)
		force = make_float3(nparams.shear*nparams.L.y*Cd1,0,0);

	float3 ipos = make_float3(integrPos[idx]);
	newPos[idx] = make_float4(ipos + force/Cd1*deltaTime, radius1);

}
__global__ void mutualMagnK(const float4* pos,
							const float4* oldMag,
							float4* newMag,
							const uint* nlist,
							const uint* numNeigh)
{
	uint idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx >= nparams.N) return;
	uint n_neigh = numNeigh[idx];
	float4 pos1 = pos[idx];
	float3 p1 = make_float3(pos1);
	//float radius1 = pos1.w;

	float4 omag = oldMag[idx];
	float3 mom1 = make_float3(omag);
	float Cp1 = omag.w;
	if(Cp1 == 0.0f) { //if nonmagnetic
		newMag[idx] = make_float4(0.0f,0.0f,0.0f,Cp1);
		return;	
	}
	float3 H = nparams.extH;
	for(uint i = 0; i < n_neigh; i++) {
		
		uint neighbor = nlist[i*nparams.N + idx];
		
		float4 pos2 = tex1Dfetch(pos_tex, neighbor);
		float3 p2 = make_float3(pos2);
		//float radius2 = pos2.w;
		
		float4 mom2 = tex1Dfetch(mom_tex, neighbor);
		float3 m2 = make_float3(mom2);
		//float Cp2 = mom2.w;

		float3 er = p1 - p2;//start it out as dr, then modify to get er
		er.x = er.x - nparams.L.x*rintf(er.x*nparams.Linv.x);
		er.z = er.z - nparams.L.x*rintf(er.z*nparams.Linv.z);
		float lsq = er.x*er.x + er.y*er.y + er.z*er.z;
		if(lsq < nparams.max_fdr_sq) {
			float invdist = rsqrtf(lsq);
			er = er*invdist;
			H += 1.0f/(4.0f*PI_F)*(3.0f*dot(m2,er)*er - m2)*invdist*invdist*invdist;
		}
	}
	newMag[idx] = make_float4(Cp1*H, Cp1);
}


__global__ void integrateRK4K(
							const float4* oldPos,
							float4* PosA,
							const float4* PosB,
							const float4* PosC,
							const float4* PosD,
							float4* forceA,
							const float4* forceB,
							const float4* forceC,
							const float4* forceD,
							const float deltaTime,
							const uint numParticles)
{
   

	uint index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index >= numParticles) return;          // handle case when no. of particles not multiple of block size
	
	float4 old = oldPos[index];
	float3 oldp = make_float3(old);
	float radius = old.w;
	//compite k1,k2, we use a factor of 2.0, because they're done with a timestep of 0.5*dt
    float3 k1 = 2.0f*(make_float3(PosA[index]) - oldp);
	float3 k2 = 2.0f*(make_float3(PosB[index]) - oldp);
	float3 k3 = make_float3(PosC[index]) - oldp;
	float3 k4 = make_float3(PosD[index]) - oldp;
	
	oldp += (1.0f/6.0f)*(k1 + 2.0f*k2 + 2.0f*k3 + k4);

	oldp.x -= nparams.L.x*rintf(oldp.x*nparams.Linv.x);
	oldp.z -= nparams.L.z*rintf(oldp.z*nparams.Linv.z);
	if (oldp.y > -1.0f*nparams.origin.y - radius ) { oldp.y = -1.0f*nparams.origin.z - radius;}
	if (oldp.y < nparams.origin.y + radius ) { oldp.y = nparams.origin.z + radius; }

	PosA[index] = make_float4(oldp, radius);

	float4 f1 = forceA[index];
	float nothin = f1.w;//doesn't actually hold any value, but might someday
	float3 force1 = make_float3(f1);
	float3 force2 = make_float3(forceB[index]);
	float3 force3 = make_float3(forceC[index]);
	float3 force4 = make_float3(forceD[index]);

	float3 fcomp = (force1 + 2*force2 + 2*force3 + force4)/6.0f;//trapezoid rule	
	forceA[index] = make_float4(fcomp, nothin);//averaged force


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
		if(cstart == 0xffffffff)//if empty, skip cell
			continue;
		uint cend = cellEnd[nhash];
		for(uint idx2 = cstart; idx2 < cend; idx2++){
			if(idx == idx2)
				continue;
			float4 pos2 = dpos[idx2];
			//float4 pos2 = tex1Dfetch(pos_tex, idx2);
			float3 p2 = make_float3(pos2);
			//float rad2 = pos2.w;
			float3 dr = p1 - p2;

			dr.x = dr.x - nparams.L.x*rintf(dr.x*nparams.Linv.x);
			dr.z = dr.z - nparams.L.z*rintf(dr.z*nparams.Linv.z);

			float lsq = dr.x*dr.x + dr.y*dr.y + dr.z*dr.z;

			if(lsq <= max_dist_sq){
				if(n_neigh < max_neigh)
					nlist[nparams.N*n_neigh + idx] = idx2;
				n_neigh++;
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
	distm_sq = rad1 > 20e-6f ? 1.1f*1.1f : distm_sq;
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
			//float4 pos2 = tex1Dfetch(pos_tex, idx2);
			float3 p2 = make_float3(pos2);
			float rad2 = pos2.w;
			float sepdist = rad1+rad2;

			float3 dr = p1 - p2;
			dr.x = dr.x - nparams.L.x*rintf(dr.x*nparams.Linv.x);
			dr.z = dr.z - nparams.L.z*rintf(dr.z*nparams.Linv.z);
			float lsq = dr.x*dr.x + dr.y*dr.y + dr.z*dr.z;
			
			if(lsq <= distm_sq*sepdist*sepdist){
				if(n_neigh < max_neigh)
					nlist[nparams.N*n_neigh + idx] = idx2;
				n_neigh++;
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
		float dist = sqrtf(er.x*er.x + er.y*er.y + er.z*er.z);
	
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

	p1.x -= nparams.L.x * rintf(p1.x*nparams.Linv.x);
	p1.z -= nparams.L.x * rintf(p1.z*nparams.Linv.z);	

	if(p1.y+radius1 > -nparams.origin.y){ 
		p1.y = -nparams.origin.y - radius1;
		v1.y*= -.03f;	
	}
    if(p1.y-radius1 <  nparams.origin.y){ 
		p1.y = nparams.origin.y + radius1;
		v1.y*= -.03f;	
	}
	
	newVel[idx] = make_float4(v1);
	newPos[idx]	= make_float4(p1, radius1); 
}



