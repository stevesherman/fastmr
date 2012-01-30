/* 
 * CUDA particle system kernel code.
 */

#ifndef _PARTICLES_KERNEL_H_
#define _PARTICLES_KERNEL_H_

#define PI 3.141592653589793f

#include <stdio.h>
#include <math.h>
#include "cutil_math.h"
#include "math_constants.h"
#include "particles_kernel.cuh"

#if USE_TEX
// textures for particle position and velocity
texture<float4, 1, cudaReadModeElementType> oldPosTex;
texture<float4, 1, cudaReadModeElementType> oldMomentTex;
texture<float4, 1, cudaReadModeElementType> oldVelTex;


texture<uint, 1, cudaReadModeElementType> gridParticleHashTex;
texture<uint, 1, cudaReadModeElementType> cellStartTex;
texture<uint, 1, cudaReadModeElementType> cellEndTex;
#endif

// simulation parameters in constant memory
__constant__ SimParams params;


// calculate position in uniform grid
__device__ int3 calcGridPos(float3 p)
{
    int3 gridPos;
    gridPos.x = floor((p.x - params.worldOrigin.x) / params.cellSize.x);
    gridPos.y = floor((p.y - params.worldOrigin.y) / params.cellSize.y);
    gridPos.z = floor((p.z - params.worldOrigin.z) / params.cellSize.z);
    return gridPos;
}

// calculate address in grid from position
__device__ uint calcGridHash(int3 gridPos)
{
    int3 agridPos;
	agridPos.x = (params.gridSize.x +  gridPos.x) % params.gridSize.x; 
    agridPos.y = (params.gridSize.y +  gridPos.y) % params.gridSize.y;
    agridPos.z = (params.gridSize.z +  gridPos.z) % params.gridSize.z;        
    return agridPos.z * params.gridSize.y * params.gridSize.x + agridPos.y * params.gridSize.x + agridPos.x;
}

__device__ 
float4 cellForce(	int3 neighborPos, 
						uint index, 
						float3 pos1, 
						float3 mom1,
					   	float radius1,	
						float xi1,
						float4* oldPos, 
						float4* oldMoment, 
						uint* cellStart, 
						uint* cellEnd)
{
	uint gridHash = calcGridHash(neighborPos);

    // get start of bucket for this cell
    uint startIndex = cellStart[gridHash];

    float3 force = make_float3(0.0f);
	float interactions = 0;
    
	float4 pos2d, mom2d;
	float3 pos2, mom2, dr, er;
	float radius2, xi2, dist, sepdist;

	if (startIndex != 0xffffffff) {        // cell is not empty
        // iterate over particles in this cell
        uint endIndex = cellEnd[gridHash];
        for(uint j=startIndex; j<endIndex; j++) {
            if (j != index) {              // check not colliding with self
				
				pos2d = oldPos[j];
				pos2 = make_float3(pos2d);
				radius2 = pos2d.w;
				
				//float3 pos1 = make_float3(pos1d);
				//float radius1 = pos1d.w;
/*
				if(neighborPos.x >= (int) params.gridSize.x)
					pos2.x += params.worldSize.x;
				if(neighborPos.x < (int) 0)
					pos2.x -= params.worldSize.x;
				if(neighborPos.z >= (int) params.gridSize.z)
					pos2.z += params.worldSize.z;
				if(neighborPos.z < (int) 0)
					pos2.z -= params.worldSize.z;
*/
				dr = (pos1 - pos2);//points from 2 to 1 
				dr.x = dr.x - params.worldSize.x*rintf(dr.x/params.worldSize.x);
				dr.z = dr.z - params.worldSize.z*rintf(dr.z/params.worldSize.z);	
				dist = length(dr);
				er = dr/dist;



                mom2d = oldMoment[j];
				mom2 = make_float3(mom2d.x, mom2d.y, mom2d.z);
				xi2 = mom2d.w;
				if(dist <= 8.0f*radius1){
              		// magnetic interaction
            	    force += 3.0f*params.uf / (4*PI*pow(dist,4)) * ( dot(mom1, mom2) * er + dot(mom1, er)*mom2 
							+ dot(mom2,er)*mom1 - 5.0f*dot(mom1,er)*dot(mom2,er)*er);

					mom1 = (xi1 == 1.0f) ? params.mup*params.externalH : mom1;
					mom2 = (xi2 == 1.0f) ? params.mup*params.externalH : mom2;
			
					//collision force
					sepdist = radius1+radius2;
					force += 3*params.uf*dot(mom1,mom2)/(2*PI*pow(sepdist,4)) * 
						exp( -params.spring*(dist/(sepdist)-1) )*er;
					//if(dist < 1.1f*sepdist)
						interactions++;
				}
            }
        }
    }
	return make_float4(force, interactions);
}

//add flow velocity
__device__ float3 addFlowVel(float3 pos, float3 force, float radius)
{
	float Cd = 6*PI*radius*params.viscosity;
	
	// flow mode - see Wereley, Pang 1997 - nondim analysis of semi-active er damper
	float ybot = pos.y - params.worldOrigin.y;
	if(params.flowmode) {
		//flow region 1
		if(ybot < params.worldSize.y/2*(1 - params.nd_plug)){
			force.x += -Cd*params.flowvel*(ybot*ybot - params.worldSize.y*(1 - params.nd_plug)*ybot);
		} else {
			//flow region 3
			if(ybot > params.worldSize.y/2*(1 + params.nd_plug)){
				force.x += -Cd*params.flowvel*(ybot*ybot - params.worldSize.y*(1 + params.nd_plug)*ybot + 
						params.worldSize.y*params.worldSize.y*params.nd_plug);
			} else { //flow region 2
				force.x += Cd*.025f*params.flowvel*pow(params.worldSize.y*(1 - params.nd_plug),2);
			}
			
		}
		//stationary at upper surface
		if(ybot > params.worldSize.y - 1.5f*radius){
			force = make_float3(0,0,0);
		}

	} else { 
		force.x += ybot *params.shear*Cd;
		if(ybot > params.worldSize.y - 1.5f*radius){//moving upper wall
			force = make_float3(params.shear*params.worldSize.y*Cd, 0.0f, 0.0f);
		}
		
	}
	//pinned at zero wall velocity in both
	if(ybot < 1.5f*radius){
		force = make_float3(0.0f, 0.0f, 0.0f);
	}
	return force;
}

// integrate particle attributes
__global__
void calcParticleForce(float4* sortedPos,//in 
               		float4* integrPos, //in
					float4* newPos,//out
					float4* forceOut,//out
			   		float4* Moments,//in
               		float deltaTime,
					uint* cellStart,
					uint* cellEnd,
               		uint numParticles)
{

	uint index = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;
    if (index >= numParticles) return;          // handle case when no. of particles not multiple of block size

	volatile float4 posData = sortedPos[index];    // ensure coalesced read
    volatile float4 momData = Moments[index];
    float radius = posData.w;
	float3 pos = make_float3(posData.x, posData.y, posData.z);
    float3 mom = make_float3(momData.x, momData.y, momData.z);
	float xi = momData.w;

	int3 gridPos = calcGridPos(pos);

	float4 force = make_float4(0.0f,0.0f,0.0f,0.0f);//get interparticle forces
	for(int z= -params.interactionr; z<=params.interactionr; z++) { //2d, so no Z
		for(int y= -params.interactionr; y<=params.interactionr; y++) {
			for(int x= -params.interactionr; x<=params.interactionr; x++) {
				int3 neighbourPos = gridPos + make_int3(x, y, z);
                force += cellForce(neighbourPos, index, pos, mom, radius, xi, sortedPos, Moments, cellStart, cellEnd);
			}
        }
    }

	forceOut[index] = force; // in the w parameter of force, include number of interactions
	float3 fcomp = make_float3(force.x,force.y,force.z);	
	//moving fluid force - shear tp - not written out, but added to position

	fcomp = addFlowVel(pos, fcomp, radius);

	
	// new position = old position + velocity * deltaTime
    volatile float4 ipos = integrPos[index];
	float3 ipos2 = make_float3(ipos.x, ipos.y, ipos.z);
	float Cd = 6*PI*radius*params.viscosity;
	newPos[index] = make_float4(ipos2 + fcomp/Cd*deltaTime, posData.w);

}




// calculate grid hash value for each particle
__global__
void calcHashD(uint*   gridParticleHash,  // output
               uint*   gridParticleIndex, // output
               float4* pos,               // input: positions
               uint    numParticles)
{
    uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if (index >= numParticles) return;
    
    volatile float4 p = pos[index];

    // get address in grid
    int3 gridPos = calcGridPos(make_float3(p.x, p.y, p.z));
    uint hash = calcGridHash(gridPos);

    // store grid hash and particle index
    gridParticleHash[index] = hash;
    gridParticleIndex[index] = index;
}

// rearrange particle data into sorted order, and find the start of each cell
// in the sorted hash array - also computes and adds the externally induced moment
__global__
void reorderDataAndFindCellStartD(uint*   cellStart,        // output: cell start index
							      uint*   cellEnd,          // output: cell end index
							      float4* sortedPos,        // output: sorted positions
                                  float4* sortedMoment,
								  uint *  gridParticleHash, // input: sorted grid hashes
                                  uint *  gridParticleIndex,// input: sorted particle indices
				                  float4* oldPos,		  // input: sorted position array
							      float4* oldMoment,
								  uint    numParticles)
{
	extern __shared__ uint sharedHash[];    // blockSize + 1 elements
    uint index = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;
	
    uint hash;
    // handle case when no. of particles not multiple of block size
    if (index < numParticles) {
        hash = gridParticleHash[index];

        // Load hash data into shared memory so that we can look 
        // at neighboring particle's hash value without loading
        // two hash values per thread
	    sharedHash[threadIdx.x+1] = hash;

	    if (index > 0 && threadIdx.x == 0)
	    {
		    // first thread in block must load neighbor particle hash
		    sharedHash[0] = gridParticleHash[index-1];
	    }
	}

	__syncthreads();
	
	if (index < numParticles) {
		// If this particle has a different cell index to the previous
		// particle then it must be the first particle in the cell,
		// so store the index of this particle in the cell.
		// As it isn't the first particle, it must also be the cell end of
		// the previous particle's cell

	    if (index == 0 || hash != sharedHash[threadIdx.x])
	    {
		    cellStart[hash] = index;
            if (index > 0)
                cellEnd[sharedHash[threadIdx.x]] = index;
	    }

        if (index == numParticles - 1)
        {
            cellEnd[hash] = index + 1;
        }

	    // Now use the sorted index to reorder the pos data and set moments
	    uint sortedIndex = gridParticleIndex[index];
	    float4 pos = FETCH(oldPos, sortedIndex);       // macro does either global read or texture fetch
        sortedPos[index] = pos;
		sortedMoment[index] = FETCH(oldMoment, sortedIndex);

	}

}


__device__ 
float3 cellH(int3 gridPos, uint index, float3 pos1, float3 moment1, uint* cellStart, uint* cellEnd, float4* oldPos, float4* oldMoment)
{
	uint gridHash = calcGridHash(gridPos);

	float4 mom2d;
	float3 mom2;
	float3 dr;
	float3 er;
	float dist;

    // get start of bucket for this cell
    uint startIndex = FETCH(cellStart, gridHash);

    float3 totalH = make_float3(0.0f);
    if (startIndex != 0xffffffff) {        // cell is not empty
        // iterate over particles in this cell
        uint endIndex = FETCH(cellEnd, gridHash);
        for(uint j=startIndex; j<endIndex; j++) {
            if (j != index) {              // check not colliding with self
               	float3 pos2 = make_float3(FETCH(oldPos, j));
				if(gridPos.x >= (int) params.gridSize.x)
					pos2.x += params.worldSize.x;
				if(gridPos.x < (int) 0)
					pos2.x -= params.worldSize.x;
				if(gridPos.z >= (int) params.gridSize.z)
					pos2.z += params.worldSize.z;
				if(gridPos.z < (int) 0)
					pos2.z -= params.worldSize.z;

	
				mom2d = oldMoment[j];
				mom2 = make_float3(mom2d);
				dr = pos1 - pos2;
				dist = length(dr);	
				er = dr/dist;
				if(dist < 8.0f*params.particleRadius[1]){
                	// if within cutoff add moments
                	totalH += (3.0f*dot(er,mom2)*er - mom2) / (4*PI*pow(dist,3));
				}
            }
        }
    }
    
	return totalH;
}


/*take in sorted positions and and output sorted mangnetic moments*/

__global__
void calcMoments(	float4* oldPos, 
					float4* oldMoment, 
					float4* newMoment,
					uint* gridParticleIndex, 
					uint* cellStart,
				   	uint* cellEnd, 
					uint numParticles)
{
	uint index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
    if (index >= numParticles) return;

   	// read particle data from sorted arrays
	// if we do variable particle radii, pack into the 4th spot of position, and i
	// then compute mup in the kernel
	volatile float4 posData = oldPos[index];
	volatile float4 momData = oldMoment[index];
	float3 moment = make_float3(momData.x, momData.y, momData.z);
	float3 pos = make_float3(posData.x, posData.y, posData.z);
	float radius = posData.w;
	float xi = momData.w;
	//float3 pos = make_float3(FETCH(oldPos, index));
	//float3 moment = make_float3(FETCH(oldMoment,index));

	// get address in grid
   	int3 gridPos = calcGridPos(pos);
	
	float3 H = params.externalH;
	for(int z=-params.interactionr; z<=params.interactionr; z++) { //loop on x y and z
   	   	for(int y=-params.interactionr; y<=params.interactionr; y++) {
       	   	for(int x=-params.interactionr; x<=params.interactionr; x++) {
				int3 neighborPos = gridPos + make_int3(x, y, z);
				//calculate H from each particle
				H += cellH(neighborPos, index, pos, moment, cellStart, cellEnd, oldPos, oldMoment);
	         }
		}
	}
	__syncthreads();
	
	float Cp = 4.0f/3.0f*PI*pow(radius,3)* 3.0f*(xi-1.0f)/(xi+2.0f);
	newMoment[index] = make_float4( Cp*H,xi);
}

__global__ void integrate(	float4* oldPos,
							float4* newPos,
							float4* forceA,
							float4* forceB,
							float deltaTime,
							uint numParticles)
{
   

	uint index = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;
    if (index >= numParticles) return;          // handle case when no. of particles not multiple of block size

	volatile float4 posData = oldPos[index];    // ensure coalesced read
    volatile float4 f1 = forceA[index];
    volatile float4 f2 = forceB[index];
	float3 pos = make_float3(posData.x, posData.y, posData.z);
    float radius = posData.w;
	float3 force1 = make_float3(f1.x, f1.y, f1.z);
	float3 force2 = make_float3(f2.x, f2.y, f2.z);
	float3 fcomp = (force1 + force2)/2;//trapezoid rule	

	fcomp = addFlowVel(pos, fcomp, radius);
	float Cd = 6*PI*params.viscosity*radius;

	//integrate	
	pos += fcomp*deltaTime/Cd;

	//periodic boundary conditions
	pos.x -= params.worldSize.x*floor((pos.x - params.worldOrigin.x)/params.worldSize.x);
	pos.y -= params.worldSize.y*floor((pos.y - params.worldOrigin.x)/params.worldSize.x);
	if (pos.z > -1.0f*params.worldOrigin.z ) { pos.z += 2.0f*params.worldOrigin.z;}
    if (pos.z < params.worldOrigin.z ) { pos.z -= 2.0f*params.worldOrigin.z; }

	newPos[index] = make_float4(pos, radius);

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

	volatile float4 posData = oldPos[index];    // ensure coalesced read
    float3 pos = make_float3(posData.x, posData.y, posData.z);
	float radius = posData.w;
	volatile float4 f1 = forceA[index];
    volatile float4 f2 = forceB[index];
	volatile float4 f3 = forceC[index];
	volatile float4 f4 = forceD[index];
	
	float3 force1 = make_float3(f1.x, f1.y, f1.z);
	float3 force2 = make_float3(f2.x, f2.y, f2.z);
	float3 force3 = make_float3(f3.x, f3.y, f3.z);
	float3 force4 = make_float3(f4.x, f4.y, f4.z);
	
	
	float3 fcomp = (force1 + 2*force2 + 2*force3 + force4)/6;//trapezoid rule	
	forceA[index] = make_float4(fcomp, f1.w);//averaged force
	
	fcomp = addFlowVel(pos, fcomp, radius);

	float Cd = 6*PI*params.viscosity*radius;
	//integrate	
	pos += fcomp*deltaTime/Cd;

	//periodic boundary conditions
   	pos.x -= params.worldSize.x*floor((pos.x - params.worldOrigin.x)/params.worldSize.x);
	pos.y -= params.worldSize.y*floor((pos.y - params.worldOrigin.x)/params.worldSize.x);
	if (pos.z > -1.0f*params.worldOrigin.z ) { pos.z += 2.0f*params.worldOrigin.z;}
    if (pos.z < params.worldOrigin.z ) { pos.z -= 2.0f*params.worldOrigin.z; }

	newPos[index] = make_float4(pos, radius);
	

}
/*
   This offers the potential ability to set the colors based on input parametsr
*/


__global__ void writeRender(const float4* pos, //never changes
							const float4* moments,
							const float4* forces,
							float4* rendPos, //output VBO - writes may cause an error in cuda-memcheck
							float4* rendColor,
							uint numParticles)
{
	uint index = blockIdx.x*blockDim.x + threadIdx.x;
	if(index >= numParticles) return;
	rendPos[index] = pos[index];
	float xi = moments[index].w;
	/*if(xi > 2.0f)
		rendColor[index] = make_float4(0.1f, 1.0f, 0.1f, 0.0f);
	else
		rendColor[index] = make_float4(0.1f, 0.1f, 1.0f, 0.0f);
	*/
	float3 force = make_float3(forces[index]);
	float3 colorOut = make_float3(1,1,1);
	float fmag = force.x + params.colorFmax/2;
	fmag = (fmag > params.colorFmax) ? params.colorFmax : fmag;
	fmag = (fmag < 0) ? 0 : fmag;
	
	const int ncolors = 3;
	float3 c[ncolors+1] = {
		make_float3(1.0, 0.0, 0.0),
		make_float3(0.0, 1.0, 0.0),
		make_float3(0.0, 0.0, 1.0),
		make_float3(0.0, 0.0, 1.0),
	};

	float fcolor = fmag/(params.colorFmax )*(ncolors-1);
	int base = (int) fcolor;
	float mix = fcolor - (float) base;
	colorOut = c[base] + mix*(c[base+1]-c[base]);
		
	if(xi == 1.0f) colorOut = make_float3(0.25f, 0.25f, 0.25f);

	rendColor[index] = make_float4(colorOut,0.0f);
}



#endif
