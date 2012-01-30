#define PI 3.141592653589793f


#include <stdio.h>
#include <math.h>
#include "cutil_math.h"
#include "math_constants.h"
#include "particles_kernel.cuh"

#if USE_TEX
// textures for particle position and velocity
texture<float4, 1, cudaReadModeElementType> oldPosTex;
texture<float4, 1, cudaReadModeElementType> oldVelTex;


texture<uint, 1, cudaReadModeElementType> gridParticleHashTex;
texture<uint, 1, cudaReadModeElementType> cellStartTex;
texture<uint, 1, cudaReadModeElementType> cellEndTex;
#endif


// integrate particle attributes
__global__
void cintegrate(float4* posArray,  // input/output
               float4* velArray,  // input/output
               float deltaTime,
               uint numParticles)
{
    uint index = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;
    if (index >= numParticles) return;          // handle case when no. of particles not multiple of block size

	float4 posData = posArray[index];    // ensure coalesced read
    float4 velData = velArray[index];
    float3 pos = make_float3(posData.x, posData.y, posData.z);
    float3 vel = make_float3(velData.x, velData.y, velData.z);
	float radius = posData.w;
    vel *= params.globalDamping;

    // new position = old position + velocity * deltaTime
    pos += vel * deltaTime;

	// wall bounce
	if (pos.x > -1.0f*params.worldOrigin.x)	{ 
		pos.x += 2.0f*params.worldOrigin.x;
	}
    if (pos.x < params.worldOrigin.x) { 
		pos.x -= 2.0f*params.worldOrigin.x;
	}
	if (pos.y > -1.0f*(params.worldOrigin.y+radius) ) { 
		pos.y = -1.0f*(params.worldOrigin.y+radius); 
		vel.y*=params.boundaryDamping;
	}
    if (pos.y < (params.worldOrigin.y+radius) ) { 
		pos.y = (params.worldOrigin.y+radius); 
		vel.y*=params.boundaryDamping;
	}
	if (pos.z > -1.0f*params.worldOrigin.z)	{ 
		pos.z += 2.0f*params.worldOrigin.z;
	}
    if (pos.z < params.worldOrigin.z) { 
		pos.z -= 2.0f*params.worldOrigin.z;
	}


    // store new position and velocity
    posArray[index] = make_float4(pos, radius);
    velArray[index] = make_float4(vel, velData.w);
}


// collide two spheres using DEM method
__device__
float3 ccollideSpheres(float3 posA, float3 posB,
                      float3 velA, float3 velB,
                      float radiusA, float radiusB)
{
	// calculate relative position
    float3 relPos = posB - posA;

    float dist = length(relPos);
    float collideDist = (radiusA + radiusB);

    float3 force = make_float3(0.0f);
    if (dist < collideDist) {
        float3 norm = relPos / dist;

		// relative velocity
        float3 relVel = velB - velA;

        // spring force
        force = -params.cspring*(collideDist - dist) * norm;
        // dashpot (damping) force
        force += params.cdamping*relVel;
    }

    return force;
}



// collide a particle against all other particles in a given cell
__device__
float3 ccollideCell(int3    gridPos,
                   uint    index,
                   float3  pos,
                   float3  vel,
                   float   radius1,
				   float4* oldPos, 
                   float4* oldVel,
                   uint*   cellStart,
                   uint*   cellEnd)
{
    uint gridHash = calcGridHash(gridPos);

    // get start of bucket for this cell
    uint startIndex = FETCH(cellStart, gridHash);
	//int interactions = 0;
    float3 force = make_float3(0.0f);
    if (startIndex != 0xffffffff) {        // cell is not empty
        // iterate over particles in this cell
        uint endIndex = FETCH(cellEnd, gridHash);
        for(uint j=startIndex; j<endIndex; j++) {
            if (j != index) {              // check not colliding with self
	            float4 posdata2 = FETCH(oldPos, j);
				float3 pos2 = make_float3(posdata2);
				float radius2 = posdata2.w;
                float3 vel2 = make_float3(FETCH(oldVel, j));
				
				if(gridPos.x >= (int) params.gridSize.x)
					pos2.x += -2.0f*params.worldOrigin.x;
				if(gridPos.x < (int) 0)
					pos2.x -= -2.0f*params.worldOrigin.x;

				if(gridPos.z >= (int) params.gridSize.z)
					pos2.z += -2.0f*params.worldOrigin.z;
				if(gridPos.z < (int) 0)
					pos2.z -= -2.0f*params.worldOrigin.z;
                
				// collide two spheres
                force += ccollideSpheres(pos, pos2, vel, vel2, 1.01f*radius1, 1.01f*radius2);
            }
        }
    }
    return force;
}


__global__
void ccollideD(float4* newVel,               // output: new velocity
              float4* oldPos,               // input: sorted positions
              float4* oldVel,               // input: sorted velocities
              uint*   gridParticleIndex,    // input: sorted particle indices
              uint*   cellStart,
              uint*   cellEnd,
              uint    numParticles)
{
    uint index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
    if (index >= numParticles) return;    
    
    // read particle data from sorted arrays
	float4 posdata = FETCH(oldPos,index);
	float3 pos = make_float3(posdata);
    float3 vel = make_float3(FETCH(oldVel, index));
	float radius = posdata.w; 
    // get address in grid
    int3 gridPos = calcGridPos(pos);

    // examine neighbouring cells
    float3 force = make_float3(0,0,0);
    for(int z=-params.interactionr; z<=params.interactionr; z++) {
        for(int y=-params.interactionr; y<=params.interactionr; y++) {
            for(int x=-params.interactionr; x<=params.interactionr; x++) {
                int3 neighbourPos = gridPos + make_int3(x, y, z);
                force += ccollideCell(neighbourPos, index, pos, vel, radius, oldPos, oldVel, cellStart, cellEnd);
            }
        }
    }


    // write new velocity back to original unsorted location
    uint originalIndex = gridParticleIndex[index];
    newVel[originalIndex] = make_float4(vel + force, 0.0f);
}


