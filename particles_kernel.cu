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


__global__ void integrate(	float4* oldPos,
							float4* newPos,
							float4* forceA,
							float4* forceB,
							float deltaTime,
							uint numParticles)
{
   

	uint index = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;
    if (index >= numParticles) return;          // handle case when no. of particles not multiple of block size

	//NOTE: volatile call because kernels aren't guaranteed to finish executing
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
	
	fcomp = addFlowVel(pos, fcomp, radius);

	float Cd = 6*PI*params.viscosity*radius;
	//integrate	
	pos += fcomp*deltaTime/Cd;

	//periodic boundary conditions
   	//pos.x -= params.worldSize.x*floor((pos.x - params.worldOrigin.x)/params.worldSize.x);
	//pos.z -= params.worldSize.z*floor((pos.z - params.worldOrigin.z)/params.worldSize.z);
	
	if (pos.x > -1.0f*params.worldOrigin.x ) { pos.x -= params.worldSize.x;}
    if (pos.x < params.worldOrigin.x ) { pos.x += params.worldSize.x;}
	if (pos.z > -1.0f*params.worldOrigin.z ) { pos.z -= params.worldSize.z;}
	if (pos.z < params.worldOrigin.z ) { pos.z += params.worldSize.z; }

	
	if (pos.y > -1.0f*params.worldOrigin.y ) { pos.y = -1.0f*params.worldOrigin.z;}
    if (pos.y < params.worldOrigin.y ) { pos.y = 1.0f*params.worldOrigin.z; }

	newPos[index] = make_float4(pos, radius);
}


__global__ void writeRender(const float4* pos, //never changes
							const float4* moments,
							const float4* forces,
							float4* rendPos, //output VBO - writes may cause an error in cuda-memcheck, b/c of leak from somewhere else?
							float4* rendColor,
							float colorFmax,
							uint numParticles)
{
	uint index = blockIdx.x*blockDim.x + threadIdx.x;
	if(index >= numParticles) return;
	
	rendPos[index] = pos[index];
	//float4 rpos = pos[index];
	float xi = moments[index].w;
	//rpos.w = (xi == 1.0f) ? 0.0 : rpos.w;
	//rendPos[index] = rpos;
	
	/*if(xi > 2.0f)
		rendColor[index] = make_float4(0.1f, 1.0f, 0.1f, 0.0f);
	else
		rendColor[index] = make_float4(0.1f, 0.1f, 1.0f, 0.0f);
	*/
	float3 force = make_float3(forces[index]);
	float3 colorOut = make_float3(1,1,1);
	float fmag = force.x + colorFmax/2.0f;
	fmag = (fmag > colorFmax) ? colorFmax : fmag;
	fmag = (fmag < 0) ? 0 : fmag;
	
	const int ncolors = 3;
	float3 c[ncolors+1] = {
		make_float3(1.0, 0.0, 0.0),
		make_float3(0.0, 1.0, 0.0),
		make_float3(0.0, 0.0, 1.0),
		make_float3(0.0, 0.0, 1.0),
	};

	float fcolor = fmag/(colorFmax )*(ncolors-1);
	int base = (int) fcolor;
	float mix = fcolor - (float) base;
	colorOut = c[base] + mix*(c[base+1]-c[base]);
		
	if(xi == 1.0f) colorOut = make_float3(0.25f, 0.25f, 0.25f);

	rendColor[index] = make_float4(colorOut,0.0f);
}



#endif
