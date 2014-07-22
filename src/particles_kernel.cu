/* 
 * CUDA particle system kernel code.
 */

#ifndef _PARTICLES_KERNEL_H_
#define _PARTICLES_KERNEL_H_

#define PI_F 3.141592653589793f

#include <stdio.h>
#include <math.h>
#include "helper_math.h"
#include "particles_kernel.h"

__constant__ SimParams params;

__global__ void writeRender(const float4* pos, 
							const float4* moments,
							const float4* forces,
							float4* rendPos, 
							float4* rendColor,
							float colorFmax,
							float scale,
							float rendercut,
							uint numParticles)
{
	uint index = blockIdx.x*blockDim.x + threadIdx.x;
	if(index >= numParticles) return;
	
	float4 rpos = pos[index];
	rpos.w = rpos.w * scale;
	if(rpos.z > rendercut*params.worldOrigin.z)	
		rpos.w = 0;
	float Cpol = moments[index].w;
	//rpos.w = (Cpol == 1.0f) ? 0.0 : rpos.w;
	//rendPos[index] = rpos;
	
	/*if(Cpol > 2.0f)
		rendColor[index] = make_float4(0.1f, 1.0f, 0.1f, 0.0f);
	else
		rendColor[index] = make_float4(0.1f, 0.1f, 1.0f, 0.0f);
	*/
	float3 force = make_float3(forces[index]);
	float3 colorOut = make_float3(1,1,1);
	float fmag = force.x + colorFmax/2.0f;
	fmag = (fmag > colorFmax) ? colorFmax : fmag;
	fmag = (fmag < 0) ? 0 : fmag;
	
//	rpos.w = length(force) > 1e-5f ? rpos.w : 0;

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
		
	if(Cpol == 0.0f) colorOut = make_float3(0.25f, 0.25f, 0.25f);
	
	rendPos[index] = rpos;
	rendColor[index] = make_float4(colorOut,0.0f);
}



#endif
