/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA) 
 * associated with this source code for terms and conditions that govern 
 * your use of this NVIDIA software.
 * 
 */

// This file contains C wrappers around the some of the CUDA API and the
// kernel functions so that they can be called from "particleSystem.cpp"
#include <cutil_inline.h>
#include "cutil_math.h"

#include <cstdlib>
#include <cstdio>
#include <string.h>

#if defined(__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

#include <cuda_gl_interop.h>

#include "thrust/device_ptr.h"
#include "thrust/sort.h"
#include "thrust/count.h"
#include "thrust/functional.h"
#include "thrust/reduce.h"
#include "thrust/inner_product.h"
#include "particles_kernel.h"
#include "new_kern.h"
#include "particleSystem.cuh"
#include "particles_kernel.cu"

using namespace thrust;

extern "C"
{

uint iDivUp(uint a, uint b)
{
	return (a%b == 0) ? (a/b) : (a/b +1);
}

void cudaInit(int argc, char **argv)
{   
    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    if( cutCheckCmdLineFlag(argc, (const char**)argv, "device") ) {
        cutilDeviceInit(argc, argv);
    } else {
        cudaSetDevice( cutGetMaxGflopsDeviceId() );
    }
}

void cudaGLInit(int argc, char **argv)
{   
    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    if( cutCheckCmdLineFlag(argc, (const char**)argv, "device") ) {
        cutilDeviceInit(argc, argv);
    } else {
        cudaGLSetGLDevice( cutGetMaxGflopsDeviceId() );
    }
}

void threadSync()
{
    cudaThreadSynchronize();
}

void copyArrayToDevice(void* device, const void* host, int offset, int size)
{
    cudaMemcpy((char *) device + offset, host, size, cudaMemcpyHostToDevice);
}

void registerGLBufferObject(uint vbo, struct cudaGraphicsResource **cuda_vbo_resource)
{
    cudaGraphicsGLRegisterBuffer(cuda_vbo_resource, vbo, 
					       cudaGraphicsMapFlagsNone);
}

void unregisterGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource)
{
    cudaGraphicsUnregisterResource(cuda_vbo_resource);	
}

void *mapGLBufferObject(struct cudaGraphicsResource **cuda_vbo_resource)
{
    void *ptr;
    cudaGraphicsMapResources(1, cuda_vbo_resource, 0);
    size_t num_bytes; 
    cudaGraphicsResourceGetMappedPointer((void **)&ptr, &num_bytes,  
						       *cuda_vbo_resource);
    return ptr;
}

void unmapGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource)
{
   cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0);
}

void copyArrayFromDevice(void* host, const void* device, 
			 struct cudaGraphicsResource **cuda_vbo_resource, int size)
{   
    if (cuda_vbo_resource)
	device = mapGLBufferObject(cuda_vbo_resource);

    cudaMemcpy(host, device, size, cudaMemcpyDeviceToHost);
    
    if (cuda_vbo_resource)
	unmapGLBufferObject(*cuda_vbo_resource);
}

void setParameters(SimParams *hostParams)
{
    // copy parameters to constant memory
     cudaMemcpyToSymbol(params, hostParams, sizeof(SimParams));
}



// compute grid and thread block size for a given number of elements
void computeGridSize(uint n, uint blockSize, uint &numBlocks, uint &numThreads)
{
    numThreads = min(blockSize, n);
    numBlocks = iDivUp(n, numThreads);
}


void sortParticles(uint *dGridParticleHash, uint *dGridParticleIndex, uint numParticles)
{
    sort_by_key(device_ptr<uint>(dGridParticleHash),
                        device_ptr<uint>(dGridParticleHash + numParticles),
                        device_ptr<uint>(dGridParticleIndex));
}

struct f4norm : public unary_function<float4, float>{
	__host__ __device__ float operator() (const float4 &f){
		return sqrtf(f.x*f.x + f.y*f.y + f.z*f.z);
	}
};

struct isOut
{
	isOut(float bmax) : bmax(bmax) {}	
	
	__host__ __device__ bool operator()(const float4 &p){
		if(isnan(p.x) || isnan(p.y) || isnan(p.z))
			return true;
		if(fabsf(p.x) > bmax )
			return true;
		if(fabsf(p.y)-p.w > bmax)//>= due to pinning BCs? not true anymore i think
			return true;
		if(fabsf(p.z) > bmax )
			return true;
		return false;
	}
	const float bmax;
};

bool isOutofBounds(float4* positions, float border, uint numParticles)
{
	int x = count_if(device_ptr<float4>(positions),
					device_ptr<float4>(positions+numParticles),
					isOut(border));
	if(x>0) printf("%d particles outofbounds\n", x);
	return x>0;
}


float3 magnetization(float4* moments, uint numParticles, float worldVol){
	float4 totalDp =  reduce(device_ptr<float4>(moments),
			device_ptr<float4>(moments+numParticles), 
			make_float4(0,0,0,0), plus<float4>() );
	return make_float3(totalDp)/worldVol;

}

uint edgeCount(float4* forces, uint numParticles){
	float4 edge = reduce(device_ptr<float4>(forces),
			device_ptr<float4>(forces+numParticles), 
			make_float4(0,0,0,0), plus<float4>());
	return (uint) edge.w/2.0f;
}
//functors for finding the top and bottom particles
struct isTop  : public binary_function<float4, float4, float3> {
	isTop(float wsize, float cut) : pin_d(cut), wsize(wsize) {}
	__host__ __device__ float3 operator()(const float4& force, const float4& pos){
		if(pos.y >= wsize - pin_d*pos.w)
			return make_float3(force);
		else 
			return make_float3(0,0,0);
	}
	const float wsize;//half the worldisze
	const float pin_d;
};

struct isBot : public binary_function<float4, float4, float3> {
	isBot(float size, float cut) : pin_d(cut), wsize(size) {}
	__host__ __device__ float3 operator()(const float4& force, const float4& pos){
		if(pos.y <= -wsize + pin_d*pos.w)
			return make_float3(force);
		else 
			return make_float3(0,0,0);
	}
	const float pin_d;
	const float wsize;
};
//the functions
float calcTopForce(float4* forces, float4* position, uint numParticles, float wsize, float cut){
	float3 tforce = inner_product(device_ptr<float4>(forces),
			device_ptr<float4>(forces+numParticles),device_ptr<float4>(position),
			make_float3(0,0,0), plus<float3>(), isTop(wsize, cut));
	return tforce.x;
}

float calcBotForce(float4* forces, float4* position, uint numParticles, float wsize, float cut){
	float3 tforce = inner_product(device_ptr<float4>(forces),
			device_ptr<float4>(forces+numParticles),device_ptr<float4>(position),
			make_float3(0,0,0), plus<float3>(), isBot(wsize, cut));
	return tforce.x;
}
//global stress functor
struct stressThing : public binary_function<float4, float4, float3>{
	stressThing(float ws, float pd) : wsize(ws), pin_d(pd) {}
	__host__ __device__ float3 operator()(const float4& force, const float4& pos){
		if(fabsf(pos.y) <= wsize - pin_d*pos.w)
			return make_float3(force.x, force.y, force.z)*pos.y;
		else
			return make_float3(0,0,0);
	}
	const float pin_d;
	const float wsize;
};

float calcGlForce(float4* forces, float4* position, uint numParticles, float wsize, float cut = 0.0f){

	float3 glf = inner_product(device_ptr<float4>(forces), 
			device_ptr<float4>(forces+numParticles), device_ptr<float4>(position), 
			make_float3(0,0,0), plus<float3>(), stressThing(wsize, cut)); 
	return glf.x;
}

uint numInteractions(uint* neighList, uint numParticles){
	return reduce(device_ptr<uint>(neighList), device_ptr<uint>(neighList+numParticles),
			0, plus<uint>() );
}

//computes v^2 - should probably add a m term lol
struct kinen : public binary_function<float4, float4, float>{
	kinen(float v, float ws, float pd): visc(v), wsize(ws), pin_d(pd) {}	
	__host__ __device__ float operator()(const float4& f, const float4& p) 
	{
		float Cd = 6*PI_F*visc*p.w;
		if(fabsf(p.y) > wsize - p.w*pin_d) {
			return 0.0f;
		} else {
			return (f.x*f.x + f.y*f.y + f.z*f.z)/(Cd*Cd)*(4.0f/3.0f*PI_F*p.w*p.w*p.w);
		}
	}
	const float visc;
	const float wsize;
	const float pin_d;
};

float calcKinEn(float4* forces, float4* position, NewParams& params){
	kinen thingy = kinen(params.visc, params.L.y*0.5f, params.pin_d);	
	float kin = inner_product(device_ptr<float4>(forces),
				device_ptr<float4>(forces+params.N), device_ptr<float4>(position),	
				0.0f, plus<float>(), thingy );
	return kin*0.5f;
}


float maxforce(float4* forces, uint numParticles) {
	return transform_reduce(device_ptr<float4>(forces), device_ptr<float4>(forces+numParticles), 
			f4norm(),0.0f, maximum<float>());
}

struct	pvel : public binary_function<float4, float4, float> {
	pvel(float v, float ws, float pdist) : visc(v), wsize(ws), pin_d(pdist) {}
	
	__host__ __device__ float operator()(const float4 &f, const float4 &p) {
		float Cd = 6*PI_F*visc*p.w;
		if(fabsf(p.y) > wsize - p.w*pin_d){
			return 0.0f;
		} else {
			return sqrtf(f.x*f.x + f.y*f.y + f.z*f.z)/Cd;
		}
	}
	const float visc;
	const float wsize;
	const float pin_d;
};


float maxvel(float4* forces, float4* positions, NewParams& params){
	//use pos.w to get radius, 
	pvel vel_calc = pvel(params.visc, params.L.y*0.5f, params.pin_d);
	return inner_product(device_ptr<float4>(forces), device_ptr<float4>(forces+params.N),
			device_ptr<float4>(positions), 0.0f, maximum<float>(), vel_calc);
}

struct isExcessForce
{
	isExcessForce(float force) : force(force) {}	
	
	__host__ __device__ bool operator()(const float4 &f){
		if(f.x*f.x + f.y*f.y + f.z*f.z > force*force )
			return true;
		return false;
	}
	const float force;
};


bool  excessForce(float4* forces, float maxforce, uint numParticles){

	int x = count_if(device_ptr<float4>(forces),
			device_ptr<float4>(forces+numParticles),
			isExcessForce(maxforce));

	if(x>0) printf("%d particles with excessive movement\n", x);
	return x>0;

}

struct mom_reset
{
	mom_reset(float3 H) : extH(H) {}
	__host__ __device__ float4 operator()(const float4& m){
		return make_float4(extH*m.w, m.w);
	}
	const float3 extH;
};

void resetMom(float4* moments, float3 extH, uint numParticles){
	transform(device_ptr<float4>(moments), device_ptr<float4>(moments+numParticles),
			device_ptr<float4>(moments), mom_reset(extH));
}



void renderStuff(const float* pos, 
				const float* moment, 
				const float* force, 
				float* rendPos, 
				float* rendColor,
				float colorFmax,
				float scale,
				uint numParticles)
{
	uint numThreads, numBlocks;
	computeGridSize(numParticles, 256, numBlocks, numThreads);
	writeRender<<<numBlocks, numThreads>>>((float4*)pos, 
											(float4*)moment,
											(float4*)force,
											(float4*)rendPos,
											(float4*)rendColor,
											colorFmax,
											scale,
											numParticles);
	cutilCheckMsg("Render Kernel execution failed");
}

}   // extern "C"
