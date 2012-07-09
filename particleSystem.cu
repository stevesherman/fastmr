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
#include "thrust/extrema.h"
#include "thrust/reduce.h"
#include "thrust/inner_product.h"

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


struct isOut
{
	isOut(float bmax) : bmax(bmax) {}	
	
	__host__ __device__ bool operator()(const float4 &p){
		if(isnan(p.x) || isnan(p.y) || isnan(p.z))
			return true;
		if(p.x*p.x > bmax*bmax )
			return true;
		if(p.y*p.y > bmax*bmax)//>= due to pinning BCs? not true anymore i think
			return true;
		if(p.z*p.z > bmax*bmax )
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


float4 magnetization(float4* moments, uint numParticles, float worldVol){
	float4 totalDp =  reduce(device_ptr<float4>(moments),
			device_ptr<float4>(moments+numParticles), 
			make_float4(0,0,0,0), plus<float4>() );
	return totalDp/worldVol;

}

uint edgeCount(float4* forces, uint numParticles){
	float4 edge = reduce(device_ptr<float4>(forces),
			device_ptr<float4>(forces+numParticles), 
			make_float4(0,0,0,0), plus<float4>());
	return (uint) edge.w/2.0f;
}

struct isTop {
	isTop(float cut) : cut(cut) {}
	__host__ __device__ float4 operator()(const float4& force, const float4& pos){
		if(pos.y > cut)
			return force;
		else 
			return make_float4(0,0,0,0);
	}
	const float cut;
};

struct isBot {
	isBot(float cut) : cut(cut) {}
	__host__ __device__ float4 operator()(const float4& force, const float4& pos){
		if(pos.y < cut)
			return force;
		else 
			return make_float4(0,0,0,0);
	}
	const float cut;
};

float calcTopForce(float4* forces, float4* position, uint numParticles, float cut){
	float4 tforce = inner_product(device_ptr<float4>(forces),
			device_ptr<float4>(forces+numParticles),device_ptr<float4>(position),
			make_float4(0,0,0,0), plus<float4>(), isTop(cut));
	return tforce.x;
}

float calcBotForce(float4* forces, float4* position, uint numParticles, float cut){
	float4 tforce = inner_product(device_ptr<float4>(forces),
			device_ptr<float4>(forces+numParticles),device_ptr<float4>(position),
			make_float4(0,0,0,0), plus<float4>(), isBot(cut));
	return tforce.x;
}

struct stressThing : public binary_function<float4, float4, float3>{
	__host__ __device__ float3 operator()(const float4& force, const float4& pos){
		return make_float3(force.x, force.y, force.z)*pos.y;
	}
};

float calcGlForce(float4* forces, float4* position, uint numParticles){

	float3 glf = inner_product(device_ptr<float4>(forces), 
			device_ptr<float4>(forces+numParticles), device_ptr<float4>(position), 
			make_float3(0,0,0), plus<float3>(), stressThing()); 
	return glf.x;
}

struct f4norm : public unary_function<float4, float>{
	__host__ __device__ float operator()(const float4& f) 
	{
		return f.x*f.x + f.y*f.y + f.z*f.z;
	}
};

float calcKinEn(float4* forces, uint numParticles){
	
	float kin = transform_reduce(device_ptr<float4>(forces),
				device_ptr<float4>(forces+numParticles), f4norm(),	
				0.0f, plus<float>());
	return kin;
}
struct forcemax 
{
	__host__ __device__ float4 operator() (const float4 &f1, const float4 &f2){
		if (sqrt(f1.x*f1.x + f1.y*f1.y + f1.z*f1.z) > sqrt(f2.x*f2.x + f2.y*f2.y + f2.z*f2.z))
			return f1;
		else
			return f2;
	}
};

float maxforce(float4* forces, uint numParticles)
{
	
	float4 max1 = reduce(device_ptr<float4>(forces), 
			device_ptr<float4>(forces+numParticles), make_float4(0,0,0,0), forcemax());
	return sqrt(max1.x*max1.x + max1.y*max1.y + max1.z*max1.z);	
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





void renderStuff(const float* pos, 
				const float* moment, 
				const float* force, 
				float* rendPos, 
				float* rendColor,
				float colorFmax,
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
											numParticles);
	cutilCheckMsg("Render Kernel execution failed");
}

}   // extern "C"
