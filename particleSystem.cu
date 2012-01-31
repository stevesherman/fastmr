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

#include "particles_kernel.cu"
#include "collisionkern.cu"

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

//Round a / b to nearest higher integer value
/*uint iDivUp(uint a, uint b){
    return (a % b != 0) ? (a / b + 1) : (a / b);
}*/

// compute grid and thread block size for a given number of elements
void computeGridSize(uint n, uint blockSize, uint &numBlocks, uint &numThreads)
{
    numThreads = min(blockSize, n);
    numBlocks = iDivUp(n, numThreads);
}

void calcForces	(float *sortedPos,//read in for calculations
                float* integrPos,//pos we read in for integration   
				float *newPos,
				float *forceOut,
				float *oldMoments,
				float deltaTime,
				uint* cellStart,
				uint* cellEnd,
				uint numParticles)
{
    uint numThreads, numBlocks;
    computeGridSize(numParticles, 64, numBlocks, numThreads);

    // execute the kernel
    calcParticleForce<<< numBlocks, numThreads >>>	((float4*)sortedPos,
												(float4*) integrPos,
												(float4*) newPos,
												(float4*) forceOut,
												(float4*) oldMoments,
                                           		deltaTime,
												cellStart,
												cellEnd,
												numParticles);
    
    // check if kernel invocation generated an error
    cutilCheckMsg("integrate kernel execution failed");
}

void calcHash(uint*  gridParticleHash,
              uint*  gridParticleIndex,
              float* pos, 
              int    numParticles)
{
    uint numThreads, numBlocks;
    computeGridSize(numParticles, 256, numBlocks, numThreads);
    // execute the kernel
    calcHashD<<< numBlocks, numThreads >>>(gridParticleHash,
                                           gridParticleIndex,
                                           (float4 *) pos,
                                           numParticles);
    
    // check if kernel invocation generated an error
    cutilCheckMsg("Hash Kernel execution failed");
}

void reorderDataAndFindCellStart(uint*  cellStart,
							     uint*  cellEnd,
							     float* sortedPos,
                                 float* newMoment,
								 uint*  gridParticleHash,
                                 uint*  gridParticleIndex,
							     float* oldPos,
							     float* oldMoment,
								 uint   numParticles,
							     uint   numCells)
{
    uint numThreads, numBlocks;
    computeGridSize(numParticles, 256, numBlocks, numThreads);
    // set all cells to empty
	//printf("CELLSTART: %p\t, numCells: %d\n", cellStart, numCells);
	cudaMemset(cellStart, 0xffffffff, numCells*sizeof(uint));

#if USE_TEX
    cudaBindTexture(0, oldPosTex, oldPos, numParticles*sizeof(float4));
	cudaBindTexture(0, oldMomentTex, oldPos, numparticles*sizeof(float4));
#endif

    uint smemSize = sizeof(uint)*(numThreads+1);
    reorderDataAndFindCellStartD<<< numBlocks, numThreads, smemSize>>>(
        cellStart,
        cellEnd,
        (float4*) sortedPos,
		(float4*) newMoment,
		gridParticleHash,
		gridParticleIndex,
        (float4 *) oldPos,
		(float4 *) oldMoment,
        numParticles);
    cutilCheckMsg("Kernel execution failed: reorderDataAndFindCellStartD");

#if USE_TEX
    cudaUnbindTexture(oldPosTex);
	cudaUnbindTexture(oldMomentTex);
#endif
}

void calcMoments(float* oldPos,
             	float* oldMoment,
				float* newMoment,
             uint*  gridParticleIndex,
             uint*  cellStart,
             uint*  cellEnd,
             uint   numParticles,
             uint   numCells)
{
#if USE_TEX
    cudaBindTexture(0, oldPosTex, oldPos, numParticles*sizeof(float4));
    cudaBindTexture(0, oldMomentTex, oldMoment, numParticles*sizeof(float4));
    cudaBindTexture(0, cellStartTex, cellStart, numCells*sizeof(uint));
    cudaBindTexture(0, cellEndTex, cellEnd, numCells*sizeof(uint));    
#endif

    // thread per particle
    uint numThreads, numBlocks;
    computeGridSize(numParticles, 64, numBlocks, numThreads);

    // execute the kernel
    calcMoments<<< numBlocks, numThreads >>>((float4*)oldPos,
                                          (float4*)oldMoment,
                                          (float4*)newMoment,
										  gridParticleIndex,
                                          cellStart,
                                          cellEnd,
                                          numParticles);

    // check if kernel invocation generated an error
    //cutilCheckMsg("Kernel execution failed");

#if USE_TEX
    cudaUnbindTexture(oldPosTex);
    cudaUnbindTexture(oldMomentTex);
    cudaUnbindTexture(cellStartTex);
    cudaUnbindTexture(cellEndTex);
#endif
}

void integrate(	float* oldPos,
				float* newPos,
				float* forceA,
				float* forceB,
				float deltaTime,
				uint numParticles)
{
	uint numThreads, numBlocks;
	computeGridSize(numParticles,128, numBlocks, numThreads);
	
	integrate<<<numBlocks, numThreads>>>((float4*) oldPos,
										(float4*) newPos,
										(float4*) forceA,
										(float4*) forceB,
										deltaTime,
										numParticles);
}

void RK4integrate(	float* oldPos,
					float* newPos,
					float* force1,
					float* force2,
					float* force3, 
					float* force4,
					float deltaTime,
					uint numParticles)
{
	uint numThreads, numBlocks;
	computeGridSize(numParticles,128,numBlocks, numThreads);

	integrateRK4 <<< numBlocks, numThreads >>> ((float4*) oldPos, 
												(float4*) newPos,
												(float4*) force1,
												(float4*) force2,
												(float4*) force3,
												(float4*) force4,
												deltaTime,
												numParticles);
}


void collIntegrateSystem(float *pos,
                     float *vel,
                     float deltaTime,
                     uint numParticles)
{
    uint numThreads, numBlocks;
    computeGridSize(numParticles, 256, numBlocks, numThreads);

    // execute the kernel
    cintegrate<<< numBlocks, numThreads >>>((float4*)pos,
                                           (float4*)vel,
                                           deltaTime,
                                           numParticles);
    
    // check if kernel invocation generated an error
    cutilCheckMsg("integrate kernel execution failed");
}


void collCollide(float* newVel,
             float* sortedPos,
             float* sortedVel,
             uint*  gridParticleIndex,
             uint*  cellStart,
             uint*  cellEnd,
             uint   numParticles,
             uint   numCells)
{
#if USE_TEX
    cudaBindTexture(0, oldPosTex, sortedPos, numParticles*sizeof(float4));
    cudaBindTexture(0, oldVelTex, sortedVel, numParticles*sizeof(float4));
    cudaBindTexture(0, cellStartTex, cellStart, numCells*sizeof(uint));
    cudaBindTexture(0, cellEndTex, cellEnd, numCells*sizeof(uint));    
#endif

    // thread per particle
    uint numThreads, numBlocks;
    computeGridSize(numParticles, 64, numBlocks, numThreads);

    // execute the kernel
    ccollideD<<< numBlocks, numThreads >>>((float4*)newVel,
                                          (float4*)sortedPos,
                                          (float4*)sortedVel,
                                          gridParticleIndex,
                                          cellStart,
                                          cellEnd,
                                          numParticles);

    // check if kernel invocation generated an error
    cutilCheckMsg("Kernel execution failed");

#if USE_TEX
    cudaUnbindTexture(oldPosTex);
    cudaUnbindTexture(oldVelTex);
    cudaUnbindTexture(cellStartTex);
    cudaUnbindTexture(cellEndTex);
#endif
}
	
void sortParticles(uint *dGridParticleHash, uint *dGridParticleIndex, uint numParticles)
{
    thrust::sort_by_key(thrust::device_ptr<uint>(dGridParticleHash),
                        thrust::device_ptr<uint>(dGridParticleHash + numParticles),
                        thrust::device_ptr<uint>(dGridParticleIndex));
}


struct isOut
{
	isOut(float bmax) : bmax(bmax) {}	
	
	__host__ __device__ bool operator()(const float4 &p){
		if(p.x*p.x >= bmax*bmax )
			return true;
		if(p.y*p.y > bmax*bmax)//>= due to pinning BCs
			return true;
		if(p.z*p.z >= bmax*bmax )
			return true;
		return false;
	}
	const float bmax;
};

bool isOutofBounds(float4* positions, float border, uint numParticles)
{
	int x = thrust::count_if(thrust::device_ptr<float4>(positions),
					thrust::device_ptr<float4>(positions+numParticles),
					isOut(border));
	if(x>0) printf("%d particles outofbounds\n", x);
	return x>0;
}


struct fuckyou 
{
	__host__ __device__ float4 operator()(const float4 &f1, const float4 &f2){
		return f1+f2;
	}
};

float4 magnetization(float4* moments, uint numParticles, float worldVol){
	float4 totalDp =  thrust::reduce(thrust::device_ptr<float4>(moments),
			thrust::device_ptr<float4>(moments+numParticles), 
			make_float4(0,0,0,0), fuckyou() );
	return totalDp/worldVol;

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
	
	float4 max1 = thrust::reduce(thrust::device_ptr<float4>(forces), 
			thrust::device_ptr<float4>(forces+numParticles), make_float4(0,0,0,0), forcemax());
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

	int x = thrust::count_if(thrust::device_ptr<float4>(forces),
			thrust::device_ptr<float4>(forces+numParticles),
			isExcessForce(maxforce));

	if(x>0) printf("%d particles with excessive movement\n", x);
	return x>0;

}


void renderStuff(const float* pos, 
				const float* moment, 
				const float* force, 
				float* rendPos, 
				float* rendColor,
				uint numParticles)
{
	uint numThreads, numBlocks;
	computeGridSize(numParticles, 256, numBlocks, numThreads);
	writeRender<<<numBlocks, numThreads>>>((float4*)pos, 
											(float4*)moment,
											(float4*)force,
											(float4*)rendPos,
											(float4*)rendColor,
											numParticles);
	cutilCheckMsg("Render Kernel execution failed");
}

}   // extern "C"
