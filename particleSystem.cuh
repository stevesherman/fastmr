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
 
 extern "C"
{
void cudaInit(int argc, char **argv);

void allocateArray(void **devPtr, int size);
void freeArray(void *devPtr);

void threadSync();

void copyArrayFromDevice(void* host, const void* device, struct cudaGraphicsResource **cuda_vbo_resource, int size);
void copyArrayToDevice(void* device, const void* host, int offset, int size);
void registerGLBufferObject(uint vbo, struct cudaGraphicsResource **cuda_vbo_resource);
void unregisterGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource);
void *mapGLBufferObject(struct cudaGraphicsResource **cuda_vbo_resource);
void unmapGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource);


void setParameters(SimParams *hostParams);

void calcForces(float *sortedPos,
				float *integrPos,
				float *newPos,
				float *force,
				float *moments,
				float deltaTime,
				uint* cellStart,
				uint* cellEnd,
                uint numParticles);

void calcHash(uint*  gridParticleHash,
              uint*  gridParticleIndex,
              float* pos, 
              int    numParticles);

void reorderDataAndFindCellStart(uint*  cellStart,
							     uint*  cellEnd,
							     float* sortedPos,
                                 float* newMoment,
								 uint*  gridParticleHash,
                                 uint*  gridParticleIndex,
							     float* oldPos,
							     float* oldMoment,
								 uint   numParticles,
							     uint   numCells);

void calcMoments(float* oldPos,
                float* oldMoment,
				float* newMoment,
             uint*  gridParticleIndex,
             uint*  cellStart,
             uint*  cellEnd,
             uint   numParticles,
             uint   numCells);

void integrate(float* oldPos,
				float* newPos,
				float* forceA,
				float* forceB,
				float deltaTime,
				uint numParticles);

void RK4integrate(float* oldPos,
				float* newPos,
				float* force1,
				float* force2,
				float* force3,
				float* force4,
				float deltaTime,
				uint numParticles);

bool isOutofBounds(float4* pos, float border, uint numparticles);
bool excessForce(float4* pos, float force, uint numparticles);
float maxforce(float4* pos, uint numpartices);
float4 magnetization(float4* pos, uint numparticles, float simVol);

void collComputeGridSize(uint n, uint blockSize, uint &numBlocks, uint &numThreads);

	
void collIntegrateSystem(float *sortedPos,
				float *vel,
				float deltaTime,
                uint numParticles);

void collCalcHash(uint*  gridParticleHash,
              uint*  gridParticleIndex,
              float* pos, 
              int    numParticles);

void collCollide(float* newVel, 
				float* sortedPos, 
				float* sortedVel,
				uint* gridParticleIndex,
				uint* cellStart,
				uint* cellEnd,
				uint numParticles,
				uint numCells);

void sortParticles(uint *dGridParticleHash, uint *dGridParticleIndex, uint numParticles);
void renderStuff(const float* post, 
				const float* moment,
				const float* force,
				float* renderPos, 
				float* renderColor,
				uint numParticles);
}
