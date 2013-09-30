#ifndef particle_system_cuh
#define particle_system_cuh

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

#include "particles_kernel.h"
#include "new_kern.h"
 extern "C"
{
void cudaInit(int argc, char **argv);
void cudaGLInit(int argc, char **argv);

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

bool isOutofBounds(float4* pos, float3 border, uint numparticles);
bool excessForce(float4* pos, float force, uint numparticles);
float maxforce(float4* pos, uint numpartices);
float3 magnetization(float4* pos, uint numparticles, float simVol);
uint edgeCount(float4* forces, uint numParticles);
float calcTopForce(float4* forces, float4* position, uint numParticles, float ws, float cut);
float calcBotForce(float4* forces, float4* position, uint numParticles, float ws, float cut);
float calcGlForce(float4* forces, float4* position, uint numParticles, float ws, float cut);

float calcKinEn(float4* forces, float4* position, NewParams& params); 
uint numInteractions(uint* m_hNeighList, uint numParticles);
float maxvel(float4* forces, float4* positions, NewParams& params);
void resetMom(float4* moments, float3 extH, uint numParticles);
void pshift(float4* positions, float3 s, uint numParticles);



void sortParticles(uint *dGridParticleHash, uint *dGridParticleIndex, uint numParticles);

void renderStuff(const float* post, 
				const float* moment,
				const float* force,
				float* renderPos, 
				float* renderColor,
				float colorFmax,
				float scale,
				float rendercut,
				uint numParticles);
}

#endif
