#include "particleSystem.h"
#include "particleSystem.cuh"
#include "particles_kernel.h"
#include "connectedgraphs.h"
#include "new_kern.h"
#include "new_kcall.h"
#include "sfc_pack.h"
#include <cutil_inline.h>
#include <cutil_math.h>

#include <assert.h>
#include <math.h>
#include <memory.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <GL/glew.h>

#define PI_F   3.141592653589793f 
#define MU_0 4e-7f*PI_F
#ifndef MU_C
#define MU_C 1
#endif

ParticleSystem::ParticleSystem(SimParams params, bool useGL, float3 worldSize):
	m_bInitialized(false),
	m_bUseOpenGL(false)
{
	newp.L = worldSize;
	m_params = params;
	m_numGridCells = m_params.gridSize.x*m_params.gridSize.y*m_params.gridSize.z;	
	
	m_params.uf = MU_C*MU_0;
	m_params.Cpol = 4.0f*PI_F*pow(m_params.pRadius[0],3)*
		(m_params.mu_p[0] - MU_C)/(m_params.mu_p[0]+2.0f*MU_C);	
	m_params.globalDamping = 0.8f; 
    m_params.cdamping = 0.03f;
	m_params.boundaryDamping = -0.03f;

	m_numParticles = m_params.numBodies;
	m_maxNeigh = (uint) ((m_params.volfr[0]+m_params.volfr[1]+m_params.volfr[2])*680.0f);

	m_bUseOpenGL = useGL;

	newp.N = m_numParticles;
	newp.gridSize = m_params.gridSize;
	newp.numCells = m_numGridCells;
	newp.cellSize = m_params.cellSize;
	newp.origin = m_params.worldOrigin;
	newp.Linv = 1/newp.L;
	newp.max_fdr_sq = 8.0f*m_params.pRadius[0]*8.0f*m_params.pRadius[0];
	newp.numAdjCells = 27;
	newp.spring = m_params.spring;
	//newp.uf = m_params.uf;
	newp.shear = m_params.shear;
	newp.visc = m_params.viscosity;
	newp.extH = m_params.externalH;
	newp.Cpol = m_params.Cpol;
	newp.pin_d = 1.5f;  //ybot < radius*pin_d

	m_contact_dist = 1.05f;	
	_initialize();

}




ParticleSystem::~ParticleSystem()
{
    _finalize();
    newp.N = 0;
}

uint
ParticleSystem::createVBO(uint size)
{
    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    return vbo;
}

// create a color ramp
void colorRamp(float /*t*/, float *r)
{
   	r[0] = .1;
	r[1] = 1;
	r[2] = .1;
}

void
ParticleSystem::_initialize()
{
    assert(!m_bInitialized);
	m_randSet = 200;
    // allocate host storage
    m_hPos = new float[newp.N*4];
    m_hMoments = new float[newp.N*4];
	m_hForces = new float[newp.N*4];
	
	memset(m_hPos, 0, newp.N*4*sizeof(float));
	memset(m_hForces, 0, newp.N*4*sizeof(float));
	memset(m_hMoments, 0, newp.N*4*sizeof(float));

   	m_hCellStart = new uint[m_numGridCells];
    memset(m_hCellStart, 0, m_numGridCells*sizeof(uint));

    m_hCellEnd = new uint[m_numGridCells];
    memset(m_hCellEnd, 0, m_numGridCells*sizeof(uint));

    // allocate GPU data
    unsigned int memSize = sizeof(float) * 4 * (newp.N);

    if (m_bUseOpenGL) {
        m_posVbo = createVBO(memSize);    
		registerGLBufferObject(m_posVbo, &m_cuda_posvbo_resource);
    } else {
       	cutilSafeCall( cudaMalloc( (void **)&m_cudaPosVBO, memSize )) ;
    }

    cudaMalloc((void**)&m_dMomentsA, memSize);
	cudaMalloc((void**)&m_dMomentsB, memSize);
	cudaMalloc((void**)&m_dForces1, memSize);
	cudaMalloc((void**)&m_dForces2, memSize);
	cudaMalloc((void**)&m_dForces3, memSize);
	cudaMalloc((void**)&m_dForces4, memSize);

	cudaMalloc((void**)&m_dPos, memSize);
    cudaMalloc((void**)&m_dSortedPos, memSize);
	cudaMalloc((void**)&m_dMidPos, memSize);

	cudaMalloc((void**)&m_dGridParticleHash, newp.N*sizeof(uint));
    cudaMalloc((void**)&m_dGridParticleIndex, newp.N*sizeof(uint));

    assert(cudaMalloc((void**)&m_dCellStart, m_numGridCells*sizeof(uint)) == cudaSuccess);
    cudaMalloc((void**)&m_dCellEnd, m_numGridCells*sizeof(uint));


	assert(cudaMalloc((void**)&m_dCellAdj, m_numGridCells*newp.numAdjCells*sizeof(uint)) == cudaSuccess);
	m_hCellAdj = new uint[m_numGridCells*newp.numAdjCells];

	m_hCellHash = new uint[m_numGridCells];
	cudaMalloc((void**)&m_dCellHash, m_numGridCells*sizeof(uint));
	
	assert(cudaMalloc((void**)&m_dNumNeigh, newp.N*sizeof(uint)) == cudaSuccess);
	m_hNumNeigh = new uint[newp.N];
	
    if (m_bUseOpenGL) {
        m_colorVBO = createVBO(newp.N*4*sizeof(float));
		registerGLBufferObject(m_colorVBO, &m_cuda_colorvbo_resource);

        // fill color buffer
        glBindBufferARB(GL_ARRAY_BUFFER, m_colorVBO);
        float *data = (float *) glMapBufferARB(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
        float *ptr = data;
        for(uint i=0; i<newp.N; i++) {
            float t = i / (float) newp.N;
            colorRamp(t, ptr);
            ptr+=3;
            *ptr++ = 1.0f;
        }
        glUnmapBufferARB(GL_ARRAY_BUFFER);
    } else {
        cutilSafeCall( cudaMalloc( (void **)&m_cudaColorVBO, sizeof(float)*newp.N*4) );
    }
 	
	assert(cudaMalloc((void**)&m_dNeighList, newp.N*m_maxNeigh*sizeof(uint)) == cudaSuccess);
		
	cutilCheckError(cutCreateTimer(&m_timer));
	
    setParameters(&m_params);
	m_bInitialized = true;
}

void
ParticleSystem::_finalize()
{
    assert(m_bInitialized);

    delete [] m_hPos;
    delete [] m_hMoments;
	delete [] m_hForces;
    delete [] m_hCellStart;
    delete [] m_hCellEnd;
	
	delete [] m_hCellAdj;
	delete [] m_hNumNeigh;
	delete [] m_hCellHash;
	
	cudaFree(m_dCellHash);
	cudaFree(m_dCellAdj);
	cudaFree(m_dNeighList);
	cudaFree(m_dNumNeigh);

    cudaFree(m_dMomentsA);
	cudaFree(m_dMomentsB);
	cudaFree(m_dForces1);
    cudaFree(m_dForces2);
	cudaFree(m_dForces3);
	cudaFree(m_dForces4);
	
	cudaFree(m_dSortedPos);
	cudaFree(m_dPos);
	cudaFree(m_dMidPos);
    
	cudaFree(m_dGridParticleHash);
    cudaFree(m_dGridParticleIndex);
    cudaFree(m_dCellStart);
    cudaFree(m_dCellEnd);
    
	if (m_bUseOpenGL) {
        unregisterGLBufferObject(m_cuda_posvbo_resource);
        glDeleteBuffers(1, (const GLuint*)&m_posVbo);
        glDeleteBuffers(1, (const GLuint*)&m_colorVBO);
    } else {
        cutilSafeCall( cudaFree(m_cudaPosVBO) );
        cutilSafeCall( cudaFree(m_cudaColorVBO) );
    }

}

// step the simulation
float ParticleSystem::update(float deltaTime, float maxdxpct)
{
    assert(m_bInitialized);
    float *dRendPos, *dRendColor;
    if (m_bUseOpenGL) 
	{
        dRendPos = (float *) mapGLBufferObject(&m_cuda_posvbo_resource);
    	dRendColor = (float *) mapGLBufferObject(&m_cuda_colorvbo_resource);
		
		renderStuff(m_dPos, m_dMomentsA, m_dForces1, dRendPos, dRendColor, m_colorFmax, newp.N);
		
		unmapGLBufferObject(m_cuda_posvbo_resource);
		unmapGLBufferObject(m_cuda_colorvbo_resource);

	} else {
        dRendPos = (float *) m_cudaPosVBO;
		dRendColor = (float*) m_cudaColorVBO; //shouldn't be a big deal, as color is only touched above
    }
	cutilCheckMsg("ohno");	
	setParameters(&m_params);
	setNParameters(&newp);

	//isOutofBounds((float4*) m_dPos, newp.origin.x, newp.N);
	comp_phash(m_dPos, m_dGridParticleHash, m_dGridParticleIndex, m_dCellHash, newp.N, m_numGridCells);
	// sort particles based on hash
	sortParticles(m_dGridParticleHash, m_dGridParticleIndex, newp.N);
	// reorder particle arrays into sorted order and
	// find start and end of each cell - also sets the fixed moment
	find_cellStart(m_dCellStart, m_dCellEnd, m_dGridParticleHash, newp.N, m_numGridCells);
	cutilCheckMsg("Cstart");
	reorder(m_dGridParticleIndex, m_dSortedPos, m_dMomentsB, m_dPos, m_dMomentsA, newp.N);	
	cutilCheckMsg("Reorder");
	float* temp = m_dMomentsB;
	m_dMomentsB = m_dMomentsA;
	m_dMomentsA = temp;
			

	if(m_randSet > 0)
	{
		
		temp = m_dForces2;
		m_dForces2 = m_dForces1;
		m_dForces1 = temp;
		NListFixed(m_dNeighList, m_dNumNeigh, m_dSortedPos, m_dGridParticleHash, 
				m_dCellStart, m_dCellEnd, m_dCellAdj, newp.N, m_maxNeigh, 3.1f*m_params.pRadius[0]);

		collision_new(m_dSortedPos, m_dForces2, m_dNeighList, m_dNumNeigh, m_dForces1, m_dPos, newp.N, 0.01f);
		deltaTime = 0;
		m_randSet--;
	} else {
		NListFixed(m_dNeighList, m_dNumNeigh, m_dSortedPos, m_dGridParticleHash, 
				m_dCellStart, m_dCellEnd, m_dCellAdj, newp.N, m_maxNeigh, 8.1f*m_params.pRadius[0]);

	
		bool solve = true;

		//if the particles are moving too much, half the timestep and resolve
		while(solve) {
			
			magForces(	m_dSortedPos,	//yin: yn 
						m_dSortedPos,	//yn
						m_dMidPos,   	//yn + 1/2*k1
						m_dForces1,   	//k1
						m_dMomentsA, m_dNeighList, m_dNumNeigh, newp.N, deltaTime/2);
			cutilCheckMsg("magForces");
			magForces(	m_dMidPos, 		//yin: yn + 1/2*k1
						m_dSortedPos, 	//yn
						m_dPos, 		//yn + 1/2*k2
						m_dForces2,		//k2
						m_dMomentsA, m_dNeighList, m_dNumNeigh, newp.N, deltaTime/2);
			magForces(	m_dPos, 		//yin: yn + 1/2*k2
						m_dSortedPos, 	//yn
						m_dMidPos, 		//yn + k3
						m_dForces3,		//k3
						m_dMomentsA, m_dNeighList, m_dNumNeigh, newp.N, deltaTime);
			magForces(	m_dMidPos, 		//yin: yn + k3
						m_dSortedPos, 	//yn
						m_dPos, 		// doesn't matter
						m_dForces4,		//k4
						m_dMomentsA, m_dNeighList, m_dNumNeigh, newp.N, deltaTime/2);

			RK4integrate(m_dSortedPos,//yn 
						m_dPos, //yn+1
						m_dForces1, //1/6*(k1 + 2*k2 + 2*k3 + k4) 
						m_dForces2, m_dForces3, m_dForces4, deltaTime, newp.N);
	
			solve = false;	
		
					
			//need some sort of controller for error
			
			//find max force
			//printf("callmax\n");
			//maxf = maxforce( (float4*) m_dForces1, newp.N);
			//maxFdx = maxdxpct*Cd*m_params.pRadius[0]/deltaTime; //force to cause a dx
			float maxDx = maxvel((float4*)m_dForces1,(float4*)m_dPos,newp)*deltaTime;
			float limDx = maxdxpct*m_params.pRadius[0];
			
			if(maxDx > limDx){
				solve = true;
			} /*else { //if not excess force, check for out of bounds
				solve = isOutofBounds((float4*)m_dPos, -newp.origin.x, newp.N);
			}*/
			if(solve){
				deltaTime *=.5f;
				assert(deltaTime != 0);
				printf("force excess ratio %.3g\treducing timestep %.3g\n", maxDx/limDx, deltaTime);
				//getBadP();	
			}
		}
	}
		
	return deltaTime;
}

void
ParticleSystem::dumpGrid()
{
    // dump grid information
    copyArrayFromDevice(m_hCellStart, m_dCellStart, 0, sizeof(uint)*m_numGridCells);
    copyArrayFromDevice(m_hCellEnd, m_dCellEnd, 0, sizeof(uint)*m_numGridCells);
    uint maxCellSize = 0;
    for(uint i=0; i<m_numGridCells; i++) {
        if (m_hCellStart[i] != 0xffffffff) {
            uint cellSize = m_hCellEnd[i] - m_hCellStart[i];
            printf("cell: %d, %d particles\n", i, cellSize);
            if (cellSize > maxCellSize) maxCellSize = cellSize;
        }
    }
   	printf("maximum particles per cell = %d\n", maxCellSize);
}
//should not be called, as it is very slow!
//use is out!
void ParticleSystem::getBadP()
{
	int nans = 0, outbounds = 0;
	copyArrayFromDevice(m_hPos, m_dPos, 0, sizeof(float)*4*newp.N);
	for(int i = 0; i < (int) newp.N; i++){
		if( isnan(m_hPos[4*i]) || isnan(m_hPos[4*i+1]) || isnan(m_hPos[4*i+2]) )
			nans++;
		if( pow(m_hPos[4*i],2) > pow(newp.origin.x,2) ||  pow(m_hPos[4*i+1],2) > pow(newp.origin.y,2) || 
				pow(m_hPos[4*i+2],2) > pow(newp.origin.z,2) )
		   outbounds++;
	}
	printf("nans: %d \toutofbounds: %d\n", nans, outbounds);

}

void ParticleSystem::getMagnetization()
{
	float3 M = magnetization((float4*) m_dMomentsA, newp.N, 
			newp.L.x*newp.L.y*newp.L.z);
	printf("M: %g %g %g\n", M.x, M.y, M.z );
}


void ParticleSystem::getGraphData(uint& graphs, uint& edges)
{
	uint maxn = NListVar(m_dNeighList, m_dNumNeigh, m_dSortedPos, m_dGridParticleHash, 
			m_dCellStart, m_dCellEnd, m_dCellAdj, newp.N, m_maxNeigh, m_contact_dist);
	edges = numInteractions(m_dNumNeigh, newp.N)/2;
	
	m_hNeighList = new uint[newp.N*maxn];
	copyArrayFromDevice(m_hNeighList, m_dNeighList, 0, sizeof(uint)*newp.N*maxn);
	copyArrayFromDevice(m_hNumNeigh,  m_dNumNeigh,  0, sizeof(uint)*newp.N);
	graphs = adjConGraphs(m_hNeighList, m_hNumNeigh, newp.N);
	delete [] m_hNeighList;
}

uint ParticleSystem::getInteractions(){
	return numInteractions(m_dNumNeigh, newp.N);
}

void
ParticleSystem::dumpParticles(uint start, uint count)
{
    // debug
    copyArrayFromDevice(m_hPos, m_dPos, 0, sizeof(float)*4*count);
	copyArrayFromDevice(m_hForces, m_dForces1,0, sizeof(float)*4*count);
	copyArrayFromDevice(m_hMoments, m_dMomentsA, 0, sizeof(float)*4*count);
	for(uint i=start; i<start+count; i++) {
        printf("Position: (%.7g, %.7g, %.7g, %.7g)\n", m_hPos[i*4+0], m_hPos[i*4+1], m_hPos[i*4+2], m_hPos[i*4+3]);
		printf("  Forces: (%.7g, %.7g, %.7g, %.7g)\n", m_hForces[i*4+0], m_hForces[i*4+1], m_hForces[i*4+2], m_hForces[i*4+3]);
		printf("  Moments: (%.7g, %.7g, %.7g, %.7g)\n", m_hMoments[i*4+0], m_hMoments[i*4+1], m_hMoments[i*4+2], m_hMoments[i*4+3]);
    }
	printf("Force cut = %g\n", sqrtf(newp.max_fdr_sq));
}

void ParticleSystem::logStuff(FILE* file, float simtime)
{
 	
	if(m_randSet != 0)  //dont log if we're setting ICs
		return;
	
	uint edges, graphs;
    getGraphData(graphs,edges);
	float3 M = magnetization((float4*) m_dMomentsA, newp.N, newp.L.x*newp.L.y*newp.L.z);

	//cuda calls for faster computation 
	float tf = calcTopForce( (float4*) m_dForces1, (float4*) m_dPos, newp.N, -newp.origin.y, newp.pin_d);
	float bf = calcBotForce( (float4*) m_dForces1, (float4*) m_dPos, newp.N, -newp.origin.y, newp.pin_d);
	float gs = calcGlForce(  (float4*) m_dForces1, (float4*) m_dPos, newp.N)*newp.Linv.x*newp.Linv.y*newp.Linv.z;
	float kinen = calcKinEn( (float4*) m_dForces1, (float4*) m_dPos, newp);
	
	fprintf(file, "%.5g\t%.5g\t%.5g\t%.5g\t%d\t%.5g\t%.5g\t%.5g\t%.5g\t%.5g\t%.5g\t%.5g\n", simtime, newp.shear, 
			newp.extH.y, (float)newp.N/graphs, edges, tf, bf, gs, kinen, M.x, M.y, M.z);
	
}

void
ParticleSystem::logParticles(FILE* file)
{
    // debug
    copyArrayFromDevice(m_hPos, m_dPos, 0, sizeof(float)*4*newp.N);
	copyArrayFromDevice(m_hForces, m_dForces1,0, sizeof(float)*4*newp.N);
	copyArrayFromDevice(m_hMoments, m_dMomentsA, 0, sizeof(float)*4*newp.N);
    for(uint i=0; i<newp.N; i++) {
        fprintf(file, "%.6g\t%.6g\t%.6g\t%.6g\t", m_hPos[i*4+0], m_hPos[i*4+1], m_hPos[i*4+2], m_hPos[i*4+3]);
		fprintf(file, "%.6g\t%.6g\t%.6g\t%.6g\t", m_hForces[i*4+0], m_hForces[i*4+1], m_hForces[i*4+2], m_hForces[i*4+3]);
		fprintf(file, "%.6g\t%.6g\t%.6g\t%.6g\n", m_hMoments[i*4+0], m_hMoments[i*4+1], m_hMoments[i*4+2], m_hMoments[i*4+3]);
    }
	fprintf(file, "-1\t-1\t-1\t-1\t-1\t-1\t-1\t-1\t\n");
}

void
ParticleSystem::logParams(FILE* file)
{
	#ifndef DATE
	#define DATE "No date"  
	#endif 
	#ifndef SVN_REV
	#define SVN_REV "no svn verion number"
	#endif
	fprintf(file, "Build Date: %s\t svn version: %s\n", DATE, SVN_REV);
	float vfrtot = m_params.volfr[0]+m_params.volfr[1]+m_params.volfr[2];
	fprintf(file, "vfrtot: %.3f\t v0: %.3f\t v1: %.3f\t v2: %.3f\n",vfrtot,	m_params.volfr[0], 
			m_params.volfr[1], m_params.volfr[2]);
	fprintf(file, "ntotal: %d\t n0: %d  \t n1: %d  \t n2: %d\n", newp.N, m_params.nump[0],
			m_params.nump[1], m_params.nump[2]);
	fprintf(file, "\t\t mu_p0: %.1f \t mu_p1: %.1f \t mu_p2: %.1f \n", m_params.mu_p[0], 
			m_params.mu_p[1], m_params.mu_p[2]);
	fprintf(file, "\t\t a0: %.2g\t a1: %.2g\t a2: %.2g\n", m_params.pRadius[0], 
			m_params.pRadius[1],m_params.pRadius[2]);
	fprintf(file, "\t\t std0: %2.g\t std1: %.25g\t std2: %.2g\n", m_params.rstd[0], m_params.rstd[1], m_params.rstd[2]);
	fprintf(file, "grid: %d x %d x %d = %d cells\n", newp.gridSize.x, newp.gridSize.y, 
			newp.gridSize.z, newp.numCells);
	fprintf(file, "worldsize: %.4gmm x %.4gmm x %.4gmm\n", newp.L.x*1e3f, 
			newp.L.y*1e3f, newp.L.z*1e3f);
	fprintf(file, "spring: %.2f visc: %.3f  ", m_params.spring, m_params.viscosity);
	fprintf(file, "Pin_d: %.4f Contact_d: %.4f\n", newp.pin_d, m_contact_dist);
	fprintf(file, "H.x: %.3g\tH.y: %.3g\tH.z: %.3g\n", newp.extH.x, newp.extH.y, newp.extH.z);

}


void ParticleSystem::zeroDevice()
{
	cudaMemset(m_dForces1, 0, 4*newp.N*sizeof(float));
	cudaMemset(m_dForces2, 0, 4*newp.N*sizeof(float));
	cudaMemset(m_dSortedPos, 0, 4*newp.N*sizeof(float));
	cudaMemset(m_dMidPos, 0, 4*newp.N*sizeof(float));
	cudaMemset(m_dPos, 0, 4*newp.N*sizeof(float));
	cudaMemset(m_dNumNeigh, 0, newp.N*sizeof(uint));
	cudaMemset(m_dNeighList, 0, newp.N*m_maxNeigh*sizeof(uint));
	cudaMemset(m_dGridParticleHash, 0, newp.N*sizeof(uint));
	cudaMemset(m_dGridParticleIndex, 0, newp.N*sizeof(uint));
}

inline float frand()
{
    return rand() / (float) RAND_MAX;
}

void
ParticleSystem::initGrid(uint3 size, float3 spacing, float3 jitter, uint numParticles)
{
    uint i = 0;
	for(uint z=0; z<size.z-0; z++) {
        for(uint y=0; y<size.y-0; y++) {
            for(uint x=0; x<size.x-0; x++) {
                if (i < numParticles) {
                    m_hPos[i*4+0] = spacing.x*(x+0.5f) + newp.origin.x + 
						(frand()*2.0f-1.0f)*jitter.x;
                    m_hPos[i*4+1] = spacing.y*(y+0.5f) + newp.origin.y + 
						(frand()*2.0f-1.0f)*jitter.y;
                    m_hPos[i*4+2] = spacing.z*(z+0.5f) + newp.origin.z + 
						(frand()*2.0f-1.0f)*jitter.z;
					m_hPos[i*4+3] = m_params.pRadius[0];
                	float mu_p = m_params.mu_p[0];
					float cpol = 4*PI_F*(mu_p - MU_C)/(mu_p+2.0f*MU_C)*m_params.pRadius[0]
						*m_params.pRadius[0]*m_params.pRadius[0];


					m_hMoments[4*i+0] = cpol*newp.extH.x;
					m_hMoments[4*i+1] = cpol*newp.extH.y;
					m_hMoments[4*i+2] = cpol*newp.extH.z;
					m_hMoments[4*i+3] = cpol;
				}
				i++;
            }
        }
    }
	if(numParticles >= 2){
		m_hPos[0*4+0] = (newp.origin.x + m_params.pRadius[0]);
		m_hPos[0*4+1] = m_params.pRadius[0];
		m_hPos[0*4+2] = 0; m_hPos[0*4+3] = m_params.pRadius[0];
		m_hPos[1*4+0] = -(newp.origin.x + 3*m_params.pRadius[0]);
		m_hPos[1*4+1] = -m_params.pRadius[0];
		m_hPos[1*4+2] = 0; m_hPos[1*4+3] = m_params.pRadius[0];
	}
}

void
ParticleSystem::reset(ParticleConfig config, uint numiter)
{
	zeroDevice();
	m_randSet = numiter;
	switch(config)
	{
	default:
	case CONFIG_RANDOM:
		{
			int ti = 0; 			
			for(int j=0; j < 3; j++) {
				float maxrad = 0, minrad = 1e8;
				int i; double radius,u,v,mu_p,cpol,norm, vol;
				double vtot = 0; 

				for(i = 0; i < (int) m_params.nump[j]; i++){
					if(m_params.rstd[j] > 0) {
						u=frand(); v=frand();
						norm = sqrt(-2.0*log(u))*cos(2.0*PI_F*v);
						float med_diam = m_params.pRadius[0]*
								expf(-0.5f*m_params.rstd[0]*m_params.rstd[0]);
						radius = exp(norm*m_params.rstd[j])*med_diam;	
					} else {
						radius = m_params.pRadius[j];
					}
					maxrad = radius > maxrad ? radius : maxrad;
					minrad = radius < minrad ? radius : minrad;

					mu_p = m_params.mu_p[j];
					vol = 4.0f/3.0f*PI_F*radius*radius*radius;
					cpol = 3.0f*(mu_p - MU_C)/(mu_p+2.0f*MU_C)*vol;
					vtot += vol;

					m_hPos[4*(i+ti)+0] = 2.0f*newp.origin.x * (frand() - 0.5f);
					m_hPos[4*(i+ti)+1] = 2.0f*(newp.origin.y+radius) * (frand() - 0.5f);
					m_hPos[4*(i+ti)+2] = 2.0f*newp.origin.z * (frand() - 0.5f);
					m_hPos[4*(i+ti)+3] = radius;
					m_hMoments[4*(i+ti)+0] = cpol*newp.extH.x;
					m_hMoments[4*(i+ti)+1] = cpol*newp.extH.y;
					m_hMoments[4*(i+ti)+2] = cpol*newp.extH.z;
					m_hMoments[4*(i+ti)+3] = cpol;
					
				}
				ti+=i;
				printf("minrad: %g maxrad: %g\n", minrad/m_params.pRadius[j], 
						maxrad/m_params.pRadius[j]);
				printf("actual vfr = %g\n", vtot*newp.Linv.x*newp.Linv.y*newp.Linv.z);

			}
				}// move randSetIter as a parameter
		
		break;

    case CONFIG_GRID:
        {
            //uint s;
			uint3 gridSize;
			float spc;		
			if(newp.L.z == 0){
				spc = sqrt(newp.L.x*newp.L.y/newp.N);
				gridSize.x=ceil(newp.L.x/spc);
				gridSize.y=ceil(newp.L.y/spc);
				gridSize.z=1;
			} else {
				spc = pow(newp.L.x*newp.L.y*newp.L.z/newp.N, 1.0f/3.0f);
				gridSize.x=ceil(newp.L.x/spc);
				gridSize.y=ceil(newp.L.y/spc);
				gridSize.z=ceil(newp.L.z/spc);
			}
			float3 spacing = newp.L/make_float3(gridSize.x,gridSize.y,gridSize.z);
			float3 jitter = 1.2*(spacing - 2*m_params.pRadius[0])/2;
			printf("gs %d %d %d\n", gridSize.x, gridSize.y, gridSize.z);
			printf("spacing: %.4g %.4g %.4g, particle radius: %g\n", spacing.x, 
					spacing.y, spacing.z, m_params.pRadius[0]);
			printf("jitter: %.4g %.4g %.4g\n", jitter.x,jitter.y,jitter.z);
			initGrid(gridSize, spacing, jitter, newp.N);
        }
        break;


	}
//	printf("gs: %d x%d x%d\n", newp.gridSize.x, newp.gridSize.y, newp.gridSize.z);
//	printf("alloced: %d", m_numGridCells*newp.numAdjCells);
	//place holder, allowing us to put in the hilbert ordered hashes
	for(uint i=0; i < m_numGridCells; i++){
		m_hCellHash[i] = i;
	}
/*  // if we can use the hilbert encoding for the grid apply it
	//this is commented out because it doesn't do a goddamn thing :(
	if( (newp.gridSize.x != 0) && !(newp.gridSize.x & (newp.gridSize.x-1))){
		printf("Using SFCSort.\n");
		getSortedOrder3D( m_hCellHash, newp.gridSize);
	}*/
	//generate the cell adjacency lists
	for(uint i=0; i < newp.gridSize.x; i++){
		for(uint j=0; j < newp.gridSize.y; j++){
			for(uint k=0; k < newp.gridSize.z; k++){
				uint idc = i + j*newp.gridSize.x + k*newp.gridSize.y*newp.gridSize.x;
				uint hash = m_hCellHash[idc];
				uint cn = 0;
				for(int kk=-1; kk<=1; kk++){
					for(int jj=-1; jj<=1; jj++){
						for(int ii=-1; ii<=1;ii++){
							int ai = ii + i;
							int aj = jj + j;
							int ak = kk + k;
							
							ai -= newp.gridSize.x*floor((double)ai/(double)newp.gridSize.x);
							aj -= newp.gridSize.y*floor((double)aj/(double)newp.gridSize.y);
							ak -= newp.gridSize.z*floor((double)ak/(double)newp.gridSize.z);

							uint cellId = ai + aj*newp.gridSize.x + ak*newp.gridSize.y*newp.gridSize.x;
							//store cellAdj with the first neighbor for each contiguous
							//m_hCellAdj[hash + cn*m_numGridCells] = m_hCellHash[cellId];
							//store cellAdj with all neighbors for a cell contiguous
							m_hCellAdj[hash*newp.numAdjCells + cn] = m_hCellHash[cellId];
							cn++;
							//printf("hi %d %d %d %d\n", cn, ii,jj,kk);
						}
					}
				}
				//this sort is super performance enhancing!
				std::sort(&m_hCellAdj[hash*newp.numAdjCells], &m_hCellAdj[hash*newp.numAdjCells+cn]);
			//	printf("idx: %d gl: %d %d %d\n", idx, i,j,k);
			}
		}
	}
	for(uint i=0; i < m_numGridCells; i++){
		if(m_hCellHash[i] >= m_numGridCells)
			printf("cell_hash entry %d has invaled entry %d\n", i, m_hCellHash[i]);
	}
	copyArrayToDevice(m_dCellAdj, m_hCellAdj,0, newp.numAdjCells*m_numGridCells*sizeof(uint));
	copyArrayToDevice(m_dCellHash, m_hCellHash, 0, m_numGridCells*sizeof(uint));
	copyArrayToDevice(m_dMomentsA, m_hMoments, 0, 4*newp.N*sizeof(float));
	copyArrayToDevice(m_dMomentsB, m_hMoments, 0, 4*newp.N*sizeof(float));
	copyArrayToDevice(m_dPos, m_hPos, 0, 4*newp.N*sizeof(float));

}

