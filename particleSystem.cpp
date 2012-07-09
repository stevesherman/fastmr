#include "particleSystem.h"
#include "particleSystem.cuh"
#include "particles_kernel.cuh"
#include "connectedgraphs.h"
#include "new_kern.cuh"
#include "new_kcall.cuh"
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

#ifndef CUDART_PI_F
#define CUDART_PI_F   3.141592653589793f 
#endif


ParticleSystem::ParticleSystem(SimParams params, bool useGL, float3 worldSize):
	m_bInitialized(false),
	m_bUseOpenGL(false)
{
	newp.L = worldSize;
	m_params = params;
	m_numGridCells = m_params.gridSize.x*m_params.gridSize.y*m_params.gridSize.z;	
	
	m_params.uf = 1.257e-6;
	m_params.mup = 4.0f*CUDART_PI_F*pow(m_params.particleRadius[0],3)*(m_params.xi[0]-1.0f)/(m_params.xi[0]+2.0f);	
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
	newp.max_fdr_sq = 8.0f*m_params.particleRadius[0]*8.0f*m_params.particleRadius[0];
	newp.numAdjCells = 27;
	newp.spring = m_params.spring;
	newp.uf = m_params.uf;
	newp.shear = m_params.shear;
	newp.visc = m_params.viscosity;
	newp.extH = m_params.externalH;
	newp.mup = m_params.mup;
	newp.contact_d_sq = 1.05f*1.05f;//lsq < contact_d_sq * sepdist*sepdist
	newp.pin_d = 1.5f;  //ybot < radius*pin_d
	
	_initialize(m_params.numBodies);

}



ParticleSystem::~ParticleSystem()
{
    _finalize();
    m_numParticles = 0;
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
ParticleSystem::_initialize(int numParticles)
{
    assert(!m_bInitialized);
    m_numParticles = numParticles;
	m_randSet = 0;
    // allocate host storage
    m_hPos = new float[m_numParticles*4];
    m_hMoments = new float[m_numParticles*4];
	m_hForces = new float[m_numParticles*4];
	
	memset(m_hPos, 0, m_numParticles*4*sizeof(float));
	memset(m_hForces, 0, m_numParticles*4*sizeof(float));
	memset(m_hMoments, 0, m_numParticles*4*sizeof(float));

   	m_hCellStart = new uint[m_numGridCells];
    memset(m_hCellStart, 0, m_numGridCells*sizeof(uint));

    m_hCellEnd = new uint[m_numGridCells];
    memset(m_hCellEnd, 0, m_numGridCells*sizeof(uint));

    // allocate GPU data
    unsigned int memSize = sizeof(float) * 4 * (m_numParticles);

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

	cudaMalloc((void**)&m_dGridParticleHash, m_numParticles*sizeof(uint));
    cudaMalloc((void**)&m_dGridParticleIndex, m_numParticles*sizeof(uint));

    assert(cudaMalloc((void**)&m_dCellStart, m_numGridCells*sizeof(uint)) == cudaSuccess);
    cudaMalloc((void**)&m_dCellEnd, m_numGridCells*sizeof(uint));


	assert(cudaMalloc((void**)&m_dCellAdj, m_numGridCells*27*sizeof(uint)) == cudaSuccess);
	m_hCellAdj = new uint[m_numGridCells*27];

	m_hCellHash = new uint[m_numGridCells];
	cudaMalloc((void**)&m_dCellHash, m_numGridCells*sizeof(uint));
	
	assert(cudaMalloc((void**)&m_dNumNeigh, m_numParticles*sizeof(uint)) == cudaSuccess);
	m_hNumNeigh = new uint[m_numParticles];
	
    if (m_bUseOpenGL) {
        m_colorVBO = createVBO(m_numParticles*4*sizeof(float));
		registerGLBufferObject(m_colorVBO, &m_cuda_colorvbo_resource);

        // fill color buffer
        glBindBufferARB(GL_ARRAY_BUFFER, m_colorVBO);
        float *data = (float *) glMapBufferARB(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
        float *ptr = data;
        for(uint i=0; i<m_numParticles; i++) {
            float t = i / (float) m_numParticles;
            colorRamp(t, ptr);
            ptr+=3;
            *ptr++ = 1.0f;
        }
        glUnmapBufferARB(GL_ARRAY_BUFFER);
    } else {
        cutilSafeCall( cudaMalloc( (void **)&m_cudaColorVBO, sizeof(float)*numParticles*4) );
    }
 	
	assert(cudaMalloc((void**)&m_dNeighList, m_numParticles*m_maxNeigh*sizeof(uint)) == cudaSuccess);
		
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
		
		renderStuff(m_dPos, m_dMomentsA, m_dForces1, dRendPos, dRendColor, m_colorFmax, m_numParticles);
		
		unmapGLBufferObject(m_cuda_posvbo_resource);
		unmapGLBufferObject(m_cuda_colorvbo_resource);

	} else {
        dRendPos = (float *) m_cudaPosVBO;
		dRendColor = (float*) m_cudaColorVBO; //shouldn't be a big deal, as color is only touched above
    }
	cutilCheckMsg("ohno");	
	setParameters(&m_params);
	setNParameters(&newp);

	//isOutofBounds((float4*) m_dPos, m_params.worldOrigin.x, m_numParticles);
	comp_phash(m_dPos, m_dGridParticleHash, m_dGridParticleIndex, m_dCellHash, m_numParticles, m_numGridCells);
	// sort particles based on hash
	sortParticles(m_dGridParticleHash, m_dGridParticleIndex, m_numParticles);
	// reorder particle arrays into sorted order and
	// find start and end of each cell - also sets the fixed moment
	find_cellStart(m_dCellStart, m_dCellEnd, m_dGridParticleHash, m_numParticles, m_numGridCells);
	cutilCheckMsg("Cstart");
	reorder(m_dGridParticleIndex, m_dSortedPos, m_dMomentsB, m_dPos, m_dMomentsA, m_numParticles);	
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
				m_dCellStart, m_dCellEnd, m_dCellAdj, m_numParticles, m_maxNeigh, 3.1f*m_params.particleRadius[0]);

		collision_new(m_dSortedPos, m_dForces2, m_dNeighList, m_dNumNeigh, m_dForces1, m_dPos, m_numParticles, 0.01f);
		deltaTime = 0;
		m_randSet--;
	} else {
		NListFixed(m_dNeighList, m_dNumNeigh, m_dSortedPos, m_dGridParticleHash, 
				m_dCellStart, m_dCellEnd, m_dCellAdj, m_numParticles, m_maxNeigh, 8.1f*m_params.particleRadius[0]);

	
		bool solve = true;
		double Cd, maxf, maxFdx;
		Cd = 6*CUDART_PI_F*m_params.viscosity*m_params.particleRadius[0];

		while(solve) {
			maxFdx = maxdxpct*Cd*m_params.particleRadius[0]/deltaTime; //force to cause a dx

			magForces(	m_dSortedPos,	//yin: yn 
						m_dSortedPos,	//yn
						m_dMidPos,   	//yn + 1/2*k1
						m_dForces1,   	//k1
						m_dMomentsA, m_dNeighList, m_dNumNeigh, m_numParticles, deltaTime/2);
			cutilCheckMsg("magForces");
			magForces(	m_dMidPos, 		//yin: yn + 1/2*k1
						m_dSortedPos, 	//yn
						m_dPos, 		//yn + 1/2*k2
						m_dForces2,		//k2
						m_dMomentsA, m_dNeighList, m_dNumNeigh, m_numParticles, deltaTime/2);
			magForces(	m_dPos, 		//yin: yn + 1/2*k2
						m_dSortedPos, 	//yn
						m_dMidPos, 		//yn + k3
						m_dForces3,		//k3
						m_dMomentsA, m_dNeighList, m_dNumNeigh, m_numParticles, deltaTime);
			magForces(	m_dMidPos, 		//yin: yn + k3
						m_dSortedPos, 	//yn
						m_dPos, 		// doesn't matter
						m_dForces4,		//k4
						m_dMomentsA, m_dNeighList, m_dNumNeigh, m_numParticles, deltaTime/2);

			RK4integrate(m_dSortedPos,//yn 
						m_dPos, //yn+1
						m_dForces1, //1/6*(k1 + 2*k2 + 2*k3 + k4) 
						m_dForces2, m_dForces3, m_dForces4, deltaTime, m_numParticles);
	
			solve = false;	
		
					
			//need some sort of controller for error
			
			//find max force
			//printf("callmax\n");
			maxf = maxforce( (float4*) m_dForces1, m_numParticles);
		
			if(maxf > maxFdx){
				solve = true;
			} else { //if not excess force, check for out of bounds
				solve = isOutofBounds((float4*)m_dPos, -m_params.worldOrigin.x, m_numParticles);
			}
			if(solve){
				deltaTime *=.5f;
				assert(deltaTime != 0);
				printf("force excess ratio %.3g\treducing timestep %.3g\n", maxf/maxFdx, deltaTime);
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
	copyArrayFromDevice(m_hPos, m_dPos, 0, sizeof(float)*4*m_numParticles);
	for(int i = 0; i < (int) m_numParticles; i++){
		if( isnan(m_hPos[4*i]) || isnan(m_hPos[4*i+1]) || isnan(m_hPos[4*i+2]) )
			nans++;
		if( pow(m_hPos[4*i],2) > pow(m_params.worldOrigin.x,2) ||  pow(m_hPos[4*i+1],2) > pow(m_params.worldOrigin.y,2) || 
				pow(m_hPos[4*i+2],2) > pow(m_params.worldOrigin.z,2) )
		   outbounds++;
	}
	printf("nans: %d \toutofbounds: %d\n", nans, outbounds);

}

void ParticleSystem::getMagnetization()
{
	float4 M = magnetization((float4*) m_dMomentsA, m_numParticles, newp.L.x*newp.L.y*newp.L.z);
	printf("M: %g %g %g\n", M.x, M.y, M.z );
}


uint ParticleSystem::getGraphs()
{
	uint maxn = NListVar(m_dNeighList, m_dNumNeigh, m_dSortedPos, m_dGridParticleHash, 
			m_dCellStart, m_dCellEnd, m_dCellAdj, m_numParticles, m_maxNeigh, 1.05f);
	printf("before copy\n");
	
	m_hNeighList = new uint[m_numParticles*maxn];
	copyArrayFromDevice(m_hNeighList, m_dNeighList, 0, sizeof(uint)*m_numParticles*maxn);
	copyArrayFromDevice(m_hNumNeigh,  m_dNumNeigh,  0, sizeof(uint)*m_numParticles);
	
	printf("after\n");
	uint ngraphs = adjConGraphs(m_hNeighList, m_hNumNeigh, m_numParticles);
	delete [] m_hNeighList;
	return ngraphs;
}

void
ParticleSystem::dumpParticles(uint start, uint count)
{
    // debug
    copyArrayFromDevice(m_hPos, m_dPos, 0, sizeof(float)*4*count);
	copyArrayFromDevice(m_hForces, m_dForces1,0, sizeof(float)*4*count);
	copyArrayFromDevice(m_hMoments, m_dMomentsA, 0, sizeof(float)*4*count);
    float numEdges = 0;
	for(uint i=start; i<start+count; i++) {
        printf("Position: (%.7g, %.7g, %.7g, %.7g)\n", m_hPos[i*4+0], m_hPos[i*4+1], m_hPos[i*4+2], m_hPos[i*4+3]);
		printf("Forces: (%.7g, %.7g, %.7g, %.7g)\n", m_hForces[i*4+0], m_hForces[i*4+1], m_hForces[i*4+2], m_hForces[i*4+3]);
		printf("Moments: (%.7g, %.7g, %.7g, %.7g)\n", m_hMoments[i*4+0], m_hMoments[i*4+1], m_hMoments[i*4+2], m_hMoments[i*4+3]);
		numEdges += m_hForces[i*4+3];
    }
	printf("IR = %d edges %f\n", m_params.interactionr, numEdges);
}

uint ParticleSystem::getEdges(){
	return edgeCount((float4*) m_dForces1, m_numParticles);

}

void ParticleSystem::logStuff(FILE* file, float simtime)
{
 	
	if(m_randSet != 0)  //dont log if we're setting ICs
		return;

	copyArrayFromDevice(m_hPos, m_dPos, 0, sizeof(float)*4*m_numParticles);
	copyArrayFromDevice(m_hForces, m_dForces1,0, sizeof(float)*4*m_numParticles);
		
	uint edges = getEdges();//i think these get us the latest force data
	float graphs = getGraphs();
	float4 M = magnetization((float4*) m_dMomentsA, m_numParticles, newp.L.x*newp.L.y*newp.L.z);

	float topcut = 	-newp.origin.y - 2*m_params.particleRadius[0];
	float topforce=0, botforce=0, speckinen=0, gstress=0;
	float Cd = 6*CUDART_PI_F*m_params.viscosity*m_params.particleRadius[0];
	
	for(uint i=0; i < m_numParticles; i++){
		if(m_hPos[i*4+1] > topcut)
			topforce += m_hForces[i*4];
		if(m_hPos[i*4+1] < -topcut)
			botforce += m_hForces[i*4];
		gstress += m_hPos[i*4+1]*m_hForces[i*4];
		speckinen += 1.0f/(2.0f*Cd*Cd)*(m_hForces[i*4]*m_hForces[i*4] + m_hForces[i*4+1]*m_hForces[i*4+1] 
				+ m_hForces[i*4+2]*m_hForces[i*4+2]);
			
	}
	gstress = gstress*newp.Linv.x*newp.Linv.y*newp.Linv.z;
	
	float tc_proper = -newp.origin.y - newp.pin_d*m_params.particleRadius[0];
	//cuda calls for faster computation - need to verify against host code
	float tf = calcTopForce( (float4*) m_dForces1, (float4*) m_dPos, m_numParticles, tc_proper);
	float bf = calcBotForce( (float4*) m_dForces1, (float4*) m_dPos, m_numParticles, -tc_proper);
	float gs = calcGlForce(  (float4*) m_dForces1, (float4*) m_dPos, m_numParticles)*newp.Linv.x*newp.Linv.y*newp.Linv.z;
	float kinen = calcKinEn( (float4*) m_dForces1, m_numParticles)/(2.0f*Cd*Cd);
	
	printf("topforce\tlocal: %g\tcuda:%g\n", topforce, tf);
	printf("botforce\tlocal: %g\t cuda:%g\n", botforce, bf);
	printf("gstress\tlocal: %g\tcuda:%g\n", gstress, gs);
	printf("Kinen local: %g cuda: %g\n", speckinen, kinen);
	
	fprintf(file, "%.5g\t%.5g\t%.5g\t%.5g\t%d\t%.5g\t%.5g\t%.5g\t%.5g\t%.5g\t%.5g\t%.5g\n", simtime, m_params.shear, 
			m_params.externalH.y, (float)m_numParticles/graphs, edges, topforce, botforce, gstress, speckinen, M.x, M.y, M.z);
	
}

void
ParticleSystem::logParticles(FILE* file)
{
    // debug
    copyArrayFromDevice(m_hPos, m_dPos, 0, sizeof(float)*4*m_numParticles);
	copyArrayFromDevice(m_hForces, m_dForces1,0, sizeof(float)*4*m_numParticles);
	copyArrayFromDevice(m_hMoments, m_dMomentsA, 0, sizeof(float)*4*m_numParticles);
    for(uint i=0; i<m_numParticles; i++) {
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
	fprintf(file, "vfrtot: %.3f\t v0: %.3f\t v1: %.3f\t v2: %.3f\n", m_params.volfr[0]+m_params.volfr[1]+m_params.volfr[2],
			m_params.volfr[0], m_params.volfr[1], m_params.volfr[2]);
	fprintf(file, "ntotal: %d\t n0: %d  \t n1: %d  \t n2: %d\n", m_params.numBodies, m_params.numParticles[0],
			m_params.numParticles[1], m_params.numParticles[2]);
	fprintf(file, "\t\t xi0: %.1f \t xi1: %.1f \t xi2: %.1f \n", m_params.xi[0], m_params.xi[1], m_params.xi[2]);
	fprintf(file, "\t\t a0: %.2g\t a1: %.2g\t a2: %.2g\n\n", m_params.particleRadius[0], m_params.particleRadius[1],
			m_params.particleRadius[2]);

	fprintf(file, "grid: %d x %d x %d = %d cells\n", m_params.gridSize.x, m_params.gridSize.y, m_params.gridSize.z, 
			m_params.gridSize.x*m_params.gridSize.y*m_params.gridSize.z);
	fprintf(file, "worldsize: %.4gmm x %.4gmm x %.4gmm\n", newp.L.x*1e3f, newp.L.y*1e3f, newp.L.z*1e3f);
	fprintf(file, "spring: %.2f\tvisc: %.4f\tdipit: %d\trcut: %f\n", m_params.spring, m_params.viscosity, m_params.mutDipIter, 4.0f);
	fprintf(file, "H.x: %.3g\tH.y: %.3g\tH.z: %.3g\n", m_params.externalH.x, m_params.externalH.y, m_params.externalH.z);

}


void ParticleSystem::zeroDevice()
{
	float xi;
	int ti = 0;
	for(int j = 0; j < 3; j++){
		int i;
		xi = m_params.xi[j];
		for ( i = 0; i < m_params.numParticles[j]; i++){
			m_hMoments[4*(i+ti)+0] = 0;
			m_hMoments[4*(i+ti)+1] = m_params.externalH.y*4.0/3.0*3.14159*
					pow(m_params.particleRadius[j],3)* 3.0*(xi-1.0)/(xi+2.0);
			m_hMoments[4*(i+ti)+2] = 0;
			m_hMoments[4*(i+ti)+3] = xi;
		}
		ti+=i;
	}

	copyArrayToDevice(m_dMomentsA, m_hMoments, 0, 4*m_numParticles*sizeof(float));
	copyArrayToDevice(m_dMomentsB, m_hMoments, 0, 4*m_numParticles*sizeof(float));
	cudaMemset(m_dForces1, 0, 4*m_numParticles*sizeof(float));
	cudaMemset(m_dForces2, 0, 4*m_numParticles*sizeof(float));
	cudaMemset(m_dSortedPos, 0, 4*m_numParticles*sizeof(float));
	cudaMemset(m_dMidPos, 0, 4*m_numParticles*sizeof(float));
	cudaMemset(m_dPos, 0, 4*m_numParticles*sizeof(float));
	cudaMemset(m_dNumNeigh, 0, m_numParticles*sizeof(uint));
	cudaMemset(m_dNeighList, 0, m_numParticles*m_maxNeigh*sizeof(uint));
	cudaMemset(m_dGridParticleHash, 0, m_numParticles*sizeof(uint));
	cudaMemset(m_dGridParticleIndex, 0, m_numParticles*sizeof(uint));
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
                    m_hPos[i*4+0] = spacing.x*(x+0.5f) + m_params.worldOrigin.x + (frand()*2.0f-1.0f)*jitter.x;
                    m_hPos[i*4+1] = spacing.y*(y+0.5f) + m_params.worldOrigin.y + (frand()*2.0f-1.0f)*jitter.y;
                    m_hPos[i*4+2] = spacing.z*(z+0.5f) + m_params.worldOrigin.z + (frand()*2.0f-1.0f)*jitter.z;
					m_hPos[i*4+3] = m_params.particleRadius[0];
                }
				i++;
            }
        }
    }
	if(numParticles >= 2){
		m_hPos[0*4+0] = (m_params.worldOrigin.x + m_params.particleRadius[0]);
		m_hPos[0*4+1] = m_params.particleRadius[0];
		m_hPos[0*4+2] = 0; m_hPos[0*4+3] = m_params.particleRadius[0];
		m_hPos[1*4+0] = -(m_params.worldOrigin.x + 3*m_params.particleRadius[0]);
		m_hPos[1*4+1] = -m_params.particleRadius[0];
		m_hPos[1*4+2] = 0; m_hPos[1*4+3] = m_params.particleRadius[0];
	}
}

void
ParticleSystem::reset(ParticleConfig config)
{
	zeroDevice();
	switch(config)
	{
	default:
	case CONFIG_RANDOM:
		{
			int ti = 0;
			for(int j=0; j < 3; j++) {
				int i;
				for(i = 0; i < (int) m_params.numParticles[j]; i++){

					float point[3];
					point[0] = frand();
					point[1] = frand();
					point[2] = frand();
					m_hPos[4*(i+ti)+0] = 2.0f*m_params.worldOrigin.x * (frand() - 0.5f);
					m_hPos[4*(i+ti)+1] = 2.0f*(m_params.worldOrigin.y+m_params.particleRadius[j]) * (frand() - 0.5f);
					m_hPos[4*(i+ti)+2] = 2.0f*m_params.worldOrigin.z * (frand() - 0.5f);
					m_hPos[4*(i+ti)+3] = m_params.particleRadius[j]; // radius
				}
				ti+=i;
			}
		}
		m_randSet = m_params.randSetIter;
		break;

    case CONFIG_GRID:
        {
            //uint s;
			uint3 gridSize;
			float spc;		
			if(newp.L.z == 0){
				spc = sqrt(newp.L.x*newp.L.y/m_numParticles);
				gridSize.x=ceil(newp.L.x/spc);
				gridSize.y=ceil(newp.L.y/spc);
				gridSize.z=1;
			} else {
				spc = pow(newp.L.x*newp.L.y*newp.L.z/m_numParticles, 1.0f/3.0f);
				gridSize.x=ceil(newp.L.x/spc);
				gridSize.y=ceil(newp.L.y/spc);
				gridSize.z=ceil(newp.L.z/spc);
			}
			float3 spacing = newp.L/make_float3(gridSize.x,gridSize.y,gridSize.z);
			float3 jitter = 1.2*(spacing - 2*m_params.particleRadius[0])/2;
			printf("gs %d %d %d\n", gridSize.x, gridSize.y, gridSize.z);
			printf("spacing: %.4g %.4g %.4g, particle radius: %g\n", spacing.x, spacing.y, spacing.z, m_params.particleRadius[0]);
			printf("jitter: %.4g %.4g %.4g\n", jitter.x,jitter.y,jitter.z);
			initGrid(gridSize, spacing, jitter, m_numParticles);
        }
        break;


	}
//	printf("gs: %d x%d x%d\n", m_params.gridSize.x, m_params.gridSize.y, m_params.gridSize.z);
//	printf("alloced: %d", m_numGridCells*27);
	//place holder, allowing us to put in the hilbert ordered hashes
	for(uint i=0; i < m_numGridCells; i++){
		m_hCellHash[i] = i;
	}
/*
	if( (m_params.gridSize.x != 0) && !(m_params.gridSize.x & (m_params.gridSize.x-1))){
		printf("Using SFCSort.\n");
		getSortedOrder3D( m_hCellHash, &m_params);
	}*/
	
	/*for(uint i = 0; i < m_numGridCells; i++){
		printf("%d,", m_hCellHash[i]);
	}*/
	printf("\n");

	for(uint i=0; i < m_params.gridSize.x; i++){
//		printf("hello\n");
		for(uint j=0; j < m_params.gridSize.y; j++){
			for(uint k=0; k < m_params.gridSize.z; k++){
				uint idc = i + j*m_params.gridSize.x + k*m_params.gridSize.y*m_params.gridSize.x;
				uint hash = m_hCellHash[idc];
				uint cn = 0;
				for(int kk=-1; kk<=1; kk++){
					for(int jj=-1; jj<=1; jj++){
						for(int ii=-1; ii<=1;ii++){
							int ai = ii + i;
							int aj = jj + j;
							int ak = kk + k;
							
							ai -= m_params.gridSize.x*floor((double)ai/(double)m_params.gridSize.x);
							aj -= m_params.gridSize.y*floor((double)aj/(double)m_params.gridSize.y);
							ak -= m_params.gridSize.z*floor((double)ak/(double)m_params.gridSize.z);

							uint cellId = ai + aj*m_params.gridSize.x + ak*m_params.gridSize.y*m_params.gridSize.x;
							//store cellAdj with the first neighbor for each contiguous
							//m_hCellAdj[hash + cn*m_numGridCells] = m_hCellHash[cellId];
							//store cellAdj with all neighbors for a cell contiguous
							m_hCellAdj[hash*27 + cn] = m_hCellHash[cellId];
							cn++;
							//printf("hi %d %d %d %d\n", cn, ii,jj,kk);
						}
					}
				}
				std::sort(&m_hCellAdj[hash*27], &m_hCellAdj[hash*27+cn]);
			//	printf("idx: %d gl: %d %d %d\n", idx, i,j,k);
			}
		}
	}
	for(uint i=0; i < m_numGridCells; i++){
		if(m_hCellHash[i] >= m_numGridCells)
			printf("cell_hash entry %d has invaled entry %d\n", i, m_hCellHash[i]);
	}
	copyArrayToDevice(m_dCellAdj, m_hCellAdj,0, 27*m_numGridCells*sizeof(uint));
	copyArrayToDevice(m_dCellHash, m_hCellHash, 0, m_numGridCells*sizeof(uint));
	float volfr;
	volfr = m_numParticles*4.0f/3.0f*CUDART_PI_F*pow(m_params.particleRadius[0],3)
		/(newp.L.x*newp.L.y*newp.L.z);
	
	copyArrayToDevice(m_dPos, m_hPos, 0, 4*m_numParticles*sizeof(float));

}

