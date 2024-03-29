#include "particleSystem.h"
#include "utilities.h"
#include "particles_kernel.h"
#include "connectedgraphs.h"
#include "new_kern.h"
#include "new_kcall.h"
#include "nlist.h"
#include "sfc_pack.h"

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_math.h>
#include <assert.h>
#include <math.h>
#include <memory.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <GL/glew.h>

#define MU_0 4e-7f*M_PI
#ifndef MU_C
#define MU_C 1
#endif

ParticleSystem::ParticleSystem(SimParams params, bool useGL, float3 worldSize,
		float fdist, float slk)
{
	m_params = params;
	m_numGridCells = m_params.gridSize.x*m_params.gridSize.y*m_params.gridSize.z;	
	
	m_params.uf = MU_C*MU_0;
	m_params.Cpol = 4.0f*M_PI*pow(1.5*m_params.pRadius[0],3)*
		(m_params.mu_p[0] - MU_C)/(m_params.mu_p[0]+2.0f*MU_C);	
	m_params.globalDamping = 0.8f; 
    m_params.cdamping = 0.03f;
	m_params.boundaryDamping = -0.03f;

	m_numParticles = m_params.numBodies;
	m_maxNeigh = (uint) ((m_params.volfr[0]+m_params.volfr[1]+m_params.volfr[2])*680.0f);

	m_bUseOpenGL = useGL;

	newp.length_scale = 2.0f*m_params.pRadius[0];

	newp.N = m_numParticles;
	newp.gridSize = m_params.gridSize;
	newp.numCells = m_numGridCells;
	newp.cellSize = m_params.cellSize;
	newp.origin = m_params.worldOrigin;
	newp.L = worldSize;
	newp.Linv = 1/newp.L;
	force_dist = fdist; //in units of length_scale
	float slack = slk; // slack factor allows for interactions of larger particles
	cdist = ceil(slack*force_dist*newp.length_scale/newp.cellSize.x);
	newp.numAdjCells = pow(2*cdist + 1,3);

	newp.max_fdr_sq = force_dist*newp.length_scale*force_dist*newp.length_scale;
	newp.forcedist_sq = force_dist*force_dist;
	newp.spring = m_params.spring;
	//newp.spring = 1.0f/(.02f*2.0f*m_params.pRadius[0]);
	//newp.uf = m_params.uf;
	newp.shear = m_params.shear;
	newp.visc = m_params.viscosity;
	newp.extH = m_params.externalH;
	newp.Cpol = m_params.Cpol;
	newp.pin_d = 0.995f;  //ybot < radius*pin_d
	newp.tanfric = 1e-5f;
	m_contact_dist = 1.05f;	
	m_cos_vert = sqrtf(3.0/5.0);
	m_tan_horz = 1.0;
	_initialize();
	zeroDevice();
	initGrid();
	rand_scale = 1.0f;
	dx_since = 1e6f;
	rebuildDist = 0.01;
	it_since_sort = 0;
	clipPlane = -1.0;
	last_dt = 0;
}

void pswap(float*& a, float*& b) {
	float* temp = b;
	b = a;
	a = temp;
}


ParticleSystem::~ParticleSystem()
{
	_finalize();
    newp.N = 0;
}

uint ParticleSystem::createVBO(uint size)
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
	m_randSet = 0;
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
       	checkCudaErrors( cudaMalloc( (void **)&m_cudaPosVBO, memSize )) ;
    }

    cudaMalloc((void**)&m_dMoments, memSize);
	cudaMalloc((void**)&m_dTemp, memSize);
	cudaMalloc((void**)&m_dForces1, memSize);
	cudaMalloc((void**)&m_dForces2, memSize);
	cudaMalloc((void**)&m_dForces3, memSize);
	cudaMalloc((void**)&m_dForces4, memSize);

    cudaMalloc((void**)&m_dSortedPos, memSize);
	cudaMalloc((void**)&m_dPos1, memSize);
	cudaMalloc((void**)&m_dPos2, memSize);
	cudaMalloc((void**)&m_dPos3, memSize);
	cudaMalloc((void**)&m_dPos4, memSize);

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
        checkCudaErrors( cudaMalloc( (void **)&m_cudaColorVBO, sizeof(float)*newp.N*4) );
    }
 	
	assert(cudaMalloc((void**)&m_dNeighList, newp.N*m_maxNeigh*sizeof(uint)) == cudaSuccess);
		
    setParameters(&m_params);
}

void
ParticleSystem::_finalize()
{
	
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

    cudaFree(m_dMoments);
	cudaFree(m_dTemp);
	cudaFree(m_dForces1);
    cudaFree(m_dForces2);
	cudaFree(m_dForces3);
	cudaFree(m_dForces4);
	
	cudaFree(m_dSortedPos);
	cudaFree(m_dPos1);
	cudaFree(m_dPos2);
	cudaFree(m_dPos3);
	cudaFree(m_dPos4);

	cudaFree(m_dGridParticleHash);
    cudaFree(m_dGridParticleIndex);
    cudaFree(m_dCellStart);
    cudaFree(m_dCellEnd);
    
	if (m_bUseOpenGL) {
        unregisterGLBufferObject(m_cuda_posvbo_resource);
        glDeleteBuffers(1, (const GLuint*)&m_posVbo);
        glDeleteBuffers(1, (const GLuint*)&m_colorVBO);
    } else {
        checkCudaErrors( cudaFree(m_cudaPosVBO) );
        checkCudaErrors( cudaFree(m_cudaColorVBO) );
    }

}

void ParticleSystem::render(RenderMode mode)
{
	float *dRendPos, *dRendColor;
	if(!m_bUseOpenGL)
		return;

	dRendPos = (float *) mapGLBufferObject(&m_cuda_posvbo_resource);
	dRendColor = (float *) mapGLBufferObject(&m_cuda_colorvbo_resource);

	if(mode == FORCE){
		//express colors as fraction of reference force, F0
		float ref_moment = M_PI/6.0f*pow(newp.length_scale,3)*
				3.0f*(m_params.mu_p[0] - MU_C)/(m_params.mu_p[0]+2.0f*MU_C)*length(newp.extH);
		float F0 = 3*MU_0*ref_moment*ref_moment/ (4*M_PI*powf(newp.length_scale,4));
		renderStuff(m_dPos1, m_dMoments, m_dForces1, dRendPos, dRendColor, m_colorFmax*F0,
				rand_scale, clipPlane,newp.N);
	}
	if(mode == GLOBAL){
		VarCond test = VarCond(m_contact_dist);
		graph_render(test, dRendPos, dRendColor);
	}
	if(mode == VERTICAL){
		VertCond test = VertCond(m_contact_dist, m_cos_vert);
		graph_render(test, dRendPos, dRendColor);
	}
	if(mode == HORIZONTAL){
		float outdist = 1.25f*(m_contact_dist - 1.0f) + 1.0f;
		//less than cos(39.2) off the vertical, and w/in tan(45 deg) of the outofplane direction
		OutOfPlane test = OutOfPlane(outdist,m_cos_vert, m_tan_horz);
		graph_render(test, dRendPos, dRendColor);
	}

	unmapGLBufferObject(m_cuda_posvbo_resource);
	unmapGLBufferObject(m_cuda_colorvbo_resource);


}

// step the simulation, limdxpct is that max distance before iteration is re-solved
float ParticleSystem::update(float dt_ref, float limdxpct)
{
	float deltaTime = dt_ref;
	setParameters(&m_params);
	setNParameters(&newp);
	bool rebuildNList = false;
	//printf("dx_since: %g\t cut: %g\n", dx_since, rebuildDist*0.5f*newp.length_scale);
	if(dx_since > rebuildDist*0.5f*newp.length_scale)
	{
		rebuildNList = true;
		sort_and_reorder();
	}else{
		pswap(m_dSortedPos, m_dPos1);//make sortespos have yn
		pswap(m_dPos1, m_dPos4); //inside pos1 has yn+1/2k1

	}

	if(m_randSet > 0)
	{
		const float rstep = .001f;
		rand_scale = rand_scale+rstep >= 1.0f ? 1.0f : rand_scale + rstep;
		rand_scale = rand_scale > .85f  && rand_scale < 1.0f ? rand_scale - 0.05f*rstep : rand_scale;
		//printf("rand_scale: %f\n", rand_scale);
		if(rebuildNList) {
			VarCond op(rand_scale*1.1f);
			funcNList(m_dNeighList, m_dNumNeigh, m_dSortedPos, m_dGridParticleHash, m_dCellStart, 
					m_dCellEnd, m_dCellAdj, newp.N, m_maxNeigh, op);
	
			dx_since = 0.0f;
		}
		dx_since += 0.04f*0.5f*newp.length_scale;
		//sorted pos and forces 1 are the old positions
		collision_new(m_dSortedPos, m_dForces1, m_dNeighList, m_dNumNeigh, m_dForces2, 
				m_dPos2, newp.N, rand_scale*1.01f, 1e-3f);
		collision_new(m_dPos2, m_dForces2, m_dNeighList, m_dNumNeigh, m_dForces3, 
				m_dPos1, newp.N, rand_scale*1.01f, 1e-3f);
		
		pswap(m_dForces1, m_dForces3);
		deltaTime = 0;
		m_randSet--;
		if(m_randSet == 0) {dx_since = 999; rand_scale = 1.0f;}
	} else {

		if (rebuildNList) {
			//force_dist is in radii, the cond operators are in diams
			VarCond asdf(force_dist + rebuildDist);
			funcNList(m_dNeighList, m_dNumNeigh, m_dSortedPos, m_dGridParticleHash,
					m_dCellStart, m_dCellEnd, m_dCellAdj, newp.N, m_maxNeigh, asdf);
			dx_since = 0.0f;
		}

		//shiny new bogacki 23
		bool solve = false;
		float dx_moved = 0.0f;
		// for radius based
		float fcoeff = 4.0/3.0*M_PI*MU_0*(3.0*newp.extH.y)*(3.0*newp.extH.y);
		do {
			if(deltaTime != last_dt || it_since_sort == 0){
				pointDip(	m_dSortedPos,	//yin: yn
						m_dSortedPos,	//yn
						m_dPos1,   	//yn + 1/2*k1
						m_dForces1,   	//k1
						m_dNeighList, m_dNumNeigh, newp.N, fcoeff,0.5f*deltaTime);
			}
			pointDip(	m_dPos1, 		//yin: yn + 1/2*k1
					m_dSortedPos, 	//yn
					m_dPos2, 		//yn + 3/4*k2
					m_dForces2,		//k2
					m_dNeighList, m_dNumNeigh, newp.N, fcoeff,0.75*deltaTime);

			pointDip(	m_dPos2, 		//yin: yn + 1/2*k2
					m_dSortedPos, 	//yn
					m_dPos3, 		//yn + k3
					m_dForces3,		//k3
					m_dNeighList, m_dNumNeigh, newp.N, fcoeff,deltaTime);

			// compute y_np1, applies periodic BCs
			bogacki_ynp1(	m_dSortedPos, m_dPos1, m_dPos2, m_dPos3, m_dPos1,
					deltaTime, newp.N);
			//compute f(y_np1) is k4/k1 at next step
			//use dt=dt_ref under assumption that this is terminating iteration
			pointDip(	m_dPos1, 		//yin: ynp1
					m_dPos1, 	//yn
					m_dPos4, 		//yn + 1/2*k1
					m_dForces4,		//k1
					m_dNeighList, m_dNumNeigh, newp.N, fcoeff,0.5*dt_ref);

			// compute error in magnetic forces?
			float err = bogacki_error((float4*)m_dForces1, (float4*)m_dForces2, (float4*)m_dForces3,
					(float4*) m_dForces4, newp.N, 3*M_PI*newp.visc*newp.length_scale,deltaTime);
			pswap(m_dForces4, m_dForces1);
			//	printf("error=%.3g\n", err/m_params.pRadius[0]);

			if(err > 0.001f*newp.length_scale) {
				solve = true;
			} else {
				solve = isOutofBounds((float4*)m_dPos1, newp.L/2.0, newp.N);
			}

			if(solve){
				deltaTime *=.5f;
				printf("error est %.3g\treducing timestep %.3g ", err/newp.length_scale, deltaTime);
				getBadP();
			}
			if(deltaTime <= 1e-30f && deltaTime != 0) {
				printf("timestep fail!, ws=%fmm\n", newp.L.y*1e3);
				getBadP();
				getMagnetization();
				NListStats();
				dumpParticles(0, newp.N);
				assert(false);
			}

		} while(solve);
		dx_moved = maxvel((float4*)m_dForces1,(float4*)m_dPos1,newp)*deltaTime;

		//1.25 is a slack factor in the particle diffusivity
		dx_since += dx_moved + newp.shear*(force_dist*1.25f)*newp.length_scale*deltaTime;
		last_dt = dt_ref;
	}
	it_since_sort++;
	//printf("dx_since= %.3g, it_since_sort=%d\n", dx_since, it_since_sort);
//	printf("dt=%.4g\n", deltaTime);
	return deltaTime;
}



void ParticleSystem::dangerousResize(double  y) {
	float Lold = newp.L.y;
	float deltay = Lold - y;
	pshift((float4*) m_dPos1, make_float3(0, deltay/2.0, 0) , newp.N);
	newp.L.y = y; 
	newp.Linv.y = 1.0/y;
	newp.origin.y = -newp.L.y/2.0;
	newp.cellSize.y = newp.L.y/newp.gridSize.y;
	if( newp.cellSize.y < force_dist*newp.length_scale) {
		newp.cellSize.y = force_dist*newp.length_scale;
		printf("Cells shrunk excessively\n");
	}

}


// takes contents of Pos1 and sorts it into m_dSortedPos, and m_dMoments into m_dTempt
void ParticleSystem::sort_and_reorder() {
		
	// sort particles based on hash
	comp_phash(m_dPos1, m_dGridParticleHash, m_dGridParticleIndex, m_dCellHash, newp.N, m_numGridCells);
	// reorder particle arrays into sorted order and
	sortParticles(m_dGridParticleHash, m_dGridParticleIndex, newp.N);
	// find start and end of each cell - also sets the fixed moment
	find_cellStart(m_dCellStart, m_dCellEnd, m_dGridParticleHash, newp.N, m_numGridCells);
	//reorder
	reorder(m_dGridParticleIndex, m_dSortedPos, m_dTemp, m_dPos1, m_dMoments, newp.N);
	pswap(m_dMoments, m_dTemp);
	//reorder and swap k4 data - for Bogacki only
	reorder(m_dGridParticleIndex, m_dPos1, m_dTemp, m_dPos4, m_dForces1, newp.N);
	pswap(m_dForces1, m_dTemp);
	//printf("it since sort: %d\n", it_since_sort);	
	it_since_sort = 0;
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
	copyArrayFromDevice(m_hPos, m_dPos1, 0, sizeof(float)*4*newp.N);
	for(int i = 0; i < (int) newp.N; i++){
		if( isnanf(m_hPos[4*i]) || isnanf(m_hPos[4*i+1]) || isnanf(m_hPos[4*i+2]) ){
			nans++;
		}
		if( pow(m_hPos[4*i],2) > pow(newp.origin.x,2) ||  pow(m_hPos[4*i+1],2) > pow(newp.origin.y,2) || 
				pow(m_hPos[4*i+2],2) > pow(newp.origin.z,2) )
		   outbounds++;
	}
	printf("nans: %d \toutofbounds: %d\n", nans, outbounds);

}

void ParticleSystem::getMagnetization()
{
	float3 M = magnetization((float4*) m_dMoments, newp.N, 
			newp.L.x*newp.L.y*newp.L.z);
	printf("M: %g %g %g\n", M.x, M.y, M.z );
}


void ParticleSystem::getGraphData(uint& graphs, uint& edges, uint& vert_edges, 
		uint& vert_graph, uint& horz_edges, uint& horz_graph)
{
	
	VarCond contact_op = VarCond(m_contact_dist);
	//no reordering, assume m_contact_dist << m_interaction_dist, so it should always be fine
	uint maxn=funcNList(m_dNeighList, m_dNumNeigh, m_dSortedPos, m_dGridParticleHash, 
			m_dCellStart, m_dCellEnd, m_dCellAdj, newp.N, m_maxNeigh, contact_op);
	edges = numInteractions(m_dNumNeigh, newp.N)/2;
	m_hNeighList = new uint[newp.N*maxn];
	copyArrayFromDevice(m_hNeighList, m_dNeighList, 0, sizeof(uint)*newp.N*maxn);
	copyArrayFromDevice(m_hNumNeigh,  m_dNumNeigh,  0, sizeof(uint)*newp.N);
	graphs = adjConGraphs(m_hNeighList, m_hNumNeigh, newp.N);
	delete [] m_hNeighList;

	VertCond test = VertCond(m_contact_dist, m_cos_vert);
	maxn = funcNList(m_dNeighList, m_dNumNeigh, m_dSortedPos, m_dGridParticleHash, 
			m_dCellStart, m_dCellEnd, m_dCellAdj, newp.N, m_maxNeigh, test);
	vert_edges = numInteractions(m_dNumNeigh, newp.N)/2;
	m_hNeighList = new uint[newp.N*maxn];
	copyArrayFromDevice(m_hNeighList, m_dNeighList, 0, sizeof(uint)*newp.N*maxn);
	copyArrayFromDevice(m_hNumNeigh,  m_dNumNeigh,  0, sizeof(uint)*newp.N);
	vert_graph = adjConGraphs(m_hNeighList, m_hNumNeigh, newp.N);
	delete [] m_hNeighList;

	float outdist = 1.25f*(m_contact_dist - 1.0f) + 1.0f;
	OutOfPlane horz_op = OutOfPlane(outdist, m_cos_vert, m_tan_horz);
	maxn = funcNList(m_dNeighList, m_dNumNeigh, m_dSortedPos, m_dGridParticleHash, 
			m_dCellStart, m_dCellEnd, m_dCellAdj, newp.N, m_maxNeigh, horz_op);
	horz_edges = numInteractions(m_dNumNeigh, newp.N)/2;
	m_hNeighList = new uint[newp.N*maxn];
	copyArrayFromDevice(m_hNeighList, m_dNeighList, 0, sizeof(uint)*newp.N*maxn);
	copyArrayFromDevice(m_hNumNeigh,  m_dNumNeigh,  0, sizeof(uint)*newp.N);
	horz_graph = adjConGraphs(m_hNeighList, m_hNumNeigh, newp.N);
	delete [] m_hNeighList;

	dx_since = 1e6f;//trips nlist regeneration
}

template <class T>
void ParticleSystem::graph_render(T cond, float* dRendPos, float* dRendColor)
{
	uint maxn = funcNList(m_dNeighList, m_dNumNeigh, m_dSortedPos, m_dGridParticleHash,
			m_dCellStart, m_dCellEnd, m_dCellAdj, newp.N, m_maxNeigh, cond);
	m_hNeighList = new uint[newp.N*maxn];
	copyArrayFromDevice(m_hNeighList, m_dNeighList, 0, sizeof(uint)*newp.N*maxn);
	copyArrayFromDevice(m_hNumNeigh,  m_dNumNeigh,  0, sizeof(uint)*newp.N);

	//use m_hMoments as temp space because idgaf
	graphColorLabel(m_hNeighList, m_hNumNeigh, newp.N, m_hMoments);

	cudaMemcpy(dRendPos, m_dPos1, sizeof(float)*4*newp.N, cudaMemcpyDeviceToDevice);
	cudaMemcpy(dRendColor, m_hMoments, sizeof(float)*4*newp.N, cudaMemcpyHostToDevice);
	delete [] m_hNeighList;

//	float4 cut = make_float4(1,1,newp.origin.z*clipPlane,1);
	float4 cut = make_float4(newp.origin.x*clipPlane,1,1,1);
	renderCutKern((float4*) dRendPos, cut,newp.N);
	dx_since = 999; //if this is called, nlist is trashed, so rebuild the nlist
}

uint ParticleSystem::getInteractions(){
	//assumes that a vanilla nlist was the last called, may fail weirdly
	return numInteractions(m_dNumNeigh, newp.N);
}

void
ParticleSystem::dumpParticles(uint start, uint count)
{
    // debug
    copyArrayFromDevice(m_hPos, m_dPos1, 0, sizeof(float)*4*count);
	copyArrayFromDevice(m_hForces, m_dForces1,0, sizeof(float)*4*count);
	copyArrayFromDevice(m_hMoments, m_dMoments, 0, sizeof(float)*4*count);
	uint n_outside = 0;
	for(uint i=start; i<start+count; i++) {
		bool islarge = sqrt(m_hForces[i*4]*m_hForces[i*4] + m_hForces[i*4+1]*m_hForces[i*4+1] 
		        + m_hForces[i*4+2]*m_hForces[i*4+2]) > 1e-6f;
		bool posnan = isnanf(m_hPos[i*4+0]) || isnanf(m_hPos[i*4+1]) || isnanf(m_hPos[i*4+2]);
		if (islarge || posnan){
			printf("Position: (%.7g, %.7g, %.7g, %.7g)\n", m_hPos[i*4+0], 
					m_hPos[i*4+1], m_hPos[i*4+2], m_hPos[i*4+3]);
			printf("  Forces: (%.7g, %.7g, %.7g, %.7g)\n", m_hForces[i*4+0], 
					m_hForces[i*4+1], m_hForces[i*4+2], m_hForces[i*4+3]);
			printf("  Moments: (%.7g, %.7g, %.7g, %.7g)\n", m_hMoments[i*4+0], 
					m_hMoments[i*4+1], m_hMoments[i*4+2], m_hMoments[i*4+3]);
		}
		if(fabs(m_hPos[i*4+1])+m_hPos[i*4+3] > newp.L.y/2.0f) n_outside++;
	}
	printf("Force cut = %g\n", sqrtf(newp.max_fdr_sq));
	printf("n_outside = %d\n", n_outside);
	getBadP();
}

void ParticleSystem::densDist(FILE* output, double dx)
{
	copyArrayFromDevice(m_hPos, m_dPos1, 0, sizeof(float)*4*newp.N);

	int np = ceil(newp.L.z/dx);
	dx = newp.L.z/np;

	double* dens = new double[np];
	if(dens == 0)
		fprintf(stderr,"Couldn't allocate 20kb wtf");
	for(int ii=0; ii<np; ii++){
		dens[ii] = 0;
	}
	//this will not work work if min(diam)/dx < 3
	for(int ii=0; ii<newp.N; ii++) {
		float zp = m_hPos[4*ii+2] - newp.origin.z;
		float rad = m_hPos[4*ii+3];
		//use mod to fix rounding errors making it neg or > np
		int cell = ((int)floor(zp/dx) + np) % np;
		float offset = zp - cell*dx;
		int numleft = ceil((rad - offset)/dx);

		// do the left cap
		float h = (cell - numleft + 1)*dx + rad - zp;
		float cap_old = 1.0/3.0*M_PI*h*h*(3.0*rad - h);
		float cap_curr = 0;
		uint idx = (cell-numleft + np) % np;
		dens[idx] += cap_old;

		for(int rc=-numleft+1; rc < 0; rc++){
			h = rad + (cell+rc+1)*dx - zp;
			cap_curr = 1.0f/3.0f*M_PI*h*h*(3*rad - h);
			idx = (cell + rc + np) % np;
			dens[idx] += cap_curr - cap_old;
			cap_old = cap_curr;
		}

		//do the middle
		h = zp+rad-(cell+1)*dx;
		float right_cap = 1.0/3.0*M_PI*h*h*(3.0*rad - h);
		//use cap_old for left cap
		dens[cell] += 4.0/3.0*M_PI*rad*rad*rad - right_cap - cap_old;
		cap_old = right_cap;

		int numright = ceil((rad - (dx-offset))/dx);
		for(int rc=1; rc<numright;rc++) {
			h = zp+rad - (cell+rc+1)*dx;
			cap_curr = 1.0/3.0*M_PI*h*h*(3.0*rad - h);
			idx = (cell+rc+np) % np;
			dens[idx] += cap_old - cap_curr;
			cap_old = cap_curr;
		}
		idx = (cell + numright + np) % np;
		dens[idx] += cap_old;
	}

	if(output == stdout){
		double npstar = m_params.volfr[0]*newp.L.x*newp.L.y*dx/20.0;
		for(uint ii=0; ii<np; ii++){
			uint nstar = floor(dens[ii]/npstar);
			fprintf(output,"z=%4.3f:  ", (float)ii/np);
			for(; nstar > 0; nstar--){
				fprintf(output,"*");
			}
			fprintf(output,"\n");
		}
	} else {
		double vol = newp.L.x*newp.L.y*newp.L.z;
		for(uint ii=0; ii<np; ii++){
			fprintf(output,"%.6g\t", dens[ii]/vol);
		}
		fprintf(output,"\n");
	}
	delete [] dens;
}

void ParticleSystem::logStuff(FILE* file, float simtime)
{
 	
	if(m_randSet != 0){  //dont log if we're setting ICs
		return;
	}
	
	uint edges=0, graphs=0, vedge=0, vgraphs=0, hedge=0,hgraphs;
    getGraphData(graphs,edges,vedge,vgraphs,hedge,hgraphs);
	float3 M = magnetization((float4*) m_dMoments, newp.N, newp.L.x*newp.L.y*newp.L.z);

	//cuda calls for faster computation 
	float tf = calcTopForce( (float4*) m_dForces1, (float4*) m_dPos1, newp.N, 
			-newp.origin.y, newp.pin_d);
	float bf = calcBotForce( (float4*) m_dForces1, (float4*) m_dPos1, newp.N, 
			-newp.origin.y, newp.pin_d);
	float gs = calcGlForce(  (float4*) m_dForces1, (float4*) m_dPos1, newp.N,
			-newp.origin.y, 0.0f)*newp.Linv.x*newp.Linv.y*newp.Linv.z;
	float kinen = calcKinEn( (float4*) m_dForces1, (float4*) m_dPos1, newp);

	fprintf(file, "%.7g\t%.6g\t%.6g\t%.6g\t%d\t%.6g\t%.6g\t%.6g\t%.6g\t%.6g\t%.6g\t%.6g\t%d\t%d\t%d\t%d\n", 
			simtime, newp.shear, newp.extH.y, (float)newp.N/graphs, edges, tf, bf, 
			gs, kinen, M.x, M.y, M.z,vedge,vgraphs,hedge,hgraphs);
}

void ParticleSystem::printStress()
{
	printf("pin_d: %.3f\t", newp.pin_d);
	float tf = calcTopForce( (float4*) m_dForces1, (float4*) m_dPos1, newp.N, 
			-newp.origin.y, newp.pin_d);
	float bf = calcBotForce( (float4*) m_dForces1, (float4*) m_dPos1, newp.N, 
			-newp.origin.y, newp.pin_d);
	float gs = calcGlForce(  (float4*) m_dForces1, (float4*) m_dPos1, newp.N, 
			-newp.origin.y, 0.0f)*newp.Linv.x*newp.Linv.y*newp.Linv.z;//so all particles get counted

	printf("stress top: %g\tbot: %g\tmom old: %g\n", tf*newp.Linv.x*newp.Linv.z, 
			bf*newp.Linv.x*newp.Linv.z, gs);
}

void ParticleSystem::NListStats()
{
	//sort_and_reorder();//make sure the particles in good positions for computing graph data
	//uint maxn = NListCut(m_dNeighList, m_dNumNeigh, m_dSortedPos, m_dMoments, m_dGridParticleHash, 
	//		m_dCellStart, m_dCellEnd, m_dCellAdj, newp.N, m_maxNeigh, 8.0f, 0.3f);
	uint maxn = 0;
	uint nInter = numInteractions(m_dNumNeigh, newp.N);
	printf("total interactions: %d\t mean interactions: %f\n", nInter, (float)nInter/newp.N);
		dx_since = 1e6f;//set this to a really large number so that the nlist is regenerated
	
	copyArrayFromDevice(m_hNumNeigh,  m_dNumNeigh,  0, sizeof(uint)*newp.N);
	copyArrayFromDevice(m_hPos, m_dSortedPos, 0, sizeof(float)*4*newp.N);
	uint overruns = 0;
	uint n_neigh;	
	for(uint ii=0; ii<newp.N; ii++){
		n_neigh = m_hNumNeigh[ii];
		if(n_neigh > m_maxNeigh)
			overruns++;
		maxn = n_neigh > maxn ? n_neigh : maxn;
	}
	printf("Number of overruns: %d\n",overruns);
	
	//print histogram with mean radius in each bin
	uint step = 20;
	uint np_star = 0.01*newp.N;
	uint nbins = ceil((double)m_maxNeigh/step);
	uint* nump = new uint[nbins];
	double* meanrad = new double[nbins];
	memset(nump,0,nbins*sizeof(uint));
	memset(meanrad,0,nbins*sizeof(double));

	for(uint ii = 0; ii < newp.N; ii++){
		uint n_neigh = m_hNumNeigh[ii];
		meanrad[n_neigh/step] += m_hPos[4*ii+3];
		nump[n_neigh/step]++;
	}

	for(uint ii = 0; ii < nbins; ii++){
		uint bin = ii*step;
		meanrad[ii] = (nump[ii] == 0) ? 0 : meanrad[ii]/nump[ii];
		if(nump != 0) {
			printf("bin %d-%d\t meanrad: %.4g\t nump: %d\t", bin, bin+step-1, meanrad[ii],nump[ii]);
			for(uint jj = 0; jj < round((float)nump[ii]/np_star); jj++)
				printf("*");
			printf("\n");
		}
	}

	//
	m_hNeighList = new uint[newp.N*m_maxNeigh];
	copyArrayFromDevice(m_hNeighList, m_dNeighList, 0, sizeof(uint)*newp.N*m_maxNeigh);
	int n_repeats = 0, n_rezeros = 0;
	for(uint ii = 0; ii < newp.N; ii++) {
		n_neigh = m_hNumNeigh[ii];
		//if(ii < 50) printf("n_neigh: %d\n", n_neigh);
		uint numzeros = 0;
		uint repeats = 0;
		for(uint jj = 0; jj <  n_neigh; jj++) {
			uint neigh = m_hNeighList[ii + newp.N*jj];
			numzeros = (neigh == 0) ? numzeros+1 : numzeros;
			if(neigh == ii){
			//	printf("Warning: self interaction on particle %d at %d/%d entry\n", ii, jj,n_neigh-1);
			}
			for(uint kk=jj+1; kk < n_neigh; kk++) {
				if(neigh == m_hNeighList[ii + newp.N*kk])
					repeats++;
			}
		}
		if(numzeros > 1){
			//printf("Excessive (%d/%d) interactions with particle 0 by particle %d\n", numzeros,n_neigh,ii);
			n_rezeros++;
		}
		if(repeats > 0){
			//printf("Particle %d has %d repeats\n", ii,repeats);
			n_repeats++;
		}
	}
	
	if(n_repeats > 0 || n_rezeros > 0)
		fprintf(stdout,"Nlist failure, %d p have repeats and %d have repeats w/ zero\n",n_repeats, n_rezeros);

	//graphs = adjConGraphs(m_hNeighList, m_hNumNeigh, newp.N);
	delete [] m_hNeighList;
	delete [] nump;
	delete [] meanrad;
	printf("Max number of neighbors currently: %d, allocated %d\n\n", maxn, m_maxNeigh);

}

void ParticleSystem::logParticles(FILE* file)
{
    // debug
    copyArrayFromDevice(m_hPos, m_dPos1, 0, sizeof(float)*4*newp.N);
	copyArrayFromDevice(m_hForces, m_dForces1,0, sizeof(float)*4*newp.N);
	copyArrayFromDevice(m_hMoments, m_dMoments, 0, sizeof(float)*4*newp.N);
    for(uint i=0; i<newp.N; i++) {
		fprintf(file, "%.9e\t%.9e\t%.9e\t%.9e\t", m_hPos[i*4+0], m_hPos[i*4+1], m_hPos[i*4+2], m_hPos[i*4+3]);
		fprintf(file, "%.9e\t%.9e\t%.9e\t%.9e\t", m_hForces[i*4+0], m_hForces[i*4+1], m_hForces[i*4+2], m_hForces[i*4+3]);
		fprintf(file, "%.9e\t%.9e\t%.9e\t%.9e\n", m_hMoments[i*4+0], m_hMoments[i*4+1], m_hMoments[i*4+2], m_hMoments[i*4+3]);
	}
	fprintf(file, "-1\t-1\t-1\t-1\t-1\t-1\t-1\t-1\t\n");
}

int ParticleSystem::loadParticles(FILE* file)
{
	zeroDevice();
	//assumes the file* is pointing to the start of the particle data
	char buff[1024]; int matches;
	for(uint i=0; i<newp.N; i++) {
		if(fgets(buff, 1024, file) != NULL){
			matches = sscanf(buff, "%f %f %f %f\t %f %f %f %f\t %f %f %f %f", &m_hPos[i*4+0], &m_hPos[i*4+1], &m_hPos[i*4+2],
					&m_hPos[i*4+3], &m_hForces[i*4+0], &m_hForces[i*4+1], &m_hForces[i*4+2], &m_hForces[i*4+3], 
					&m_hMoments[i*4+0], &m_hMoments[i*4+1], &m_hMoments[i*4+2], &m_hMoments[i*4+3]);
		   if(matches != 12) {
			   fprintf(stderr, "Failed to read in particles, i=%d matches=%d\n", i, matches);
			   return -1;
		   }
		   if(isnanf(m_hPos[i*4+0]) || isnanf(m_hPos[i*4+1]) || isnanf(m_hPos[i*4+2]) || isnanf(m_hPos[i*4+3]))
			   printf("bad pos: %f %f %f %f\n", m_hPos[i*4], m_hPos[i*4+1], m_hPos[i*4+2], m_hPos[i*4+3]);
		} else {
			fprintf(stderr, "particle loader failed to read in a line, i=%d\n", i);
			return -1;
		}
	}
	int endcheck[8];
	int endref[8] = {-1, -1, -1, -1, -1, -1, -1, -1 };  
	if(fgets(buff, 1024, file) != NULL){
		matches = sscanf(buff, "%d %d %d %d\t %d %d %d %d ", endcheck, endcheck+1, endcheck+2, endcheck+3,
				endcheck+4, endcheck+5,endcheck+6, endcheck+7);
		if(matches != 8)
			fprintf(stderr, "not enough entries at end, matches = %d\n", matches);
		for(int i = 0; i < 8; i++){
			if(	endcheck[i] != endref[i]){
				fprintf(stderr, "particle loader failed to find file terminator in expected location\n");
				fprintf(stderr, "%d %d %d %d %d %d %d %d\n", endcheck[0], endcheck[1], endcheck[2], endcheck[3],
						endcheck[4], endcheck[5], endcheck[6], endcheck[7]);
				return -1;
			}
		}
	} else {
		fprintf(stderr, "particle loader file too short, failed to find terminator\n");
		return -1;
	}

	copyArrayToDevice(m_dPos1, m_hPos, 0, 4*newp.N*sizeof(float));
	copyArrayToDevice(m_dMoments, m_hMoments, 0, 4*newp.N*sizeof(float));
	copyArrayToDevice(m_dForces1, m_hForces, 0, 4*newp.N*sizeof(float));	
	return 0;
}


void ParticleSystem::logParams(FILE* file)
{
	#ifndef VERSION_NUMBER
	#define VERSION_NUMBER "revno not found"
	#endif
	fprintf(file, "Build Date: %s %s\t version: %s\n", __DATE__, __TIME__,VERSION_NUMBER);
	float vfrtot = m_params.volfr[0]+m_params.volfr[1]+m_params.volfr[2];
	fprintf(file, "vfrtot: %.4f\t v0: %.4f\t v1: %.3f\t v2: %.4f\n",vfrtot,	m_params.volfr[0], 
			m_params.volfr[1], m_params.volfr[2]);
	fprintf(file, "ntotal: %d\t n0: %d\t n1: %d  \t n2: %d\n", newp.N, m_params.nump[0],
			m_params.nump[1], m_params.nump[2]);
	fprintf(file, "\t\t mu0: %.1f \t mu1: %.1f \t mu2: %.1f \n", m_params.mu_p[0], 
			m_params.mu_p[1], m_params.mu_p[2]);
	fprintf(file, "\t\t a0: %.3g\t a1: %.3g\t a2: %.3g\n", m_params.pRadius[0], 
			m_params.pRadius[1],m_params.pRadius[2]);
	fprintf(file, "\t\t std0: %.3g\t std1: %.3g\t std2: %.3g\n", m_params.rstd[0], m_params.rstd[1], m_params.rstd[2]);
	fprintf(file, "grid: %d x %d x %d = %d cells\n", newp.gridSize.x, newp.gridSize.y, 
			newp.gridSize.z, newp.numCells);
	fprintf(file, "worldsize: %.4gmm x %.4gmm x %.4gmm\n", newp.L.x*1e3f, 
			newp.L.y*1e3f, newp.L.z*1e3f);
	fprintf(file, "spring: %.2f visc: %.5f ", m_params.spring, m_params.viscosity);
	fprintf(file, "Pin_d: %.3f Contact_d: %.3f\n", newp.pin_d, m_contact_dist);
	fprintf(file, "rebuildDist: %.4g fdist %.2f cdist %u\n", rebuildDist, force_dist, cdist);
	fprintf(file, "H.x: %.3g\tH.y: %.6g\tH.z: %.3g\n", newp.extH.x, newp.extH.y, newp.extH.z);

}


void ParticleSystem::zeroDevice()
{
	cudaMemset(m_dForces1, 0, 4*newp.N*sizeof(float));
	cudaMemset(m_dForces2, 0, 4*newp.N*sizeof(float));
	cudaMemset(m_dSortedPos, 0, 4*newp.N*sizeof(float));
	cudaMemset(m_dPos4, 0, 4*newp.N*sizeof(float));
	cudaMemset(m_dPos3, 0, 4*newp.N*sizeof(float));
	cudaMemset(m_dPos2, 0, 4*newp.N*sizeof(float));
	cudaMemset(m_dPos1, 0, 4*newp.N*sizeof(float));
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
ParticleSystem::initParticleGrid(uint3 size, float3 spacing, float3 jitter, uint numParticles)
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
					float cpol = 4*M_PI*(mu_p - MU_C)/(mu_p+2.0f*MU_C)*m_params.pRadius[0]
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
		m_hPos[1*4+0] = -(newp.origin.x + 2*m_params.pRadius[0]);
		m_hPos[1*4+1] = -1.5f*m_params.pRadius[0];
		m_hPos[1*4+2] = 0; m_hPos[1*4+3] = m_params.pRadius[0];
	}
}

void
ParticleSystem::resetParticles(uint numiter, float scale_start)
{
	zeroDevice();
	dx_since = 1e6f;
	it_since_sort = 0;
	m_randSet = numiter;//how many iterations
	rand_scale = scale_start;//how large we start them at
	
	int ti = 0;	
	for(int j=0; j < 3; j++) {
		float maxrad = 0, minrad = 1e8;
		int i; double radius,u,v,mu_p,cpol,norm, vol;
		double vtot = 0; 

		for(i = 0; i < (int) m_params.nump[j]; i++){
			if(m_params.rstd[j] > 0) {
				u=frand(); v=frand();
				norm = sqrt(-2.0*log(u))*cos(2.0*M_PI*v);
				float med_diam = m_params.pRadius[j];
				radius = exp(norm*m_params.rstd[j])*med_diam;	
			} else {
				radius = m_params.pRadius[j];
			}
			maxrad = radius > maxrad ? radius : maxrad;
			minrad = radius < minrad ? radius : minrad;

			mu_p = m_params.mu_p[j];
			vol = 4.0f/3.0f*M_PI*radius*radius*radius;
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
		float vfr_actual = vtot*newp.Linv.x*newp.Linv.y*newp.Linv.z;
		m_params.volfr[j] = vfr_actual;
		printf("minrad: %g maxrad: %g\n", minrad/m_params.pRadius[j], 
				maxrad/m_params.pRadius[j]);
		printf("actual vfr = %g\n", vfr_actual);

	}
	copyArrayToDevice(m_dMoments, m_hMoments, 0, 4*newp.N*sizeof(float));
	copyArrayToDevice(m_dPos1, m_hPos, 0, 4*newp.N*sizeof(float));

}



void ParticleSystem::initGrid()
{	

    
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
				//const int cdist = 2;
				for(int kk=-cdist; kk<=cdist; kk++){
					for(int jj=-cdist; jj<=cdist; jj++){
						for(int ii=-cdist; ii<=cdist;ii++){
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

}



