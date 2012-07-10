
#ifndef __PARTICLESYSTEM_H__
#define __PARTICLESYSTEM_H__

#define DEBUG_GRID 0
#define DO_TIMING 0

#include "particles_kernel.cuh"
#include "new_kern.cuh"
#include "vector_functions.h"
#include <cstdio>
#include <cstdlib>

// Particle system class
class ParticleSystem
{
public:
    ~ParticleSystem();
	
	ParticleSystem(SimParams params, bool useGL, float3 worldSize);

    enum ParticleConfig
    {
	    CONFIG_RANDOM,
	    CONFIG_GRID,
	    _NUM_CONFIGS
    };

    enum ParticleArray
    {
        POSITION,
        VELOCITY,
    };
    float update(float deltaTime, float maxdxpct);
    void reset(ParticleConfig config);

    int    getNumParticles() const { return m_numParticles; }

    unsigned int getCurrentReadBuffer() const { return m_posVbo; }
    unsigned int getColorBuffer()       const { return m_colorVBO; }

    void * getCudaPosVBO()              const { return (void *)m_cudaPosVBO; }
    void * getCudaColorVBO()            const { return (void *)m_cudaColorVBO; }

    void dumpGrid();
    void dumpParticles(uint start, uint count);
	void logParticles(FILE* file);
	void logParams(FILE* file);
	void logStuff(FILE* file, float simtime);
	void getBadP();
	void getMagnetization();

	void zeroDevice();
   
	void setCollideSpring(float x) { m_params.cspring = x;}
	void setCollideDamping(float x) { m_params.cdamping = x; }
	void setGlobalDamping(float x) { m_params.globalDamping = x; }
	
	void setRepelSpring(float x) { m_params.spring = x; newp.spring = x;}
    void setShear(float x) { m_params.shear = x; newp.shear = x;}
	void setViscosity(float x) { m_params.viscosity = x; newp.visc = x;}
	void setColorFmax(float x) { m_params.colorFmax=x; m_colorFmax = x;}

	void setDipIt(uint x) {m_params.mutDipIter = x;}
    void setInteractionRadius(uint x) {m_params.interactionr = x;}
	void setExternalH(float3 x) { m_params.externalH = x; newp.extH = x;}

	float getParticleRadius() { return m_params.particleRadius[0]; }
    uint3 getGridSize() { return m_params.gridSize; }
    float3 getWorldOrigin() { return m_params.worldOrigin; }
    float3 getCellSize() { return m_params.cellSize; }
	
	uint getEdges();
	uint getGraphs();

	int getInteractionRadius() { return m_params.interactionr;}	


protected: // methods
    ParticleSystem() {}
    uint createVBO(uint size);

    void _initialize(int numParticles);
    void _finalize();

    void initGrid(uint3 size, float3 spacing, float3 jitter, uint numParticles);

protected: // data
    bool m_bInitialized, m_bUseOpenGL;
    bool m_2d;
	uint m_numParticles;
	uint m_maxNeigh;
	int m_randSet;

    // params
    SimParams m_params;
    NewParams newp;
	uint3 m_gridSize;
    float3 m_worldSize;
	uint m_numGridCells;
	float m_colorFmax;
    uint m_timer;


	// CPU data
    float* m_hPos;              // particle positions
	float* m_hMoments;
	float* m_hForces;

    uint* m_hParticleHash;
    uint* m_hCellStart;
    uint* m_hCellEnd;
	uint* m_hCellAdj;
	uint* m_hCellHash;
	uint* m_hNumNeigh;
	uint* m_hNeighList;
    // GPU data
    float* m_dPos;
    float* m_dMidPos;
	float* m_dVel;
	float* m_dMomentsA;
	float* m_dMomentsB;
	float* m_dForces1;
	float* m_dForces2;
	float* m_dForces3;
	float* m_dForces4;

	uint* m_dNeighList;
	uint* m_dNumNeigh;

    float* m_dSortedPos;
	float* m_dSortedVel;

    // grid data for sorting method
    uint* m_dGridParticleHash; // grid hash value for each particle
    uint* m_dGridParticleIndex;// particle index for each particle
    uint* m_dCellStart;        // index of start of each cell in sorted list
    uint* m_dCellEnd;          // index of end of cell
	uint* m_dCellHash;
	uint* m_dCellAdj;

    uint   m_gridSortBits;

    uint   m_posVbo;            // vertex buffer object for particle positions
    uint   m_colorVBO;          // vertex buffer object for colors
    
    float *m_cudaPosVBO;        // these are the CUDA deviceMem Pos
    float *m_cudaColorVBO;      // these are the CUDA deviceMem Color

    struct cudaGraphicsResource *m_cuda_posvbo_resource; // handles OpenGL-CUDA exchange
    struct cudaGraphicsResource *m_cuda_colorvbo_resource; // handles OpenGL-CUDA exchange



};

#endif // __PARTICLESYSTEM_H__
