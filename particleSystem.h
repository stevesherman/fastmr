
#ifndef __PARTICLESYSTEM_H__
#define __PARTICLESYSTEM_H__

#define DEBUG_GRID 0
#define DO_TIMING 0

#include "particles_kernel.h"
#include "new_kern.h"
#include "vector_functions.h"
#include <cstdio>
#include <cstdlib>


#include "nlist.h"

enum RenderMode {FORCE, GLOBAL, VERTICAL, HORIZONTAL};

// Particle system class
class ParticleSystem
{
public:
    ~ParticleSystem();
	
	ParticleSystem(SimParams params, bool useGL, float3 worldSize,
			float fdist, float slk);

    float update(float deltaTime, float limdxpct);
    void render(RenderMode mode);

    void resetParticles(uint numiter, float scale_start);

    int    getNumParticles() const { return m_numParticles; }

    unsigned int getCurrentReadBuffer() const { return m_posVbo; }
    unsigned int getColorBuffer()       const { return m_colorVBO; }

    void * getCudaPosVBO()              const { return (void *)m_cudaPosVBO; }
    void * getCudaColorVBO()            const { return (void *)m_cudaColorVBO; }

    void dumpGrid();
    void dumpParticles(uint start, uint count);
	void logParticles(FILE* file);
	int loadParticles(FILE* file);
	void logParams(FILE* file);
	void logStuff(FILE* file, float simtime);
	void getBadP();
	void getMagnetization();
	void printStress();
	void NListStats();
	void zeroDevice();
   
	void setCollideSpring(float x) { m_params.cspring = x;}
	void setCollideDamping(float x) { m_params.cdamping = x; }
	void setGlobalDamping(float x) { m_params.globalDamping = x; }
	
	void setRepelSpring(float x) { m_params.spring = x; newp.spring = x;}
    void setShear(float x) { m_params.shear = x; newp.shear = x;}
	void setViscosity(float x) { m_params.viscosity = x; newp.visc = x;}
	void setColorFmax(float x) { m_params.colorFmax=x; m_colorFmax = x;}
	void setClipPlane(float x) { clipPlane = x;}
	void setDipIt(uint x) {m_params.mutDipIter = x;}
    void setInteractionRadius(uint x) {m_params.interactionr = x;}
	void setExternalH(float3 x) { m_params.externalH = x; newp.extH = x;}
	void setPinDist(float x) { newp.pin_d = x;}
	void setContactDist(float x) {m_contact_dist = x;}
	void setRebuildDist(float x) {rebuildDist = x;}
	void setForceDist(float x) {force_dist = x;}
	void dangerousResize(double  y);
	void densDist(FILE* output, double dx);
	float3 getWorldSize() { return newp.L;}
	float getParticleRadius() { return m_params.pRadius[0]; }
    uint3 getGridSize() { return m_params.gridSize; }
    float3 getWorldOrigin() { return m_params.worldOrigin; }
    float3 getCellSize() { return m_params.cellSize; }
	
	void getGraphData(uint& graphs, uint& edges, uint& vedge, uint& vgraph, 
			uint& hedge, uint& horzgraph);
	uint getInteractions();
	int getInteractionRadius() { return m_params.interactionr;}

protected: // methods
    uint createVBO(uint size);
	void _initialize();
	void initGrid();
	void _finalize();
	void sort_and_reorder();
	void initParticleGrid(uint3 size, float3 spacing, float3 jitter, uint numParticles);

	template<class T> void graph_render(T cond, float* dRendPos, float* dRendColor);

protected: // data
	uint it_since_sort;
	bool m_bUseOpenGL;
	bool m_2d;
	uint m_numParticles;
	uint m_maxNeigh;
	int m_randSet;
	float rand_scale;
    float dx_since;
	float rebuildDist;	//the distance in units of rad[0] traveled before we resolve
	// params
    SimParams m_params;
    NewParams newp;
	uint3 m_gridSize;
    float3 m_worldSize;
	uint m_numGridCells;
	int cdist;
	float m_colorFmax;
	float clipPlane;
	float m_contact_dist;
	float force_dist;

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
    float* m_dPos1;
    float* m_dPos2;
	float* m_dPos3;
	float* m_dPos4;
	float* m_dVel;
	float* m_dMoments;
	float* m_dTemp;
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
