#include "new_kern.h"

extern "C" {

void setNParameters(NewParams* hostParams);

void comp_phash(float* dpos, uint* d_pHash, uint* d_pIndex, uint* d_cell_hash, 
		uint numParticles, uint numGridCells);

void find_cellStart(uint* cellStart, uint* cellEnd, uint* phash, uint numParticles,
		uint numCells);

void reorder(uint* sortedIndex, float* sortedA, float* sortedB,
		float* oldA, float* oldB, uint numParticles);

uint vertEdge(uint* connections, const uint* nlist, const uint* num_neigh, const float* dPos, 
		float maxth, float maxdist, uint numParticles);

void magForces(const float* dSortedPos, const float* dIntPos, float* newPos, 
		float* dForce, const float* dMom, const uint* nlist, const uint* num_neigh, 
		uint numParticles, float deltaTime);

void finiteDip(const float* dSortedPos, const float* dIntPos, float* newPos, float* dForce,
		const uint* nlist, const uint* num_neigh, uint numParticles,
		float dipole_d, float F0, float sigma_0, float deltaTime);

void pointDip(const float* dSortedPos, const float* dIntPos, float* newPos, float* dForce,
		const uint* nlist, const uint* num_neigh, uint numParticles,
		float forceFactor, float deltaTime);

void magFricForces(const float* dSortedPos, const float* dIntPos, float* newPos, 
		float* dForceOut, const float* dMom, const float* dForceIn,const uint* nlist,
		const uint* num_neigh, uint numParticles, float static_fric, float deltaTime);

void collision_new(	const float* dSortedPos, const float* dOldVel, 
		const uint* nlist, const uint* num_neigh, float* dNewVel, 
		float* dNewPos, uint numParticles, float raxExp, float deltaTime);

void mutualMagn(const float* pos, const float* oldMag, float* newMag, 
		const uint* nlist, const uint* numNeigh, uint numParticles);

void integrateRK4(	const float* oldPos,
					float* PosA,
					const float* PosB,
					const float* PosC,
					const float* PosD,
					float* forceA,
					const float* forceB,
					const float* forceC,
					const float* forceD,
					float deltaTime,
					uint numParticles);

void bogacki_ynp1(	const float* d_yn, const float* d_k1, const float* d_k2,
					const float* d_k3, float* d_ynp1, float deltaTime, uint numParticles);

}
