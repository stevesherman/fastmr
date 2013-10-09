
extern "C" {

void setNParameters(NewParams* hostParams);

void comp_phash(float* dpos, uint* d_pHash, uint* d_pIndex, uint* d_cell_hash, 
		uint numParticles, uint numGridCells);

void find_cellStart(uint* cellStart, uint* cellEnd, uint* phash, uint numParticles,
		uint numCells);

void reorder(uint* sortedIndex, float* sortedPos, float* sortedMom, float* oldPos, 
		float* oldMom, uint numParticles);

uint vertEdge(uint* connections, const uint* nlist, const uint* num_neigh, const float* dPos, 
		float maxth, float maxdist, uint numParticles);

void magForces(const float* dSortedPos, const float* dIntPos, float* newPos, 
		float* dForce, const float* dMom, const uint* nlist, const uint* num_neigh, 
		uint numParticles, float deltaTime);

void magFricForces(const float* dSortedPos, const float* dIntPos, float* newPos, 
		float* dForceOut, const float* dMom, const float* dForceIn,
		const uint* nlist, const uint* num_neigh, uint numParticles, float deltaTime);

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
}
