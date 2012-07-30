
extern "C" {

void setNParameters(NewParams* hostParams);

void comp_phash(float* dpos, uint* d_pHash, uint* d_pIndex, uint* d_cell_hash, uint numParticles, uint numGridCells);

void find_cellStart(uint* cellStart, uint* cellEnd, uint* phash, uint numParticles,uint numCells);

void reorder(uint* sortedIndex, float* sortedPos, float* sortedMom, float* oldPos, float* oldMom, uint numParticles);

uint NListFixed(uint*& nlist, uint* num_neigh, float* dpos, 
		uint* phash, uint* cellStart, uint* cellEnd, 
		uint* cellAdj, uint numParticles, 
		uint& max_neigh, float max_dist);


uint NListVar(uint*& nlist, uint* num_neigh, float* dpos, 
		uint* phash, uint* cellStart, uint* cellEnd, 
		uint* cellAdj, uint numParticles, 
		uint& max_neigh, float max_dist);

void magForces(	float* dSortedPos, float* dIntPos, float* newPos, float* dForce, float* dMom, uint* nlist, uint* num_neigh, uint numParticles, float deltaTime);

void collision_new(	const float* dSortedPos, const float* dOldVel, 
					const uint* nlist, const uint* num_neigh, float* dNewVel, float* dNewPos, 
					uint numParticles, float deltaTime);
void mutualMagn(const float* pos, const float* oldMag, float* newMag, const uint* nlist, const uint* numNeigh, uint numParticles);

void RK4integrate(float* oldPos,
				float* newPos,
				float* force1,
				float* force2,
				float* force3,
				float* force4,
				float deltaTime,
				uint numParticles);

void integrateRK4Proper(
							const float* oldPos,
							float* PosA,
							const float* PosB,
							const float* PosC,
							const float* PosD,
							float* forceA,
							const float* forceB,
							const float* forceC,
							const float* forceD,
							const float deltaTime,
							const uint numParticles);
}
