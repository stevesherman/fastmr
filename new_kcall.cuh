
extern "C" {

void setNParameters(newParams* hostParams);

void comp_phash(float* dpos, uint* d_pHash, uint* d_pIndex, uint* d_cell_hash, uint numParticles, uint numGridCells);

void find_cellStart(uint* cellStart, uint* cellEnd, uint* phash, uint numParticles,uint numCells);

void reorder(uint* sortedIndex, float* sortedPos, float* sortedMom, float* oldPos, float* oldMom, uint numParticles);

int buildNList(uint* nlist, uint* num_neigh, float* dpos, uint* phash, uint* cellStart, uint* cellEnd, uint* cellAdj, uint numParticles);

void magForces(	float* dSortedPos, float* dIntPos, float* newPos, float* dForce, float* dMom, uint* nlist, uint* num_neigh, uint numParticles, float deltaTime);
}


