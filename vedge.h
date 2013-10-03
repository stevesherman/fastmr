extern "C" {
uint vertNListVar(uint*& nlist, uint* num_neigh, float* dpos, float* dmom, 
		uint* phash, uint* cellStart, uint* cellEnd, uint* cellAdj, 
		uint numParticles, uint& max_neigh, float max_dist, float maxcosth);
}
