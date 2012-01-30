typedef unsigned int uint;


class AdjPair
{
	public:
		int node;
		int edge;
};

int3 calcGridPos(float3 p, SimParams params);
uint calcGridHash(int3 gridPos, SimParams params);
void makeAdjList(float4* points, uint* cellStart, uint* cellEnd, AdjPair* adjlist, int* adjstart, 
		const SimParams &params, uint adjlistsize);

int stackConGraphs(AdjPair* adjlist, const int* adjstart, const uint numParticles, const uint adjlistisze);

