class NListDistCond {
public:

	__host__ __device__ bool operator()(const float rad1, const float rad2, 
			const float3 er, const float distsq) {
		float sepdist = rad1 + rad2;
		return distsq < 4.0*4.0*sepdist*sepdist;
	}
};

class VarCond: public NListDistCond{
	VarCond(float v) : max_distsq(v) {}

	__host__ __device__ bool operator()(const float rad1, const float rad2,
			const float3, const float distsq) const {
		float sepdist = rad1 + rad2;
		return distsq < max_distsq*sepdist*sepdist;
	}

	const float max_distsq;
};

class VertCond: public NListDistCond{
public:
	VertCond(float d, float c) : max_distsq(d), max_costh(c) {}
   
	__host__ __device__ bool operator()(const float rad1, const float rad2, 
			const float3 er, const float distsq) const
	{
		float sepdist = rad1 + rad2;
		return (distsq < max_distsq*sepdist*sepdist) && er.y > (max_costh);
	}
	const float max_distsq;	
	const float max_costh;
};	


extern "C" {


uint funcNList(	uint*& nlist,
				uint* num_neigh, 
				const float* dpos, 
				const uint* phash, 
				const uint* cellStart, 
				const uint* cellEnd, 
				const uint* cellAdj, 
				const uint numParticles, 
				uint& max_neigh, 
				NListDistCond op);


}



