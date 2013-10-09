#include <helper_math.h>
#ifndef NLIST_H
#define NLIST_H


class NListDistCond {
public:

	__host__ __device__ virtual bool operator()(const float rad1, const float rad2, 
			const float3 er, const float distsq) const {
		float sepdist = rad1 + rad2;
		//return false;
		return distsq < 4.0*4.0*sepdist*sepdist;
	}
};

class VarCond: public NListDistCond{
public:
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
		return (distsq < max_distsq*sepdist*sepdist) && (fabs(er.y) >= max_costh);
	}
	const float max_distsq;	
	const float max_costh;
};	

class OutOfPlane: public NListDistCond{
public:
	OutOfPlane(float d, float v, float h) : max_distsq(d), vert(v), horiz(h) {}

	__host__ __device__ bool operator()(const float rad1, const float rad2,
			const float3 er, const float distsq) const 
	{
		float sepdist = rad1 + rad2;
		return (distsq < max_distsq*sepdist*sepdist) 
			&& fabs(er.y) < vert && fabs(er.z) >= horiz;
	}

	const float max_distsq;
	const float vert;
	const float horiz;
};

class MomVar {
public:
	MomVar(float d, float n) : max_distsq(d), nm_dist(n) {}
	__host__ __device__ bool operator()(const float rad1, const float rad2,
			const float3 er, const float distsq, const float Cp1, const float Cp2) const
	{
		float sepdist = rad1 + rad2;
		float dcon = max_distsq;
		if (Cp1 == 0 || Cp2 == 0) 
			dcon = nm_dist;
		return (sepdist < dcon*sepdist*sepdist);
	}
	const float max_distsq;
	const float nm_dist;
};

class MomCut {
public:
	MomCut(float max_dist, float big_pct, float n) : bigrad(max_dist*big_pct), 
			lilrad(max_dist*(1.0f - big_pct)), nm_dist(n) {}
	
	__host__ __device__ bool operator()(const float rad1, const float rad2,
			const float3 er, const float distsq, const float Cp1, const float Cp2) const
	{
		float sepdist;
		if(rad1 > rad2){
			sepdist = bigrad*rad1 + lilrad*rad2;
		} else {
			sepdist = bigrad*rad2 + lilrad*rad1;
		}
		
		if(Cp1 == 0.0f || Cp2 == 0.0f)
			sepdist = nm_dist*(rad1 + rad2);

		return (distsq  < sepdist*sepdist);

	}

	const float bigrad;
	const float lilrad;
	const float nm_dist;
};

template <class O>
uint funcNList(	uint*& nlist,
				uint* num_neigh, 
				const float* dpos, 
				const uint* phash, 
				const uint* cellStart, 
				const uint* cellEnd, 
				const uint* cellAdj, 
				const uint numParticles, 
				uint& max_neigh, 
				O op);

template<class O>
uint momNList(	uint*& nlist, //reference to the nlist pointer
				uint* num_neigh, 
				const float* dpos, 
				const float* dmom,
				const uint* phash, 
				const uint* cellStart, 
				const uint* cellEnd, 
				const uint* cellAdj, 
				const uint numParticles, 
				uint& max_neigh, 
				O op);

#endif
