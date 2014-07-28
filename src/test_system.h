/*
 * test_system.h
 *
 *  Created on: Jul 28, 2014
 *      Author: steve
 */

#ifndef TEST_SYSTEM_H_
#define TEST_SYSTEM_H_

#include "nlist.h"
#include "new_kcall.h"
#include "new_kern.h"
#include "utilities.h"

class BasicPsystem {

public:
	NewParams hp;

	uint *dNList;
	uint nListSize;
	uint *dNumNeigh;
	float4* dpos;
	uint* phash;

	uint* cellStart;
	uint* cellEnd;
	uint* cellAdj;

	bool setDPos(float4* in_pos) {
		return cudaMemcpy(dpos, in_pos, sizeof(float4)*hp.N, cudaMemcpyHostToDevice) == cudaSuccess;
	}

	bool allocNList(uint sz) {
		nListSize = sz;
		bool flag = cudaMalloc(&dNList, hp.N*nListSize*sizeof(uint)) == cudaSuccess;
		cudaMemset(&dNList, 0, nListSize*hp.N*sizeof(uint));
		return flag;
	}

	void setDevParams() { setNParameters(&hp); }

	template <class Op>
	uint callNList(Op o) {
		return funcNList(dNList, dNumNeigh,(float*) dpos, phash,
				cellStart,cellEnd,cellAdj, hp.N, nListSize, o);
	}

	bool validateNListLength(std::vector<uint> ref) {
		std::vector<uint> hNumNeigh(hp.N);
		cudaMemcpy(hNumNeigh.data(), dNumNeigh, sizeof(uint)*hp.N, cudaMemcpyDeviceToHost);
		return ref == hNumNeigh;
	}

};


#endif /* TEST_SYSTEM_H_ */
