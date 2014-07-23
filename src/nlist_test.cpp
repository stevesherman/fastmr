#include "gtest/gtest.h"
#include "nlist.h"
#include "new_kcall.h"
#include "new_kern.h"
#include "utilities.h"
#include <cuda_runtime.h>

class BasicPsystem{

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

};

class TwoPTest : public BasicPsystem, public ::testing::TestWithParam<float> {
public:
	TwoPTest() {
		hp.N = 2;
		hp.L = make_float3(100,100,100); hp.Linv = 1/hp.L;
		hp.numAdjCells = 1;
		setDevParams();

		nListSize = 0;
		dNList = NULL;

		cudaMalloc(&dNumNeigh, hp.N*sizeof(uint));
		cudaMemset(&dNumNeigh,0, hp.N*sizeof(uint));

		cudaMalloc(&dpos, hp.N*sizeof(float4));
		cudaMalloc(&phash, hp.N*sizeof(uint));
		cudaMemset(phash,0, hp.N*sizeof(uint));

		// Everything is in one cell, so NOT a test of cell placement
		cudaMalloc( &cellStart, sizeof(int));
		cudaMalloc( &cellEnd, sizeof(int));
		cudaMalloc( &cellAdj, sizeof(uint));
		uint st = 0, end = 2;
		cudaMemcpy(cellStart, &st, sizeof(uint), cudaMemcpyHostToDevice);
		cudaMemcpy(cellEnd, &end, sizeof(uint), cudaMemcpyHostToDevice);
		cudaMemset(cellAdj, 0, sizeof(uint));
	}

	~TwoPTest(){
		cudaFree(dNList);
		cudaFree(dNumNeigh);
		cudaFree(dpos);
		cudaFree(phash);
		cudaFree(cellStart);
		cudaFree(cellEnd);
		cudaFree(cellAdj);
	}
};


TEST_F(TwoPTest, VarCondBasic) {
	float4 fakepos[2] = {make_float4(0,0,0,1), make_float4(2,0,0,1)};
	ASSERT_TRUE(setDPos(fakepos));

	//sepdist is 2, distance is 2
	uint maxn = funcNList(dNList, dNumNeigh,(float*) dpos, phash,
			cellStart,cellEnd,cellAdj, hp.N, nListSize, VarCond(1.01));
	ASSERT_EQ(maxn,1);

	maxn = funcNList(dNList, dNumNeigh,(float*) dpos, phash,
			cellStart,cellEnd,cellAdj, hp.N, nListSize, VarCond(0.99));
	ASSERT_EQ(maxn,0);
}

TEST_F(TwoPTest, VarCondFar) {
	float4 fakepos[2] = {make_float4(0,0,0,1), make_float4(4,0,0,1)};
	ASSERT_TRUE(setDPos(fakepos));

	//var cond identifies points with dist < param*(a1+a2);
	//sepdist is 2, distance is 4
	EXPECT_EQ(0, funcNList(dNList, dNumNeigh,(float*) dpos, phash,
			cellStart,cellEnd,cellAdj, hp.N, nListSize, VarCond(1.01)));

	//exactly equal, so NOT neighbors
	EXPECT_EQ(0, funcNList(dNList, dNumNeigh,(float*) dpos, phash,
			cellStart,cellEnd,cellAdj, hp.N, nListSize, VarCond(2.0)));

	EXPECT_EQ(1, funcNList(dNList, dNumNeigh,(float*) dpos, phash,
			cellStart,cellEnd,cellAdj, hp.N, nListSize, VarCond(2.01)));
}

TEST_P(TwoPTest, VarCondRadius) {
	float4 fakepos[2] = {make_float4(0,0,0,1.0),
			make_float4(1.0f+GetParam(),0,0,GetParam() )};
	ASSERT_TRUE(setDPos(fakepos));

	EXPECT_EQ(0,funcNList(dNList, dNumNeigh,(float*) dpos, phash,
			cellStart,cellEnd,cellAdj, hp.N, nListSize, VarCond(0.99f)));

	//exactly equal, so NOT neighbors but to floating prec?
	EXPECT_EQ(0, funcNList(dNList, dNumNeigh,(float*) dpos, phash,
			cellStart,cellEnd,cellAdj, hp.N, nListSize, VarCond(1.0f)));

	EXPECT_EQ(1, funcNList(dNList, dNumNeigh,(float*) dpos, phash,
			cellStart,cellEnd,cellAdj, hp.N, nListSize, VarCond(1.1f)));

	float4 fp2[2] = {make_float4(1.0f+GetParam(),0,0,GetParam() ),
			make_float4(0,0,0,1.0)};
	ASSERT_TRUE(setDPos(fp2));

	EXPECT_EQ(0,funcNList(dNList, dNumNeigh,(float*) dpos, phash,
			cellStart,cellEnd,cellAdj, hp.N, nListSize, VarCond(0.99f)));

	//exactly equal, so NOT neighbors but to floating prec?
	EXPECT_EQ(0, funcNList(dNList, dNumNeigh,(float*) dpos, phash,
			cellStart,cellEnd,cellAdj, hp.N, nListSize, VarCond(1.0f)));

	EXPECT_EQ(1, funcNList(dNList, dNumNeigh,(float*) dpos, phash,
			cellStart,cellEnd,cellAdj, hp.N, nListSize, VarCond(1.1f)));
}

INSTANTIATE_TEST_CASE_P(NearOne, TwoPTest, ::testing::Range(0.2f,5.0f,0.1f));

TEST_F(TwoPTest, VarCondDiffRadius) {

}
