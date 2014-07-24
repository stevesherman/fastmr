/*
 * some really shitty unit testing code
 */

#include "gtest/gtest.h"
#include "nlist.h"
#include "new_kcall.h"
#include "new_kern.h"
#include "utilities.h"
#include <cuda_runtime.h>


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

	//ensure
	template <class Op> void NLTest(int n_neigh, Op o) {
		SCOPED_TRACE(testing::Message()<< "want n_neigh==" << n_neigh
				<< ", at dist=" << sqrt(o.max_distsq));
		EXPECT_EQ(n_neigh, callNList(o));
		EXPECT_TRUE( validateNListLength(std::vector<uint>(hp.N,n_neigh) ) );
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

TEST_F(TwoPTest, VarCondFar) {
	float4 fakepos[2] = {make_float4(0,0,0,1), make_float4(4,0,0,1)};
	ASSERT_TRUE(setDPos(fakepos));

	//var cond identifies points with dist < param*(a1+a2);
	//sepdist is 2, distance is 4
	NLTest(0,VarCond(1.0+1e-5f));
	//exactly equal, so NOT neighbors
	NLTest(0,VarCond(2.0) );
	NLTest(1,VarCond(2.0+1e-5f));
}

TEST_P(TwoPTest, VarCondRadius) {
	float sep = 1.0f+GetParam();
	float4 fakepos[2] = {make_float4(0,0,0,1.0),
			make_float4(sep,0,0,GetParam() )};
	ASSERT_TRUE(setDPos(fakepos));

	EXPECT_EQ(0,callNList( VarCond(1.0f - 1e-5f)));
	//exactly equal, so NOT neighbors but to floating prec?
	NLTest(0,VarCond(1.0f));
	NLTest(1,VarCond(1.0f+1e-5f));

	float4 fp2[2] = {make_float4(sep,0,0,GetParam() ),
			make_float4(0,0,0,1.0)};
	ASSERT_TRUE(setDPos(fp2));

	EXPECT_EQ(0,callNList( VarCond(1.0f - 1e-5f)));
	//exactly equal, so NOT neighbors but to floating prec?
	NLTest(0,VarCond(1.0f));
	NLTest(1,VarCond(1.0f+1e-5f));
}

INSTANTIATE_TEST_CASE_P(NearOne, TwoPTest, ::testing::Range(0.2f,5.0f,0.2f));

TEST_F(TwoPTest, VarPeriodicBC) {

	//hp.L = make_float3(20.0f,20.0f,20.0f); hp.Linv = 1/hp.L;
	//setDevParams();

	float4 fp[2] = {make_float4(0,0,0,1.0),
	make_float4(hp.L.x - 2.0,0,0, 1.0 )};
	ASSERT_TRUE(setDPos(fp));

	NLTest(0,VarCond(1.0f));
	NLTest(1,VarCond(1.0f+1e-5f));

	float4 fp2[2] = {make_float4(0,0,0,1.0),
			make_float4(0,hp.L.y - 2.0,0, 1.0 )};
	ASSERT_TRUE(setDPos(fp2));
	//test that it is NOT vertically periodic
	NLTest(0,VarCond(1.0f));
	NLTest(0,VarCond(1.0f+1e-5f));

	float4 fp3[2] = {make_float4(0,0,0,1.0),
			make_float4(0,0,hp.L.z - 2.0,1.0 )};
	ASSERT_TRUE(setDPos(fp3));
	//test that it is horiz periodic
	NLTest(0,VarCond(1.0f));
	NLTest(1,VarCond(1.0f+1e-5f));
}

TEST_P(TwoPTest, VertRadius)
{
	float sep = 1.0f+GetParam();
	float mult = 2.0f;
	//shear adjacent so it fails
	float4 fpx[2] = {make_float4(0,0,0,1.0),
			make_float4(mult*sep,0,0,GetParam() )};
	ASSERT_TRUE(setDPos(fpx));
	NLTest(0, VertCond(mult,sqrt(3.0/5.0)));
	NLTest(0, VertCond(mult+1e-5f,sqrt(3.0/5.0)));

	//vertically adjacent so 2nd succeeds
	float4 fpy[2] = {make_float4(0,0,0,1.0),
			make_float4(0,mult*sep,0,GetParam() )};
	ASSERT_TRUE(setDPos(fpy));
	NLTest(0, VertCond(mult,sqrt(3.0/5.0)));
	NLTest(1, VertCond(mult+1e-5f,sqrt(3.0/5.0)));

	//oop adjacent so it fails
	float4 fpz[2] = {make_float4(0,0,0,1.0),
			make_float4(0,0,mult*sep,GetParam() )};
	ASSERT_TRUE(setDPos(fpz));
	NLTest(0, VertCond(mult,sqrt(3.0/5.0)));
	NLTest(0, VertCond(mult+1e-5f,sqrt(3.0/5.0)));

	//flip order to check
}

TEST_P(TwoPTest, OOPRadius)
{
	float sep = 1.0f+GetParam();
	float mult = 2.0; //tests may fail if mult not fp exactly representable
	//shear adjacent so it fails
	float4 fpx[2] = {make_float4(0,0,0,1.0),
			make_float4(mult*sep,0,0,GetParam() )};
	ASSERT_TRUE(setDPos(fpx));
	NLTest(0, OutOfPlane(mult,sqrt(3.0/5.0),1));
	NLTest(0, OutOfPlane(mult+1e-5f,sqrt(3.0/5.0),1));

	//vertically adjacent so fails
	float4 fpy[2] = {make_float4(0,0,0,1.0),
			make_float4(0,mult*sep,0,GetParam() )};
	ASSERT_TRUE(setDPos(fpy));
	NLTest(0, OutOfPlane(mult,sqrt(3.0/5.0),1));
	NLTest(0, OutOfPlane(mult+1e-5f,sqrt(3.0/5.0),1));

	//oop adjacent so suceeds
	float4 fpz[2] = {make_float4(0,0,0,1.0),
			make_float4(0,0,mult*sep,GetParam() )};
	ASSERT_TRUE(setDPos(fpz));
	NLTest(0, OutOfPlane(mult,sqrt(3.0/5.0),1));
	NLTest(1, OutOfPlane(mult+1e-5f,sqrt(3.0/5.0),1));


}
