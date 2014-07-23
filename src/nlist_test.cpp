#include "gtest/gtest.h"
#include "nlist.h"
#include "new_kcall.h"
#include "new_kern.h"
#include "utilities.h"
#include <cuda_runtime.h>


TEST(NlistTest, SimpleCuda) {
	uint* nlist;
	ASSERT_EQ(cudaMalloc((void**) &nlist, 2*sizeof(float)), cudaSuccess);
	ASSERT_EQ(cudaFree(nlist), cudaSuccess);
}


TEST(NlistTest, twop_adjacent_resize)
{
	NewParams host;
	host.L = make_float3(10,10,10);
	host.Linv = 1/host.L;
	host.N = 2;
	host.numAdjCells = 1;
	uint *dnlist, *dnumneigh;
	uint nlist_size = 0;

	cudaMalloc(&dnumneigh, 2*sizeof(uint));
	uint* phash;
	cudaMalloc(&phash, sizeof(uint)*2);
	cudaMemset(phash,0, 2*sizeof(uint));

	float4* dpos;
	float4 hpos[2] = {make_float4(0,0,0,1), make_float4(2,0,0,1)};
	cudaMalloc( &dpos, 2*sizeof(float4));
	cudaMemcpy(dpos,hpos,sizeof(float4)*host.N,cudaMemcpyHostToDevice);

	uint* cellStart, *cellEnd;
	cudaMalloc( &cellStart, sizeof(int));
	cudaMalloc( &cellEnd, sizeof(int));
	int st = 0, end = 2;
	cudaMemcpy(cellStart, &st, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(cellEnd, &end, sizeof(int), cudaMemcpyHostToDevice);

	uint* cellAdj;
	cudaMalloc(&cellAdj, sizeof(uint));
	cudaMemset(cellAdj, 0, sizeof(uint));


	setNParameters(&host);
	uint maxn = funcNList(dnlist, dnumneigh, (float*) dpos, phash, cellStart,
			cellEnd, cellAdj, host.N, nlist_size,VarCond(1.1));
	ASSERT_EQ(maxn,1);


	maxn = funcNList(dnlist, dnumneigh, (float*) dpos, phash, cellStart,
				cellEnd, cellAdj, host.N, nlist_size,VarCond(0.9));
	ASSERT_EQ(maxn,0);


	cudaFree(dpos);
	cudaFree(dnlist);
	cudaFree(dnumneigh);
	cudaFree(phash);
	cudaFree(cellStart);
	cudaFree(cellEnd);
	cudaFree(cellAdj);
}


/*
int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}*/
