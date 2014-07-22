#include "connectedgraphs.h"
#include "gtest/gtest.h"


//two particle chain
TEST(ConnectedGraphs,2pchain){
	//two particle chain
	uint fake_nlist[2] = {1, 0};
	uint num_neigh[2] = { 1 ,1};
	uint num_graphs = adjConGraphs(fake_nlist,num_neigh,2);
	ASSERT_EQ(num_graphs,1);
}

//two particle chain and stray
TEST(ConnectedGraphs,2pchain_andstray){
	//two particle chain
	const uint num_particles = 3;
	uint fake_nlist[6] =   {1, 0, 0,
							0, 2, 0};
	uint num_neigh[num_particles] = {1,1,0};
	uint num_graphs = adjConGraphs(fake_nlist,num_neigh,num_particles);
	ASSERT_EQ(num_graphs,2);
}


//three particle chain
TEST(ConnectedGraphs,3pchain){
	//two particle chain
	const uint num_particles = 3;
	uint fake_nlist[6] =   {1, 0, 1,
							0, 2, 0};
	uint num_neigh[num_particles] = {1,2,1};
	uint num_graphs = adjConGraphs(fake_nlist,num_neigh,num_particles);
	ASSERT_EQ(num_graphs,1);
}

//three particle loop
TEST(ConnectedGraphs,3ploop){
	//two particle chain
	const uint num_particles = 3;
	uint fake_nlist[6] =   {1, 2, 0,
							2, 0, 1};
	uint num_neigh[num_particles] = {2,2,2};
	uint num_graphs = adjConGraphs(fake_nlist,num_neigh,num_particles);
	ASSERT_EQ(num_graphs,1);
}

//three particle loop with stray
TEST(ConnectedGraphs,3ploopandstray){
	//two particle chain
	const uint num_particles = 4;
	uint fake_nlist[8] =   {1, 2, 0, 0,
							2, 0, 1, 0};
	uint num_neigh[num_particles] = {2,2,2,0};
	uint num_graphs = adjConGraphs(fake_nlist,num_neigh,num_particles);
	ASSERT_EQ(num_graphs,2);
}

//three particle loop with stray, reversed to make sure graph order doenst matter
TEST(ConnectedGraphs,3ploopandstray_flipped){
	//two particle chain
	const uint num_particles = 4;
	uint fake_nlist[8] =   {2, 0, 1, 0,
							1, 2, 0, 0};
	uint num_neigh[num_particles] = {2,2,2,0};
	uint num_graphs = adjConGraphs(fake_nlist,num_neigh,num_particles);
	ASSERT_EQ(num_graphs,2);
}
