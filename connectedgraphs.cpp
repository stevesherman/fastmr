#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <memory.h>
#include "connectedgraphs.h"
#include <assert.h>
#include <stack>
#include <vector>
#include <algorithm>
#include <helper_math.h>

using namespace std;

/*
 * Takes in a nlist, puts the graph ID in visited, and the corresponding length
 * in length_list.  visited and length are assumed to empty and preinitialized
 */

void graphLabeler(const uint* nlist,const uint* num_neigh, uint numParticles,
		vector<uint>& length_list, int* visited)
{
	stack<uint> S;
	uint numGraphs = 0;
	uint curr = 0;
	uint chainl = 0;

	//depth first search
	for(uint i=0; i < numParticles; i++){
		//if we haven't visited, prime the stack, and add to chain counter
		chainl = 0;
		if(visited[i] == 0){
			chainl++;
			numGraphs++;
			visited[i] = numGraphs;
			for(uint j = 0; j < num_neigh[i]; j++){
				S.push(nlist[numParticles*j + i]);
			}
		}
		while(!S.empty() ) {
			curr = S.top();
			S.pop();
			if(visited[curr] == 0){
				visited[curr] = numGraphs;
				chainl++;
				for(uint j=0; j < num_neigh[curr];j++){
					S.push(nlist[numParticles*j + curr]);
				}
			}
		}
		if(chainl > 0){
			//	printf("chain %d has length %d\n", i, chainl);
			length_list.push_back(chainl);
		}
	}
}




uint adjConGraphs(uint* nlist, uint* num_neigh, uint numParticles)
{
	int* visited = new int[numParticles];
	memset(visited, 0, sizeof(int)*numParticles);
	vector<uint> length_list;

	graphLabeler(nlist, num_neigh, numParticles, length_list, visited);

	uint numGraphs = length_list.size();

	delete [] visited;
	return numGraphs;
}	

float frand()
{
	return (float) rand() / (float) RAND_MAX;
}

void graphColorLabel(const uint* nlist, const uint* num_neigh,
					uint numParticles, float* hColor)
{
	float4* f4hColor = (float4*) hColor;
	int* visited = new int[numParticles];
	memset(visited, 0, sizeof(int)*numParticles);
	vector<uint> length_list;

	graphLabeler(nlist, num_neigh, numParticles, length_list, visited);

	//desaturated vals from maltab's distinguishable_colors.m
	int numGraphs = length_list.size();
	float4* colorList = new float4[numGraphs];
	srand(2); //so the colors flash less
	for (int ii=0; ii < numGraphs; ii++) {
		colorList[ii] = make_float4(frand()*0.9 + 0.05, frand()*0.9 + 0.05,
				frand()*0.9 + 0.05, 0.0);
	}

	int graph_id, col_id;
	bool flag = false;
	for(int ii=0; ii < numParticles; ii++) {
		graph_id = visited[ii];
		if(graph_id == 0)
			flag = true;
		f4hColor[ii] = colorList[graph_id-1];
	}

	if(flag)
		printf("Oh hey graph labeling is FUCKED\n");

	delete [] visited;
	delete [] colorList;
}



