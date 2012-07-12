#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <memory.h>
#include "connectedgraphs.h"
#include <assert.h>
#include <stack>
#include <vector>
#include <algorithm>
using namespace std;

uint adjConGraphs(uint* nlist, uint* num_neigh, uint numParticles)
{
	bool* visited = new bool[numParticles];
	memset(visited, 0, sizeof(bool)*numParticles);
	stack<uint> S;
	uint numGraphs = 0;
	uint curr = 0;
	uint chainl = 0;
	vector<uint> length_list;

	for(uint i=0; i < numParticles; i++){
		//if we haven't visited, prime the stack, and add to chain counter
		chainl = 0;
		if(!visited[i]){
			visited[i] = true;
			for(uint j = 0; j < num_neigh[i]; j++){
				S.push(nlist[numParticles*j + i]);
			}
			chainl++;
			numGraphs++;
		}
		while(!S.empty() ) {
			curr = S.top();
			S.pop();
			if(!visited[curr]){
				visited[curr] = true;
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
	nth_element(length_list.begin(), length_list.begin()+numGraphs/2, length_list.end());
	//printf("median chain length: %d\n", length_list[numGraphs/2]);
	delete visited;
	return numGraphs;
}	







