#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <memory.h>
#include "connectedgraphs.h"
#include <assert.h>
#include <stack>
using namespace std;

uint adjConGraphs(uint* nlist, uint* num_neigh, uint numParticles)
{
	bool* visited = new bool[numParticles];
	memset(visited, 0, sizeof(bool)*numParticles);
	stack<uint> S;
	uint numGraphs = 0;
	uint curr = 0;
	uint chainl = 0;

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
	}
	delete visited;
	return numGraphs;
}	







