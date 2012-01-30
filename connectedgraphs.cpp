#include <cstdio>
#include <math.h>
#include "cutil_inline.h"
#include "cutil_math.h"
#include "vector_functions.h"
#include "particles_kernel.cuh"
#include "connectedgraphs.h"
#include <assert.h>
#include <stack>
using namespace std;


int3 calcGridPos(float3 p, SimParams params){
	int3 gridPos;
	gridPos.x = floor((p.x - params.worldOrigin.x) / params.cellSize.x);
	gridPos.y = floor((p.y - params.worldOrigin.y) / params.cellSize.y);
	gridPos.z = floor((p.z - params.worldOrigin.z) / params.cellSize.z);
	return gridPos;
}


uint calcGridHash(int3 gridPos, SimParams params)
{
	int3 agridPos;
	agridPos.x = (params.gridSize.x + gridPos.x) % params.gridSize.x;
	agridPos.y = (params.gridSize.y + gridPos.y) % params.gridSize.y;
	agridPos.z = (params.gridSize.z + gridPos.z) % params.gridSize.z;
	return agridPos.z*params.gridSize.y*params.gridSize.x + agridPos.y*params.gridSize.x + agridPos.x;
}
//adj list is in groups of two, starting at zero, containing particles numbers as in a->b
//adjstart[ii] contains the beginning of the adjacencye ntries in adjlist for particle[ii]
void makeAdjList(
		float4* points, uint* cellStart, uint* cellEnd, 
		AdjPair* adjlist, int* adjstart, 
		const SimParams &params, uint adjlistsize)
{
	int solos = 0;	
	uint adjcount = 0;
	bool firstEdge = true;
	for(uint ii = 0; ii < params.numBodies; ii++){
		firstEdge = true;	
		float3 pos1 = make_float3(points[ii].x, points[ii].y, points[ii].z);
		float radius1 = points[ii].w;
		
		int3 gridPos = calcGridPos(pos1, params);
		for(int z=-1; z <=1; z++){
			for(int y=-1; y<=1; y++){
				for(int x=-1; x<=1; x++){
					int3 neighborPos = gridPos + make_int3(x,y,z);
					uint gridHash = calcGridHash(neighborPos, params);
					uint startIndex = cellStart[gridHash];
					uint endIndex = cellEnd[gridHash];
					for(uint j = startIndex; j<endIndex; j++){
						if(j != ii){
							float3 pos2 = make_float3(points[j]);
							float radius2 = points[j].w;
							//periodic BCs
							if(neighborPos.x >= (int) params.gridSize.x)
								pos2.x += -2.0f*params.worldOrigin.x;
							if(neighborPos.x < 0)
								pos2.x -= -2.0f*params.worldOrigin.x;
							if(neighborPos.y >= (int) params.gridSize.y)
								pos2.y += -2.0f*params.worldOrigin.y;
							if(neighborPos.y < 0)
								pos2.y -= -2.0f*params.worldOrigin.y;
							
							float sepdist = radius1+radius2;
							if ( length(pos2-pos1) < 1.1*sepdist){
								assert(adjcount < adjlistsize);
								adjlist[adjcount].node = ii;
								adjlist[adjcount].edge = j;
								if(firstEdge){
									adjstart[ii] = adjcount;
								}
								firstEdge = false;
								adjcount++;

							}
						}
					}
				}
			}
		}
		if(firstEdge){ //has no edges, make it self connected so it still gets flagged
			assert(adjcount < adjlistsize);
			adjlist[adjcount].node = ii;
			adjlist[adjcount].edge = ii;
			adjstart[ii] = adjcount;
			adjcount++;
			solos++;
		}
	}
	//printf("%d particles\t%d solos\t adjcount %d\n", numParticles,solos, adjcount);
}


int stackConGraphs(AdjPair* adjlist, const int* adjstart, const uint numParticles, const uint adjlistsize)
{
	bool* visited = new bool[numParticles];
	memset(visited, 0, sizeof(bool)*numParticles);
	stack<AdjPair> S;
	uint numgraphs = 0;

	for(int ii=0; ii < (int) numParticles; ii++){
		if(!visited[ii]){
			assert(S.empty());
			for(int jj = adjstart[ii]; adjlist[jj].node == ii; jj++){
				//prime the stack with all the entries of a node
				S.push(adjlist[jj]);
			}
			numgraphs++;
		}
		visited[ii]=true;
		int curr = 0;// a node number
		while( !S.empty() ){
			curr = S.top().edge;
			S.pop();
			if(!visited[curr]){
				for(uint jj = adjstart[curr]; adjlist[jj].node == curr; jj++){
					//add the edges of particles adjacent
					assert(jj < adjlistsize);
					S.push(adjlist[jj]);
				}
			}
			visited[curr]=true;
		}
	}
	delete visited;
	return numgraphs;
}
