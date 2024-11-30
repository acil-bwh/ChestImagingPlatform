/** \file
 *  \ingroup commandLineTools 
 */

//#include "stdafx.h"
#include "cipChestConventions.h"
#include "cipHelper.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <iostream>
#include <fstream>
#include "GCoptimization.h"

#include "GraphCutsOptimizationCLP.h"

using namespace std;

std::vector<int> GraphCutsOptimization(int num_els, int num_labels, vector<int> source, vector<int> sink, 
							vector< vector<int> > adj, int method)
{
	std::vector<int> result;   // stores result of optimization

	// first set up the array for data costs
	int *data = new int[num_els*num_labels];
	for (int i = 0; i < num_els; i++){
		for (int l = 0; l < num_labels; l++){
			if (l == 0)	data[i*num_labels+l] = sink[i];
			else data[i*num_labels + l] = source[i];
		}
	}
	
	try{
		GCoptimizationGeneralGraph *gc = new GCoptimizationGeneralGraph(num_els, num_labels);
		gc->setDataCost(data);

		vector<int> ir;
		vector<int> jc;
		for (int col = 0; col < adj[0].size(); col++){
			bool nonzeros = 0;
			for (int row = 0; row < adj.size(); row++){
				if (adj[row][col] != 0){
					ir.push_back(row);
					nonzeros = 1;
					if (jc.size() == 0) jc.push_back(col);
				}
			}
			if (!nonzeros){
				if (jc.size() == 0) jc.push_back(0);
			}
			jc.push_back(ir.size());
		}

		for (int c = 0; c < num_els; c++) {
			int rowStart = jc[c];
			int rowEnd = jc[c+1];

			for (int ri = rowStart; ri < rowEnd; ri++){
				int r = ir[ri];
				if (r < c ){
					gc->setNeighbors(r, c, int(adj[r][c]));
				}
					
			}
		}

		//gc->setVerbosity(2);
		if (method == 0){ //Swap method
			gc->swap();
		}
		else{ //Expansion method
			gc->expansion();
		}

		for (int i = 0; i < num_els; i++){
			if (gc->whatLabel(i) == 0) result.push_back(1);
			else result.push_back(0);
		}
		
		delete gc;
	}
	catch (GCException e){
		e.Report();
	}

	delete[] data;

	return result;
}

void SaveData(std::vector<int> res, string namefile, string meth){
	std::cout << "    Saving " << meth << " Output..." << std::endl;
	ofstream outfile(namefile.c_str());
	if (outfile.is_open())
	{
		outfile << res[0];
		for (int i = 1; i < res.size(); i++){
			outfile << " " << res[i];
		}
		outfile.close();
	}
	else cout << "Unable to open file";
}


int main( int argc, char *argv[] )
{  
  PARSE_ARGS;
  
  int num_labels = 2;
  int m = 0; // Swap method by default

  if (Method == "Expansion"){
	  m = 1;
  }

  ifstream adjFile(adjMat.c_str());
  string line;
  vector< vector<int> > adjMatrix;
  int i = 0;

  while (getline(adjFile, line))
  {
	  int value;
	  stringstream ss(line);

	  adjMatrix.push_back(std::vector<int>());

	  while (ss >> value)
	  {
		  adjMatrix[i].push_back(value);
	  }
	  ++i;
  }
  
  std::vector<int> w_source;
  ifstream sourceFile(WSource.c_str());

  float sourceNum;
  while (sourceFile >> sourceNum){
	  w_source.push_back(sourceNum);
  }

  std::vector<int> w_sink;
  ifstream sinkFile(WSink.c_str());

  float sinkNum;
  while (sinkFile >> sinkNum){
	  w_sink.push_back(sinkNum);
  }

  std::vector<int> GCResult = GraphCutsOptimization(w_source.size(), num_labels, w_source, w_sink, adjMatrix, m);
  
  SaveData(GCResult, OutputFileName, Method);
  std::cout << "    DONE" << std::endl;

  //printf("\n  Finished %d (%d) clock per sec %d \n", clock() / CLOCKS_PER_SEC, clock(), CLOCKS_PER_SEC);

  
  return cip::EXITSUCCESS;
}


