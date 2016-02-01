/** \file
 *  \ingroup commandLineTools 
 *  \details This program reads either an airway particles dataset or a vessel
 *   particles dataset and uses Kruskall's min-spanning tree algorithm to
 *   define a topology on the particles points. The output polydata is
 *   equivalent to the input polydata but with polylines defined indicating
 *   the edges between particle points found by the min spanning tree
 *   algorithm. The connected dataset is rendered and then optionally written
 *   to file
 */

#ifndef DOXYGEN_SHOULD_SKIP_THIS

#include "vtkSmartPointer.h"
#include "vtkPolyData.h"
#include "vtkPointData.h"
#include "vtkPolyDataReader.h"
#include "vtkPolyDataWriter.h"
#include "vtkGraphToPolyData.h"
#include "vtkExtractSelectedGraph.h"
#include "vtkBoostKruskalMinimumSpanningTree.h"
#include "vtkMutableUndirectedGraph.h"
#include "vtkDataSetAttributes.h"
#include "vtkDoubleArray.h"
#include "cipChestConventions.h"
#include "cipHelper.h"
#include "ReadParticlesWriteConnectedParticlesCLP.h"

vtkSmartPointer<vtkMutableUndirectedGraph> GetMinimumSpanningTree(vtkSmartPointer<vtkPolyData>, double, std::string);
bool GetEdgeWeight(unsigned int, unsigned int, vtkSmartPointer<vtkPolyData>, double*, double, std::string);

int main( int argc, char *argv[] )
{
  PARSE_ARGS;

  std::string particlesType;
  vtkSmartPointer<vtkPolyData> particles = vtkSmartPointer<vtkPolyData>::New();

  vtkSmartPointer<vtkPolyDataReader> particlesReader = vtkSmartPointer<vtkPolyDataReader>::New();
  if (vesselParticlesFileName.compare("NA") != 0)
    {
    particlesType = "vessel";

    particlesReader->SetFileName(vesselParticlesFileName.c_str());
    }
  else if (airwayParticlesFileName.compare("NA") != 0)
    {
    particlesType = "airway";

    particlesReader->SetFileName(airwayParticlesFileName.c_str());
    }
  else
    {
    std::cerr << "Must either specify an airway particles file name or a vessel particles file name" << std::endl;
    return cip::ARGUMENTPARSINGERROR;
    }
  std::cout << "Reading particles..." << std::endl;
  particlesReader->Update();

  std::cout << "Constructing minimum spanning tree..." << std::endl;
  vtkSmartPointer<vtkMutableUndirectedGraph> minimumSpanningTree = 
    GetMinimumSpanningTree(particlesReader->GetOutput(), particleDistanceThreshold, particlesType);

  if (visualize)
    {
      std::cout << "Visualizing graph..." << std::endl;
      cip::ViewGraphAsPolyData(minimumSpanningTree);
    }

  // If the user has specified an output file name, write the
  // connected particles to fiel
  if (outParticlesFileName.compare("NA") != 0)
    {
    // First convert the graph to polydata
    vtkSmartPointer<vtkGraphToPolyData> graphToPolyData = vtkSmartPointer<vtkGraphToPolyData>::New();
      graphToPolyData->SetInputData(minimumSpanningTree);
      graphToPolyData->Update();

    cip::GraftPointDataArrays( particlesReader->GetOutput(), graphToPolyData->GetOutput() );    
    cip::TransferFieldData( particlesReader->GetOutput(), graphToPolyData->GetOutput() );

    std::cout << "Writing connected particles..." << std::endl;
    vtkSmartPointer<vtkPolyDataWriter> writer = vtkSmartPointer<vtkPolyDataWriter>::New();
      writer->SetInputData(graphToPolyData->GetOutput());
      writer->SetFileName(outParticlesFileName.c_str());
      writer->Update();
    }

  std::cout << "DONE." << std::endl;

  return cip::EXITSUCCESS;
}

vtkSmartPointer<vtkMutableUndirectedGraph> GetMinimumSpanningTree(vtkSmartPointer<vtkPolyData> particles, double distanceThreshold, 
                                                                  std::string particlesType)
{ 
  // Now create the weighted graph that will be passed to the minimum 
  // spanning tree filter
  std::map<unsigned int, unsigned int> particleIDToNodeIDMap;
  std::map<unsigned int, unsigned int> nodeIDToParticleIDMap;

  vtkSmartPointer<vtkMutableUndirectedGraph> weightedGraph =  
    vtkSmartPointer<vtkMutableUndirectedGraph>::New();

  for (unsigned int i=0; i<particles->GetNumberOfPoints(); i++)
    {
    vtkIdType nodeID = weightedGraph->AddVertex();

    particleIDToNodeIDMap[i]      = nodeID;
    nodeIDToParticleIDMap[nodeID] = i;
    }

  vtkSmartPointer<vtkDoubleArray> edgeWeights = vtkSmartPointer<vtkDoubleArray>::New();
    edgeWeights->SetNumberOfComponents(1);
    edgeWeights->SetName("Weights");

  for (unsigned int i=0; i<particles->GetNumberOfPoints(); i++)
    {
    for (unsigned int j=i+1; j<particles->GetNumberOfPoints(); j++)
      {
      double weight;
	  
      if (GetEdgeWeight(i, j, particles, &weight, distanceThreshold, particlesType))
        {
        weightedGraph->AddEdge(particleIDToNodeIDMap[i], particleIDToNodeIDMap[j]);
        edgeWeights->InsertNextValue(weight);
        }
      }
    }

  weightedGraph->GetEdgeData()->AddArray(edgeWeights);
  weightedGraph->SetPoints(particles->GetPoints());

  vtkSmartPointer<vtkBoostKruskalMinimumSpanningTree> minimumSpanningTreeFilter = 
    vtkSmartPointer<vtkBoostKruskalMinimumSpanningTree>::New();
    minimumSpanningTreeFilter->SetInputData(weightedGraph);
    minimumSpanningTreeFilter->SetEdgeWeightArrayName("Weights");
    minimumSpanningTreeFilter->Update();

  vtkSmartPointer<vtkExtractSelectedGraph> extractSelection = vtkSmartPointer<vtkExtractSelectedGraph>::New();
    extractSelection->SetInputData(0, weightedGraph);
    extractSelection->SetInputData(1, minimumSpanningTreeFilter->GetOutput()) ;
    extractSelection->Update();

  return vtkMutableUndirectedGraph::SafeDownCast(extractSelection->GetOutput());
}

bool GetEdgeWeight(unsigned int particleID1, unsigned int particleID2, vtkSmartPointer<vtkPolyData> particles, double* weight, 
                   double distanceThreshold, std::string particlesType)
{
  std::string vecName;
  if (particlesType.compare("vessel") == 0)
    {
    vecName = "hevec0";
    }
  else
    {
    vecName = "hevec2";
    }

  // Used in the function for determing what weight to assign to each edge
  double edgeWeightAngleSigma = 1.0;

  // Determine the vector connecting the two particles
  double point1[3];
    point1[0] = particles->GetPoint(particleID1)[0];
    point1[1] = particles->GetPoint(particleID1)[1];
    point1[2] = particles->GetPoint(particleID1)[2];

  double point2[3];
    point2[0] = particles->GetPoint(particleID2)[0];
    point2[1] = particles->GetPoint(particleID2)[1];
    point2[2] = particles->GetPoint(particleID2)[2];

  cip::VectorType connectingVec(3);
    connectingVec[0] = point1[0] - point2[0];
    connectingVec[1] = point1[1] - point2[1];
    connectingVec[2] = point1[2] - point2[2];

  double connectorMagnitude = cip::GetVectorMagnitude(connectingVec);

  if (connectorMagnitude > distanceThreshold)
    {
    return false;
    }

  cip::VectorType particle1Hevec2(3);
    particle1Hevec2[0] = particles->GetPointData()->GetArray(vecName.c_str())->GetTuple(particleID1)[0];
    particle1Hevec2[1] = particles->GetPointData()->GetArray(vecName.c_str())->GetTuple(particleID1)[1];
    particle1Hevec2[2] = particles->GetPointData()->GetArray(vecName.c_str())->GetTuple(particleID1)[2];

  cip::VectorType particle2Hevec2(3);
    particle2Hevec2[0] = particles->GetPointData()->GetArray(vecName.c_str())->GetTuple(particleID2)[0];
    particle2Hevec2[1] = particles->GetPointData()->GetArray(vecName.c_str())->GetTuple(particleID2)[1];
    particle2Hevec2[2] = particles->GetPointData()->GetArray(vecName.c_str())->GetTuple(particleID2)[2];

  double angle1 = cip::GetAngleBetweenVectors(particle1Hevec2, connectingVec, true);
  double angle2 = cip::GetAngleBetweenVectors(particle2Hevec2, connectingVec, true);

  if (angle1 < angle2)
    {
    *weight = connectorMagnitude*(1.0 + exp(-pow((90.0 - angle1)/edgeWeightAngleSigma, 2)));
    }
  else
    {
    *weight = connectorMagnitude*(1.0 + exp(-pow((90.0 - angle2)/edgeWeightAngleSigma, 2)));
    }

  return true;
}

#endif
