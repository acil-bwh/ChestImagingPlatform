/**
 *
 *  $Date: 2012-09-17 18:32:23 -0400 (Mon, 17 Sep 2012) $
 *  $Revision: 268 $
 *  $Author: jross $
 *
 */

#include "vtkCIPAirwayParticlesToConnectedAirwayParticlesFilter.h"

#include "vtkProperty.h" 
#include "vtkDataSetAttributes.h"
#include "vtkObjectFactory.h"
#include "vtkDoubleArray.h"
#include "vtkStreamingDemandDrivenPipeline.h"
#include "vtkInformationVector.h"
#include "vtkInformation.h"
#include "vtkInteractorStyleTrackballCamera.h"
#include "vtkDataObject.h"
#include "vtkPoints.h"
#include "vtkFloatArray.h"
#include "vtkPointData.h"
//#include "vtkGraphLayoutView.h"
#include "vtkRenderWindow.h"
#include "vtkRenderWindowInteractor.h"
#include "vtkBoostKruskalMinimumSpanningTree.h"
#include "vtkRenderer.h"
#include "vtkPolyDataMapper.h"
#include "vtkEdgeListIterator.h"
#include "vtkActor.h"
#include "vtkGraphToPolyData.h"
#include "vtkSphereSource.h"
#include "vtkGlyph3D.h"
#include "vtkMutableUndirectedGraph.h"
#include <cfloat>
#include <math.h>


vtkStandardNewMacro(vtkCIPAirwayParticlesToConnectedAirwayParticlesFilter);
 
vtkCIPAirwayParticlesToConnectedAirwayParticlesFilter::vtkCIPAirwayParticlesToConnectedAirwayParticlesFilter()
{
  this->EdgeWeightAngleSigma = 1.0;

  this->SetNumberOfInputPorts(1);
  this->SetNumberOfOutputPorts(1);

  //  this->InterParticleDistance = 1.7; // Typically used for particles sampling

  // this->InternalInputPolyData = vtkSmartPointer< vtkPolyData >::New();
  // this->DataStructureImage    = vtkSmartPointer< vtkImageData >::New();
  // this->CompositeGraph        = vtkSmartPointer< vtkMutableDirectedGraph >::New();

  // this->ConnectorPoints = vtkSmartPointer< vtkPoints >::New(); //DEB
}
 
vtkCIPAirwayParticlesToConnectedAirwayParticlesFilter::~vtkCIPAirwayParticlesToConnectedAirwayParticlesFilter()
{
 
}
 
int vtkCIPAirwayParticlesToConnectedAirwayParticlesFilter::RequestData(vtkInformation *vtkNotUsed(request),
                                                                       vtkInformationVector **inputVector,
                                                                       vtkInformationVector *outputVector)
{
  vtkInformation* inInfo  = inputVector[0]->GetInformationObject(0);
  vtkInformation* outInfo = outputVector->GetInformationObject(0);
 
  vtkSmartPointer< vtkPolyData > inputParticles  = vtkPolyData::SafeDownCast( inInfo->Get(vtkDataObject::DATA_OBJECT()) );
  vtkSmartPointer< vtkPolyData > outputParticles = vtkPolyData::SafeDownCast( outInfo->Get(vtkDataObject::DATA_OBJECT()) );
 
  unsigned int numberInputParticles = inputParticles->GetNumberOfPoints();

  this->NumberInputParticles    = inputParticles->GetNumberOfPoints();
  this->NumberOfPointDataArrays = inputParticles->GetPointData()->GetNumberOfArrays();

  //
  // Initialize the data structure image that will make accessing a
  // particle's neighbors more efficient, and also initialize the
  // corresponding internal poly data that will be used throughout the
  // rest of this filter
  //
  // std::cout << "Initializing..." << std::endl;
  // this->InitializeDataStructureImageAndInternalInputPolyData( inputParticles );


  //
  // Now create the weighted graph that will be passed to the minimum 
  // spanning tree filter
  // 
  std::map< unsigned int, unsigned int > particleIDToNodeIDMap;
  std::map< unsigned int, unsigned int > nodeIDToParticleIDMap;

  vtkSmartPointer< vtkMutableUndirectedGraph > weightedGraph =  
    vtkSmartPointer< vtkMutableUndirectedGraph >::New();

  for ( unsigned int i=0; i<this->NumberInputParticles; i++ )
    {
      vtkIdType nodeID = weightedGraph->AddVertex();

      particleIDToNodeIDMap[i]      = nodeID;
      nodeIDToParticleIDMap[nodeID] = i;
    }

  vtkSmartPointer< vtkDoubleArray > edgeWeights = vtkSmartPointer<vtkDoubleArray>::New();
    edgeWeights->SetNumberOfComponents( 1 );
    edgeWeights->SetName( "Weights" );

  for ( unsigned int i=0; i<this->NumberInputParticles; i++ )
    {
      for ( unsigned int j=i+1; j<this->NumberInputParticles; j++ )
	{
	  double weight;
	  
	  if ( GetEdgeWeight( i, j, inputParticles, &weight ) )
	    {
	      weightedGraph->AddEdge( particleIDToNodeIDMap[i], particleIDToNodeIDMap[j] );
	      edgeWeights->InsertNextValue( weight );
	    }
	}
    }

  weightedGraph->GetEdgeData()->AddArray( edgeWeights );

  std::cout << "Computing minimum spanning tree..." << std::endl;
  vtkSmartPointer< vtkBoostKruskalMinimumSpanningTree > minimumSpanningTreeFilter = 
    vtkSmartPointer< vtkBoostKruskalMinimumSpanningTree >::New();
    minimumSpanningTreeFilter->SetInputData( weightedGraph );
    minimumSpanningTreeFilter->SetEdgeWeightArrayName( "Weights" );
    minimumSpanningTreeFilter->Update();

  //------------------------------------
  // std::cout << "Querying..." << std::endl;
  // for ( unsigned int x=0; x<this->DataImageSize[0]; x++ )
  //   {
  //   for ( unsigned int y=0; y<this->DataImageSize[1]; y++ )
  //     {
  //     for ( unsigned int z=0; z<this->DataImageSize[2]; z++ )
  //       {
  //       unsigned int* pixel = static_cast< unsigned int *>(this->DataStructureImage->GetScalarPointer(x,y,z));

  //       if ( pixel[0] != 0 )
  //         {
  //         //
  //         // We have found a particle that has not yet been added to
  //         // any subgraph. Create a unique subgraph to contain this
  //         // particle and its connected neighbors. We'll also need to
  //         // keep track of the mapping between the sub graphs node IDs
  //         // and the particle IDs, so create a map to do this. We'll
  //         // also want to associate the particles' spatial coordinates
  //         // with the sub-graph nodes, so create a points
  //         // container. Note that the call to 'QueryNeighborhood' is
  //         // recursive, and it zeros out DataStructureImage voxel
  //         // coordinates as it finds connected particles. So if we're
  //         // at this point in the loop, we can be guaranteed that the
  //         // particle corresponding to the voxel location being fed to
  //         // 'QueryNeighborhood' has not yet been touched.
  //         //
  //         vtkSmartPointer< vtkMutableDirectedGraph > subGraph = vtkSmartPointer< vtkMutableDirectedGraph >::New();
          
  //         std::map< vtkIdType, unsigned int > nodeIDToParticleIDMap;
          
  //         //
  //         // Get the ID of the particle at the data image's voxel location,
  //         // then set the voxel value to zero indicating that this particle
  //         // has been visited.
  //         //
  //         unsigned int* pixel = static_cast< unsigned int *>(this->DataStructureImage->GetScalarPointer( x, y, z));
  //         unsigned int  particleID = pixel[0]-1;
  //         pixel[0] = 0;
          
  //         //
  //         // Add a vertex to the subgraph, and map the graph node ID to the
  //         // particle ID. Also add the particle's spatial location to the
  //         // points containter
  //         //
  //         vtkIdType nodeID = subGraph->AddVertex();
  //         nodeIDToParticleIDMap[nodeID] = particleID;
          
  //         this->QueryNeighborhood( nodeID, particleID, subGraph, &nodeIDToParticleIDMap, x, y, z );

  //         if ( static_cast< unsigned int >( subGraph->GetNumberOfVertices() ) >= this->SegmentCardinalityThreshold )
  //           {
  //           this->SubGraphsVec.push_back( subGraph );
  //           this->SubGraphNodeIDToParticleIDMapVec.push_back( nodeIDToParticleIDMap );

  //           for ( unsigned int i=0; i<static_cast< unsigned int >( subGraph->GetNumberOfVertices() ); i++ )
  //             {
  //             unsigned int particleID = nodeIDToParticleIDMap[i];
  //             this->ParticleIDToSubGraphMap[particleID] = this->SubGraphsVec.size()-1;

  //             if ( subGraph->GetDegree(i) == 1 )
  //               {
  //               this->SubGraphLeafParticleIDs.push_back( particleID );
  //               }
  //             }
  //           }
  //         }
  //       }
  //     }
  //   }


  // //
  // // Now create the output poly data
  // //
  // std::vector< vtkFloatArray* > arrayVec;

  // for ( unsigned int i=0; i<this->NumberOfPointDataArrays; i++ )
  //   {
  //   vtkSmartPointer< vtkFloatArray > array = vtkSmartPointer< vtkFloatArray >::New();
  //     array->SetNumberOfComponents( this->InternalInputPolyData->GetPointData()->GetArray(i)->GetNumberOfComponents() );
  //     array->SetName( this->InternalInputPolyData->GetPointData()->GetArray(i)->GetName() );

  //   arrayVec.push_back( array );
  //   }

  // vtkPoints* outputPoints  = vtkPoints::New();  

  // unsigned int inc = 0;
  // for ( unsigned int i=0; i<static_cast< unsigned int >( this->CompositeGraph->GetNumberOfVertices() ); i++ )
  //   {
  //   unsigned int particleID = this->CompositeGraphNodeIDToParticleIDMap[i];

  //   outputPoints->InsertNextPoint( this->InternalInputPolyData->GetPoint(particleID) );
        
  //   for ( unsigned int k=0; k<this->NumberOfPointDataArrays; k++ )
  //     {
  //     arrayVec[k]->InsertTuple( inc, this->InternalInputPolyData->GetPointData()->GetArray(k)->GetTuple(particleID) );
  //     }
    
  //   inc++;
  //   }

  // this->CompositeGraph->SetPoints( outputPoints );

  // vtkSmartPointer< vtkGraphToPolyData > compositeToPolyData = vtkSmartPointer<vtkGraphToPolyData>::New();
  //   compositeToPolyData->SetInput( this->CompositeGraph );
  //   compositeToPolyData->Update();

  // for ( unsigned int j=0; j<this->NumberOfPointDataArrays; j++ )
  //   {
  //   compositeToPolyData->GetOutput()->GetPointData()->AddArray( arrayVec[j] );
  //   }

  // outputParticles->ShallowCopy( compositeToPolyData->GetOutput() );

  return 1;
}
 

// int vtkCIPAirwayParticlesToConnectedAirwayParticlesFilter::RequestData(vtkInformation *vtkNotUsed(request),
//                                                                        vtkInformationVector **inputVector,
//                                                                        vtkInformationVector *outputVector)
// {
//   vtkInformation* inInfo  = inputVector[0]->GetInformationObject(0);
//   vtkInformation* outInfo = outputVector->GetInformationObject(0);
 
//   vtkPolyData* inputParticles  = vtkPolyData::SafeDownCast( inInfo->Get(vtkDataObject::DATA_OBJECT()) );
//   vtkPolyData* outputParticles = vtkPolyData::SafeDownCast( outInfo->Get(vtkDataObject::DATA_OBJECT()) );
 
//   unsigned int numberInputParticles = inputParticles->GetNumberOfPoints();

//   this->NumberInputParticles    = inputParticles->GetNumberOfPoints();
//   this->NumberOfPointDataArrays = inputParticles->GetPointData()->GetNumberOfArrays();

//   //
//   // Initialize the data structure image that will make accessing a
//   // particle's neighbors more efficient, and also initialize the
//   // corresponding internal poly data that will be used throughout the
//   // rest of this filter
//   //
//   std::cout << "Initializing..." << std::endl;
//   this->InitializeDataStructureImageAndInternalInputPolyData( inputParticles );

//   //this->DataExplorer();
  
//   std::cout << "Querying..." << std::endl;
//   for ( unsigned int x=0; x<this->DataImageSize[0]; x++ )
//     {
//     for ( unsigned int y=0; y<this->DataImageSize[1]; y++ )
//       {
//       for ( unsigned int z=0; z<this->DataImageSize[2]; z++ )
//         {
//         unsigned int* pixel = static_cast< unsigned int *>(this->DataStructureImage->GetScalarPointer(x,y,z));

//         if ( pixel[0] != 0 )
//           {
//           //
//           // We have found a particle that has not yet been added to
//           // any subgraph. Create a unique subgraph to contain this
//           // particle and its connected neighbors. We'll also need to
//           // keep track of the mapping between the sub graphs node IDs
//           // and the particle IDs, so create a map to do this. We'll
//           // also want to associate the particles' spatial coordinates
//           // with the sub-graph nodes, so create a points
//           // container. Note that the call to 'QueryNeighborhood' is
//           // recursive, and it zeros out DataStructureImage voxel
//           // coordinates as it finds connected particles. So if we're
//           // at this point in the loop, we can be guaranteed that the
//           // particle corresponding to the voxel location being fed to
//           // 'QueryNeighborhood' has not yet been touched.
//           //
//           vtkSmartPointer< vtkMutableDirectedGraph > subGraph = vtkSmartPointer< vtkMutableDirectedGraph >::New();
          
//           std::map< vtkIdType, unsigned int > nodeIDToParticleIDMap;
          
//           //
//           // Get the ID of the particle at the data image's voxel location,
//           // then set the voxel value to zero indicating that this particle
//           // has been visited.
//           //
//           unsigned int* pixel = static_cast< unsigned int *>(this->DataStructureImage->GetScalarPointer( x, y, z));
//           unsigned int  particleID = pixel[0]-1;
//           pixel[0] = 0;
          
//           //
//           // Add a vertex to the subgraph, and map the graph node ID to the
//           // particle ID. Also add the particle's spatial location to the
//           // points containter
//           //
//           vtkIdType nodeID = subGraph->AddVertex();
//           nodeIDToParticleIDMap[nodeID] = particleID;
          
//           this->QueryNeighborhood( nodeID, particleID, subGraph, &nodeIDToParticleIDMap, x, y, z );

//           if ( static_cast< unsigned int >( subGraph->GetNumberOfVertices() ) >= this->SegmentCardinalityThreshold )
//             {
//             this->SubGraphsVec.push_back( subGraph );
//             this->SubGraphNodeIDToParticleIDMapVec.push_back( nodeIDToParticleIDMap );

//             for ( unsigned int i=0; i<static_cast< unsigned int >( subGraph->GetNumberOfVertices() ); i++ )
//               {
//               unsigned int particleID = nodeIDToParticleIDMap[i];
//               this->ParticleIDToSubGraphMap[particleID] = this->SubGraphsVec.size()-1;

//               if ( subGraph->GetDegree(i) == 1 )
//                 {
//                 this->SubGraphLeafParticleIDs.push_back( particleID );
//                 }
//               }
//             }
//           }
//         }
//       }
//     }

//   //
//   // For every leaf node, determine the nearest particle that it
//   // connects to (provided that it actually forms a connection)
//   //
//   std::cout << "this->SubGraphLeafParticleIDs.size():\t" << this->SubGraphLeafParticleIDs.size() << std::endl;
//   for ( unsigned int i=0; i<this->SubGraphLeafParticleIDs.size(); i++ )
//     {
//     unsigned int leafParticleID  = this->SubGraphLeafParticleIDs[i];
//     unsigned int currentSubGraph = this->ParticleIDToSubGraphMap[leafParticleID];

//     double leafPoint[3];
//       leafPoint[0] = this->InternalInputPolyData->GetPoint( leafParticleID )[0];
//       leafPoint[1] = this->InternalInputPolyData->GetPoint( leafParticleID )[1];
//       leafPoint[2] = this->InternalInputPolyData->GetPoint( leafParticleID )[2];

//     double leafDirection[3];
//     this->GetLeafNodeDirection( leafParticleID, leafDirection );

//     bool   connectionFound  = false;
//     double smallestDistance = DBL_MAX;
//     unsigned int connectingParticleID = 0;

//     for ( unsigned int g=0; g<this->SubGraphsVec.size(); g++ )
//       {
//       if ( g != currentSubGraph )
//         {
//         for ( unsigned int n=0; n<static_cast< unsigned int >( this->SubGraphsVec[g]->GetNumberOfVertices() ); n++ )
//           {
//           unsigned int subGraphParticleID = this->SubGraphNodeIDToParticleIDMapVec[g][n];

//           double nodePoint[3];
//             nodePoint[0] = this->InternalInputPolyData->GetPoint( subGraphParticleID )[0];
//             nodePoint[1] = this->InternalInputPolyData->GetPoint( subGraphParticleID )[1];
//             nodePoint[2] = this->InternalInputPolyData->GetPoint( subGraphParticleID )[2];

//           double connectingVec[3];
//             connectingVec[0] = leafPoint[0] - nodePoint[0];
//             connectingVec[1] = leafPoint[1] - nodePoint[1];
//             connectingVec[2] = leafPoint[2] - nodePoint[2];

//           double connectorMagnitude = this->GetVectorMagnitude( connectingVec );

//           if ( connectorMagnitude <= this->SegmentDistanceThreshold )
//             {
//             if ( this->GetAngleBetweenVectors( leafDirection, connectingVec, true ) <= this->SegmentAngleThreshold ) 
//               {
//               if ( connectorMagnitude < smallestDistance )
//                 {
//                 connectionFound = true;
//                 smallestDistance = connectorMagnitude;
//                 connectingParticleID = subGraphParticleID;
//                 }
//               }
//             }
//           }
//         }
//       }
    
//     if ( connectionFound )
//       {
//       //
//       // A connection between the current subGraph and some other
//       // graph has been found. However, we only want to form a
//       // connection provided that the smallest distance between the
//       // connecting nodes is smaller than the smallest distance
//       // between any two leaf nodes of the two graphs to merge. We
//       // make the assumption that if the smallest distance between two
//       // graphs is between leaf nodes, the connection between the two
//       // graphs should be there and nowhere else.
//       //
//       unsigned int connectingSubGraph = this->ParticleIDToSubGraphMap[connectingParticleID];

//       if ( this->GetMinLeafNodeDistanceBetweenGraphs( connectingSubGraph, currentSubGraph ) >= smallestDistance )
//         {
//         //
//         // We also don't want to form a link if doing so would form a
//         // cycle
//         //
//         if ( !this->EvaluateGraphConnectedness(this->ParticleIDToSubGraphMap[leafParticleID], this->ParticleIDToSubGraphMap[connectingParticleID]) &&
//              !this->EvaluateGraphConnectedness(this->ParticleIDToSubGraphMap[connectingParticleID], this->ParticleIDToSubGraphMap[leafParticleID]) )
//           {
//           this->ParticleIDToConnectingParticleIDMap[leafParticleID] = connectingParticleID;
//           }
//         }
//       }
//     }    

//   //
//   // At this stage we have created subgraphs representing groups of
//   // connected particles. We now attempt to merge subgraphs into
//   // larger subgraphs based on connection criterion between
//   // subgraphs. To differentiate these larger subgraphs from the
//   // previously created subgraphs, we will use the designation
//   // 'subTree'. Begin by creating a map from the subGraph IDs to
//   // boolean values to indicate whether or not the sub graph has been
//   // added to a sub tree or not.
//   //
//   std::map< unsigned int, bool > subGraphToAddedToSubTreeMap;
//   for ( unsigned int i=0; i<this->SubGraphsVec.size(); i++ )
//     {
//     subGraphToAddedToSubTreeMap[i] = false;
//     }

//   //
//   // Now merge the connected subGraphs into subTrees 
//   //
//   std::cout << "Merging..." << std::endl;
//   for ( unsigned int i=0; i<this->SubGraphsVec.size(); i++ )
//     {
//     if ( subGraphToAddedToSubTreeMap[i] == false )
//       {
//       vtkSmartPointer< vtkMutableDirectedGraph > subTree = vtkSmartPointer< vtkMutableDirectedGraph >::New();
//       std::map< vtkIdType, unsigned int > nodeIDToParticleIDMap;

//       this->AddSubGraphToSubTree( i, subTree, &nodeIDToParticleIDMap );
//       subGraphToAddedToSubTreeMap[i] = true;

//       for ( unsigned int j=0; j<this->SubGraphsVec.size(); j++ )
//         {
//         subGraphToAddedToSubTreeMap[j] = this->AttemptAddSubGraphToSubTree( j, subTree, &nodeIDToParticleIDMap );
//         }      

//       this->SubTreeNodeIDToParticleIDMapVec.push_back( nodeIDToParticleIDMap );
//       this->SubTreesVec.push_back( subTree );
//       }
    
//     }

//   std::cout << "this->SubGraphsVec.size():\t" << this->SubGraphsVec.size() << std::endl;
// //  this->ViewGraphs( this->SubGraphsVec, this->SubGraphNodeIDToParticleIDMapVec, this->SegmentCardinalityThreshold );
//   this->ViewGraphs( this->SubTreesVec, this->SubTreeNodeIDToParticleIDMapVec, this->TreeCardinalityThreshold );

//   this->FillCompositeGraphWithSubTrees();

//   //
//   // Now create the output poly data
//   //
//   std::cout << "Creating output..." << std::endl;  

//   std::vector< vtkFloatArray* > arrayVec;

//   for ( unsigned int i=0; i<this->NumberOfPointDataArrays; i++ )
//     {
//     vtkFloatArray* array = vtkFloatArray::New();
//       array->SetNumberOfComponents( this->InternalInputPolyData->GetPointData()->GetArray(i)->GetNumberOfComponents() );
//       array->SetName( this->InternalInputPolyData->GetPointData()->GetArray(i)->GetName() );

//     arrayVec.push_back( array );
//     }

//   vtkPoints* outputPoints  = vtkPoints::New();  

//   unsigned int inc = 0;
//   for ( unsigned int i=0; i<static_cast< unsigned int >( this->CompositeGraph->GetNumberOfVertices() ); i++ )
//     {
//     unsigned int particleID = this->CompositeGraphNodeIDToParticleIDMap[i];

//     outputPoints->InsertNextPoint( this->InternalInputPolyData->GetPoint(particleID) );
        
//     for ( unsigned int k=0; k<this->NumberOfPointDataArrays; k++ )
//       {
//       arrayVec[k]->InsertTuple( inc, this->InternalInputPolyData->GetPointData()->GetArray(k)->GetTuple(particleID) );
//       }
    
//     inc++;
//     }

//   this->CompositeGraph->SetPoints( outputPoints );

//   vtkSmartPointer< vtkGraphToPolyData > compositeToPolyData = vtkSmartPointer<vtkGraphToPolyData>::New();
//     compositeToPolyData->SetInput( this->CompositeGraph );
//     compositeToPolyData->Update();

//   for ( unsigned int j=0; j<this->NumberOfPointDataArrays; j++ )
//     {
//     compositeToPolyData->GetOutput()->GetPointData()->AddArray( arrayVec[j] );
//     }

//   outputParticles->ShallowCopy( compositeToPolyData->GetOutput() );


//   //
//   // Visualize the final product
//   //
// //   vtkSmartPointer< vtkGraphToPolyData > graphToPolyData = vtkSmartPointer<vtkGraphToPolyData>::New();
// //   graphToPolyData->SetInput( this->CompositeGraph );
// //   graphToPolyData->Update();
      
// //   vtkSmartPointer< vtkPolyDataMapper > mapper = vtkSmartPointer< vtkPolyDataMapper >::New();
// //   mapper->SetInputConnection( graphToPolyData->GetOutputPort() );
 
// //   vtkSmartPointer< vtkActor > actor = vtkSmartPointer< vtkActor >::New();
// //   actor->SetMapper(mapper);
        
// //   vtkSmartPointer< vtkRenderer > renderer = vtkSmartPointer< vtkRenderer >::New();
// //   renderer->SetBackground( 0, 0, 0); 
// //   renderer->AddActor(actor);

// //   vtkSmartPointer< vtkRenderWindow > renderWindow = vtkSmartPointer< vtkRenderWindow >::New();
// //     renderWindow->AddRenderer(renderer);

// //   vtkSmartPointer< vtkInteractorStyleTrackballCamera > trackball = vtkSmartPointer< vtkInteractorStyleTrackballCamera >::New();

// //   vtkSmartPointer< vtkRenderWindowInteractor > renderWindowInteractor = vtkSmartPointer< vtkRenderWindowInteractor >::New();
// //     renderWindowInteractor->SetRenderWindow(renderWindow);
// //     renderWindowInteractor->SetInteractorStyle( trackball );

// //   renderWindow->Render();
// //   renderWindowInteractor->Start();

//   return 1;
// }
 

bool vtkCIPAirwayParticlesToConnectedAirwayParticlesFilter::GetEdgeWeight( unsigned int particleID1, unsigned int particleID2, 
									   vtkSmartPointer< vtkPolyData > particles, double* weight )
{
  //
  // Determine the vector connecting the two particles
  //
  double point1[3];
    point1[0] = particles->GetPoint( particleID1 )[0];
    point1[1] = particles->GetPoint( particleID1 )[1];
    point1[2] = particles->GetPoint( particleID1 )[2];

  double point2[3];
    point2[0] = particles->GetPoint( particleID2 )[0];
    point2[1] = particles->GetPoint( particleID2 )[1];
    point2[2] = particles->GetPoint( particleID2 )[2];

  double connectingVec[3];
    connectingVec[0] = point1[0] - point2[0];
    connectingVec[1] = point1[1] - point2[1];
    connectingVec[2] = point1[2] - point2[2];

  double connectorMagnitude = this->GetVectorMagnitude( connectingVec );

  if ( connectorMagnitude > this->ParticleDistanceThreshold )
    {
    return false;
    }

  double particle1Hevec2[3];
    particle1Hevec2[0] = particles->GetPointData()->GetArray( "hevec2" )->GetTuple( particleID1 )[0];
    particle1Hevec2[1] = particles->GetPointData()->GetArray( "hevec2" )->GetTuple( particleID1 )[1];
    particle1Hevec2[2] = particles->GetPointData()->GetArray( "hevec2" )->GetTuple( particleID1 )[2];

  double particle2Hevec2[3];
    particle2Hevec2[0] = particles->GetPointData()->GetArray( "hevec2" )->GetTuple( particleID2 )[0];
    particle2Hevec2[1] = particles->GetPointData()->GetArray( "hevec2" )->GetTuple( particleID2 )[1];
    particle2Hevec2[2] = particles->GetPointData()->GetArray( "hevec2" )->GetTuple( particleID2 )[2];

  double angle1 =  this->GetAngleBetweenVectors( particle1Hevec2, connectingVec, true );
  double angle2 =  this->GetAngleBetweenVectors( particle2Hevec2, connectingVec, true );

  if ( angle1 < angle2 )
    {
      *weight = connectorMagnitude*(1.0 + exp(-pow( (90.0 - angle1)/this->EdgeWeightAngleSigma, 2 )));
    }
  else
    {
      *weight = connectorMagnitude*(1.0 + exp(-pow( (90.0 - angle2)/this->EdgeWeightAngleSigma, 2 )));
    }

  return true;
}

//
// This method initializes the data structure image. Each voxel in the
// image will contain the index to one of the particles in input
// particles poly data. An image is used for this purpose to more
// efficiently access a selected particle's neighbors. Note that this
// method also creates an internal version of the input particles poly
// data. This is to accomodate the fact that multiple particls that
// happen to be very close together may get mapped to the same voxel
// location in the image data structure. If multiple particles are
// mapped to the same voxel location, only one of the particles is
// retained (arbitrarily, the one most recently added). The internal
// poly data is used throughout the rest of this filter after it is
// initialized here. 
//
// void vtkCIPAirwayParticlesToConnectedAirwayParticlesFilter::InitializeDataStructureImageAndInternalInputPolyData( vtkPolyData* particles )
// {
//   //
//   // First determine the min and spatial particle coordinates so we
//   // can determine the extent of the image
//   //
//   double xMin = DBL_MAX;
//   double yMin = DBL_MAX;
//   double zMin = DBL_MAX;

//   double xMax = -DBL_MAX;
//   double yMax = -DBL_MAX;
//   double zMax = -DBL_MAX;

//   for ( int i=0; i<particles->GetNumberOfPoints(); i++ )
//     {
//     if ( (particles->GetPoint(i))[0] > xMax )
//       {
//       xMax = (particles->GetPoint(i))[0];
//       }
//     if ( (particles->GetPoint(i))[1] > yMax )
//       {
//       yMax = (particles->GetPoint(i))[1];
//       }
//     if ( (particles->GetPoint(i))[2] > zMax )
//       {
//       zMax = (particles->GetPoint(i))[2];
//       }

//     if ( (particles->GetPoint(i))[0] < xMin )
//       {
//       xMin = (particles->GetPoint(i))[0];
//       }
//     if ( (particles->GetPoint(i))[1] < yMin )
//       {
//       yMin = (particles->GetPoint(i))[1];
//       }
//     if ( (particles->GetPoint(i))[2] < zMin )
//       {
//       zMin = (particles->GetPoint(i))[2];
//       }
//     }

//   this->DataImageSpacing = this->InterParticleDistance/2.0;

//   this->DataImageSize[0] = static_cast< unsigned int >( ceil( (xMax-xMin)/this->DataImageSpacing ) ) + 1;
//   this->DataImageSize[1] = static_cast< unsigned int >( ceil( (yMax-yMin)/this->DataImageSpacing ) ) + 1;
//   this->DataImageSize[2] = static_cast< unsigned int >( ceil( (zMax-zMin)/this->DataImageSpacing ) ) + 1;

//   this->DataStructureImage->SetOrigin( xMin, yMin, zMin );
//   this->DataStructureImage->SetSpacing( this->DataImageSpacing, this->DataImageSpacing, this->DataImageSpacing );
//   this->DataStructureImage->SetDimensions( this->DataImageSize[0], this->DataImageSize[1], this->DataImageSize[2] );
//   this->DataStructureImage->SetNumberOfScalarComponents( 1 );
//   this->DataStructureImage->SetScalarTypeToUnsignedInt();
  
//   //
//   // Now that the extent of the image has been established, initialize
//   // all voxels values to zero
//   //
//   for ( unsigned int x=0; x<this->DataImageSize[0]; x++ )
//     {
//     for ( unsigned int y=0; y<this->DataImageSize[1]; y++ )
//       {
//       for ( unsigned int z=0; z<this->DataImageSize[2]; z++ )
//         {
//         unsigned int* pixel = static_cast< unsigned int *>(this->DataStructureImage->GetScalarPointer(x,y,z));
//         pixel[0] = 0;
//         }
//       }
//     }

//   //
//   // Now fill the data structure image with the particle IDs. Note
//   // that it is permissible for a particle ID to overwrite another ID
//   // that was previously set in the same voxel location
//   //
//   double point[3];
//   int    index[3];
//   for ( int i=0; i<particles->GetNumberOfPoints(); i++ )
//     {      
//     point[0] = particles->GetPoint(i)[0];
//     point[1] = particles->GetPoint(i)[1];
//     point[2] = particles->GetPoint(i)[2];

//     index[0] = static_cast< int >( (point[0]-xMin)/this->DataImageSpacing );
//     index[1] = static_cast< int >( (point[1]-yMin)/this->DataImageSpacing );
//     index[2] = static_cast< int >( (point[2]-zMin)/this->DataImageSpacing );

//     unsigned int* pixel = static_cast< unsigned int* >(this->DataStructureImage->GetScalarPointer(index[0], index[1], index[2]));
//     pixel[0] = i+1;
//     }

//   //
//   // Now that the data structure image has been created, we can fill
//   // the internal input poly data to be used throughout the rest of
//   // the filter. The need for doing this is that only a subset of the
//   // input particles are actually registered in the data structure
//   // image (given that some particles overwrite old particles as the
//   // image is filled). So we need our m_InternalInputPolyData to
//   // refer to those particles that remain.
//   //
//   vtkPoints* points  = vtkPoints::New();

//   std::vector< vtkFloatArray* > arrayVec;

//   for ( unsigned int i=0; i<this->NumberOfPointDataArrays; i++ )
//     {
//     vtkFloatArray* array = vtkFloatArray::New();
//       array->SetNumberOfComponents( particles->GetPointData()->GetArray(i)->GetNumberOfComponents() );
//       array->SetName( particles->GetPointData()->GetArray(i)->GetName() );

//     arrayVec.push_back( array );
//     }

//   unsigned int inc = 0;

//   for ( unsigned int x=0; x<this->DataImageSize[0]; x++ )
//     {
//     for ( unsigned int y=0; y<this->DataImageSize[1]; y++ )
//       {
//       for ( unsigned int z=0; z<this->DataImageSize[2]; z++ )
//         {
//         unsigned int* pixel = static_cast< unsigned int* >(this->DataStructureImage->GetScalarPointer( x, y, z ));

//         if ( pixel[0] != 0 )
//           {
//           unsigned int i = pixel[0]-1;

//           points->InsertNextPoint( particles->GetPoint(i) );

//           for ( unsigned int j=0; j<this->NumberOfPointDataArrays; j++ )
//             {
//             arrayVec[j]->InsertTuple( inc, particles->GetPointData()->GetArray(j)->GetTuple(i) );
//             }

//           inc++;    
//           pixel[0] = inc; // Ensures that image's voxel value points to new
//                           // particle structure, not the old one.
//           }
//         }
//       }
//     }

//   this->NumberInternalInputParticles = inc;
//   this->InternalInputPolyData->SetPoints( points );

//   for ( unsigned int j=0; j<this->NumberOfPointDataArrays; j++ )
//     {
//     this->InternalInputPolyData->GetPointData()->AddArray( arrayVec[j] ); 
//     }
// }


// void vtkCIPAirwayParticlesToConnectedAirwayParticlesFilter::QueryNeighborhood( vtkIdType inNodeID, unsigned int inParticleID,
//                                                                                vtkSmartPointer< vtkMutableDirectedGraph > subGraph, 
//                                                                                std::map< vtkIdType, unsigned int >* nodeIDToParticleIDMap,
//                                                                                unsigned int x, unsigned int y, unsigned int z )
// {
//   //
//   // Define the neighborhood search radius to consider. It needs to be
//   // large enough to consider all particles that are within
//   // 'ParticleDistanceThreshold' away.
//   //
//   int index[3];

//   int searchRadius = static_cast< int >( ceil( this->ParticleDistanceThreshold/this->DataImageSpacing ) );

//   for ( int i=-searchRadius; i<=searchRadius; i++ )
//     {
//     index[0] = static_cast< int >(x) + i;

//     for ( int j=-searchRadius; j<=searchRadius; j++ )
//       {
//       index[1] = static_cast< int >(y) + j;

//       for ( int k=-searchRadius; k<=searchRadius; k++ )
//         {
//         index[2] = static_cast< int >(z) + k;
        
//         if ( index[0] >= 0 && index[0] < static_cast< int >( this->DataImageSize[0] ) &&
//              index[1] >= 0 && index[1] < static_cast< int >( this->DataImageSize[1] ) &&
//              index[2] >= 0 && index[2] < static_cast< int >( this->DataImageSize[2] ) )
//           {
//           unsigned int* pixel = static_cast< unsigned int *>(this->DataStructureImage->GetScalarPointer(index[0],index[1],index[2]));

//           if ( pixel[0] != 0 )
//             {
//             unsigned int neighborParticleID = pixel[0] - 1;

//             bool connected = this->EvaluateParticleConnectedness( inParticleID, neighborParticleID );

//             if ( connected )
//               {
//               //
//               // Get the ID of the particle at the data image's voxel location,
//               // then set the voxel value to zero indicating that this particle
//               // has been visited.
//               //
//               unsigned int* pixel = static_cast< unsigned int *>(this->DataStructureImage->GetScalarPointer(index[0],index[1],index[2]));
//               unsigned int neighborParticleID = pixel[0]-1;
//               pixel[0] = 0;

//               //
//               // Add a vertex to the subgraph, and map the graph node ID to the
//               // particle ID. 
//               //
//               vtkIdType neighborNodeID = subGraph->AddVertex();
//               (*nodeIDToParticleIDMap)[neighborNodeID] = neighborParticleID;
  
//               //
//               // The nodes are connected, so create a link between
//               // them.
//               //
//               subGraph->AddEdge( neighborNodeID, inNodeID );
              
//               //
//               // And now query the neighborhood of the new particle
//               //
//               this->QueryNeighborhood( neighborNodeID, neighborParticleID, subGraph, nodeIDToParticleIDMap, 
//                                        index[0], index[1], index[2] );
//              }
//             }
//           }
//         }
//       }
//     }
// }


// bool vtkCIPAirwayParticlesToConnectedAirwayParticlesFilter::EvaluateGraphConnectedness( unsigned int subGraphID1, unsigned int subGraphID2 )
// {
//   bool connected = false;

//   std::map< unsigned int, unsigned int >::iterator it = this->ParticleIDToConnectingParticleIDMap.begin();

//   while ( it != this->ParticleIDToConnectingParticleIDMap.end() )
//     {
//     if ( this->ParticleIDToSubGraphMap[it->first] == subGraphID1 )
//       {
//       if ( this->ParticleIDToSubGraphMap[it->second] == subGraphID2 )
//         {
//         return true;
//         }
//       else if ( this->ParticleIDToConnectingParticleIDMap.find(it->second) != this->ParticleIDToConnectingParticleIDMap.end() )
//         {
//         connected = this->EvaluateGraphConnectedness( this->ParticleIDToSubGraphMap[it->second], subGraphID2 );
//         }
//       }

//     ++it;
//     }
         
//   return connected;
// }


// bool vtkCIPAirwayParticlesToConnectedAirwayParticlesFilter::TestForCycle( unsigned int particleID1, unsigned int particleID2 )
// {
//   unsigned int graph1Index = this->ParticleIDToSubGraphMap[particleID1];
//   unsigned int graph2Index = this->ParticleIDToSubGraphMap[particleID2];

//   unsigned int id 

//   if ( this->ParticleIDToConnectingParticleIDMap.find(particleID2) != this->ParticleIDToConnectingParticleIDMap.end() ) 
//     {

//     }
// }


// bool vtkCIPAirwayParticlesToConnectedAirwayParticlesFilter::EvaluateParticleConnectedness( unsigned int particleIndex1, 
//                                                                                            unsigned int particleIndex2 )
// {
//   //
//   // Determine the vector connecting the two particles
//   //
//   double point1[3];
//     point1[0] = this->InternalInputPolyData->GetPoint( particleIndex1 )[0];
//     point1[1] = this->InternalInputPolyData->GetPoint( particleIndex1 )[1];
//     point1[2] = this->InternalInputPolyData->GetPoint( particleIndex1 )[2];

//   double point2[3];
//     point2[0] = this->InternalInputPolyData->GetPoint( particleIndex2 )[0];
//     point2[1] = this->InternalInputPolyData->GetPoint( particleIndex2 )[1];
//     point2[2] = this->InternalInputPolyData->GetPoint( particleIndex2 )[2];

//   double connectingVec[3];
//     connectingVec[0] = point1[0] - point2[0];
//     connectingVec[1] = point1[1] - point2[1];
//     connectingVec[2] = point1[2] - point2[2];

//   double connectorMagnitude = this->GetVectorMagnitude( connectingVec );

//   if ( connectorMagnitude > this->ParticleDistanceThreshold )
//     {
//     return false;
//     }

//   double particle1Hevec2[3];
//     particle1Hevec2[0] = this->InternalInputPolyData->GetPointData()->GetArray( "hevec2" )->GetTuple( particleIndex1 )[0];
//     particle1Hevec2[1] = this->InternalInputPolyData->GetPointData()->GetArray( "hevec2" )->GetTuple( particleIndex1 )[1];
//     particle1Hevec2[2] = this->InternalInputPolyData->GetPointData()->GetArray( "hevec2" )->GetTuple( particleIndex1 )[2];

//   double particle2Hevec2[3];
//     particle2Hevec2[0] = this->InternalInputPolyData->GetPointData()->GetArray( "hevec2" )->GetTuple( particleIndex2 )[0];
//     particle2Hevec2[1] = this->InternalInputPolyData->GetPointData()->GetArray( "hevec2" )->GetTuple( particleIndex2 )[1];
//     particle2Hevec2[2] = this->InternalInputPolyData->GetPointData()->GetArray( "hevec2" )->GetTuple( particleIndex2 )[2];

//   if ( this->GetAngleBetweenVectors( particle1Hevec2, connectingVec, true ) > this->ParticleAngleThreshold )
//     {
//     return false;
//     }

//   if ( this->GetAngleBetweenVectors( particle2Hevec2, connectingVec, true ) > this->ParticleAngleThreshold )
//     {
//     return false;
//     }

//   return true;
// }


// bool vtkCIPAirwayParticlesToConnectedAirwayParticlesFilter::AttemptAddSubGraphToSubTree( unsigned int subGraphID, vtkSmartPointer< vtkMutableDirectedGraph > subTree,
//                                                                                          std::map< vtkIdType, unsigned int >* subTreeNodeIDToParticleIDMap )
// {
//   bool connected = false;

//   unsigned int subTreeParticleID, subGraphParticleID;
//   std::map< unsigned int, unsigned int >::iterator mapIt;

//   for ( unsigned int i=0; i<static_cast< unsigned int >( subTree->GetNumberOfVertices() ); i++ )
//     {
//     subTreeParticleID = (*subTreeNodeIDToParticleIDMap)[i];

//     if ( !connected && this->ParticleIDToConnectingParticleIDMap.find( subTreeParticleID ) != this->ParticleIDToConnectingParticleIDMap.end() )
//       {
//       //
//       // The tree connects to some subGraph. See if connects to this
//       // one
//       //
//       for ( unsigned int j=0; j<static_cast< unsigned int >( this->SubGraphsVec[subGraphID]->GetNumberOfVertices() ); j++ )
//         {
//         subGraphParticleID = this->SubGraphNodeIDToParticleIDMapVec[subGraphID][j];
        
//         if ( this->ParticleIDToConnectingParticleIDMap[subTreeParticleID] == subGraphParticleID )
//           {
//           //
//           // The subGraph connects to the subTree
//           //
//           connected = true;

//           this->AddSubGraphToSubTree( subGraphID, subTree, subTreeNodeIDToParticleIDMap, subGraphParticleID, subTreeParticleID );
//           }
//         }
//       }
//     }

//   return connected;
// }


// void vtkCIPAirwayParticlesToConnectedAirwayParticlesFilter::FillCompositeGraphWithSubTrees()
// {
//   for ( unsigned int t=0; t<this->SubTreesVec.size(); t++ )
//     {
//     std::map< vtkIdType, vtkIdType > subGraphNodeIDToSubTreeNodeIDMap;

//     if ( static_cast< unsigned int >( this->SubTreesVec[t]->GetNumberOfVertices() ) >= this->TreeCardinalityThreshold )
//       {
//       for ( unsigned int v=0; v<static_cast< unsigned int >( this->SubTreesVec[t]->GetNumberOfVertices() ); v++ )
//         {
//         vtkIdType compNodeID = this->CompositeGraph->AddVertex();
//         unsigned int particleID = this->SubTreeNodeIDToParticleIDMapVec[t][v];
//         this->CompositeGraphNodeIDToParticleIDMap[compNodeID] = particleID;
//         subGraphNodeIDToSubTreeNodeIDMap[v] = compNodeID;
//         }

//       //
//       // Now get the edges of the subGraph and create corresponding edges
//       // in the subTree
//       //
//       vtkSmartPointer< vtkEdgeListIterator > edgeIterator = vtkSmartPointer< vtkEdgeListIterator >::New();
      
//       this->SubTreesVec[t]->GetEdges( edgeIterator );

//       while ( edgeIterator->HasNext() )
//         {
//         vtkEdgeType edge = edgeIterator->Next();
//         vtkIdType subTreeSource = edge.Source;
//         vtkIdType subTreeTarget = edge.Target;
        
//         vtkIdType compSource = subGraphNodeIDToSubTreeNodeIDMap[subTreeSource];
//         vtkIdType compTarget = subGraphNodeIDToSubTreeNodeIDMap[subTreeTarget];
        
//         this->CompositeGraph->AddEdge( compSource, compTarget );
//         }
//       }
//     }
// }


// void vtkCIPAirwayParticlesToConnectedAirwayParticlesFilter::AddSubGraphToSubTree( unsigned int subGraphID, vtkSmartPointer< vtkMutableDirectedGraph > subTree,
//                                                                                   std::map< vtkIdType, unsigned int >* subTreeNodeIDToParticleIDMap )
// {
//   std::map< vtkIdType, vtkIdType > subGraphNodeIDToSubTreeNodeIDMap;

//   //
//   // First add all the subGraph nodes to subTree, and keep track of
//   // the mapping between the subTree nodes and the corresponding
//   // particle ID. Also keep a mapping of the subTree's node IDs and
//   // the subGraph's node IDs
//   //
//   for ( unsigned int i=0; i<static_cast< unsigned int >( this->SubGraphsVec[subGraphID]->GetNumberOfVertices() ); i++ )
//     {
//     vtkIdType treeNodeID = subTree->AddVertex();
//     unsigned int particleID = this->SubGraphNodeIDToParticleIDMapVec[subGraphID][i];
//     (*subTreeNodeIDToParticleIDMap)[treeNodeID] = particleID;
//     subGraphNodeIDToSubTreeNodeIDMap[i] = treeNodeID;
//     }

//   //
//   // Now get the edges of the subGraph and create corresponding edges
//   // in the subTree
//   //
//   vtkSmartPointer< vtkEdgeListIterator > edgeIterator = vtkSmartPointer< vtkEdgeListIterator >::New();

//   this->SubGraphsVec[subGraphID]->GetEdges( edgeIterator );

//   while ( edgeIterator->HasNext() )
//     {
//     vtkEdgeType edge = edgeIterator->Next();
//     vtkIdType subGraphSource = edge.Source;
//     vtkIdType subGraphTarget = edge.Target;

//     vtkIdType subTreeSource = subGraphNodeIDToSubTreeNodeIDMap[subGraphSource];
//     vtkIdType subTreeTarget = subGraphNodeIDToSubTreeNodeIDMap[subGraphTarget];

//     subTree->AddEdge( subTreeSource, subTreeTarget );
//     }
// }


// void vtkCIPAirwayParticlesToConnectedAirwayParticlesFilter::AddSubGraphToSubTree( unsigned int subGraphID, vtkSmartPointer< vtkMutableDirectedGraph > subTree,
//                                                                                   std::map< vtkIdType, unsigned int >*  subTreeNodeIDToParticleIDMap,
//                                                                                   unsigned int subGraphParticleID, unsigned int subTreeParticleID )
// {
//   vtkIdType fromNode, toNode;

//   this->AddSubGraphToSubTree( subGraphID, subTree, subTreeNodeIDToParticleIDMap );

//   for ( unsigned int i=0; i<static_cast< unsigned int >( subTree->GetNumberOfVertices() ); i++ )
//     {
//     if ( (*subTreeNodeIDToParticleIDMap)[i] == subGraphParticleID )
//       {
//       fromNode = i;
//       }
//     if ( (*subTreeNodeIDToParticleIDMap)[i] == subTreeParticleID )
//       {
//       toNode = i;
//       }
//     }

//   subTree->AddEdge( fromNode, toNode );
// }


double vtkCIPAirwayParticlesToConnectedAirwayParticlesFilter::GetVectorMagnitude( double vector[3] )
{
  double magnitude = sqrt( pow( vector[0], 2 ) + pow( vector[1], 2 ) + pow( vector[2], 2 ) );

  return magnitude;
}


double vtkCIPAirwayParticlesToConnectedAirwayParticlesFilter::GetAngleBetweenVectors( double vec1[3], double vec2[3], bool returnDegrees )
{
  double vec1Mag = this->GetVectorMagnitude( vec1 );
  double vec2Mag = this->GetVectorMagnitude( vec2 );

  double arg = (vec1[0]*vec2[0] + vec1[1]*vec2[1] + vec1[2]*vec2[2])/(vec1Mag*vec2Mag);

  if ( std::abs( arg ) > 1.0 )
    {
    arg = 1.0;
    }

  double angle = acos( arg );

  if ( !returnDegrees )
    {
    return angle;
    }

  double angleInDegrees = (180.0/3.14159265358979323846)*angle;

  if ( angleInDegrees > 90.0 )
    {
    angleInDegrees = 180.0 - angleInDegrees;
    }

  return angleInDegrees;
}


// double vtkCIPAirwayParticlesToConnectedAirwayParticlesFilter::GetMinLeafNodeDistanceBetweenGraphs( unsigned int graphIndex1, 
//                                                                                                    unsigned int graphIndex2 )
// {
//   double minDistance = DBL_MAX;

//   unsigned int particleID1, particleID2;

//   double point1[3];
//   double point2[3];
//   double connectingVec[3];

//   double connectorMagnitude;

//   for ( unsigned int i=0; i<static_cast< unsigned int >( this->SubGraphsVec[graphIndex1]->GetNumberOfVertices() ); i++ )
//     {
//     if ( this->SubGraphsVec[graphIndex1]->GetDegree(i) == 1 )
//       {
//       particleID1 = this->SubGraphNodeIDToParticleIDMapVec[graphIndex1][i];

//       for ( unsigned int j=0; j<static_cast< unsigned int >( this->SubGraphsVec[graphIndex2]->GetNumberOfVertices() ); j++ )
//         {
//         particleID2 = this->SubGraphNodeIDToParticleIDMapVec[graphIndex2][j];

//         if ( this->SubGraphsVec[graphIndex2]->GetDegree(j) == 1 )
//           { 
//           point1[0] = this->InternalInputPolyData->GetPoint( particleID1 )[0];
//           point1[1] = this->InternalInputPolyData->GetPoint( particleID1 )[1];
//           point1[2] = this->InternalInputPolyData->GetPoint( particleID1 )[2];

//           point2[0] = this->InternalInputPolyData->GetPoint( particleID2 )[0];
//           point2[1] = this->InternalInputPolyData->GetPoint( particleID2 )[1];
//           point2[2] = this->InternalInputPolyData->GetPoint( particleID2 )[2];

//           connectingVec[0] = point1[0] - point2[0];
//           connectingVec[1] = point1[1] - point2[1];
//           connectingVec[2] = point1[2] - point2[2];

//           connectorMagnitude = this->GetVectorMagnitude( connectingVec );

//           if ( connectorMagnitude < minDistance )
//             {
//             minDistance = connectorMagnitude;
//             }
//           }
//         }
//       }
//     } 

//   return minDistance;
// }


// bool vtkCIPAirwayParticlesToConnectedAirwayParticlesFilter::GetLeafNodeDirection( unsigned int leafParticleID, double* leafDirection )
// {
//   bool computedDirection = false;

//   //
//   // First find the subGraph that the leafParticleID corresponds to 
//   //
//   unsigned int subGraphID = this->ParticleIDToSubGraphMap[leafParticleID];

//   vtkSmartPointer< vtkEdgeListIterator > edgeIterator = vtkSmartPointer< vtkEdgeListIterator >::New();

//   this->SubGraphsVec[subGraphID]->GetEdges( edgeIterator );

//   while ( edgeIterator->HasNext() )
//     {
//     vtkEdgeType edge = edgeIterator->Next();
//     vtkIdType subGraphSource = edge.Source;
//     vtkIdType subGraphTarget = edge.Target;
    
//     unsigned int sourceParticleID = this->SubGraphNodeIDToParticleIDMapVec[subGraphID][subGraphSource];
//     unsigned int targetParticleID = this->SubGraphNodeIDToParticleIDMapVec[subGraphID][subGraphTarget];

//     if ( sourceParticleID == leafParticleID || targetParticleID == leafParticleID )
//       {
//       double sPoint[3];
//         sPoint[0] = this->InternalInputPolyData->GetPoint( sourceParticleID )[0];
//         sPoint[1] = this->InternalInputPolyData->GetPoint( sourceParticleID )[1];
//         sPoint[2] = this->InternalInputPolyData->GetPoint( sourceParticleID )[2];

//       double tPoint[3];
//         tPoint[0] = this->InternalInputPolyData->GetPoint( targetParticleID )[0];
//         tPoint[1] = this->InternalInputPolyData->GetPoint( targetParticleID )[1];
//         tPoint[2] = this->InternalInputPolyData->GetPoint( targetParticleID )[2];

//       leafDirection[0] = sPoint[0] - tPoint[0];
//       leafDirection[1] = sPoint[1] - tPoint[1];
//       leafDirection[2] = sPoint[2] - tPoint[2];

//       computedDirection = true;
//       }
//     }

//   return computedDirection;
// }


void vtkCIPAirwayParticlesToConnectedAirwayParticlesFilter::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);
}


void vtkCIPAirwayParticlesToConnectedAirwayParticlesFilter::ViewGraphs( std::vector< vtkSmartPointer< vtkMutableDirectedGraph > > graphsVec,
                                                                        std::vector< std::map< vtkIdType, unsigned int > > graphIDsMapVec,
                                                                        unsigned int cardinalityThreshold )
{
//   vtkSmartPointer< vtkRenderer > renderer = vtkSmartPointer< vtkRenderer >::New();
// //    renderer->SetBackground(.3, .6, .3); 
//     renderer->SetBackground( 0, 0, 0); 

//   vtkSmartPointer< vtkPoints > allTreesPoints = vtkSmartPointer< vtkPoints >::New();

//   for ( unsigned int i=0; i<graphsVec.size(); i++ )
//     {
//     if ( static_cast< unsigned int >( graphsVec[i]->GetNumberOfVertices() ) >= cardinalityThreshold )
//       {
//       vtkSmartPointer< vtkPoints > points = vtkSmartPointer< vtkPoints >::New();

//       for ( unsigned int j=0; j<static_cast< unsigned int >( graphsVec[i]->GetNumberOfVertices() ); j++ )
//         {      
//         points->InsertNextPoint( this->InternalInputPolyData->GetPoint( graphIDsMapVec[i][j] ) );
//         allTreesPoints->InsertNextPoint( this->InternalInputPolyData->GetPoint( graphIDsMapVec[i][j] ) );
//         }
      
//       graphsVec[i]->SetPoints( points );

//       vtkSmartPointer< vtkGraphToPolyData > graphToPolyData = vtkSmartPointer<vtkGraphToPolyData>::New();
//         graphToPolyData->SetInput( graphsVec[i] );
//         graphToPolyData->Update();
      
//       vtkSmartPointer< vtkPolyDataMapper > mapper = vtkSmartPointer< vtkPolyDataMapper >::New();
//         mapper->SetInputConnection( graphToPolyData->GetOutputPort() );
 
//       vtkSmartPointer< vtkActor > actor = vtkSmartPointer< vtkActor >::New();
//         actor->SetMapper(mapper);

//         renderer->AddActor(actor);
//       }
//     }

//   vtkSmartPointer< vtkSphereSource > sphereSource = vtkSmartPointer< vtkSphereSource >::New();
//     sphereSource->SetRadius( 0.2 );
//     sphereSource->SetCenter( 0, 0, 0 );

//   //
//   // Get the leaf points
//   //
//   vtkSmartPointer< vtkPoints > leafPoints = vtkSmartPointer< vtkPoints >::New();
//   for ( unsigned int i=0; i<this->SubGraphLeafParticleIDs.size(); i++ )
//     {
//     unsigned int leafParticleID  = this->SubGraphLeafParticleIDs[i];

//     double leafPoint[3];
//     leafPoint[0] = this->InternalInputPolyData->GetPoint( leafParticleID )[0];
//     leafPoint[1] = this->InternalInputPolyData->GetPoint( leafParticleID )[1];
//     leafPoint[2] = this->InternalInputPolyData->GetPoint( leafParticleID )[2];

//     leafPoints->InsertNextPoint( leafPoint[0], leafPoint[1], leafPoint[2] );
//     }

//   vtkSmartPointer< vtkPolyData > leafPoly = vtkSmartPointer< vtkPolyData >::New();
//     leafPoly->SetPoints( leafPoints );

//   vtkSmartPointer< vtkGlyph3D > leafGlyph = vtkSmartPointer< vtkGlyph3D >::New();
//     leafGlyph->SetInput( leafPoly );
//     leafGlyph->SetSource( sphereSource->GetOutput() );
//     leafGlyph->Update();

//   vtkSmartPointer< vtkPolyDataMapper > leafMapper = vtkSmartPointer< vtkPolyDataMapper >::New();
//     leafMapper->SetInput( leafGlyph->GetOutput() );

//   vtkSmartPointer<vtkActor> leafActor = vtkSmartPointer<vtkActor>::New();
//     leafActor->SetMapper( leafMapper );
//     leafActor->GetProperty()->SetColor( 0, 0, 1 );

//   vtkSmartPointer< vtkPolyData > particlesPoly = vtkSmartPointer< vtkPolyData >::New();
//     particlesPoly->SetPoints( allTreesPoints );

//   vtkSmartPointer< vtkGlyph3D > glyph = vtkSmartPointer< vtkGlyph3D >::New();
//     glyph->SetInput( particlesPoly );
//     glyph->SetSource( sphereSource->GetOutput() );
//     glyph->Update();

//   vtkSmartPointer< vtkPolyDataMapper > particlesMapper = vtkSmartPointer< vtkPolyDataMapper >::New();
//     particlesMapper->SetInput( glyph->GetOutput() );

//   vtkSmartPointer<vtkActor> particlesActor = vtkSmartPointer<vtkActor>::New();
//     particlesActor->SetMapper( particlesMapper );
//     particlesActor->GetProperty()->SetColor( 1, 0, 0 );

//   renderer->AddActor( particlesActor );
//   renderer->AddActor( leafActor );

//   vtkSmartPointer< vtkRenderWindow > renderWindow = vtkSmartPointer< vtkRenderWindow >::New();
//     renderWindow->AddRenderer(renderer);

//   vtkSmartPointer< vtkInteractorStyleTrackballCamera > trackball = vtkSmartPointer< vtkInteractorStyleTrackballCamera >::New();

//   vtkSmartPointer< vtkRenderWindowInteractor > renderWindowInteractor = vtkSmartPointer< vtkRenderWindowInteractor >::New();
//     renderWindowInteractor->SetRenderWindow(renderWindow);
//     renderWindowInteractor->SetInteractorStyle( trackball );

//   renderWindow->Render();
//   renderWindowInteractor->Start();
}



// void vtkCIPAirwayParticlesToConnectedAirwayParticlesFilter::DataExplorer()
// {
//   double angle1, angle2, particleParticleAngle;

//   for ( unsigned int i=0; i<static_cast< unsigned int >( this->InternalInputPolyData->GetNumberOfPoints() ); i++ )
//     {
//     double x1 = this->InternalInputPolyData->GetPoint(i)[0];
//     double y1 = this->InternalInputPolyData->GetPoint(i)[1];
//     double z1 = this->InternalInputPolyData->GetPoint(i)[2];

//     double minDist = DBL_MAX;
//     for ( unsigned int j=i+1; j<static_cast< unsigned int >( this->InternalInputPolyData->GetNumberOfPoints() ); j++ )
//       {
//       double x2 = this->InternalInputPolyData->GetPoint(j)[0];
//       double y2 = this->InternalInputPolyData->GetPoint(j)[1];
//       double z2 = this->InternalInputPolyData->GetPoint(j)[2];

//       double dist = sqrt( pow( x1-x2, 2 ) + pow( y1-y2, 2 ) + pow( z1-z2, 2 ) );

//       if ( dist < minDist )
//         {
//         double particle1Hevec2[3];
//         particle1Hevec2[0] = this->InternalInputPolyData->GetPointData()->GetArray( "hevec2" )->GetTuple(i)[0];
//         particle1Hevec2[1] = this->InternalInputPolyData->GetPointData()->GetArray( "hevec2" )->GetTuple(i)[1];
//         particle1Hevec2[2] = this->InternalInputPolyData->GetPointData()->GetArray( "hevec2" )->GetTuple(i)[2];

//         double particle2Hevec2[3];
//         particle2Hevec2[0] = this->InternalInputPolyData->GetPointData()->GetArray( "hevec2" )->GetTuple(j)[0];
//         particle2Hevec2[1] = this->InternalInputPolyData->GetPointData()->GetArray( "hevec2" )->GetTuple(j)[1];
//         particle2Hevec2[2] = this->InternalInputPolyData->GetPointData()->GetArray( "hevec2" )->GetTuple(j)[2];

//         double connectingVec[3];
//         connectingVec[0] = particle1Hevec2[0] - particle2Hevec2[0];
//         connectingVec[1] = particle1Hevec2[1] - particle2Hevec2[1];
//         connectingVec[2] = particle1Hevec2[2] - particle2Hevec2[2];

//         particleParticleAngle = this->GetAngleBetweenVectors( particle1Hevec2, particle2Hevec2, true );
//         angle1 = this->GetAngleBetweenVectors( particle1Hevec2, connectingVec, true );
//         angle2 = this->GetAngleBetweenVectors( particle2Hevec2, connectingVec, true );

//         minDist = dist;
//         }
//       }
//     std::cout << particleParticleAngle  << std::endl;
// //    std::cout << minDist << std::endl;
//     }
//}
