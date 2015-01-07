/**
 *
 *  $Date$
 *  $Revision$
 *  $Author$
 *
 *  TODO:
 *
 */

#include "cipVesselDataInteractor.h"
#include "vtkSmartPointer.h"
#include "vtkPropPicker.h"
#include "vtkRendererCollection.h"
#include "vtkDataSetMapper.h"
#include "vtkPointData.h"
#include "vtkProperty.h"
#include "vtkExtractSelectedGraph.h"
#include "vtkDoubleArray.h"
#include "cipHelper.h"
#include "vtkFloatArray.h"
#include "vtkDijkstraGraphGeodesicPath.h"
#include "vtkGraphToPolyData.h"
#include "vtkIdList.h"
#include "vtkPolyDataWriter.h"

cipVesselDataInteractor::cipVesselDataInteractor()
{
  this->InteractorCallbackCommand = vtkCallbackCommand::New();
  this->InteractorCallbackCommand->SetCallback( InteractorKeyCallback );
  this->InteractorCallbackCommand->SetClientData( (void*)this );

  // cipVesselDataInteractor inherits from cipChestDataViewer. The
  // cipChestDataViewer constructor is called before the
  // cipVesselDataInteractor constructor, and in the parent constructor,
  // the ViewerCallbackCommand is set. We want to remove it and set
  // the InteractorCallbackCommand in order for the key bindings
  // specific to interaction can take effect
  this->RenderWindowInteractor->RemoveObserver( this->ViewerCallbackCommand );
  this->RenderWindowInteractor->AddObserver( vtkCommand::KeyPressEvent, this->InteractorCallbackCommand );

  this->SelectedChestRegion = (unsigned char)(cip::UNDEFINEDREGION);
  this->SelectedChestType   = (unsigned char)(cip::UNDEFINEDTYPE);

  this->NumberInputParticles      = 0;
  this->NumberOfPointDataArrays   = 0;
  this->EdgeWeightAngleSigma      = 1.0;
  this->ParticleDistanceThreshold = 3.0;
  this->ActorColor                = new double[3];

  this->VesselModelActor = vtkSmartPointer< vtkActor >::New();
  this->VesselModel = vtkSmartPointer< vtkPolyData >::New();
  this->VesselModelShowing = false;
}

void cipVesselDataInteractor::SetRootNode( vtkActor* actor )
{
  this->MinimumSpanningTreeRootNode = this->ActorToParticleIDMap[actor];
}

void cipVesselDataInteractor::SetFileName( std::string fileName )
{
  this->FileName = fileName;
}

void cipVesselDataInteractor::UpdateVesselBranchCode( char c )
{
  std::stringstream ss;
  std::string code;
  ss << c;
  ss >> code;

  if ( code.compare( "a" ) == 0 )
    {
    this->Conventions->GetChestTypeColor( (unsigned char)(cip::ARTERY), this->ActorColor );
    this->SelectedChestRegion = (unsigned char)(cip::UNDEFINEDREGION);
    this->SelectedChestType   = (unsigned char)(cip::ARTERY);
    }
  if ( code.compare( "v" ) == 0 )
    {
    this->Conventions->GetChestTypeColor( (unsigned char)(cip::VEIN), this->ActorColor );
    this->SelectedChestRegion = (unsigned char)(cip::UNDEFINEDREGION);
    this->SelectedChestType   = (unsigned char)(cip::VEIN);
    }
}

void cipVesselDataInteractor::UndoAndRender()
{
  unsigned int lastModification = this->LabeledParticleIDs.size() - 1;

  float tmpRegion = float(cip::UNDEFINEDREGION);
  float tmpType   = float(cip::UNDEFINEDTYPE);

  for ( unsigned int i=0; i<this->LabeledParticleIDs[lastModification].size(); i++ )
    {
    unsigned int id = this->LabeledParticleIDs[lastModification][i];
    this->VesselParticles->GetPointData()->GetArray("ChestType")->SetTuple( id, &tmpType );
    this->VesselParticles->GetPointData()->GetArray("ChestRegion")->SetTuple( id, &tmpRegion );

    this->ParticleIDToActorMap[id]->GetProperty()->SetColor( 1.0, 1.0, 1.0 );
    }

  this->LabeledParticleIDs[lastModification].clear();
  this->LabeledParticleIDs.erase( this->LabeledParticleIDs.end() );

  this->RenderWindow->Render();
}

void cipVesselDataInteractor::SetIntermediateNode( vtkActor* actor )
{
  this->MinimumSpanningTreeIntermediateNode = this->ActorToParticleIDMap[actor];  

  // vtkSmartPointer< vtkGraphToPolyData > graphToPolyData = vtkSmartPointer< vtkGraphToPolyData >::New();
  //   graphToPolyData->SetInput( this->MinimumSpanningTree );
  //   graphToPolyData->Update();
 
  vtkSmartPointer< vtkDijkstraGraphGeodesicPath > dijkstra = vtkSmartPointer< vtkDijkstraGraphGeodesicPath >::New();
  //dijkstra->SetInputConnection( graphToPolyData->GetOutputPort() );
    dijkstra->SetInputData( this->VesselParticles );
    dijkstra->SetStartVertex( this->MinimumSpanningTreeIntermediateNode );
    dijkstra->SetEndVertex( this->MinimumSpanningTreeRootNode );
    dijkstra->Update();

  vtkIdList* idList = dijkstra->GetIdList();

  // The following will store particle IDs that are being
  // labeled. We'll need this in case we want to undo the labeling.
  std::vector< unsigned int > labeledIDs;
  
  cip::VectorType refVec(3);
  for ( unsigned int i=0; i<idList->GetNumberOfIds(); i++ )
    {
      if ( this->VesselParticles->GetPointData()->GetArray("ChestType")->GetTuple(idList->GetId(i))[0] == float(cip::UNDEFINEDTYPE) )
	{
	  float tmpRegion = (float)(this->SelectedChestRegion);
	  float tmpType   = (float)(this->SelectedChestType);
	  this->ParticleIDToActorMap[idList->GetId(i)]->GetProperty()->SetColor( this->ActorColor[0], this->ActorColor[1], this->ActorColor[2] );
	  this->VesselParticles->GetPointData()->GetArray("ChestRegion")->SetTuple(idList->GetId(i), &tmpRegion );
	  this->VesselParticles->GetPointData()->GetArray("ChestType")->SetTuple(idList->GetId(i), &tmpType );
	  labeledIDs.push_back(idList->GetId(i));
	  
	  // Re-orient the particle's minor eigenvector. This is necessary for other algorithms
	  // (specifically, some particles registration algorithms) which require that all the 
	  // particles be oriented in a consistent manner -- in this case, all minor eigenvectors
	  // will point from leaf node to root node.     
	  if ( i < idList->GetNumberOfIds() - 1 )
	    {
	      refVec[0] = this->VesselParticles->GetPoint(idList->GetId(i+1))[0] - this->VesselParticles->GetPoint(idList->GetId(i))[0];
	      refVec[1] = this->VesselParticles->GetPoint(idList->GetId(i+1))[1] - this->VesselParticles->GetPoint(idList->GetId(i))[1];
	      refVec[2] = this->VesselParticles->GetPoint(idList->GetId(i+1))[2] - this->VesselParticles->GetPoint(idList->GetId(i))[2];
	    }
	  else
	    {
	      refVec[0] = this->VesselParticles->GetPoint(idList->GetId(i))[0] - this->VesselParticles->GetPoint(idList->GetId(i-1))[0];
	      refVec[1] = this->VesselParticles->GetPoint(idList->GetId(i))[1] - this->VesselParticles->GetPoint(idList->GetId(i-1))[1];
	      refVec[2] = this->VesselParticles->GetPoint(idList->GetId(i))[2] - this->VesselParticles->GetPoint(idList->GetId(i-1))[2];
	    }
	  
	  this->OrientParticle( idList->GetId(i), refVec );
	}
    }
  
  this->LabeledParticleIDs.push_back( labeledIDs );  
  this->RenderWindow->Render();
}

// This function re-orients a particle's minor eigenvector so that it is more parallel than
// anti-parallel to the specified reference vector. This function is used during the labeling
// process to ensure that all labeled particles have minor eigenvectors oriented in a consistent
// way, which is necessary for some registration routines that take as input the labeled
// particles datasets.
void cipVesselDataInteractor::OrientParticle( unsigned int particleID, const cip::VectorType& refVec )
{
  cip::VectorType hevec0(3);
    hevec0[0] = this->VesselParticles->GetPointData()->GetArray( "hevec0" )->GetTuple( particleID )[0];
    hevec0[1] = this->VesselParticles->GetPointData()->GetArray( "hevec0" )->GetTuple( particleID )[1];
    hevec0[2] = this->VesselParticles->GetPointData()->GetArray( "hevec0" )->GetTuple( particleID )[2];

  double angle = cip::GetAngleBetweenVectors( hevec0, refVec, false );

  if ( angle > vnl_math::pi/2.0 )
    {
      float* hevec0flipped = new float[3];
        hevec0flipped[0] = -hevec0[0];
	hevec0flipped[1] = -hevec0[1];
	hevec0flipped[2] = -hevec0[2];

      this->VesselParticles->GetPointData()->GetArray("hevec0")->SetTuple( particleID, hevec0flipped );
    }
}

void cipVesselDataInteractor::UpdateVesselGenerationAndRender( vtkActor* actor, int generation )
{
  // // Save data to undo stack    
  // this->m_UndoState.actor = actor;
  // actor->GetProperty()->GetColor( this->m_UndoState.color );
  // this->m_UndoState.opacity = actor->GetProperty()->GetOpacity(); 
    
  // std::map< std::string, vtkActor* >::iterator it = this->VesselParticlesActorMap.begin();

  // double* color = new double[3];

  // while ( it != this->VesselParticlesActorMap.end() )
  //   {
  //   if ( it->second == actor )
  //     {
  //     if ( generation == 0 )
  //       {
  //       this->Conventions->GetChestTypeColor( static_cast< unsigned char >( cip::TRACHEA ), color );
  //       }
  //     if ( generation == 1 )
  //       {
  //       this->Conventions->GetChestTypeColor( static_cast< unsigned char >( cip::MAINBRONCHUS ), color );
  //       }
  //     if ( generation == 2 )
  //       {
  //       this->Conventions->GetChestTypeColor( static_cast< unsigned char >( cip::UPPERLOBEBRONCHUS ), color );
  //       }
  //     if ( generation == 3 )
  //       {
  //       this->Conventions->GetChestTypeColor( static_cast< unsigned char >( cip::AIRWAYGENERATION3 ), color );
  //       }
  //     if ( generation == 4 )
  //       {
  //       this->Conventions->GetChestTypeColor( static_cast< unsigned char >( cip::AIRWAYGENERATION4 ), color );
  //       }
  //     if ( generation == 5 )
  //       {
  //       this->Conventions->GetChestTypeColor( static_cast< unsigned char >( cip::AIRWAYGENERATION5 ), color );
  //       }
  //     if ( generation == 6 )
  //       {
  //       this->Conventions->GetChestTypeColor( static_cast< unsigned char >( cip::AIRWAYGENERATION6 ), color );
  //       }
  //     if ( generation == 7 )
  //       {
  //       this->Conventions->GetChestTypeColor( static_cast< unsigned char >( cip::AIRWAYGENERATION7 ), color );
  //       }
  //     if ( generation == 8 )
  //       {
  //       this->Conventions->GetChestTypeColor( static_cast< unsigned char >( cip::AIRWAYGENERATION8 ), color );
  //       }
  //     if ( generation == 9 )
  //       {
  // 	this->Conventions->GetChestTypeColor( static_cast< unsigned char >( cip::AIRWAYGENERATION9 ), color );
  //       }

  //     actor->GetProperty()->SetColor( color[0], color[1], color[2] );
  //     actor->GetProperty()->SetOpacity( 1.0 );

  //     break;
  //     }

  //   ++it;
  //   }
  
  // delete[] color;

  // this->RenderWindow->Render();
}

void cipVesselDataInteractor::SetVesselModel( vtkSmartPointer< vtkPolyData > model )
{
  this->VesselModel = model;
  this->VesselModelActor = this->SetPolyData( model, "vesselModel" );
  this->SetActorColor( "vesselModel", 1.0, 1.0, 1.0 );
  this->SetActorOpacity( "vesselModel", 0.3 );

  this->VesselModelShowing = true;
}

void cipVesselDataInteractor::ShowVesselModel()
{
  if ( !this->VesselModelShowing )
    {
      this->VesselModelShowing = true;

      this->Renderer->AddActor( this->VesselModelActor);
      this->RenderWindow->Render();
    }
}

void cipVesselDataInteractor::HideVesselModel()
{
  if ( this->VesselModelShowing )
    {
      this->VesselModelShowing = false;

      this->Renderer->RemoveActor( this->VesselModelActor );
      this->RenderWindow->Render();
    }
}

void cipVesselDataInteractor::UndoUpdateAndRender()
{
  this->m_UndoState.actor->GetProperty()->SetColor( this->m_UndoState.color);
  this->m_UndoState.actor->GetProperty()->SetOpacity( this->m_UndoState.opacity);
    
  this->RenderWindow->Render();
}

void cipVesselDataInteractor::RemoveActorAndRender( vtkActor* actor )
{
  std::map< std::string, vtkActor* >::iterator it;

  for ( it = this->ActorMap.begin(); it != this->ActorMap.end(); it++ )
    {
    if ( (*it).second == actor )
      {
      this->ActorMap.erase( it );

      break;
      }
    }

  this->Renderer->RemoveActor( actor );
  this->RenderWindow->Render();
}

void cipVesselDataInteractor::HideActorAndRender( vtkActor* actor )
{
  actor->GetProperty()->SetOpacity( 0.0 );
  this->RenderWindow->Render();
}

void cipVesselDataInteractor::ColorActorByChestTypeAndRender( vtkActor* actor, unsigned char cipType )
{
  double color[3];
  this->Conventions->GetColorFromChestRegionChestType((unsigned char)(cip::UNDEFINEDREGION), cipType, color);
  actor->GetProperty()->SetColor( color[0], color[1], color[2] );
  this->RenderWindow->Render();
}

void cipVesselDataInteractor::SetConnectedVesselParticles( vtkSmartPointer< vtkPolyData > particles, 
							   double particleSize )
{
  this->VesselParticles = particles;

  this->NumberInputParticles    = particles->GetNumberOfPoints();
  this->NumberOfPointDataArrays = particles->GetPointData()->GetNumberOfArrays();

  // First create a minimum spanning tree representation
  //this->InitializeMinimumSpanningTree( particles );

  // Now we want to loop over all particles and create an individual
  // actor for each.

  for ( unsigned int p = 0; p < this->NumberInputParticles; p++ )
    {
    vtkPolyData* tmpPolyData = vtkPolyData::New();

    vtkPoints* point  = vtkPoints::New();
      point->InsertNextPoint( particles->GetPoint(p) );
 
    std::stringstream stream;
    stream << p;
    std::string name = stream.str();

    std::vector< vtkFloatArray* > arrayVec;

    for ( unsigned int i=0; i<this->NumberOfPointDataArrays; i++ )
      {
      vtkFloatArray* array = vtkFloatArray::New();
        array->SetNumberOfComponents( particles->GetPointData()->GetArray(i)->GetNumberOfComponents() );
        array->SetName( particles->GetPointData()->GetArray(i)->GetName() );

      arrayVec.push_back( array );
      arrayVec[i]->InsertTuple( 0, particles->GetPointData()->GetArray(i)->GetTuple(p) );
      }

    tmpPolyData->SetPoints( point );
    for ( unsigned int j=0; j<this->NumberOfPointDataArrays; j++ )
      {
      tmpPolyData->GetPointData()->AddArray( arrayVec[j] );
      }

    vtkActor* actor = this->SetVesselParticlesAsDiscs( tmpPolyData, particleSize, name );
      this->SetActorColor( name, 1.0, 1.0, 1.0 );
      this->SetActorOpacity( name, 1.0 );
      
    this->ActorToParticleIDMap[actor] = p;
    this->ParticleIDToActorMap[p]     = actor;
    }

  vtkSmartPointer< vtkPolyDataMapper > mapper = vtkSmartPointer< vtkPolyDataMapper >::New();
    mapper->SetInputData( particles );

  vtkSmartPointer< vtkActor > actor = vtkSmartPointer< vtkActor >::New();
    actor->SetMapper( mapper );
    actor->GetProperty()->SetColor( 0.0, 0.0, 0.0 );

  this->ActorMap["vesselParticles"] = actor;
  this->Renderer->AddActor( this->ActorMap["vesselParticles"] );
}


void cipVesselDataInteractor::InitializeMinimumSpanningTree( vtkSmartPointer< vtkPolyData > particles )
{
  // Now create the weighted graph that will be passed to the minimum 
  // spanning tree filter
  std::map< unsigned int, unsigned int > particleIDToNodeIDMap;
  std::map< unsigned int, unsigned int > nodeIDToParticleIDMap;

  vtkSmartPointer< vtkMutableUndirectedGraph > weightedGraph = vtkSmartPointer< vtkMutableUndirectedGraph >::New();

  for ( unsigned int i=0; i<this->NumberInputParticles; i++ )
    {
      vtkIdType nodeID = weightedGraph->AddVertex();

      particleIDToNodeIDMap[i]      = nodeID;
      nodeIDToParticleIDMap[nodeID] = i;
    }

  vtkSmartPointer< vtkDoubleArray > edgeWeights = vtkSmartPointer< vtkDoubleArray >::New();
    edgeWeights->SetNumberOfComponents( 1 );
    edgeWeights->SetName( "Weights" );

  for ( unsigned int i=0; i<this->NumberInputParticles; i++ )
    {
    for ( unsigned int j=i+1; j<this->NumberInputParticles; j++ )
      {
      double weight;
      
      if ( this->GetEdgeWeight( i, j, particles, &weight ) )
        {
        weightedGraph->AddEdge( particleIDToNodeIDMap[i], particleIDToNodeIDMap[j] );
        edgeWeights->InsertNextValue( weight );
        }
      }
    }
  weightedGraph->GetEdgeData()->AddArray( edgeWeights );
  weightedGraph->SetPoints( particles->GetPoints() );

  vtkSmartPointer< vtkBoostKruskalMinimumSpanningTree > minimumSpanningTreeFilter = vtkSmartPointer< vtkBoostKruskalMinimumSpanningTree >::New();
    minimumSpanningTreeFilter->SetInputData( weightedGraph );
    minimumSpanningTreeFilter->SetEdgeWeightArrayName( "Weights" );
    minimumSpanningTreeFilter->Update();

  vtkSmartPointer< vtkExtractSelectedGraph > extractSelection = vtkSmartPointer< vtkExtractSelectedGraph >::New();
    extractSelection->SetInputData( 0, weightedGraph );
    extractSelection->SetInputData( 1, minimumSpanningTreeFilter->GetOutput()) ;
    extractSelection->Update();

  this->MinimumSpanningTree = vtkMutableUndirectedGraph::SafeDownCast( extractSelection->GetOutput() );

  //cip::ViewGraphAsPolyData( this->MinimumSpanningTree );
}

void cipVesselDataInteractor::Write()
{
  std::cout << "---Writing labeled particles..." << std::endl;
  vtkSmartPointer< vtkPolyDataWriter > writer = vtkSmartPointer< vtkPolyDataWriter >::New();
    writer->SetFileName( this->FileName.c_str() );
    writer->SetInputData( this->VesselParticles );
    writer->SetFileTypeToASCII();
    writer->Write();  
}

bool cipVesselDataInteractor::GetEdgeWeight( unsigned int particleID1, unsigned int particleID2, 
                                             vtkSmartPointer< vtkPolyData > particles, double* weight )
{
  // Determine the vector connecting the two particles
  double point1[3];
    point1[0] = particles->GetPoint( particleID1 )[0];
    point1[1] = particles->GetPoint( particleID1 )[1];
    point1[2] = particles->GetPoint( particleID1 )[2];

  double point2[3];
    point2[0] = particles->GetPoint( particleID2 )[0];
    point2[1] = particles->GetPoint( particleID2 )[1];
    point2[2] = particles->GetPoint( particleID2 )[2];

  cip::VectorType connectingVec(3);
    connectingVec[0] = point1[0] - point2[0];
    connectingVec[1] = point1[1] - point2[1];
    connectingVec[2] = point1[2] - point2[2];

  double connectorMagnitude = cip::GetVectorMagnitude( connectingVec );

  if ( connectorMagnitude > this->ParticleDistanceThreshold )
    {
    return false;
    }

  cip::VectorType particle1Hevec0(3);
    particle1Hevec0[0] = particles->GetPointData()->GetArray( "hevec0" )->GetTuple( particleID1 )[0];
    particle1Hevec0[1] = particles->GetPointData()->GetArray( "hevec0" )->GetTuple( particleID1 )[1];
    particle1Hevec0[2] = particles->GetPointData()->GetArray( "hevec0" )->GetTuple( particleID1 )[2];

  cip::VectorType particle2Hevec0(3);
    particle2Hevec0[0] = particles->GetPointData()->GetArray( "hevec0" )->GetTuple( particleID2 )[0];
    particle2Hevec0[1] = particles->GetPointData()->GetArray( "hevec0" )->GetTuple( particleID2 )[1];
    particle2Hevec0[2] = particles->GetPointData()->GetArray( "hevec0" )->GetTuple( particleID2 )[2];

  double angle1 =  cip::GetAngleBetweenVectors( particle1Hevec0, connectingVec, true );
  double angle2 =  cip::GetAngleBetweenVectors( particle2Hevec0, connectingVec, true );

  if ( angle1 < angle2 )
    {
      //*weight = 2.0*(10.0/15.0)*connectorMagnitude + (10.0/45.0)*angle1;
      *weight = connectorMagnitude*(1.0 + 1.1*exp(-pow( (90.0 - angle1)/this->EdgeWeightAngleSigma, 2 )));
    }
  else
    {
      //*weight = 2.0*(10.0/15.0)*connectorMagnitude + (10.0/45.0)*angle2;
      *weight = connectorMagnitude*(1.0 + 1.1*exp(-pow( (90.0 - angle2)/this->EdgeWeightAngleSigma, 2 )));
    }

  return true;
}


void InteractorKeyCallback( vtkObject* obj, unsigned long b, void* clientData, void* d )
{
  cipVesselDataInteractor* dataInteractor = reinterpret_cast< cipVesselDataInteractor* >( clientData );

  char pressedKey = dataInteractor->GetRenderWindowInteractor()->GetKeyCode(); 

  if ( pressedKey == 'k' || pressedKey == 'a' || pressedKey == 'v' || pressedKey == 'h' )
    {
    int* clickPos = dataInteractor->GetRenderWindowInteractor()->GetEventPosition();

    vtkSmartPointer< vtkPropPicker > picker = vtkSmartPointer< vtkPropPicker >::New();

    picker->Pick( clickPos[0], clickPos[1], 0, 
                  dataInteractor->GetRenderWindowInteractor()->GetRenderWindow()->GetRenderers()->GetFirstRenderer() );

    vtkActor* actor = picker->GetActor();

    if ( actor != NULL )
      {
	if ( pressedKey == 'h' )
	  {
	    dataInteractor->HideActorAndRender( actor );
	  }
	if ( pressedKey == 'k' )
	  {
	    dataInteractor->RemoveActorAndRender( actor );
	  }
	else if ( pressedKey == 'a' )
	  {
	    dataInteractor->ColorActorByChestTypeAndRender( actor, (unsigned char)(cip::ARTERY) );
	  }
	else
	  {
	    dataInteractor->ColorActorByChestTypeAndRender( actor, (unsigned char)(cip::VEIN) );
	  }
      }
    }
  // else if ( pressedKey == 's' )
  //   {
  //     //dataInteractor->Write();
  //   }
  // else if ( pressedKey == 'h' )
  //   {
  //     dataInteractor->HideVesselModel();
  //   }
  // else if ( pressedKey == 'u' )
  //   {
  //     //dataInteractor->UndoAndRender();
  //   }
  else if ( pressedKey == '!' || pressedKey == '@' || pressedKey == '#' || pressedKey == '$' || pressedKey == '%' || 
       pressedKey == '^' || pressedKey == '&' || pressedKey == '*' || pressedKey == '(' || pressedKey == ')' ||
       pressedKey == 'o' || pressedKey == 'm' )
    {
    int* clickPos = dataInteractor->GetRenderWindowInteractor()->GetEventPosition();

    vtkSmartPointer< vtkPropPicker > picker = vtkSmartPointer< vtkPropPicker >::New();

    picker->Pick( clickPos[0], clickPos[1], 0, 
                  dataInteractor->GetRenderWindowInteractor()->GetRenderWindow()->GetRenderers()->GetFirstRenderer() );

    vtkActor* actor = picker->GetActor();

    if ( actor != NULL )
      {
      if ( pressedKey == '!' )
        {
        dataInteractor->UpdateVesselGenerationAndRender( actor, 1 );
        }
      if ( pressedKey == '@' )
        {
        dataInteractor->UpdateVesselGenerationAndRender( actor, 2 );
        }
      if ( pressedKey == '#' )
        {
        dataInteractor->UpdateVesselGenerationAndRender( actor, 3 );
        }
      if ( pressedKey == '$' )
        {
        dataInteractor->UpdateVesselGenerationAndRender( actor, 4 );
        }
      if ( pressedKey == '%' )
        {
        dataInteractor->UpdateVesselGenerationAndRender( actor, 5 );
        }
      if ( pressedKey == '^' )
        {
        dataInteractor->UpdateVesselGenerationAndRender( actor, 6 );
        }
      if ( pressedKey == '&' )
        {
        dataInteractor->UpdateVesselGenerationAndRender( actor, 7 );
        }
      if ( pressedKey == '*' )
        {
        dataInteractor->UpdateVesselGenerationAndRender( actor, 8 );
        }
      if ( pressedKey == '(' )
        {
        dataInteractor->UpdateVesselGenerationAndRender( actor, 9 );
        }
      if ( pressedKey == ')' )
        {
        dataInteractor->UpdateVesselGenerationAndRender( actor, 0 );
        }
      if ( pressedKey == 'o' )
        {
        dataInteractor->SetRootNode( actor );
        }
      if ( pressedKey == 'm' )
        {
        dataInteractor->SetIntermediateNode( actor );
        }
      }
    }

  if ( pressedKey == 'x' )
    {
    if ( dataInteractor->GetPlaneWidgetXShowing() ) 
      {
      dataInteractor->SetPlaneWidgetXShowing( false );
      }
    else
      {
      dataInteractor->SetPlaneWidgetXShowing( true );
      }
    }
  
  if ( pressedKey == 'y' )
    {
    if ( dataInteractor->GetPlaneWidgetYShowing() ) 
      {
      dataInteractor->SetPlaneWidgetYShowing( false );
      }
    else
      {
      dataInteractor->SetPlaneWidgetYShowing( true );
      }
    }
  
  if ( pressedKey == 'z' )
    {
    if ( dataInteractor->GetPlaneWidgetZShowing() ) 
      {
      dataInteractor->SetPlaneWidgetZShowing( false );
      }
    else
      {
      dataInteractor->SetPlaneWidgetZShowing( true );
      }
    }  
}

