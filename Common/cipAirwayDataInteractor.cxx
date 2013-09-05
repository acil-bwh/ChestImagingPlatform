/**
 *
 *  $Date$
 *  $Revision$
 *  $Author$
 *
 *  TODO:
 *
 */

#include "cipAirwayDataInteractor.h"
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

cipAirwayDataInteractor::cipAirwayDataInteractor()
{
  this->InteractorCallbackCommand = vtkCallbackCommand::New();
  this->InteractorCallbackCommand->SetCallback( InteractorKeyCallback );
  this->InteractorCallbackCommand->SetClientData( (void*)this );

  //
  // cipAirwayDataInteractor inherits from cipChestDataViewer. The
  // cipChestDataViewer constructor is called before the
  // cipAirwayDataInteractor constructor, and in the parent constructor,
  // the ViewerCallbackCommand is set. We want to remove it and set
  // the InteractorCallbackCommand in order for the key bindings
  // specific to interaction can take effect
  //
  this->RenderWindowInteractor->RemoveObserver( this->ViewerCallbackCommand );
  this->RenderWindowInteractor->AddObserver( vtkCommand::KeyPressEvent, this->InteractorCallbackCommand );

  this->NumberInputParticles      = 0;
  this->NumberOfPointDataArrays   = 0;
  this->EdgeWeightAngleSigma      = 1.0;
  this->ParticleDistanceThreshold = 20.0;
  this->AirwayBranchCode          = "";
  this->ActorColor                = new double[3];
}

void cipAirwayDataInteractor::SetRootNode( vtkActor* actor )
{
  this->MinimumSpanningTreeRootNode = this->ActorToParticleIDMap[actor];
}

void cipAirwayDataInteractor::UpdateAirwayBranchCode( char c )
{
  if ( c == 'C' )
    {
    std::cout << "Clearing airway branch label..." << std::endl;
    this->AirwayBranchCode.clear();
    return;
    }
  if ( c == 'T' )
    {
    if ( this->AirwayBranchCode.length() == 0 )
      {
      std::cout << "Setting airway branch to trachea..." << std::endl;
      this->Conventions->GetChestTypeColor( static_cast< unsigned char >( cip::AIRWAYGENERATION0 ), this->ActorColor );
      }
    return;
    }    

  this->AirwayBranchCode.append( &c );
  
  std::cout << "Code:\t" << this->AirwayBranchCode << std::endl;
  if ( this->AirwayBranchCode.compare( "LMB" ) == 0 )
    {
    std::cout << "Setting airway branch to left main bronchi..." << std::endl;
    //TODO: Need to create LMB in conventions
    this->Conventions->GetChestTypeColor( static_cast< unsigned char >( cip::AIRWAYGENERATION1 ), this->ActorColor );
    }
  if ( this->AirwayBranchCode.compare( "IMB" ) == 0 )
    {
    std::cout << "Setting airway branch to left main bronchi..." << std::endl;
    //TODO: Need to create RMB in conventions
    this->Conventions->GetChestTypeColor( static_cast< unsigned char >( cip::AIRWAYGENERATION1 ), this->ActorColor );
    }
}

void cipAirwayDataInteractor::SetIntermediateNode( vtkActor* actor )
{
  this->MinimumSpanningTreeIntermediateNode = this->ActorToParticleIDMap[actor];  

 vtkSmartPointer< vtkGraphToPolyData > graphToPolyData = vtkSmartPointer< vtkGraphToPolyData >::New();
   graphToPolyData->SetInput( this->MinimumSpanningTree );
   graphToPolyData->Update();
 
 vtkSmartPointer< vtkDijkstraGraphGeodesicPath > dijkstra = vtkSmartPointer< vtkDijkstraGraphGeodesicPath >::New();
   dijkstra->SetInputConnection( graphToPolyData->GetOutputPort() );
   dijkstra->SetStartVertex( this->MinimumSpanningTreeIntermediateNode );
   dijkstra->SetEndVertex( this->MinimumSpanningTreeRootNode );
   dijkstra->Update();

 vtkIdList* idList = dijkstra->GetIdList();

 for ( unsigned int i=0; i<idList->GetNumberOfIds(); i++ )
   {
   this->ParticleIDToActorMap[idList->GetId(i) ]->GetProperty()->SetColor( this->ActorColor[0], this->ActorColor[1], this->ActorColor[2] );
   }

 this->RenderWindow->Render();
}


void cipAirwayDataInteractor::UpdateAirwayGenerationAndRender( vtkActor* actor, int generation )
{
  // Save data to undo stack    
  this->m_UndoState.actor = actor;
  actor->GetProperty()->GetColor( this->m_UndoState.color );
  this->m_UndoState.opacity = actor->GetProperty()->GetOpacity(); 
    
  std::map< std::string, vtkActor* >::iterator it = this->AirwayParticlesActorMap.begin();

  double* color = new double[3];

  while ( it != this->AirwayParticlesActorMap.end() )
    {
    if ( it->second == actor )
      {
      if ( generation == 0 )
        {
        this->Conventions->GetChestTypeColor( static_cast< unsigned char >( cip::AIRWAYGENERATION0 ), color );
        }
      if ( generation == 1 )
        {
        this->Conventions->GetChestTypeColor( static_cast< unsigned char >( cip::AIRWAYGENERATION1 ), color );
        }
      if ( generation == 2 )
        {
        this->Conventions->GetChestTypeColor( static_cast< unsigned char >( cip::AIRWAYGENERATION2 ), color );
        }
      if ( generation == 3 )
        {
        this->Conventions->GetChestTypeColor( static_cast< unsigned char >( cip::AIRWAYGENERATION3 ), color );
        }
      if ( generation == 4 )
        {
        this->Conventions->GetChestTypeColor( static_cast< unsigned char >( cip::AIRWAYGENERATION4 ), color );
        }
      if ( generation == 5 )
        {
        this->Conventions->GetChestTypeColor( static_cast< unsigned char >( cip::AIRWAYGENERATION5 ), color );
        }
      if ( generation == 6 )
        {
        this->Conventions->GetChestTypeColor( static_cast< unsigned char >( cip::AIRWAYGENERATION6 ), color );
        }
      if ( generation == 7 )
        {
        this->Conventions->GetChestTypeColor( static_cast< unsigned char >( cip::AIRWAYGENERATION7 ), color );
        }
      if ( generation == 8 )
        {
        this->Conventions->GetChestTypeColor( static_cast< unsigned char >( cip::AIRWAYGENERATION8 ), color );
        }
      if ( generation == 9 )
        {
	this->Conventions->GetChestTypeColor( static_cast< unsigned char >( cip::AIRWAYGENERATION9 ), color );
        }

      actor->GetProperty()->SetColor( color[0], color[1], color[2] );
      actor->GetProperty()->SetOpacity( 1.0 );

      break;
      }

    ++it;
    }
  
  delete[] color;

  this->RenderWindow->Render();
}


void cipAirwayDataInteractor::UndoUpdateAndRender()
{
  this->m_UndoState.actor->GetProperty()->SetColor( this->m_UndoState.color);
  this->m_UndoState.actor->GetProperty()->SetOpacity( this->m_UndoState.opacity);
    
  this->RenderWindow->Render();
}

void cipAirwayDataInteractor::RemoveActorAndRender( vtkActor* actor )
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


void cipAirwayDataInteractor::SetAirwayParticlesAsMinimumSpanningTree( vtkSmartPointer< vtkPolyData > particles )
{
  this->NumberInputParticles    = particles->GetNumberOfPoints();
  this->NumberOfPointDataArrays = particles->GetPointData()->GetNumberOfArrays();

  // First create a minimum spanning tree representation
  this->InitializeMinimumSpanningTree( particles );

  // Now we want to loop over all particles and create an individual
  // actor for each.
  double particleSize = 1.0;

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

    vtkActor* actor = this->SetAirwayParticlesAsCylinders( tmpPolyData, particleSize, name );
      this->SetActorColor( name, 1.0, 1.0, 1.0 );
      this->SetActorOpacity( name, 1.0 );

    this->ActorToParticleIDMap[actor] = p;
    this->ParticleIDToActorMap[p]     = actor;
    }
}


void cipAirwayDataInteractor::InitializeMinimumSpanningTree( vtkSmartPointer< vtkPolyData > particles )
{
  // Now create the weighted graph that will be passed to the minimum 
  // spanning tree filter
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

  vtkSmartPointer< vtkBoostKruskalMinimumSpanningTree > minimumSpanningTreeFilter = 
    vtkSmartPointer< vtkBoostKruskalMinimumSpanningTree >::New();
    minimumSpanningTreeFilter->SetInput( weightedGraph );
    minimumSpanningTreeFilter->SetEdgeWeightArrayName( "Weights" );
    minimumSpanningTreeFilter->Update();

  vtkSmartPointer< vtkExtractSelectedGraph > extractSelection = vtkSmartPointer< vtkExtractSelectedGraph >::New();
    extractSelection->SetInput( 0, weightedGraph );
    extractSelection->SetInput( 1, minimumSpanningTreeFilter->GetOutput()) ;
    extractSelection->Update();

  this->MinimumSpanningTree = vtkMutableUndirectedGraph::SafeDownCast( extractSelection->GetOutput() );

  //cip::ViewGraphAsPolyData( this->MinimumSpanningTree );
}


bool cipAirwayDataInteractor::GetEdgeWeight( unsigned int particleID1, unsigned int particleID2, 
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

  double connectingVec[3];
    connectingVec[0] = point1[0] - point2[0];
    connectingVec[1] = point1[1] - point2[1];
    connectingVec[2] = point1[2] - point2[2];

  double connectorMagnitude = cip::GetVectorMagnitude( connectingVec );

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

  double angle1 =  cip::GetAngleBetweenVectors( particle1Hevec2, connectingVec, true );
  double angle2 =  cip::GetAngleBetweenVectors( particle2Hevec2, connectingVec, true );

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


void InteractorKeyCallback( vtkObject* obj, unsigned long b, void* clientData, void* d )
{
  cipAirwayDataInteractor* dataInteractor = reinterpret_cast< cipAirwayDataInteractor* >( clientData );

  char pressedKey = dataInteractor->GetRenderWindowInteractor()->GetKeyCode(); 

  if ( pressedKey == 'T' || pressedKey == 'I' || pressedKey == 'M' || pressedKey == 'B' ||
       pressedKey == 'C' || pressedKey == 'L' )
    {
    dataInteractor->UpdateAirwayBranchCode( pressedKey );
    }

  std::cout << pressedKey << std::endl;
  if ( pressedKey == 'k' )
    {
    int* clickPos = dataInteractor->GetRenderWindowInteractor()->GetEventPosition();

    vtkSmartPointer< vtkPropPicker > picker = vtkSmartPointer< vtkPropPicker >::New();

    picker->Pick( clickPos[0], clickPos[1], 0, 
                  dataInteractor->GetRenderWindowInteractor()->GetRenderWindow()->GetRenderers()->GetFirstRenderer() );

    vtkActor* actor = picker->GetActor();

    if ( actor != NULL )
      {
      dataInteractor->RemoveActorAndRender( actor );
      }
    }

  if ( pressedKey == '!' || pressedKey == '@' || pressedKey == '#' || pressedKey == '$' || pressedKey == '%' || 
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
        dataInteractor->UpdateAirwayGenerationAndRender( actor, 1 );
        }
      if ( pressedKey == '@' )
        {
        dataInteractor->UpdateAirwayGenerationAndRender( actor, 2 );
        }
      if ( pressedKey == '#' )
        {
        dataInteractor->UpdateAirwayGenerationAndRender( actor, 3 );
        }
      if ( pressedKey == '$' )
        {
        dataInteractor->UpdateAirwayGenerationAndRender( actor, 4 );
        }
      if ( pressedKey == '%' )
        {
        dataInteractor->UpdateAirwayGenerationAndRender( actor, 5 );
        }
      if ( pressedKey == '^' )
        {
        dataInteractor->UpdateAirwayGenerationAndRender( actor, 6 );
        }
      if ( pressedKey == '&' )
        {
        dataInteractor->UpdateAirwayGenerationAndRender( actor, 7 );
        }
      if ( pressedKey == '*' )
        {
        dataInteractor->UpdateAirwayGenerationAndRender( actor, 8 );
        }
      if ( pressedKey == '(' )
        {
        dataInteractor->UpdateAirwayGenerationAndRender( actor, 9 );
        }
      if ( pressedKey == ')' )
        {
        dataInteractor->UpdateAirwayGenerationAndRender( actor, 0 );
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
  
  if ( pressedKey == 'u' )
    {
    dataInteractor->UndoUpdateAndRender();  
    }
}

