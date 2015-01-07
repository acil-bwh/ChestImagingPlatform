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

  // cipAirwayDataInteractor inherits from cipChestDataViewer. The
  // cipChestDataViewer constructor is called before the
  // cipAirwayDataInteractor constructor, and in the parent constructor,
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
  this->ParticleDistanceThreshold = 20.0;
  this->AirwayBranchCode          = "";
  this->ActorColor                = new double[3];

  this->AirwayModelActor = vtkSmartPointer< vtkActor >::New();
  this->AirwayModel = vtkSmartPointer< vtkPolyData >::New();
  this->AirwayModelShowing = false;
}

void cipAirwayDataInteractor::SetRootNode( vtkActor* actor )
{
  this->MinimumSpanningTreeRootNode = this->ActorToParticleIDMap[actor];
  std::cout << "Root node particle ID:\t" << this->MinimumSpanningTreeRootNode << std::endl;
  std::cout << this->AirwayParticles->GetPoint(this->MinimumSpanningTreeRootNode)[0] << "\t";
  std::cout << this->AirwayParticles->GetPoint(this->MinimumSpanningTreeRootNode)[1] << "\t";
  std::cout << this->AirwayParticles->GetPoint(this->MinimumSpanningTreeRootNode)[2] << std::endl;
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
      this->SelectedChestRegion = (unsigned char)(cip::UNDEFINEDREGION);
      this->SelectedChestType   = (unsigned char)(cip::TRACHEA);
      this->Conventions->GetChestTypeColor( static_cast< unsigned char >( cip::TRACHEA ), this->ActorColor );
      }
    return;
    }    

  this->AirwayBranchCode.append( 1, c );
  
  std::cout << "Code:\t" << this->AirwayBranchCode << std::endl;
  if ( this->AirwayBranchCode.compare( "LMB" ) == 0 )
    {
    std::cout << "Setting airway branch to left main bronchi..." << std::endl;
    this->Conventions->GetChestTypeColor( (unsigned char)(cip::MAINBRONCHUS), this->ActorColor );
    this->SelectedChestRegion = (unsigned char)(cip::LEFT);
    this->SelectedChestType   = (unsigned char)(cip::MAINBRONCHUS);
    }
  if ( this->AirwayBranchCode.compare( "IMB" ) == 0 )
    {
    std::cout << "Setting airway branch to right main bronchi..." << std::endl;
    this->SelectedChestRegion = (unsigned char)(cip::RIGHT);
    this->SelectedChestType   = (unsigned char)(cip::MAINBRONCHUS);
    this->Conventions->GetChestTypeColor( static_cast< unsigned char >( cip::MAINBRONCHUS ), this->ActorColor );
    }
  if ( this->AirwayBranchCode.compare( "IIB" ) == 0 )
    {
    std::cout << "Setting airway branch to intermediate bronchus (right lung)..." << std::endl;
    this->SelectedChestRegion = (unsigned char)(cip::RIGHTLUNG);
    this->SelectedChestType   = (unsigned char)(cip::INTERMEDIATEBRONCHUS);
    this->Conventions->GetChestTypeColor( static_cast< unsigned char >( cip::INTERMEDIATEBRONCHUS ), this->ActorColor );
    }
  if ( this->AirwayBranchCode.compare( "LLB" ) == 0 )
    {
    std::cout << "Setting airway branch to left lower lobe bronchus..." << std::endl;
    this->SelectedChestRegion = (unsigned char)(cip::LEFTLUNG);
    this->SelectedChestType   = (unsigned char)(cip::LOWERLOBEBRONCHUS);
    this->Conventions->GetChestTypeColor( static_cast< unsigned char >( cip::LOWERLOBEBRONCHUS ), this->ActorColor );
    }
  if ( this->AirwayBranchCode.compare( "ILB" ) == 0 )
    {
    std::cout << "Setting airway branch to right lower lobe bronchus..." << std::endl;
    this->SelectedChestRegion = (unsigned char)(cip::RIGHTLUNG);
    this->SelectedChestType   = (unsigned char)(cip::LOWERLOBEBRONCHUS);
    this->Conventions->GetChestTypeColor( static_cast< unsigned char >( cip::LOWERLOBEBRONCHUS ), this->ActorColor );
    }
  if ( this->AirwayBranchCode.compare( "SD" ) == 0 )
    {
    std::cout << "Setting airway branch to superior division bronchus (left lung)..." << std::endl;
    this->SelectedChestRegion = (unsigned char)(cip::LEFTLUNG);
    this->SelectedChestType   = (unsigned char)(cip::SUPERIORDIVISIONBRONCHUS);
    this->Conventions->GetChestTypeColor( static_cast< unsigned char >( cip::SUPERIORDIVISIONBRONCHUS ), this->ActorColor );
    }
  if ( this->AirwayBranchCode.compare( "LB" ) == 0 )
    {
    std::cout << "Setting airway branch to lingular bronchus (left lung)..." << std::endl;
    this->SelectedChestRegion = (unsigned char)(cip::LEFTLUNG);
    this->SelectedChestType   = (unsigned char)(cip::LINGULARBRONCHUS);
    this->Conventions->GetChestTypeColor( static_cast< unsigned char >( cip::LINGULARBRONCHUS ), this->ActorColor );
    }
  if ( this->AirwayBranchCode.compare( "ML" ) == 0 )
    {
    std::cout << "Setting airway branch to middle lobe bronchus (right lung)..." << std::endl;
    this->SelectedChestRegion = (unsigned char)(cip::RIGHTLUNG);
    this->SelectedChestType   = (unsigned char)(cip::MIDDLELOBEBRONCHUS);
    this->Conventions->GetChestTypeColor( static_cast< unsigned char >( cip::MIDDLELOBEBRONCHUS ), this->ActorColor );
    }
  if ( this->AirwayBranchCode.compare( "LUB" ) == 0 )
    {
    std::cout << "Setting airway branch to left upper lobe bronchus..." << std::endl;
    this->SelectedChestRegion = (unsigned char)(cip::LEFTLUNG);
    this->SelectedChestType   = (unsigned char)(cip::UPPERLOBEBRONCHUS);
    this->Conventions->GetChestTypeColor( static_cast< unsigned char >( cip::UPPERLOBEBRONCHUS ), this->ActorColor );
    }
  if ( this->AirwayBranchCode.compare( "IUB" ) == 0 )
    {
    std::cout << "Setting airway branch to right upper lobe bronchus..." << std::endl;
    this->SelectedChestRegion = (unsigned char)(cip::RIGHTLUNG);
    this->SelectedChestType   = (unsigned char)(cip::UPPERLOBEBRONCHUS);
    this->Conventions->GetChestTypeColor( static_cast< unsigned char >( cip::UPPERLOBEBRONCHUS ), this->ActorColor );
    }
  if ( this->AirwayBranchCode.compare( "L#" ) == 0 )
    {
    std::cout << "Setting airway branch to left lung generation 3..." << std::endl;
    this->SelectedChestRegion = (unsigned char)(cip::LEFTLUNG);
    this->SelectedChestType   = (unsigned char)(cip::AIRWAYGENERATION3);
    this->Conventions->GetChestTypeColor( static_cast< unsigned char >( cip::AIRWAYGENERATION3 ), this->ActorColor );
    }
  if ( this->AirwayBranchCode.compare( "L$" ) == 0 )
    {
    std::cout << "Setting airway branch to left lung generation 4..." << std::endl;
    this->SelectedChestRegion = (unsigned char)(cip::LEFTLUNG);
    this->SelectedChestType   = (unsigned char)(cip::AIRWAYGENERATION4);
    this->Conventions->GetChestTypeColor( static_cast< unsigned char >( cip::AIRWAYGENERATION4 ), this->ActorColor );
    }
  if ( this->AirwayBranchCode.compare( "L%" ) == 0 )
    {
    std::cout << "Setting airway branch to left lung generation 5..." << std::endl;
    this->SelectedChestRegion = (unsigned char)(cip::LEFTLUNG);
    this->SelectedChestType   = (unsigned char)(cip::AIRWAYGENERATION5);
    this->Conventions->GetChestTypeColor( static_cast< unsigned char >( cip::AIRWAYGENERATION5 ), this->ActorColor );
    }
  if ( this->AirwayBranchCode.compare( "I#" ) == 0 )
    {
    std::cout << "Setting airway branch to right lung generation 3..." << std::endl;
    this->SelectedChestRegion = (unsigned char)(cip::RIGHTLUNG);
    this->SelectedChestType   = (unsigned char)(cip::AIRWAYGENERATION3);
    this->Conventions->GetChestTypeColor( static_cast< unsigned char >( cip::AIRWAYGENERATION3 ), this->ActorColor );
    }
  if ( this->AirwayBranchCode.compare( "I$" ) == 0 )
    {
    std::cout << "Setting airway branch to right lung generation 4..." << std::endl;
    this->SelectedChestRegion = (unsigned char)(cip::RIGHTLUNG);
    this->SelectedChestType   = (unsigned char)(cip::AIRWAYGENERATION4);
    this->Conventions->GetChestTypeColor( static_cast< unsigned char >( cip::AIRWAYGENERATION4 ), this->ActorColor );
    }
  if ( this->AirwayBranchCode.compare( "I%" ) == 0 )
    {
    std::cout << "Setting airway branch to right lung generation 5..." << std::endl;
    this->SelectedChestRegion = (unsigned char)(cip::RIGHTLUNG);
    this->SelectedChestType   = (unsigned char)(cip::AIRWAYGENERATION5);
    this->Conventions->GetChestTypeColor( static_cast< unsigned char >( cip::AIRWAYGENERATION5 ), this->ActorColor );
    }
}

void cipAirwayDataInteractor::UndoAndRender()
{
  unsigned int lastModification = this->LabeledParticleIDs.size() - 1;

  float tmpRegion = float(cip::UNDEFINEDREGION);
  float tmpType   = float(cip::UNDEFINEDTYPE);

  for ( unsigned int i=0; i<this->LabeledParticleIDs[lastModification].size(); i++ )
    {
    unsigned int id = this->LabeledParticleIDs[lastModification][i];
    this->AirwayParticles->GetPointData()->GetArray("ChestType")->SetTuple( id, &tmpType );
    this->AirwayParticles->GetPointData()->GetArray("ChestRegion")->SetTuple( id, &tmpRegion );

    this->ParticleIDToActorMap[id]->GetProperty()->SetColor( 1.0, 1.0, 1.0 );
    }

  this->LabeledParticleIDs[lastModification].clear();
  this->LabeledParticleIDs.erase( this->LabeledParticleIDs.end() );

  this->RenderWindow->Render();
}

void cipAirwayDataInteractor::SetIntermediateNode( vtkActor* actor )
{
  this->MinimumSpanningTreeIntermediateNode = this->ActorToParticleIDMap[actor];  

 vtkSmartPointer< vtkGraphToPolyData > graphToPolyData = vtkSmartPointer< vtkGraphToPolyData >::New();
   graphToPolyData->SetInputData( this->MinimumSpanningTree );
   graphToPolyData->Update();
 
 vtkSmartPointer< vtkDijkstraGraphGeodesicPath > dijkstra = vtkSmartPointer< vtkDijkstraGraphGeodesicPath >::New();
   dijkstra->SetInputConnection( graphToPolyData->GetOutputPort() );
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
   if ( this->AirwayParticles->GetPointData()->GetArray("ChestType")->GetTuple(idList->GetId(i))[0] == float(cip::UNDEFINEDTYPE) )
     {
     float tmpRegion = (float)(this->SelectedChestRegion);
     float tmpType   = (float)(this->SelectedChestType);
     this->ParticleIDToActorMap[idList->GetId(i)]->GetProperty()->SetColor( this->ActorColor[0], this->ActorColor[1], this->ActorColor[2] );
     this->AirwayParticles->GetPointData()->GetArray("ChestRegion")->SetTuple(idList->GetId(i), &tmpRegion );
     this->AirwayParticles->GetPointData()->GetArray("ChestType")->SetTuple(idList->GetId(i), &tmpType );
     labeledIDs.push_back(idList->GetId(i));

     // Re-orient the particle's minor eigenvector. This is necessary for other algorithms
     // (specifically, some particles registration algorithms) which require that all the 
     // particles be oriented in a consistent manner -- in this case, all minor eigenvectors
     // will point from leaf node to root node.     
     if ( i < idList->GetNumberOfIds() - 1 )
       {
	 refVec[0] = this->AirwayParticles->GetPoint(idList->GetId(i+1))[0] - this->AirwayParticles->GetPoint(idList->GetId(i))[0];
	 refVec[1] = this->AirwayParticles->GetPoint(idList->GetId(i+1))[1] - this->AirwayParticles->GetPoint(idList->GetId(i))[1];
	 refVec[2] = this->AirwayParticles->GetPoint(idList->GetId(i+1))[2] - this->AirwayParticles->GetPoint(idList->GetId(i))[2];
       }
     else
       {
	 refVec[0] = this->AirwayParticles->GetPoint(idList->GetId(i))[0] - this->AirwayParticles->GetPoint(idList->GetId(i-1))[0];
	 refVec[1] = this->AirwayParticles->GetPoint(idList->GetId(i))[1] - this->AirwayParticles->GetPoint(idList->GetId(i-1))[1];
	 refVec[2] = this->AirwayParticles->GetPoint(idList->GetId(i))[2] - this->AirwayParticles->GetPoint(idList->GetId(i-1))[2];
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
void cipAirwayDataInteractor::OrientParticle( unsigned int particleID, cip::VectorType& refVec )
{
  cip::VectorType hevec2(3);
    hevec2[0] = this->AirwayParticles->GetPointData()->GetArray( "hevec2" )->GetTuple( particleID )[0];
    hevec2[1] = this->AirwayParticles->GetPointData()->GetArray( "hevec2" )->GetTuple( particleID )[1];
    hevec2[2] = this->AirwayParticles->GetPointData()->GetArray( "hevec2" )->GetTuple( particleID )[2];

  double angle = cip::GetAngleBetweenVectors( hevec2, refVec, false );

  if ( angle > vnl_math::pi/2.0 )
    {
      float hevec2flipped[3];
        hevec2flipped[0] = -hevec2[0];
	hevec2flipped[1] = -hevec2[1];
	hevec2flipped[2] = -hevec2[2];

      this->AirwayParticles->GetPointData()->GetArray("hevec2")->SetTuple( particleID, hevec2flipped );
    }
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
        this->Conventions->GetChestTypeColor( static_cast< unsigned char >( cip::TRACHEA ), color );
        }
      if ( generation == 1 )
        {
        this->Conventions->GetChestTypeColor( static_cast< unsigned char >( cip::MAINBRONCHUS ), color );
        }
      if ( generation == 2 )
        {
        this->Conventions->GetChestTypeColor( static_cast< unsigned char >( cip::UPPERLOBEBRONCHUS ), color );
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

void cipAirwayDataInteractor::SetAirwayModel( vtkSmartPointer< vtkPolyData > model )
{
  this->AirwayModel = model;
  this->AirwayModelActor = this->SetPolyData( model, "airwayModel" );
  this->SetActorColor( "airwayModel", 1.0, 1.0, 1.0 );
  this->SetActorOpacity( "airwayModel", 0.3 );

  this->AirwayModelShowing = true;
}

void cipAirwayDataInteractor::ShowAirwayModel()
{
  if ( !this->AirwayModelShowing )
    {
      this->AirwayModelShowing = true;

      this->Renderer->AddActor( this->AirwayModelActor);
      this->RenderWindow->Render();
    }
}

void cipAirwayDataInteractor::HideAirwayModel()
{
  if ( this->AirwayModelShowing )
    {
      this->AirwayModelShowing = false;

      this->Renderer->RemoveActor( this->AirwayModelActor );
      this->RenderWindow->Render();
    }
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


void cipAirwayDataInteractor::SetAirwayParticlesAsMinimumSpanningTree( vtkSmartPointer< vtkPolyData > particles, 
								       double particleSize )
{
  this->AirwayParticles = particles;

  this->NumberInputParticles    = particles->GetNumberOfPoints();
  this->NumberOfPointDataArrays = particles->GetPointData()->GetNumberOfArrays();

  // First create a minimum spanning tree representation
  this->InitializeMinimumSpanningTree( particles );

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

    vtkActor* actor = this->SetAirwayParticlesAsDiscs( tmpPolyData, particleSize, name );
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

  cip::VectorType connectingVec(3);
    connectingVec[0] = point1[0] - point2[0];
    connectingVec[1] = point1[1] - point2[1];
    connectingVec[2] = point1[2] - point2[2];

  double connectorMagnitude = cip::GetVectorMagnitude( connectingVec );

  if ( connectorMagnitude > this->ParticleDistanceThreshold )
    {
    return false;
    }

  cip::VectorType particle1Hevec2(3);
    particle1Hevec2[0] = particles->GetPointData()->GetArray( "hevec2" )->GetTuple( particleID1 )[0];
    particle1Hevec2[1] = particles->GetPointData()->GetArray( "hevec2" )->GetTuple( particleID1 )[1];
    particle1Hevec2[2] = particles->GetPointData()->GetArray( "hevec2" )->GetTuple( particleID1 )[2];

  cip::VectorType particle2Hevec2(3);
    particle2Hevec2[0] = particles->GetPointData()->GetArray( "hevec2" )->GetTuple( particleID2 )[0];
    particle2Hevec2[1] = particles->GetPointData()->GetArray( "hevec2" )->GetTuple( particleID2 )[1];
    particle2Hevec2[2] = particles->GetPointData()->GetArray( "hevec2" )->GetTuple( particleID2 )[2];

  double angle1 =  cip::GetAngleBetweenVectors( particle1Hevec2, connectingVec, true );
  double angle2 =  cip::GetAngleBetweenVectors( particle2Hevec2, connectingVec, true );

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
  cipAirwayDataInteractor* dataInteractor = reinterpret_cast< cipAirwayDataInteractor* >( clientData );

  char pressedKey = dataInteractor->GetRenderWindowInteractor()->GetKeyCode(); 
  std::cout << pressedKey << std::endl;

  if ( pressedKey == 'T' || pressedKey == 'I' || pressedKey == 'M' || pressedKey == 'B' ||
       pressedKey == 'C' || pressedKey == 'L' || pressedKey == 'S' || pressedKey == 'D' ||
       pressedKey == 'M' || pressedKey == 'U' || pressedKey == '$' || pressedKey == '%' ||
       pressedKey == '#' )
    {
    dataInteractor->UpdateAirwayBranchCode( pressedKey );
    }
  else if ( pressedKey == 'k' )
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
  else if ( pressedKey == 's' )
    {
      dataInteractor->ShowAirwayModel();
    }
  else if ( pressedKey == 'h' )
    {
      dataInteractor->HideAirwayModel();
    }
  else if ( pressedKey == 'u' )
    {
    dataInteractor->UndoAndRender();
    }
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
}

