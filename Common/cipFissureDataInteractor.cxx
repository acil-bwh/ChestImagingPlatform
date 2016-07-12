/**
 *
 *  $Date$
 *  $Revision$
 *  $Author$
 *
 *  TODO:
 *
 */

#include "cipFissureDataInteractor.h"
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

cipFissureDataInteractor::cipFissureDataInteractor()
{
  this->InteractorCallbackCommand = vtkCallbackCommand::New();
  this->InteractorCallbackCommand->SetCallback( InteractorKeyCallback );
  this->InteractorCallbackCommand->SetClientData( (void*)this );

  // cipFissureDataInteractor inherits from cipChestDataViewer. The
  // cipChestDataViewer constructor is called before the
  // cipFissureDataInteractor constructor, and in the parent constructor,
  // the ViewerCallbackCommand is set. We want to remove it and set
  // the InteractorCallbackCommand in order for the key bindings
  // specific to interaction can take effect
  this->RenderWindowInteractor->RemoveObserver( this->ViewerCallbackCommand );
  this->RenderWindowInteractor->AddObserver( vtkCommand::KeyPressEvent, this->InteractorCallbackCommand );

  this->NumberInputParticles      = 0;
  this->NumberOfPointDataArrays   = 0;
  this->ActorColor                = new double[3];
}

void cipFissureDataInteractor::SetFileName( std::string fileName )
{
  this->FileName = fileName;
}

void cipFissureDataInteractor::RemoveActorAndRender( vtkActor* actor )
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

void cipFissureDataInteractor::HideActorAndRender( vtkActor* actor )
{
  actor->GetProperty()->SetOpacity( 0.0 );
  this->RenderWindow->Render();
}

void cipFissureDataInteractor::ColorActorAndRender( vtkActor* actor, unsigned char cipType )
{
  double color[3];
  Conventions.GetColorFromChestRegionChestType( (unsigned char)(cip::UNDEFINEDREGION), cipType, color );

  actor->GetProperty()->SetColor( color[0], color[1], color[2] );
  this->RenderWindow->Render();
}

void cipFissureDataInteractor::SetConnectedFissureParticles( vtkSmartPointer< vtkPolyData > particles, 
							     double particleSize )
{
  this->FissureParticles = particles;

  this->NumberInputParticles    = particles->GetNumberOfPoints();
  this->NumberOfPointDataArrays = particles->GetPointData()->GetNumberOfArrays();


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

    this->SetFissureParticlesAsDiscs( tmpPolyData, particleSize, name );
    this->SetActorColor( name, 1.0, 1.0, 1.0 );
    this->SetActorOpacity( name, 1.0 );
      
    // this->ActorToParticleIDMap[actor] = p;
    // this->ParticleIDToActorMap[p]     = actor;
    }

  vtkSmartPointer< vtkPolyDataMapper > mapper = vtkSmartPointer< vtkPolyDataMapper >::New();
    mapper->SetInputData( particles );

  vtkSmartPointer< vtkActor > actor = vtkSmartPointer< vtkActor >::New();
    actor->SetMapper( mapper );
    actor->GetProperty()->SetColor( 0.0, 0.0, 0.0 );

  this->ActorMap["fissureParticles"] = actor;
  this->Renderer->AddActor( this->ActorMap["fissureParticles"] );
}

void cipFissureDataInteractor::Write()
{
  std::cout << "---Writing labeled particles..." << std::endl;
  vtkSmartPointer< vtkPolyDataWriter > writer = vtkSmartPointer< vtkPolyDataWriter >::New();
    writer->SetFileName( this->FileName.c_str() );
    writer->SetInputData( this->FissureParticles );
    writer->SetFileTypeToASCII();
    writer->Write();  
}

void InteractorKeyCallback( vtkObject* obj, unsigned long b, void* clientData, void* d )
{
  cipFissureDataInteractor* dataInteractor = reinterpret_cast< cipFissureDataInteractor* >( clientData );

  char pressedKey = dataInteractor->GetRenderWindowInteractor()->GetKeyCode(); 

  if ( pressedKey == 'k' || pressedKey == 'u' || pressedKey == 'o' || pressedKey == 'h' )
    {
    int* clickPos = dataInteractor->GetRenderWindowInteractor()->GetEventPosition();

    vtkSmartPointer< vtkPropPicker > picker = vtkSmartPointer< vtkPropPicker >::New();

    picker->Pick( clickPos[0], clickPos[1], 0, 
                  dataInteractor->GetRenderWindowInteractor()->GetRenderWindow()->GetRenderers()->GetFirstRenderer() );

    vtkActor* actor = picker->GetActor();

    if ( actor != NULL )
      {
	if ( pressedKey == 'k' )
	  {
	    dataInteractor->RemoveActorAndRender( actor );
	  }
	else if ( pressedKey == 'u' )
	  {
	    dataInteractor->ColorActorAndRender( actor, (unsigned char)(cip::FISSURE) );
	  }
	else if ( pressedKey == 'o' )
	  {
	    dataInteractor->ColorActorAndRender( actor, (unsigned char)(cip::OBLIQUEFISSURE) );
	  }
	else if ( pressedKey == 'h' )
	  {
	    dataInteractor->ColorActorAndRender( actor, (unsigned char)(cip::HORIZONTALFISSURE) );
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

