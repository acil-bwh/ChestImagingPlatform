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


void InteractorKeyCallback( vtkObject* obj, unsigned long b, void* clientData, void* d )
{
  cipAirwayDataInteractor* dataInteractor = reinterpret_cast< cipAirwayDataInteractor* >( clientData );

  char pressedKey = dataInteractor->GetRenderWindowInteractor()->GetKeyCode(); 

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
       pressedKey == '^' || pressedKey == '&' || pressedKey == '*' || pressedKey == '(' || pressedKey == ')' )
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

