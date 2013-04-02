/**
 *
 *  $Date: 2012-09-17 21:03:47 -0400 (Mon, 17 Sep 2012) $
 *  $Revision: 272 $
 *  $Author: jross $
 *
 *  TODO:
 *
 */

#include "cipVesselDataInteractor.h"
#include "vtkSmartPointer.h"
#include "vtkPropPicker.h"
#include "vtkRendererCollection.h"
#include "vtkTriangleStrip.h"
#include "vtkUnstructuredGrid.h"
#include "vtkDataSetMapper.h"
#include "vtkFloatArray.h"
#include "vtkPointData.h"
#include "vtkProperty.h"


cipVesselDataInteractor::cipVesselDataInteractor()
{
  this->InteractorCallbackCommand = vtkCallbackCommand::New();
  this->InteractorCallbackCommand->SetCallback( InteractorKeyCallback );
  this->InteractorCallbackCommand->SetClientData( (void*)this );

  //
  // cipVesselDataInteractor inherits from cipChestDataViewer. The
  // cipChestDataViewer constructor is called before the
  // cipVesselDataInteractor constructor, and in the parent constructor,
  // the ViewerCallbackCommand is set. We want to remove it and set
  // the InteractorCallbackCommand in order for the key bindings
  // specific to interaction can take effect
  //
  this->RenderWindowInteractor->RemoveObserver( this->ViewerCallbackCommand );
  this->RenderWindowInteractor->AddObserver( vtkCommand::KeyPressEvent, this->InteractorCallbackCommand );
}


void cipVesselDataInteractor::UpdateVesselGenerationAndRender( vtkActor* actor, int generation )
{
  // Save data to undo stack    
  this->m_UndoState.actor = actor;
  actor->GetProperty()->GetColor( this->m_UndoState.color );
  this->m_UndoState.opacity = actor->GetProperty()->GetOpacity(); 
    
  std::map< std::string, vtkActor* >::iterator it = this->VesselParticlesActorMap.begin();

  double* color = new double[3];

  while ( it != this->VesselParticlesActorMap.end() )
    {
    if ( it->second == actor )
      {
      if ( generation == 0 )
        {
        this->Conventions->GetChestTypeColor( static_cast< unsigned char >( VESSELGENERATION0 ), color );
        }
      if ( generation == 1 )
        {
        this->Conventions->GetChestTypeColor( static_cast< unsigned char >( VESSELGENERATION1 ), color );
        }
      if ( generation == 2 )
        {
        this->Conventions->GetChestTypeColor( static_cast< unsigned char >( VESSELGENERATION2 ), color );
        }
      if ( generation == 3 )
        {
        this->Conventions->GetChestTypeColor( static_cast< unsigned char >( VESSELGENERATION3 ), color );
        }
      if ( generation == 4 )
        {
        this->Conventions->GetChestTypeColor( static_cast< unsigned char >( VESSELGENERATION4 ), color );
        }
      if ( generation == 5 )
        {
        this->Conventions->GetChestTypeColor( static_cast< unsigned char >( VESSELGENERATION5 ), color );
        }
      if ( generation == 6 )
        {
        this->Conventions->GetChestTypeColor( static_cast< unsigned char >( VESSELGENERATION6 ), color );
        }
      if ( generation == 7 )
        {
        this->Conventions->GetChestTypeColor( static_cast< unsigned char >( VESSELGENERATION7 ), color );
        }
      if ( generation == 8 )
        {
        this->Conventions->GetChestTypeColor( static_cast< unsigned char >( VESSELGENERATION8 ), color );
        }
      if ( generation == 9 )
        {
        this->Conventions->GetChestTypeColor( static_cast< unsigned char >( VESSELGENERATION9 ), color );
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


void cipVesselDataInteractor::UpdateVeinParticlesAndRender( vtkActor* actor )
{
  // Save data to undo stack
  this->m_UndoState.actor = actor;
  actor->GetProperty()->GetColor( this->m_UndoState.color );
  this->m_UndoState.opacity = actor->GetProperty()->GetOpacity();  
  // Update actor  
  actor->GetProperty()->SetColor( 1, 0, 0 );
  actor->GetProperty()->SetOpacity( 1.0 );

  this->RenderWindow->Render();
}


void cipVesselDataInteractor::UpdateArteryParticlesAndRender( vtkActor* actor )
{
  // Save data to undo stack
  this->m_UndoState.actor = actor;
  actor->GetProperty()->GetColor( this->m_UndoState.color );
  this->m_UndoState.opacity = actor->GetProperty()->GetOpacity(); 
  // Update actor   
  actor->GetProperty()->SetColor( 0, 0, 1 );
  actor->GetProperty()->SetOpacity( 1.0 );

  this->RenderWindow->Render();
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


void InteractorKeyCallback( vtkObject* obj, unsigned long b, void* clientData, void* d )
{
  cipVesselDataInteractor* dataInteractor = reinterpret_cast< cipVesselDataInteractor* >( clientData );

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
  
  if ( pressedKey == 'a' )
    {
    int* clickPos = dataInteractor->GetRenderWindowInteractor()->GetEventPosition();
    
    vtkSmartPointer< vtkPropPicker > picker = vtkSmartPointer< vtkPropPicker >::New();
    
    picker->Pick( clickPos[0], clickPos[1], 0, 
                  dataInteractor->GetRenderWindowInteractor()->GetRenderWindow()->GetRenderers()->GetFirstRenderer() );
    
    vtkActor* actor = picker->GetActor();
    
    if ( actor != NULL )
      {
      dataInteractor->UpdateArteryParticlesAndRender( actor );
      }
    }
  
  if ( pressedKey == 'v' )
    {
    int* clickPos = dataInteractor->GetRenderWindowInteractor()->GetEventPosition();
    
    vtkSmartPointer< vtkPropPicker > picker = vtkSmartPointer< vtkPropPicker >::New();
    
    picker->Pick( clickPos[0], clickPos[1], 0, 
                  dataInteractor->GetRenderWindowInteractor()->GetRenderWindow()->GetRenderers()->GetFirstRenderer() );
    
    vtkActor* actor = picker->GetActor();
    
    if ( actor != NULL )
      {
      dataInteractor->UpdateVeinParticlesAndRender( actor );
      }
    }
  
  if ( pressedKey == 'u' )
    {
    dataInteractor->UndoUpdateAndRender();  
    }
}

