/**
 *  \class cipVesselDataInteractor
 *  \ingroup common
 *  \brief  This class inherits the viewing capabilty of the
 *  cipChestDataViewer but also provides the ability to interact with
 *  vessel data (e.g. labeling particles as being veins or arteries,
 *  indicating generation numbers, etc).
 *
 *  $Date: 2012-06-27 11:39:23 -0700 (Wed, 27 Jun 2012) $
 *  $Revision: 191 $
 *  $Author: rjosest $
 *
 *  TODO:
 *
 */

#ifndef __cipVesselDataInteractor_h
#define __cipVesselDataInteractor_h

#include "cipChestDataViewer.h"


void InteractorKeyCallback( vtkObject*, unsigned long, void*, void* );

class UndoState {
  public:
    vtkActor *actor;
    double color[3];
    double opacity;
};

class cipVesselDataInteractor : public cipChestDataViewer
{
public:
  ~cipVesselDataInteractor(){};
  cipVesselDataInteractor();

  void RemoveActorAndRender( vtkActor* );

  void UpdateVesselGenerationAndRender( vtkActor*, int );
  void UpdateArteryParticlesAndRender( vtkActor* );
  void UpdateVeinParticlesAndRender( vtkActor* );
  void UndoUpdateAndRender ();

private:
  vtkCallbackCommand* InteractorCallbackCommand;
  UndoState m_UndoState;
      
};

#endif
