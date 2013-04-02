/**
 *  \class cipAirwayDataInteractor
 *  \ingroup common
 *  \brief  This class inherits the viewing capabilty of the
 *  cipChestDataViewer but also provides the ability to interact with
 *  airway data (e.g. labeling particles by generation)
 *
 *  $Date$
 *  $Revision$
 *  $Author$
 *
 *  TODO:
 *
 */

#ifndef __cipAirwayDataInteractor_h
#define __cipAirwayDataInteractor_h

#include "cipChestDataViewer.h"


void InteractorKeyCallback( vtkObject*, unsigned long, void*, void* );

class UndoState {
  public:
    vtkActor *actor;
    double color[3];
    double opacity;
};

class cipAirwayDataInteractor : public cipChestDataViewer
{
public:
  ~cipAirwayDataInteractor(){};
  cipAirwayDataInteractor();

  void RemoveActorAndRender( vtkActor* );
  void UpdateAirwayGenerationAndRender( vtkActor*, int );
  void UndoUpdateAndRender ();

private:
  vtkCallbackCommand* InteractorCallbackCommand;
  UndoState m_UndoState;      
};

#endif
