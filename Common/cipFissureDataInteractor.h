/**
 *  \class cipFissureDataInteractor
 *  \ingroup common
 *  \brief  This class inherits the viewing capabilty of the
 *  cipChestDataViewer but also provides the ability to interact with
 *  fissure data (e.g. labeling particles by generation)
 *
 *  $Date$
 *  $Revision$
 *  $Author$
 *
 *  TODO:
 *
 */

#ifndef __cipFissureDataInteractor_h
#define __cipFissureDataInteractor_h

#include "vtkSmartPointer.h"
#include "vtkBoostKruskalMinimumSpanningTree.h"
#include "cipChestDataViewer.h"
#include "vtkMutableUndirectedGraph.h"
#include "cipChestConventions.h"

void InteractorKeyCallback( vtkObject*, unsigned long, void*, void* );

class cipFissureDataInteractor : public cipChestDataViewer
{
public:
  ~cipFissureDataInteractor(){};
  cipFissureDataInteractor();

  void RemoveActorAndRender( vtkActor* );
  void HideActorAndRender( vtkActor* );
  void ColorActorAndRender( vtkActor*, unsigned char );

  /** Set fissure particles poly data. Once read in, a minimum
   *  spanning tree representation will be created. This
   *  representation will enable labeling based on root node and
   *  intermediate node specification: once a root node is specified,
   *  a label will assigned to every node between a specified
   *  intermediate node and the root node. */
  void SetConnectedFissureParticles( vtkSmartPointer< vtkPolyData >, double );

  /** Set the output file name. The user can safe work to this file
   *  as he/she goes along. */
  void SetFileName( std::string );
  void Write();

private:
  std::map< vtkActor*, unsigned int > ActorToParticleIDMap;
  std::map< unsigned int, vtkActor* > ParticleIDToActorMap;

  vtkCallbackCommand* InteractorCallbackCommand;

  vtkSmartPointer< vtkMutableUndirectedGraph > MinimumSpanningTree;
  vtkSmartPointer< vtkPolyData > FissureParticles;
  unsigned int NumberInputParticles;
  unsigned int NumberOfPointDataArrays;

  std::string FileName;

  double* ActorColor;

  cip::ChestConventions Conventions;
};

#endif
