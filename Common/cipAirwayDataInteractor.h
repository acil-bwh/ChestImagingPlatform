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

#include "vtkSmartPointer.h"
#include "vtkBoostKruskalMinimumSpanningTree.h"
#include "cipChestDataViewer.h"
#include "vtkMutableUndirectedGraph.h"

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
  void UndoUpdateAndRender();
  void SetRootNode( vtkActor* );
  void SetIntermediateNode( vtkActor* );
  void UpdateAirwayBranchCode( char );
  void UndoAndRender();

  /** Set airway particles poly data. Once read in, a minimum
   *  spanning tree representation will be created. This
   *  representation will enable labeling based on root node and
   *  intermediate node specification: once a root node is specified,
   *  a label will assigned to every node between a specified
   *  intermediate node and the root node. */
  void SetAirwayParticlesAsMinimumSpanningTree( vtkSmartPointer< vtkPolyData >, double );

  void SetAirwayModel( vtkSmartPointer< vtkPolyData > );
  void HideAirwayModel();
  void ShowAirwayModel();

  bool AirwayModelShowing;

private:
  void InitializeMinimumSpanningTree( vtkSmartPointer< vtkPolyData > );
  bool GetEdgeWeight( unsigned int, unsigned int, vtkSmartPointer< vtkPolyData >, double* );
  void OrientParticle( unsigned int, cip::VectorType& );

  std::map< vtkActor*, unsigned int > ActorToParticleIDMap;
  std::map< unsigned int, vtkActor* > ParticleIDToActorMap;

  vtkSmartPointer< vtkPolyData > AirwayModel;
  vtkSmartPointer< vtkActor > AirwayModelActor;
  vtkCallbackCommand* InteractorCallbackCommand;
  UndoState m_UndoState; 

  vtkSmartPointer< vtkMutableUndirectedGraph > MinimumSpanningTree;
  vtkSmartPointer< vtkPolyData > AirwayParticles;
  unsigned int NumberInputParticles;
  unsigned int NumberOfPointDataArrays;
  unsigned int MinimumSpanningTreeRootNode;
  unsigned int MinimumSpanningTreeIntermediateNode;
  double ParticleDistanceThreshold;
  double EdgeWeightAngleSigma;
  unsigned char SelectedChestRegion;
  unsigned char SelectedChestType;
  std::vector< std::vector< unsigned int > > LabeledParticleIDs;

  double* ActorColor;
  std::string AirwayBranchCode;
};

#endif
