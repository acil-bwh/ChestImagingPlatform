/**
 *  \class cipVesselDataInteractor
 *  \ingroup common
 *  \brief  This class inherits the viewing capabilty of the
 *  cipChestDataViewer but also provides the ability to interact with
 *  vessel data (e.g. labeling particles by generation)
 *
 *  $Date$
 *  $Revision$
 *  $Author$
 *
 *  TODO:
 *
 */

#ifndef __cipVesselDataInteractor_h
#define __cipVesselDataInteractor_h

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

class cipVesselDataInteractor : public cipChestDataViewer
{
public:
  ~cipVesselDataInteractor(){};
  cipVesselDataInteractor();

  void RemoveActorAndRender( vtkActor* );
  void HideActorAndRender( vtkActor* );
  void ColorActorByChestTypeAndRender( vtkActor*, unsigned char );
  void UpdateVesselGenerationAndRender( vtkActor*, int );
  void UndoUpdateAndRender();
  void SetRootNode( vtkActor* );
  void SetIntermediateNode( vtkActor* );
  void UpdateVesselBranchCode( char );
  void UndoAndRender();

  /** Set vessel particles poly data. Once read in, a minimum
   *  spanning tree representation will be created. This
   *  representation will enable labeling based on root node and
   *  intermediate node specification: once a root node is specified,
   *  a label will assigned to every node between a specified
   *  intermediate node and the root node. */
  void SetConnectedVesselParticles( vtkSmartPointer< vtkPolyData >, double );

  void SetVesselModel( vtkSmartPointer< vtkPolyData > );
  void HideVesselModel();
  void ShowVesselModel();

  /** Set the output file name. The user can safe work to this file
   *  as he/she goes along. */
  void SetFileName( std::string );
  void Write();

  bool VesselModelShowing;

private:
  void InitializeMinimumSpanningTree( vtkSmartPointer< vtkPolyData > );
  bool GetEdgeWeight( unsigned int, unsigned int, vtkSmartPointer< vtkPolyData >, double* );
  void OrientParticle( unsigned int, const cip::VectorType& );

  std::map< vtkActor*, unsigned int > ActorToParticleIDMap;
  std::map< unsigned int, vtkActor* > ParticleIDToActorMap;

  vtkSmartPointer< vtkPolyData > VesselModel;
  vtkSmartPointer< vtkActor > VesselModelActor;
  vtkCallbackCommand* InteractorCallbackCommand;
  UndoState m_UndoState; 

  vtkSmartPointer< vtkMutableUndirectedGraph > MinimumSpanningTree;
  vtkSmartPointer< vtkPolyData > VesselParticles;
  unsigned int NumberInputParticles;
  unsigned int NumberOfPointDataArrays;
  unsigned int MinimumSpanningTreeRootNode;
  unsigned int MinimumSpanningTreeIntermediateNode;
  double ParticleDistanceThreshold;
  double EdgeWeightAngleSigma;
  unsigned char SelectedChestRegion;
  unsigned char SelectedChestType;
  std::vector< std::vector< unsigned int > > LabeledParticleIDs;

  std::string FileName;

  double* ActorColor;
  std::string VesselBranchCode;
};

#endif
