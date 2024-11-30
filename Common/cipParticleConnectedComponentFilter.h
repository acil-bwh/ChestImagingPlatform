/**
 *  \class cipParticleConnectedComponentFilter
 *  \ingroup common
 *  \brief  This is a base class designed to incapsulate
 *  commonality between several particle filtering schemes (airway,
 *  fissure, and vessel). These filters are designed to filter noisy
 *  particles. They are based on connected component concepts. 
 *
 *  The input to the filter is a VTK poly data file. Assumptions are
 *  made about the contents of this file -- see inherited class headers
 *  for further information.
 *
 *  The filter proceeds by first organizing the particle data in an
 *  a vtkPointLocator object. This data structure is traversed and
 *  for each particle, a small neighborhood around that particle is then 
 *  searched, and if other particles are found, their connectivity to the 
 *  first particle is checked. If connected, a small neighborhood around the 
 *  second particle is checked for additional connections. This continues 
 *  recursively until no more particles are found to be connected to that 
 *  component. See inherited classes for specific criteria used for 
 *  connectivity decisions.
 *
 *  $Date: 2012-08-28 17:54:18 -0400 (Tue, 28 Aug 2012) $
 *  $Revision: 212 $
 *  $Author: jross $
 *
 */

#ifndef __cipParticleConnectedComponentFilter_h
#define __cipParticleConnectedComponentFilter_h

#include "vtkPolyData.h"
#include "vtkPointLocator.h"
#include "itkImage.h"

class cipParticleConnectedComponentFilter
{
public:
  ~cipParticleConnectedComponentFilter(){};
  cipParticleConnectedComponentFilter();

  void SetInput( vtkPolyData* );

  /** This allows the user to isolate a specific component to be
      pulled out in the output. It is used mainly for debugging
      purposes */
  void         SetSelectedComponent( unsigned int );
  unsigned int GetSelectedComponent();

  /** This allows the user to indicate how many particles must be in a
      component to survive the filtering process. */
  void         SetComponentSizeThreshold( unsigned int );
  unsigned int GetComponentSizeThreshold();

  /** These two methods allow you to get a particle's associated
      component label and the size of that label. Note these methods
      can only be called after the filter has been executed. */
  unsigned int GetComponentSizeFromParticleID( unsigned int );
  unsigned int GetComponentLabelFromParticleID( unsigned int );

  /** The maximum allowable distance between particles can be
      set/retrieved with these methods. The default values are set
      within the constructors of the inheriting classes. Particles
      that are within this distance from one another will be
      considered for connectivity. Otherwise, they will not. */
  void   SetParticleDistanceThreshold( double );
  double GetParticleDistanceThreshold();

  /** Set/Get the particle angle threshold. This value will be
      used to test whether two particles should be considered for
      connectivity. The vector connecting two particles is
      computed. The orientation of each particle is considered with
      respect to the connecting vec. Depending on the structure of
      interest (fissure, vessel, or airway), the threshold will be
      interpreted differently  */
  void   SetParticleAngleThreshold( double );
  double GetParticleAngleThreshold();

  /** Set/Get the maximum component size. No component will be allowed
      to grow larger than this */
  void         SetMaximumComponentSize( unsigned int );
  unsigned int GetMaximumComponentSize();

  unsigned int GetNumberOfOutputParticles();

  void Update();

  vtkPolyData* GetOutput();

  vtkPolyData* GetComponent( unsigned int );

protected:
  /** This is method needs to be overridden in inherint classes. By default it 
      simply returns true */
  virtual bool EvaluateParticleConnectedness( unsigned int, unsigned int );

  std::map< unsigned int, unsigned int >   ParticleToComponentMap;
  std::map< unsigned int, unsigned int >   ComponentSizeMap;

  double        GetVectorMagnitude( double[3] );
  double        GetVectorMagnitude( double[3], double[3] );
  double        GetAngleBetweenVectors( double[3], double[3], bool );
  unsigned int  GetComponentSize( unsigned int );
  void          ComputeComponentSizes();
  void          GetComponentParticleIndices( unsigned int, std::vector< unsigned int >* );
  void          QueryNeighborhood( unsigned int, unsigned int, unsigned int* );
  
  vtkPolyData* InputPolyData;
  vtkPolyData* OutputPolyData;
  vtkPointLocator* Locator;

  //double InterParticleSpacing;
  double ParticleDistanceThreshold;
  double ParticleAngleThreshold;

  unsigned int MaximumComponentSize;
  unsigned int NumberOfPointDataArrays;
  unsigned int ComponentSizeThreshold;
  unsigned int NumberInputParticles;
  unsigned int NumberOutputParticles;
  unsigned int SelectedComponent;
  unsigned int LargestComponentLabel;
};

#endif
