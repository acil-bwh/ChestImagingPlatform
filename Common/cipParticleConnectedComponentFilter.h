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
 *  image-based data structure. This facilitates traversal and access
 *  of the data. In order to construct this dataset, a spacing value
 *  must be determined. The current implementation uses a spacing value
 *  of 1/2 the specified inter-particles distance. This data structure
 *  is common to all inherited classes.
 *
 *  In order to filter the particles, the data structure image is
 *  traversed until a particle is found. A small neighborhood around
 *  that particle is then searched, and if other particles are found,
 *  their connectivity to the first particle is checked. If connected,
 *  a small neighborhood around the second particle is checked for
 *  additional connections. This continues recursively until no more
 *  particles are found to be connected to that component. During this
 *  time, connected particles are removed from further consideration
 *  from the data structure image. See inherited classes for specific
 *  criteria used for connectivity decisions.
 *
 *  $Date: 2012-08-28 17:54:18 -0400 (Tue, 28 Aug 2012) $
 *  $Revision: 212 $
 *  $Author: jross $
 *
 */

#ifndef __cipParticleConnectedComponentFilter_h
#define __cipParticleConnectedComponentFilter_h


#include "vtkPolyData.h"
#include "itkImage.h"
#include "itkImageRegionIteratorWithIndex.h"


class cipParticleConnectedComponentFilter
{
public:
  ~cipParticleConnectedComponentFilter(){};
  cipParticleConnectedComponentFilter();

  void SetInput( vtkPolyData* );

  /** This function should be set to the inter-particle spacing value
      that was used to generate the input particles. It is used to
      specify the spacing used for the data structure image */
  void   SetInterParticleSpacing( double );
  double GetInterParticleSpacing();

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
      can only be called after the filter has been executed. Also, the
      particle IDs are with respect to the internal poly data, which
      could be different from the input poly data */
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
  typedef itk::Image< unsigned int, 3 >                   ImageType;
  typedef itk::ImageRegionIteratorWithIndex< ImageType >  IteratorType;

  ImageType::Pointer DataStructureImage;

  /** This is method needs to be overridden in inherint classes. By default it 
      simply returns true */
  virtual bool EvaluateParticleConnectedness( unsigned int, unsigned int );

  std::map< unsigned int, unsigned int >   ParticleToComponentMap;
  std::map< unsigned int, unsigned int >   ComponentSizeMap;

  double        GetVectorMagnitude( double[3] );
  double        GetVectorMagnitude( double[3], double[3] );
  double        GetAngleBetweenVectors( double[3], double[3], bool );
  unsigned int  GetComponentSize( unsigned int );
  void          InitializeDataStructureImageAndInternalInputPolyData();
  void          ComputeComponentSizes();
  void          GetComponentParticleIndices( unsigned int, std::vector< unsigned int >* );
  void          QueryNeighborhood( ImageType::IndexType, unsigned int, unsigned int* );
  
  vtkPolyData* InputPolyData;
  vtkPolyData* InternalInputPolyData;
  vtkPolyData* OutputPolyData;

  double InterParticleSpacing;
  double ParticleDistanceThreshold;
  double ParticleAngleThreshold;

  unsigned int MaximumComponentSize;
  unsigned int NumberOfPointDataArrays;
  unsigned int ComponentSizeThreshold;
  unsigned int NumberInputParticles;
  unsigned int NumberOutputParticles;
  unsigned int NumberInternalInputParticles;
  unsigned int SelectedComponent;
  unsigned int LargestComponentLabel;
};

#endif
