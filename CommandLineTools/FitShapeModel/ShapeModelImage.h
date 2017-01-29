#ifndef _ShapeModelImage_h_
#define _ShapeModelImage_h_

#include "FitShapeModelTypes.h"
#include "ShapeModelObject.h"
#include "PoissonRecon/PoissonRecon.h"

// 
// abstract base class to be used as an interface for different image types
// chose inheritance over templates to use this as a parameter for different
// objects (ShapeModelOptimizer, ShapeModelVisualizer etc.) without making them
// template classes
// example derived classes: ShapeModelImageShort, ShapeModelImageFloat, which
// are thin wrappers for the template class ShapeModelImageITK with actual 
// implementation of the functionalites
// 
class ShapeModelImage : public ShapeModelObject
{
public:
  ShapeModelImage() {};
  virtual ~ShapeModelImage() {};
  virtual void read( const std::string& fileName ) = 0;
  virtual void getCenter( double* center ) const = 0;
  virtual void getSpacing( double* spacing ) const = 0;
  virtual void createBinaryMeshImage ( MeshType::Pointer mesh,
                                       const std::string& outputName ) const = 0;
  virtual void createBinaryVolumeImage ( std::vector< PoissonRecon::VolumeData* > volumes,
                                         const std::string& outputName ) const = 0;
  virtual void createGradientMagnitudeImage ( double sigma,
                                              const std::string& outputName ) const = 0;
};

#endif