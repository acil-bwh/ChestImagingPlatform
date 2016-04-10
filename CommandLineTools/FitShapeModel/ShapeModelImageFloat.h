#ifndef _ShapeModelImageFloat_h_
#define _ShapeModelImageFloat_h_

#include "ShapeModelImage.h"
#include "ShapeModelImageITK.h"

class ShapeModelImageFloat : public ShapeModelImage
{
public:
  ShapeModelImageFloat() {};
  virtual ~ShapeModelImageFloat() {};
  virtual void read( const std::string& fileName ) 
  {
    _image.read( fileName ); 
  }
  virtual void getCenter( double* center ) const
  {
    _image.getCenter( center );
  }
  virtual void getSpacing( double* spacing ) const
  {
    _image.getSpacing( spacing );
  }
  virtual void createBinaryMeshImage( MeshType::Pointer mesh,
                                      const std::string& outputName ) const
  {
    _image.createBinaryMeshImage( mesh, outputName );
  }
  virtual void createBinaryVolumeImage ( PoissonRecon::VolumeData& volume,
                                         const std::string& outputName ) const
  {
    _image.createBinaryVolumeImage( volume, outputName );
  }
  virtual void createGradientMagnitudeImage( double sigma,
                                             const std::string& outputName ) const
  {
    _image.createGradientMagnitudeImage( sigma, outputName );
  }
  FitShapeModelType< float >::ImageType::Pointer getImageITK() const 
  {
    return _image.getImageITK();
  }
private:
  ShapeModelImageITK< float > _image;
};

#endif // _ShapeModelImageFloat_h_
