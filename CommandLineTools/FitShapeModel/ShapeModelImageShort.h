#ifndef _ShapeModelmageShort_h_
#define _ShapeModelmageShort_h_

#include "ShapeModelImage.h"
#include "ShapeModelImageITK.h"

class ShapeModelImageShort : public ShapeModelImage
{
public:
  ShapeModelImageShort() {};
  virtual ~ShapeModelImageShort() {};
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
  virtual void createBinaryVolumeImage ( std::vector< PoissonRecon::VolumeData* > volumes,
                                         const std::string& outputName ) const
  {
    _image.createBinaryVolumeImage( volumes, outputName );
  }
  virtual void createGradientMagnitudeImage( double sigma,
                                             const std::string& outputName ) const
  {
    _image.createGradientMagnitudeImage( sigma, outputName );
  }
  FitShapeModelType< short >::ImageType::Pointer getImageITK() const 
  {
    return _image.getImageITK();
  }
private:
  ShapeModelImageITK< short > _image;
};

#endif // _ShapeModelmageShort_h_
