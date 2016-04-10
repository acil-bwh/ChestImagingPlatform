#ifndef _ShapeModelImageITK_h_
#define _ShapeModelImageITK_h_

#include "FitShapeModelTypes.h"
#include "PoissonRecon/PoissonRecon.h"

// -----------------------------------------------------------------------------
// declaration
// -----------------------------------------------------------------------------
template < typename T >
class ShapeModelImageITK
{
public:
  typedef FitShapeModelType< T > FT;
  void read( const std::string& fileName );
  void getCenter( double* ) const;
  void getSpacing( double* ) const;
  void createBinaryMeshImage( MeshType::Pointer mesh,
                              const std::string& outputName ) const;
  void createBinaryVolumeImage ( PoissonRecon::VolumeData& volume,
                                 const std::string& outputName ) const;
  void createGradientMagnitudeImage( double sigma,
                                     const std::string& outputName ) const;
  typename FT::ImageType::Pointer getImageITK() const { return _image; }
private:
  typename FT::ImageType::Pointer _image;
};

// -----------------------------------------------------------------------------
// definition
// -----------------------------------------------------------------------------
template < typename T >
void
ShapeModelImageITK< T >::read( const std::string& fileName )
{
  // Read the input ct image
  std::cout << "Reading CT image..." << std::endl;
  typename FT::ImageReaderType::Pointer imageReader = FT::ImageReaderType::New();
  imageReader->SetFileName( fileName.c_str() );
  try
  {
    imageReader->Update();
  }
  catch (itk::ExceptionObject& e)
  {
    throw std::runtime_error( e.what() );
  }

  _image = imageReader->GetOutput();
  typename FT::ImageType::SizeType sz = _image->GetLargestPossibleRegion().GetSize();

  typename FT::ImageType::IndexType originIndex = {0, 0, 0}, centerIndex = {sz[0]/2, sz[1]/2, sz[2]/2};
  typename FT::ImageType::PointType originPoint, centerPoint;
  _image->TransformIndexToPhysicalPoint(originIndex, originPoint);
  _image->TransformIndexToPhysicalPoint(centerIndex, centerPoint);

  std::cout << "origin point: " << originPoint << std::endl;
  std::cout << "center point: " << centerPoint << std::endl;
  std::cout << "spacing: " << _image->GetSpacing() << std::endl;
}

template < typename T >
void
ShapeModelImageITK< T >::getCenter( double* center ) const
{
  typename FT::ImageType::SizeType sz = _image->GetLargestPossibleRegion().GetSize();
  typename FT::ImageType::IndexType centerIndex = { sz[0]/2, sz[1]/2, sz[2]/2} ;

  typename FT::ImageType::PointType centerPoint;
  _image->TransformIndexToPhysicalPoint( centerIndex, centerPoint );
  
  for (unsigned int k = 0; k < 3; k++)
  {
    center[k] = centerPoint[k];
  }
}

template < typename T >
void
ShapeModelImageITK< T >::getSpacing( double* spacing ) const
{
  typename FT::ImageType::SpacingType itk_spacing = _image->GetSpacing();
  
  for (unsigned int k = 0; k < 3; k++)
  {
    spacing[k] = itk_spacing[k];
  }
}

template < typename T >
void
ShapeModelImageITK< T >::createBinaryMeshImage( MeshType::Pointer mesh, 
                                                const std::string& outputName ) const
{
  typename FT::TriangleMeshToBinaryImageFilterType::Pointer meshFilter = 
    FT::TriangleMeshToBinaryImageFilterType::New();

  meshFilter->SetTolerance( 1.0 );
  meshFilter->SetSpacing( _image->GetSpacing() );
  meshFilter->SetOrigin( _image->GetOrigin() );
  meshFilter->SetSize( _image->GetLargestPossibleRegion().GetSize() );
  meshFilter->SetIndex( _image->GetLargestPossibleRegion().GetIndex() );
  meshFilter->SetDirection( _image->GetDirection() );
  meshFilter->SetInput( mesh );

  try
  {
    meshFilter->Update();
  }
  catch (itk::ExceptionObject &e)
  {
    throw std::runtime_error( e.what() );
  }

  typename FT::ImageType::Pointer meshBinaryImage = meshFilter->GetOutput();
  typename FT::IteratorType iit( meshBinaryImage, meshBinaryImage->GetLargestPossibleRegion() );
  typename FT::ConstIteratorType it( _image, _image->GetLargestPossibleRegion() );

  PixelType maxValue = SHRT_MIN;
  PixelType minValue = SHRT_MAX;

  // find scalar value range of the image
  for (it.GoToBegin(); !it.IsAtEnd(); ++it)
  {
    const PixelType& value = it.Get();
    if (value < minValue)
    {
      minValue = value;
    }
    if (value > maxValue)
    {
      maxValue = value;
    }
  }

  // Overlay on top of the the input image
  bool useInputImage = true;

  if (useInputImage)
  {
    for (it.GoToBegin(), iit.GoToBegin(); !iit.IsAtEnd(); ++it, ++iit)
    {
      iit.Value() = it.Value() + (iit.Value() * 0.2 * maxValue); // match the highest intensity of the original image
    }

    // Write the output to file (or new volume in memory)
    std::cout << "Writing output..." << std::endl;
    typename FT::ImageWriterType::Pointer writer = FT::ImageWriterType::New();
    writer->SetFileName( outputName.c_str() );
    writer->UseCompressionOff();
    writer->SetInput( meshBinaryImage );

    try
    {
      writer->Update();
    }
    catch (itk::ExceptionObject& e)
    {
      throw std::runtime_error( e.what() );
    }
  }
}

template < typename T >
void 
ShapeModelImageITK< T >::createBinaryVolumeImage ( PoissonRecon::VolumeData& volume,
                                                   const std::string& outputName ) const
{
  unsigned int N = volume.res; // input (reference) volume resolution

  // create itk image from input volume data
  FloatImageType::RegionType region;
  FloatImageType::IndexType start;
  start[0] = start[1] = start[2] = 0;
 
  FloatImageType::SizeType size;
  size[0] = size[1] = size[2] = N;
 
  region.SetSize( size );
  region.SetIndex( start );
 
  FloatImageType::Pointer volumeImage = FloatImageType::New();
  volumeImage->SetRegions( region );
  volumeImage->Allocate();

  itk::ImageRegionIterator< FloatImageType > vit( volumeImage, volumeImage->GetLargestPossibleRegion() );
  unsigned int i = 0;
  for (vit.GoToBegin(); !vit.IsAtEnd(); ++vit)
  {
    vit.Value() = volume.data[i++];
  }
  
  typedef itk::LinearInterpolateImageFunction< FloatImageType, typename PointType::CoordRepType >
    LinearInterpolatorType;
  LinearInterpolatorType::Pointer volumeImageInterpolator = LinearInterpolatorType::New();
  volumeImageInterpolator->SetInputImage( volumeImage );
    
  // duplicate the input image and use it as a target composite image
  typename FT::ImageDuplicatorType::Pointer duplicator = FT::ImageDuplicatorType::New();
  duplicator->SetInputImage( _image );
  duplicator->Update();
  
  typename FT::ConstIteratorType it( _image, _image->GetLargestPossibleRegion() );
  typename FT::ImageType::Pointer compositeImage = duplicator->GetOutput();
  typename FT::IteratorType iit( compositeImage, compositeImage->GetLargestPossibleRegion() );

  PixelType maxValue = SHRT_MIN;
  PixelType minValue = SHRT_MAX;

  // find scalar value range of the image
  for (it.GoToBegin(); !it.IsAtEnd(); ++it)
  {
    const PixelType& value = it.Get();
    if (value < minValue)
    {
      minValue = value;
    }
    if (value > maxValue)
    {
      maxValue = value;
    }
  }

  // Overlay on top of the the input image
  bool useInputImage = true;

  if (useInputImage)
  {
    PointType vpt; // input volume index = position
    
    for (it.GoToBegin(), iit.GoToBegin(); !iit.IsAtEnd(); ++it, ++iit)
    {
      PointType pt;
      _image->TransformIndexToPhysicalPoint( iit.GetIndex(), pt );
      
      // transform point from input image space to volume image space
      for (int k = 0; k < 3; k++)
      {
        vpt[k] = (pt[k] - volume.center[k]) * N / volume.scale;
      }
      
      float binary_value = 0;
      FloatImageType::IndexType vid;
      bool inside = volumeImage->TransformPhysicalPointToIndex( vpt, vid );
      if (inside)
      {
        if (volumeImageInterpolator->Evaluate( vpt ) > 0) // point is inside of iso-surface of volume
        {
          binary_value = 1.0;
        }
      }
      
      iit.Value() = it.Value() + (binary_value * 0.2 * maxValue); // match the highest intensity of the original image
    }
    
    // Write the output to file (or new volume in memory)
    std::cout << "Writing output..." << std::endl;
    typename FT::ImageWriterType::Pointer writer = FT::ImageWriterType::New();
    writer->SetFileName( outputName.c_str() );
    writer->UseCompressionOff();
    writer->SetInput( compositeImage );

    try
    {
      writer->Update();
    }
    catch (itk::ExceptionObject& e)
    {
      throw std::runtime_error( e.what() );
    }
  }
}

template < typename T >
void
ShapeModelImageITK< T >::createGradientMagnitudeImage( double sigma,
                                                       const std::string& outputName ) const
{
  typename FT::GradientMagnitudeRecursiveGaussianImageFilterType::Pointer
    gradientMagnitudeFilter = FT::GradientMagnitudeRecursiveGaussianImageFilterType::New();
  gradientMagnitudeFilter->SetInput( _image );
  gradientMagnitudeFilter->SetSigma( sigma );
  try
  {
    std::cout << "Running gradient magnitude filter..." << std::endl;
    gradientMagnitudeFilter->Update();
    std::cout << "Done." << std::endl;
  }
  catch (itk::ExceptionObject& e)
  {
    throw std::runtime_error( e.what() );
  }
  typename FT::ImageType::Pointer gradientMagnitudeImage = gradientMagnitudeFilter->GetOutput();
  
  /*
  IteratorType git( _gradientMagnitudeImage, _gradientMagnitudeImage->GetLargestPossibleRegion() );
  for (git.GoToBegin(), iit.GoToBegin(); !iit.IsAtEnd(); ++git, ++iit)
  {
    git.Value() = git.Value() + (iit.Value() * 200); // match the highest intensity of the original image
  }
  */

  // Write the output to file (or new volume in memory)
  std::cout << "Writing output with gradient magnitude image..." << std::endl;
  typename FT::ImageWriterType::Pointer writer = FT::ImageWriterType::New();
  std::string magOutputName = outputName + ".nrrd";
  writer->SetFileName( magOutputName.c_str() );
  writer->UseCompressionOff();
  writer->SetInput( gradientMagnitudeImage );

  try
  {
    writer->Update();
  }
  catch (itk::ExceptionObject& e)
  {
    throw std::runtime_error( e.what() );
  }
}

#endif // _ShapeModelImageITK_h_
