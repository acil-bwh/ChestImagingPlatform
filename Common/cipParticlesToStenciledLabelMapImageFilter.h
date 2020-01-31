/**
 *  \class cipParticlesToStenciledLabelMapImageFilter
 *  \ingroup common
 *  \brief This filter takes a set of particles and produces a label
 *  map based on the provided stencil. 'SetInput' takes an input label
 *  map, but it is only used to retrieve the spacing, origin, and
 *  dimensions needed for the output label map.
 *
 *  $Date: 2012-06-11 17:58:50 -0700 (Mon, 11 Jun 2012) $
 *  $Version$
 *  $Author: jross $
 *
 *  TODO: Sphere and cylinder stencil scaling using particle scale is
 *  implement for vessels, but not for airways or fissures yet.
 * 
 */

#ifndef __cipParticlesToStenciledLabelMapImageFilter_h
#define __cipParticlesToStenciledLabelMapImageFilter_h

#include "itkImageToImageFilter.h" 
#include "itkImageRegionIteratorWithIndex.h"
#include "vtkPolyData.h"
#include "vtkSmartPointer.h"
#include "cipChestConventions.h"
#include "cipStencil.h"

template < class TInputImage >
class cipParticlesToStenciledLabelMapImageFilter :
  public itk::ImageToImageFilter< TInputImage, itk::Image< unsigned short, 3 > > 
{
public:
  /** Extract dimension from input and output image. */
  itkStaticConstMacro( InputImageDimension, unsigned int, TInputImage::ImageDimension );
  itkStaticConstMacro( OutputImageDimension, unsigned int, 3 );

  /** Convenient typedefs for simplifying declarations. */
  typedef TInputImage                       InputImageType;
  typedef itk::Image< unsigned short, 3 >   OutputImageType;

  /** Standard class typedefs. */
  typedef cipParticlesToStenciledLabelMapImageFilter                  Self;
  typedef itk::ImageToImageFilter< InputImageType, OutputImageType >  Superclass;
  typedef itk::SmartPointer< Self >                                   Pointer;
  typedef itk::SmartPointer< const Self >                             ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(cipParticlesToStenciledLabelMapImageFilter, ImageToImageFilter);

  /** Image typedef support. */
  typedef unsigned short                                        LabelMapPixelType;
  typedef typename InputImageType::PixelType                    InputPixelType;
  typedef typename OutputImageType::PixelType                   OutputPixelType;
  typedef typename InputImageType::RegionType                   InputImageRegionType;
  typedef typename OutputImageType::RegionType                  OutputImageRegionType;
  typedef typename InputImageType::SizeType                     InputSizeType;
  typedef itk::ImageRegionIteratorWithIndex< OutputImageType >  IteratorType;

  void PrintSelf( std::ostream& os, itk::Indent indent ) const override;

  cipParticlesToStenciledLabelMapImageFilter();
  virtual ~cipParticlesToStenciledLabelMapImageFilter() {}

  void GenerateData() override;
  
  /** */
  void SetStencil( cipStencil* );

  /** Use the ChestConvention's ChestType enum to define whether the
   * input particles are vessels or airways. */
  void SetChestParticleType( unsigned int );
  unsigned int GetChestParticleType();

  /** Use this class's ParticleType enum to define the type of input
   * particles. */
  void SetParticleType( unsigned int );
  unsigned int GetParticleType();

  /** Set/Get the particles data */
  void SetParticlesData( vtkSmartPointer< vtkPolyData > );
  vtkSmartPointer< vtkPolyData > GetParticlesData();

  /** Set/Get the CT point spread function. This value can be used to
   *  scale stencil patterns appropriately so that they have sizes
   *  (radii) that indicate true structure size */
  void SetCTPointSpreadFunctionSigma( double sigma )
    {
      CTPointSpreadFunctionSigma = sigma;
    }
  double SetCTPointSpreadFunctionSigma()
    {
      return CTPointSpreadFunctionSigma;
    }

  void SetScaleStencilPatternByParticleScale( bool scale )
    {
      ScaleStencilPatternByParticleScale = scale;
    }
  void SetScaleStencilPatternByParticleDNNRadius( bool dnn_radius )
  {
      ScaleStencilPatternByParticleDNNRadius = dnn_radius;
  }
  void SetDNNRadiusName( std::string dnn_radius_name )
  {
    DNNRadiusName = dnn_radius_name;
  }

protected:
  void UpdateLabelMapRegion( vtkIdType );

private:
  enum ParticleType {
    RIDGELINE,
    VALLEYLINE,
    RIDGESURFACE,
    VALLEYSURFACE,
  };

  cipParticlesToStenciledLabelMapImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  unsigned int ChestParticleType;
  unsigned int SelectedParticleType;

  cipStencil* Stencil;

  bool   ScaleStencilPatternByParticleScale;
  bool   ScaleStencilPatternByParticleDNNRadius;
  std::string DNNRadiusName;
  double CTPointSpreadFunctionSigma;

  vtkSmartPointer< vtkPolyData > ParticlesData;
};

 
#ifndef ITK_MANUAL_INSTANTIATION
#include "cipParticlesToStenciledLabelMapImageFilter.txx"
#endif

#endif
