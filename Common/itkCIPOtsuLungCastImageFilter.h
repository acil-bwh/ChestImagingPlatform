/** \class CIPOtsuLungCastImageFilter
 *  \brief This filter produces a segmentation of the lungs and
 *  airways using the Otsu thresholding approach. This filter is 
 *  so named because the output is a "cast" from which more 
 *  refined segmentations can be derived.
 */

#ifndef __itkCIPOtsuLungCastImageFilter_h
#define __itkCIPOtsuLungCastImageFilter_h

#include "itkImageToImageFilter.h"
#include "itkImage.h"
#include "itkOtsuThresholdImageFilter.h"
#include "itkConnectedComponentImageFilter.h"
#include "itkRelabelComponentImageFilter.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkExtractImageFilter.h"

namespace itk
{
template <class TInputImage, class TOutputImage = itk::Image<unsigned short, 3> >
class ITK_EXPORT CIPOtsuLungCastImageFilter :
    public ImageToImageFilter< TInputImage, TOutputImage >
{
public:
  /** Extract dimension from input and output image. */
  itkStaticConstMacro( InputImageDimension, unsigned int, TInputImage::ImageDimension );
  itkStaticConstMacro( OutputImageDimension, unsigned int, 3 );

  /** Convenient typedefs for simplifying declarations. */
  typedef TInputImage    InputImageType;
  typedef TOutputImage   OutputImageType;

  /** Standard class typedefs. */
  typedef CIPOtsuLungCastImageFilter                             Self;
  typedef ImageToImageFilter< InputImageType, OutputImageType >  Superclass;
  typedef SmartPointer< Self >                                   Pointer;
  typedef SmartPointer< const Self >                             ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(CIPOtsuLungCastImageFilter, ImageToImageFilter);
  
  /** Image typedef support. */
  typedef typename InputImageType::PixelType           InputPixelType;
  typedef typename OutputImageType::PixelType          OutputPixelType;
  typedef typename InputImageType::RegionType          InputImageRegionType;
  typedef typename OutputImageType::RegionType         OutputImageRegionType;
  typedef typename InputImageType::SizeType            InputSizeType;

  void PrintSelf( std::ostream& os, Indent indent ) const override;

protected:
  typedef itk::Image< unsigned short, 3 >                                                ComponentImageType;
  typedef itk::Image< unsigned short, 2 >                                                ComponentSliceType;
  typedef itk::Image< OutputPixelType, 2 >                                               OutputImageSliceType;
  typedef itk::OtsuThresholdImageFilter< InputImageType, OutputImageType >               OtsuThresholdType;
  typedef itk::ConnectedComponentImageFilter< OutputImageType, ComponentImageType >      ConnectedComponent3DType;
  typedef itk::ConnectedComponentImageFilter< OutputImageSliceType, ComponentSliceType > ConnectedComponent2DType;
  typedef itk::RelabelComponentImageFilter< ComponentImageType, ComponentImageType >     Relabel3DType;
  typedef itk::ImageRegionIteratorWithIndex< ComponentImageType >                        ComponentIteratorType;
  typedef itk::ImageRegionIteratorWithIndex< OutputImageType >                           OutputIteratorType;
  typedef itk::ExtractImageFilter< OutputImageType, OutputImageSliceType >               OutputImageExtractorType;
  typedef itk::ImageRegionIteratorWithIndex< ComponentSliceType >                        ComponentSliceIteratorType;

  CIPOtsuLungCastImageFilter();
  virtual ~CIPOtsuLungCastImageFilter() {}

  /** This method will consider each slice in turn after the initial Otsu thresholding
      and will remove all objects that touch one of the four corners in the slice 
      (which should never occurr -- if it does occur, it means the FOV does not fully
      contain the lung region, a situation that we don't handle) */
  void RemoveCornerObjects();

  void GenerateData() override;

private:
  CIPOtsuLungCastImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

};
  
} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkCIPOtsuLungCastImageFilter.txx"
#endif

#endif
