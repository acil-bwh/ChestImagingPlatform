/*=========================================================================

=========================================================================*/
#ifndef __itkCIPOtsuLungCastImageFilter_h
#define __itkCIPOtsuLungCastImageFilter_h

#include "itkImageToImageFilter.h"
#include "itkImage.h"
#include "itkOtsuThresholdImageFilter.h"
#include "itkConnectedComponentImageFilter.h"
#include "itkRelabelComponentImageFilter.h"
#include "itkImageRegionIteratorWithIndex.h"


namespace itk
{
/** \class CIPOtsuLungCastImageFilter
 * \brief Brief description here
 */
template <class TInputImage>
class ITK_EXPORT CIPOtsuLungCastImageFilter :
    public ImageToImageFilter< TInputImage, itk::Image< unsigned short, 3 > >
{
public:
  /** Extract dimension from input and output image. */
  itkStaticConstMacro( InputImageDimension, unsigned int, TInputImage::ImageDimension );
  itkStaticConstMacro( OutputImageDimension, unsigned int, 3 );

  /** Convenient typedefs for simplifying declarations. */
  typedef TInputImage                       InputImageType;
  typedef itk::Image< unsigned short, 3 >   OutputImageType;

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
  typedef unsigned short                               LabelMapPixelType;
  typedef typename InputImageType::PixelType           InputPixelType;
  typedef typename OutputImageType::PixelType          OutputPixelType;
  typedef typename InputImageType::RegionType          InputImageRegionType;
  typedef typename OutputImageType::RegionType         OutputImageRegionType;
  typedef typename InputImageType::SizeType            InputSizeType;


  void PrintSelf( std::ostream& os, Indent indent ) const;

protected:
  typedef itk::Image< unsigned long, 3 >                                              ComponentImageType;
  typedef itk::Image< LabelMapPixelType, 3 >                                          LabelMapType;
  typedef itk::OtsuThresholdImageFilter< InputImageType, OutputImageType >            OtsuThresholdType;
  typedef itk::ConnectedComponentImageFilter< LabelMapType, ComponentImageType >      ConnectedComponent3DType;
  typedef itk::RelabelComponentImageFilter< ComponentImageType, ComponentImageType >  Relabel3DType;
  typedef itk::ImageRegionIteratorWithIndex< ComponentImageType >                     ComponentIteratorType;
  typedef itk::ImageRegionIteratorWithIndex< LabelMapType >                           LabelMapIteratorType;

  CIPOtsuLungCastImageFilter();
  virtual ~CIPOtsuLungCastImageFilter() {}

  void GenerateData();

private:
  CIPOtsuLungCastImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

};
  
} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkCIPOtsuLungCastImageFilter.txx"
#endif

#endif
