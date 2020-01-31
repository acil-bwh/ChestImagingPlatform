/** \class CIPAutoThresholdAirwaySegmentationImageFilter
 *  \brief This filter segments the airways with a region growing
 *  algorithm. The thresholds used by the algorithm are automatically
 *  adjusted between two pre-defined extremes until an airway volume
 *  is achieved that is a close to possible as a maximum specified
 *  volume without going over. After region growing, morphological 
 *  closing is performed to fill in holes. The foreground value of
 *  the output depends on the output image type: if unsigned short,
 *  the value will correspond to (UndefinedRegion, Airway); if
 *  unsigned char, the value will correspond to Airway. See the
 *  cipChestConventions for details about the corresponding numerical
 *  values.
 */

#ifndef __itkCIPAutoThresholdAirwaySegmentationImageFilter_h
#define __itkCIPAutoThresholdAirwaySegmentationImageFilter_h

#include "itkImageToImageFilter.h"
#include "itkImage.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkConnectedThresholdImageFilter.h"
#include "cipChestConventions.h"
#include "itkBinaryDilateImageFilter.h"
#include "itkBinaryBallStructuringElement.h"
#include "itkBinaryThresholdImageFilter.h"
#include "itkBinaryErodeImageFilter.h"

namespace itk
{

template <class TInputImage, class TOutputImage = itk::Image<unsigned short, 3> >
class ITK_EXPORT CIPAutoThresholdAirwaySegmentationImageFilter :
  public ImageToImageFilter< TInputImage, TOutputImage >
{
public:
  /** Extract dimension from input and output image. */
  itkStaticConstMacro( InputImageDimension, unsigned int, TInputImage::ImageDimension );
  itkStaticConstMacro( OutputImageDimension, unsigned int, 3 );

  /** Convenient typedefs for simplifying declarations. */
  typedef TInputImage  InputImageType;
  typedef TOutputImage OutputImageType;

  /** Standard class typedefs. */
  typedef CIPAutoThresholdAirwaySegmentationImageFilter          Self;
  typedef ImageToImageFilter< InputImageType, OutputImageType >  Superclass;
  typedef SmartPointer< Self >                                   Pointer;
  typedef SmartPointer< const Self >                             ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(CIPAutoThresholdAirwaySegmentationImageFilter, ImageToImageFilter);
  
  /** Image typedef support. */
  typedef unsigned short                               LabelMapPixelType;
  typedef typename InputImageType::PixelType           InputPixelType;
  typedef typename OutputImageType::PixelType          OutputPixelType;
  typedef typename InputImageType::RegionType          InputImageRegionType;
  typedef typename OutputImageType::RegionType         OutputImageRegionType;
  typedef typename InputImageType::SizeType            InputSizeType;

  /** The algorithm performs region growing to find a threshold that
   *  produces an airway tree with a volume as close to this specified
   *  value as possible, without going over. This is an optional input
   *  and is set to a reasonable value by default. The volume should
   *  be specified in mm^3 (not to be confused with milliliters) */
  itkSetMacro( MaxAirwayVolume, double );
  itkGetMacro( MaxAirwayVolume, double );

  /** Set the maximum threshold value to use during region growing 
   *  segmentation. This is a required value -- if no value is 
   *  specified, an exception will be thrown. For CT images, this value
   *  should be in the neighborhood of -800 HU. */
  void SetMaxIntensityThreshold( InputPixelType );
  itkGetMacro( MaxIntensityThreshold, InputPixelType );

  /** Set the minimum threshold value to use during region growing 
   *  segmentation. This is a required value -- if no value is 
   *  specified, an exception will be thrown. For CT images, this value
   *  should probably be -1024 HU. */
  void SetMinIntensityThreshold( InputPixelType );
  itkGetMacro( MinIntensityThreshold, InputPixelType );

  /** Set a seed (multiple seeds may be specified) for the region
   * growing */
  void AddSeed( typename TOutputImage::IndexType );

  void PrintSelf( std::ostream& os, Indent indent ) const override;

protected:
  typedef itk::Image< LabelMapPixelType, 3 >                                             LabelMapType;
  typedef itk::ImageRegionIteratorWithIndex< LabelMapType >                              LabelMapIteratorType;
  typedef itk::ImageRegionIteratorWithIndex< OutputImageType >                           OutputIteratorType;
  typedef itk::ConnectedThresholdImageFilter< InputImageType, OutputImageType >          SegmentationType;
  typedef itk::BinaryBallStructuringElement< OutputPixelType, 3 >                        ElementType;
  typedef itk::BinaryDilateImageFilter< OutputImageType, OutputImageType, ElementType >  DilateType;
  typedef itk::BinaryErodeImageFilter< OutputImageType, OutputImageType, ElementType >   ErodeType;

  CIPAutoThresholdAirwaySegmentationImageFilter();
  virtual ~CIPAutoThresholdAirwaySegmentationImageFilter() {}

  void GenerateData() override;
  void Test();

private:
  CIPAutoThresholdAirwaySegmentationImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  std::vector< typename OutputImageType::IndexType > m_SeedVec;

  cip::ChestConventions  m_ChestConventions;
  double                 m_MaxAirwayVolume;
  InputPixelType         m_MaxIntensityThreshold;
  bool                   m_MaxIntensityThresholdSet;
  InputPixelType         m_MinIntensityThreshold;
  bool                   m_MinIntensityThresholdSet;
};
  
} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkCIPAutoThresholdAirwaySegmentationImageFilter.txx"
#endif

#endif
