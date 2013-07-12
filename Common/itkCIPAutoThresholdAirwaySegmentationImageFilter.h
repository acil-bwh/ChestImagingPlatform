/** \class CIPAutoThresholdAirwaySegmentationImageFilter
 * \brief This filter segments the airways with an algorithm that
 * consists of region growing in which the threshold is adapted to
 * find the best result without leakage.  The starting threshold value
 * is the darkest value among the seeds.  The threshold is increased
 * until leakage is detected; the final threshold value is then
 * selected to be the value that is just below the value producing
 * leakage. After region growing, morphological closing is performed
 * to fill in holes.
 *  $Date:  $
 *  $Revision: $
 *  $Author: $
 */

#ifndef __itkCIPAutoThresholdAirwaySegmentationImageFilter_h
#define __itkCIPAutoThresholdAirwaySegmentationImageFilter_h

#include "itkImageToImageFilter.h"
#include "itkImage.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkConnectedThresholdImageFilter.h"
#include "cipConventions.h"
#include "itkBinaryDilateImageFilter.h"
#include "itkBinaryBallStructuringElement.h"
#include "itkBinaryThresholdImageFilter.h"
#include "itkBinaryErodeImageFilter.h"


namespace itk
{

template <class TInputImage>
class ITK_EXPORT CIPAutoThresholdAirwaySegmentationImageFilter :
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
  typedef CIPAutoThresholdAirwaySegmentationImageFilter             Self;
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

  /** Reasonable airway segmentations have been empiracally found to
      have volumes of at least 10.0 cc (default).  The value set by
      this method indicates the minimum airway volume expected in cc.
      This value is used during airway segmentation (when trying to
      find the best threshold for connected threshold segmentation) */
  itkSetMacro( MinAirwayVolume, double );
  itkGetMacro( MinAirwayVolume, double );

  /** Region growing can "explode slowly". That is, the lung region
   *  can slowly be filled by the region growing algorithm in a way
   *  that goes undetected by the other explosion control
   *  mechanism. By setting the max volume of the airway segmentation,
   *  we enable an additional check to safegaurd against explosions. */
  itkSetMacro( MaxAirwayVolume, double );
  itkGetMacro( MaxAirwayVolume, double );

  /** MaxAirwayVolumeIncreaseRate is used during airway segmentation
      while trying to find the optimal threshold to use with the
      connected threshold algorithm. Change in airway volume over
      change in threshold increment less than 2.0 (default) has been
      empirically found to be stable.  Leakage typically produces
      values much larger than 2.0.  If you see a little leakage in the
      airway segmentation output, it is likely due to "slow" leakage.
      Decreasing from 2.0 might fix the problem in this
      case. Satisfactory values for this parameter will likely be
      quite close to 2.0. */
  itkSetMacro( MaxAirwayVolumeIncreaseRate, double );
  itkGetMacro( MaxAirwayVolumeIncreaseRate, double );

  /** Set a seed (multiple seeds may be specified) for the region
   * growing */
  void AddSeed( OutputImageType::IndexType );

  void PrintSelf( std::ostream& os, Indent indent ) const;

protected:
  typedef itk::Image< LabelMapPixelType, 3 >                                         LabelMapType;
  typedef itk::ImageRegionIteratorWithIndex< LabelMapType >                          LabelMapIteratorType;
  typedef itk::ConnectedThresholdImageFilter< InputImageType, OutputImageType >      SegmentationType;
  typedef itk::BinaryBallStructuringElement< LabelMapPixelType, 3 >                  ElementType;
  typedef itk::BinaryDilateImageFilter< LabelMapType, LabelMapType, ElementType >    DilateType;
  typedef itk::BinaryErodeImageFilter< LabelMapType, LabelMapType, ElementType >     ErodeType;

  CIPAutoThresholdAirwaySegmentationImageFilter();
  virtual ~CIPAutoThresholdAirwaySegmentationImageFilter() {}

  void GenerateData();
  void Test();

private:
  CIPAutoThresholdAirwaySegmentationImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  std::vector< OutputImageType::IndexType > m_SeedVec;

  cip::ChestConventions  m_LungConventions;
  double           m_MinAirwayVolume;
  double           m_MaxAirwayVolume;
  double           m_MaxAirwayVolumeIncreaseRate;
};
  
} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkCIPAutoThresholdAirwaySegmentationImageFilter.txx"
#endif

#endif
