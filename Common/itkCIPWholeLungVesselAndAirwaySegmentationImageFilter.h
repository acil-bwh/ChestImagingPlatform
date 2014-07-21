/** \class WholeLungVesselAndAirwaySegmentationImageFilter
 * \brief This filter produces a lung label map given a grayscale
 * input image. The output image consists of the following labelings:
 * AIRWAY, VESSEL, and WHOLELUNG (in keeping with the conventions
 * outlined in itkLungConventions.h). No attempt is made to split the
 * lung halves in two, nor is there an attempt to label the lungs by
 * thirds, etc. Users of this filter may wish to supply their own
 * airway label map, which will then be incorporated into the overall
 * segmentation routine. If no airway label map is provided, one will
 * be computed automatically.
 */
#ifndef __itkCIPWholeLungVesselAndAirwaySegmentationImageFilter_h
#define __itkCIPWholeLungVesselAndAirwaySegmentationImageFilter_h

#include "itkImageToImageFilter.h"
#include "itkImage.h"
#include "itkOtsuThresholdImageFilter.h"
#include "itkRelabelComponentImageFilter.h"
#include "itkConnectedComponentImageFilter.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkExtractImageFilter.h"
#include "cipChestConventions.h"
#include "itkBinaryDilateImageFilter.h"
#include "itkBinaryBallStructuringElement.h"
#include "itkBinaryThresholdImageFilter.h"
#include "itkBinaryErodeImageFilter.h"
#include "itkCIPAutoThresholdAirwaySegmentationImageFilter.h"
#include "itkCIPExtractChestLabelMapImageFilter.h"


namespace itk
{
template <class TInputImage>
class ITK_EXPORT CIPWholeLungVesselAndAirwaySegmentationImageFilter :
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
  typedef CIPWholeLungVesselAndAirwaySegmentationImageFilter        Self;
  typedef ImageToImageFilter< InputImageType, OutputImageType >  Superclass;
  typedef SmartPointer< Self >                                   Pointer;
  typedef SmartPointer< const Self >                             ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(CIPWholeLungVesselAndAirwaySegmentationImageFilter, ImageToImageFilter);
  
  /** Image typedef support. */
  typedef unsigned short                               LabelMapPixelType;
  typedef typename InputImageType::PixelType           InputPixelType;
  typedef typename OutputImageType::PixelType          OutputPixelType;
  typedef typename InputImageType::RegionType          InputImageRegionType;
  typedef typename OutputImageType::RegionType         OutputImageRegionType;
  typedef typename InputImageType::SizeType            InputSizeType;

  /** This variable indicates whether or not the patient was scanned
   *  in the head-first position (default is true) */
  itkSetMacro( HeadFirst, bool );
  itkGetMacro( HeadFirst, bool );

  /** If the user specifies an airway label map, it will be used in
   * in the overall segmentation. If no airway label map is specified,
   * one will be automatically determined. */
  void SetAirwayLabelMap( OutputImageType::Pointer );

  /** This method allows a user to specify seeds to be used during the
   * airway segmentation process. If no seeds are specified, the
   * filter will attempt to find them automatically */
  void AddAirwaySegmentationSeed( OutputImageType::IndexType );

  /** Reasonable airway segmentations have been empiracally found to
      have volumes of at least 10.0 cc (default).  The value set by
      this method indicates the minimum airway volume expected in cc.
      This value is used during airway segmentation (when trying to
      find the best threshold for connected threshold segmentation) */
  itkSetMacro( MinAirwayVolume, double );
  itkGetMacro( MinAirwayVolume, double );

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

  /** Set the neighborhood defining the structuring element for
   *  morphological closing (3 dimensions) */
  void SetClosingNeighborhood( unsigned long * );

  void PrintSelf( std::ostream& os, Indent indent ) const;

protected:
  typedef typename itk::Image< LabelMapPixelType, 2 >  LabelMapSliceType;
  typedef typename itk::Image< InputPixelType, 2 >     InputSliceType;
  typedef typename InputSliceType::Pointer             InputSlicePointerType;
  typedef typename LabelMapSliceType::IndexType        LabelMapSliceIndexType;

  typedef itk::Image< LabelMapPixelType, 3 >                                                     LabelMapType;
  typedef itk::Image< unsigned long, 3 >                                                         ComponentImageType;
  typedef itk::OtsuThresholdImageFilter< InputImageType, OutputImageType >                       OtsuThresholdType;
  typedef itk::ConnectedComponentImageFilter< LabelMapSliceType, LabelMapSliceType >             ConnectedComponent2DType;
  typedef itk::ConnectedComponentImageFilter< LabelMapType, ComponentImageType >                 ConnectedComponent3DType;
  typedef itk::RelabelComponentImageFilter< LabelMapSliceType, LabelMapSliceType >               Relabel2DType;
  typedef itk::RelabelComponentImageFilter< ComponentImageType, ComponentImageType >             Relabel3DType;
  typedef itk::ImageRegionIteratorWithIndex< LabelMapType >                                      LabelMapIteratorType;
  typedef itk::ImageRegionIteratorWithIndex< ComponentImageType >                                ComponentIteratorType;
  typedef itk::ImageRegionIteratorWithIndex< LabelMapSliceType >                                 LabelMapSliceIteratorType;
  typedef itk::ExtractImageFilter< LabelMapType, LabelMapSliceType >                             LabelMapExtractorType;
  typedef itk::BinaryBallStructuringElement< LabelMapPixelType, 2 >                              Element2DType;
  typedef itk::BinaryBallStructuringElement< LabelMapPixelType, 3 >                              Element3DType;
  typedef itk::BinaryDilateImageFilter< LabelMapSliceType, LabelMapSliceType, Element2DType >    Dilate2DType;
  typedef itk::BinaryDilateImageFilter< LabelMapType, LabelMapType, Element3DType >              Dilate3DType;
  typedef itk::BinaryThresholdImageFilter< LabelMapSliceType, LabelMapSliceType >                Threshold2DType;
  typedef itk::BinaryErodeImageFilter< LabelMapType, LabelMapType, Element3DType >               Erode3DType;
  typedef itk::CIPAutoThresholdAirwaySegmentationImageFilter< InputImageType >                   AirwaySegmentationType;
  typedef itk::CIPExtractChestLabelMapImageFilter                                                ExtractLabelMapType;

  CIPWholeLungVesselAndAirwaySegmentationImageFilter();
  virtual ~CIPWholeLungVesselAndAirwaySegmentationImageFilter() {}

  void ApplyOtsuThreshold();
  void FillAndRecordVessels();
  std::vector< OutputImageType::IndexType > GetAirwaySeeds();
  void SetNonLungAirwayRegion();
  void SetLungType( OutputImageType::IndexType, unsigned char );
  void SetLungRegion( OutputImageType::IndexType, unsigned char );
  void ExtractLabelMapSlice( LabelMapType::Pointer, LabelMapSliceType::Pointer, int );

  void GenerateData();

private:
  CIPWholeLungVesselAndAirwaySegmentationImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  std::vector< OutputImageType::IndexType > m_VesselIndexVec;
  std::vector< OutputImageType::IndexType > m_AirwayIndexVec;
  std::vector< OutputImageType::IndexType > m_AirwaySegmentationSeedVec;

  LabelMapType::Pointer m_AirwayLabelMap;

  cip::ChestConventions  m_LungConventions;
  bool                   m_HeadFirst;
  unsigned long          m_ClosingNeighborhood[3];
  double                 m_MinAirwayVolume;
  double                 m_MaxAirwayVolumeIncreaseRate; 

};
  
} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkCIPWholeLungVesselAndAirwaySegmentationImageFilter.txx"
#endif

#endif
