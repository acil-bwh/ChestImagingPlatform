/** \class CIPPartialLungLabelMapImageFilter
 * \brief This filter produces a lung label map given a grayscale
 * input image. The designation 'Partial' is meant to emphasize that
 * certain structures (namely airways and vessels) are only roughly
 * generated and that other structures (e.g. the lobes) are not produced
 * at all. This filter will attempt to produce a lung label map with
 * the left and right lungs labeled by thirds. If no left-right
 * distinction can be made (i.e. if the lungs can't be separated), 
 * then the output labeling will consist only the lung region labeled
 * by thirds but with no left-right distinction.
 */

#ifndef __itkCIPPartialLungLabelMapImageFilter_h
#define __itkCIPPartialLungLabelMapImageFilter_h

#include "itkImageToImageFilter.h"
#include "itkImage.h"
#include "cipConventions.h"
#include "itkCIPWholeLungVesselAndAirwaySegmentationImageFilter.h"
#include "itkCIPSplitLeftLungRightLungImageFilter.h"
#include "itkCIPLabelLungRegionsImageFilter.h"
#include "itkCIPAutoThresholdAirwaySegmentationImageFilter.h"
#include "itkOtsuThresholdImageFilter.h"
#include "itkConnectedComponentImageFilter.h"
#include "itkRelabelComponentImageFilter.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkExtractImageFilter.h"
#include "itkBinaryDilateImageFilter.h"
#include "itkBinaryBallStructuringElement.h"
#include "itkBinaryThresholdImageFilter.h"
#include "itkBinaryErodeImageFilter.h"
#include "itkCIPOtsuLungCastImageFilter.h"

namespace itk
{
template <class TInputImage>
class ITK_EXPORT CIPPartialLungLabelMapImageFilter :
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
  typedef CIPPartialLungLabelMapImageFilter                         Self;
  typedef ImageToImageFilter< InputImageType, OutputImageType >  Superclass;
  typedef SmartPointer< Self >                                   Pointer;
  typedef SmartPointer< const Self >                             ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(CIPPartialLungLabelMapImageFilter, ImageToImageFilter);
  
  /** Image typedef support. */
  typedef unsigned short                               LabelMapPixelType;
  typedef typename InputImageType::PixelType           InputPixelType;
  typedef typename OutputImageType::PixelType          OutputPixelType;
  typedef typename InputImageType::RegionType          InputImageRegionType;
  typedef typename OutputImageType::RegionType         OutputImageRegionType;
  typedef typename InputImageType::SizeType            InputSizeType;

  /** Region growing can "explode slowly". That is, the lung region
   *  can slowly be filled by the region growing algorithm in a way
   *  that goes undetected by the other explosion control
   *  mechanism. By setting the max volume of the airway segmentation,
   *  we enable an additional check to safegaurd against explosions. */
  itkSetMacro( MaxAirwayVolume, double );
  itkGetMacro( MaxAirwayVolume, double );

  /** This variable indicates whether or not the patient was scanned
   *  in the supine position (default is true) */
  itkSetMacro( Supine, bool );
  itkGetMacro( Supine, bool );

  /** This variable indicates whether or not the patient was scanned
   *  in the head-first position (default is true) */
  itkSetMacro( HeadFirst, bool );
  itkGetMacro( HeadFirst, bool );

  /** By default, the left-right lung splitting routine makes
   *  assumptions to make the process as fast as possible.  In some
   *  cases, however, this can result in left and right lungs that are
   *  still merged.  By setting 'AggressiveLeftRightSplitter' to true,
   *  the splitting routing will take longer, but will be more robust. */
  itkSetMacro( AggressiveLeftRightSplitter, bool ); 
  itkGetMacro( AggressiveLeftRightSplitter, bool );

  /** In order to split the left and right lungs, a min cost path
   *  algorithm is used.  To do this, a section of the image is
   *  converted to a graph and weights are assigned to the indices
   *  based on an exponential function ( f=A*exp{t/tau) ).  For the
   *  task of splitting the lungs, it's assumed that dark voxels are
   *  to be penalized much more than bright voxels.  For images
   *  ranging from -1024 to 1024, the default
   *  'ExponentialTimeConstant' and 'ExponentialCoefficient' values of
   *  -700 and 200 have been found to work well. If you find that the
   *  lungs are merged after running this filter, it migh help to
   *  double check the values you're setting for these parameters */
  itkSetMacro( ExponentialTimeConstant, double );
  itkGetMacro( ExponentialTimeConstant, double );

  /** See not for 'ExponentialTimeConstant' above */
  itkSetMacro( ExponentialCoefficient, double );
  itkGetMacro( ExponentialCoefficient, double );


  /** If the left and right lungs are merged in a certain section, 
   * graph methods are used to find a min cost path (i.e. the brightest
   * path) that passes through the merge region. This operation returns
   * a set of indices corresponding to points along the path. When the
   * lungs are actually split in two, a radius (essentially an erosion
   * radius) is used to separate the lungs.  The larged the radius, the
   * more aggressive the splitting. The default value is 3. */
  itkSetMacro( LeftRightLungSplitRadius, int );
  itkGetMacro( LeftRightLungSplitRadius, int );
    
  /** If the user specifies an airway label map, it will be used in
   * in the overall segmentation. If no airway label map is specified,
   * one will be automatically determined. */
  void SetAirwayLabelMap( OutputImageType::Pointer );

  /** This method allows a user to specify seeds to be used during the
   * airway segmentation process. If no seeds are specified, the
   * filter will attempt to find them automatically */
  void AddAirwaySegmentationSeed( OutputImageType::IndexType );

  /** Set the neighborhood defining the structuring element for
   *  morphological closing (3 dimensions) */
  void SetClosingNeighborhood( unsigned long * );

  /** Set the maximum threshold value to use during airway region growing 
   *  segmentation. This is a required value -- if no value is 
   *  specified, an exception will be thrown. For CT images, this value
   *  should be in the neighborhood of -800 HU. */
  void SetMaxIntensityThreshold( InputPixelType );
  itkGetMacro( MaxIntensityThreshold, InputPixelType );

  /** Set the minimum threshold value to use during airway region growing 
   *  segmentation. This is a required value -- if no value is 
   *  specified, an exception will be thrown. For CT images, this value
   *  should probably be -1024 HU. */
  void SetMinIntensityThreshold( InputPixelType );
  itkGetMacro( MinIntensityThreshold, InputPixelType );

  /** This function allows to set a "helper" mask. This mask is
   *  assumed to be binary, with 1 as the foreground. It is assumed
   *  that the lung region has been reasonably well threshold, and the
   *  lungs have been split (into left and right). The airways are
   *  assumed to be foreground as well. Passing a helper mask is
   *  optional and is intended to be used for recovering failure modes
   */
  void SetHelperMask( OutputImageType::Pointer );

  void PrintSelf( std::ostream& os, Indent indent ) const;

protected:
  typedef itk::Image< unsigned long, 3 >                                              ComponentImageType;
  typedef typename itk::Image< LabelMapPixelType, 2 >                                 LabelMapSliceType;
  typedef itk::ImageRegionIteratorWithIndex< LabelMapSliceType >                      LabelMapSliceIteratorType;
  typedef itk::Image< LabelMapPixelType, 3 >                                          LabelMapType;
  typedef itk::CIPWholeLungVesselAndAirwaySegmentationImageFilter< InputImageType >   WholeLungVesselAndAirwayType;
  typedef itk::CIPSplitLeftLungRightLungImageFilter< InputImageType >                 SplitterType;
  typedef itk::CIPLabelLungRegionsImageFilter                                         LungRegionLabelerType;
  typedef itk::CIPAutoThresholdAirwaySegmentationImageFilter< InputImageType >        AirwaySegmentationType;
  typedef itk::OtsuThresholdImageFilter< InputImageType, OutputImageType >            OtsuThresholdType;
  typedef itk::BinaryThresholdImageFilter< InputImageType, OutputImageType >          BinaryThresholdType;
  typedef itk::ConnectedComponentImageFilter< LabelMapSliceType, LabelMapSliceType >  ConnectedComponent2DType;
  typedef itk::ConnectedComponentImageFilter< LabelMapType, ComponentImageType >      ConnectedComponent3DType;
  typedef itk::RelabelComponentImageFilter< LabelMapSliceType, LabelMapSliceType >    Relabel2DType;
  typedef itk::RelabelComponentImageFilter< ComponentImageType, ComponentImageType >  Relabel3DType;
  typedef itk::ImageRegionIteratorWithIndex< ComponentImageType >                     ComponentIteratorType;
  typedef itk::ImageRegionIteratorWithIndex< LabelMapType >                           LabelMapIteratorType;
  typedef itk::ExtractImageFilter< LabelMapType, LabelMapSliceType >                  LabelMapExtractorType;
  typedef itk::BinaryBallStructuringElement< LabelMapPixelType, 3 >                   Element3DType;
  typedef itk::BinaryDilateImageFilter< LabelMapType, LabelMapType, Element3DType >   Dilate3DType;
  typedef itk::BinaryErodeImageFilter< LabelMapType, LabelMapType, Element3DType >    Erode3DType;
  typedef itk::CIPOtsuLungCastImageFilter< InputImageType >                           OtsuCastType;

  CIPPartialLungLabelMapImageFilter();
  virtual ~CIPPartialLungLabelMapImageFilter() {}

  void GenerateData();
  void ApplyOtsuThreshold();
  void ApplyHelperMask();
  void RecordAndRemoveAirways( LabelMapType::Pointer );
  void RemoveTracheaAndMainBronchi();
  void ExtractLabelMapSlice( LabelMapType::Pointer, LabelMapSliceType::Pointer, int );
  void CloseLabelMap( unsigned short );
  void ExpandLungRegionsInSlices( LabelMapType::Pointer, short );
  void ExpandLungRegionInSlice( LabelMapType::Pointer, LabelMapType::IndexType, unsigned char, short );
  void ExpandLeftRight( LabelMapType::IndexType, short, unsigned short, unsigned int );
  void ConditionalDilation( short );
  std::vector< OutputImageType::IndexType > GetAirwaySeeds();


private:
  CIPPartialLungLabelMapImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  std::vector< OutputImageType::IndexType > m_AirwaySegmentationSeedVec;
  std::vector< OutputImageType::IndexType > m_AirwayIndicesVec;

  LabelMapType::Pointer m_AirwayLabelMap;
  LabelMapType::Pointer m_HelperMask;

  InputPixelType         m_MaxIntensityThreshold;
  bool                   m_MaxIntensityThresholdSet;
  InputPixelType         m_MinIntensityThreshold;
  bool                   m_MinIntensityThresholdSet;

  double           m_MinAirwayVolume;
  double           m_MaxAirwayVolume;
  double           m_MaxAirwayVolumeIncreaseRate;
  double           m_ExponentialCoefficient;
  double           m_ExponentialTimeConstant;
  bool             m_HeadFirst;
  bool             m_Supine;
  bool             m_AggressiveLeftRightSplitter;
  unsigned long    m_ClosingNeighborhood[3];
  int              m_LeftRightLungSplitRadius;
  short            m_OtsuThreshold;
};
  
} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkCIPPartialLungLabelMapImageFilter.txx"
#endif

#endif
