/**
 *  \class itkCIPChestLabelMapToChestLobeLabelMapImageFilter
 *  \ingroup common
 *  \brief This class reads a label map (conforming to labeling
 *  schemes outlined in cipChestConventions.h) and produces a label map
 *  with the lobes of the lung labeled.
 *
 *  Detailed description 
 *
 *  Date:     $Date: 2012-04-24 17:06:09 -0700 (Tue, 24 Apr 2012) $
 *  Version:  $Revision: 93 $
 *
 *  TODO:
 *  Needs commenting
 */

#ifndef __itkCIPChestLabelMapToChestLobeLabelMapImageFilter_h
#define __itkCIPChestLabelMapToChestLobeLabelMapImageFilter_h

#include "itkImageToImageFilter.h"
#include "itkImage.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkImageRegionConstIterator.h"
#include "cipChestConventions.h"
#include "itkConnectedComponentImageFilter.h"
#include "cipThinPlateSplineSurface.h"


namespace itk
{

class ITK_EXPORT CIPChestLabelMapToChestLobeLabelMapImageFilter :
    public ImageToImageFilter< itk::Image<unsigned short, 3>, itk::Image<unsigned short, 3> >
{
public:
  /** Extract dimension from input and output image. */
  itkStaticConstMacro( InputImageDimension, unsigned int, 3 );
  itkStaticConstMacro( OutputImageDimension, unsigned int, 3 );

  /** Convenient typedefs for simplifying declarations. */
  typedef itk::Image< unsigned short, 3 >   InputImageType;
  typedef itk::Image< unsigned short, 3 >   OutputImageType;

  /** Standard class typedefs. */
  typedef CIPChestLabelMapToChestLobeLabelMapImageFilter              Self;
  typedef ImageToImageFilter< InputImageType, OutputImageType >  Superclass;
  typedef SmartPointer<Self>                                     Pointer;
  typedef SmartPointer<const Self>                               ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(CIPChestLabelMapToChestLobeLabelMapImageFilter, ImageToImageFilter);
  
  /** Image typedef support. */
  typedef InputImageType::PixelType  InputPixelType;
  typedef OutputImageType::PixelType OutputPixelType;

  typedef InputImageType::RegionType  InputImageRegionType;
  typedef OutputImageType::RegionType OutputImageRegionType;

  typedef InputImageType::SizeType InputSizeType;

  typedef itk::ImageRegionIteratorWithIndex< OutputImageType > OutputIteratorType;
  typedef itk::ImageRegionConstIterator< InputImageType >      InputIteratorType;

  void SetLeftObliqueFissureIndices( std::vector< InputImageType::IndexType > );

  void SetRightObliqueFissureIndices( std::vector< InputImageType::IndexType > );

  void SetRightHorizontalFissureIndices( std::vector< InputImageType::IndexType > );

  void SetLeftObliqueFissurePoints( std::vector< double* >* );

  void SetRightObliqueFissurePoints( std::vector< double* >* );

  void SetRightHorizontalFissurePoints( std::vector< double* >* );

  void SetLeftObliqueThinPlateSplineSurface( ThinPlateSplineSurface* );

  void SetRightObliqueThinPlateSplineSurface( ThinPlateSplineSurface* );

  void SetRightHorizontalThinPlateSplineSurface( ThinPlateSplineSurface* );

  void PrintSelf( std::ostream& os, Indent indent ) const;

  /** This filter needs first to label the output image with the left
   *  lung and right lung regions.  This takes a little time -- if this
   *  operation is performed ahead of time, and the input image only
   *  has two values (indicating the left and right lung regions),
   *  then set this to true.  By default it is set to false. */
  itkSetMacro( InputIsLeftLungRightLung, bool );
  itkGetMacro( InputIsLeftLungRightLung, bool );

protected:
  CIPChestLabelMapToChestLobeLabelMapImageFilter();
  virtual ~CIPChestLabelMapToChestLobeLabelMapImageFilter() {}

  typedef itk::ConnectedComponentImageFilter< OutputImageType, OutputImageType >  ConnectedComponentType;

  void GenerateData();

private:
  CIPChestLabelMapToChestLobeLabelMapImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  unsigned short m_FissureSurfaceValue;
  bool           m_InputIsLeftLungRightLung;

  std::vector< InputImageType::IndexType >  m_LeftObliqueFissureIndices;
  std::vector< InputImageType::IndexType >  m_RightObliqueFissureIndices;
  std::vector< InputImageType::IndexType >  m_RightHorizontalFissureIndices;

  std::vector< double* >  m_LeftObliqueFissurePoints;
  std::vector< double* >  m_RightObliqueFissurePoints;
  std::vector< double* >  m_RightHorizontalFissurePoints;

  cipThinPlateSplineSurface* m_LeftObliqueThinPlateSplineSurface;
  cipThinPlateSplineSurface* m_RightObliqueThinPlateSplineSurface;
  cipThinPlateSplineSurface* m_RightHorizontalThinPlateSplineSurface;
};
  
} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkCIPChestLabelMapToChestLobeLabelMapImageFilter.txx"
#endif

#endif
