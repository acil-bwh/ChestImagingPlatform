/**
 *  \class cipLabelMapToLungLobeLabelMapImageFilter
 *  \ingroup common
 *  \brief This filter takes as input a label map image (adhering to
 *  the labeling conventions laid out in cipConventions) as well as
 *  information needed to define thin plate spline boundary surfaces
 *  and produces a label map image with the lung lobes defined.
 *
 *  This filter assumes that the input label map has the left and
 *  right lungs properly defined.
 *
 *  Detailed description 
 *
 *  Date:     $Date: 2012-04-24 17:06:09 -0700 (Tue, 24 Apr 2012) $
 *  Version:  $Revision: 93 $
 *
 *  TODO:
 *  
 */

#ifndef __cipLabelMapToLungLobeLabelMapImageFilter_h
#define __cipLabelMapToLungLobeLabelMapImageFilter_h

#include "itkImageToImageFilter.h"
#include "itkImage.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkImageRegionConstIterator.h"
#include "cipConventions.h"
#include "itkConnectedComponentImageFilter.h"
#include "cipThinPlateSplineSurface.h"


class ITK_EXPORT cipLabelMapToLungLobeLabelMapImageFilter :
  public itk::ImageToImageFilter< itk::Image<unsigned short, 3>, itk::Image<unsigned short, 3> >
{
public:
  /** Extract dimension from input and output image. */
  itkStaticConstMacro( InputImageDimension, unsigned int, 3 );
  itkStaticConstMacro( OutputImageDimension, unsigned int, 3 );

  /** Convenient typedefs for simplifying declarations. */
  typedef itk::Image< unsigned short, 3 >   InputImageType;
  typedef itk::Image< unsigned short, 3 >   OutputImageType;

  /** Standard class typedefs. */
  typedef cipLabelMapToLungLobeLabelMapImageFilter                    Self;
  typedef itk::ImageToImageFilter< InputImageType, OutputImageType >  Superclass;
  typedef itk::SmartPointer<Self>                                     Pointer;
  typedef itk::SmartPointer<const Self>                               ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(cipLabelMapToLungLobeLabelMapImageFilter, ImageToImageFilter);
  
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

  void SetLeftObliqueThinPlateSplineSurface( cipThinPlateSplineSurface* );

  void SetRightObliqueThinPlateSplineSurface( cipThinPlateSplineSurface* );

  void SetRightHorizontalThinPlateSplineSurface( cipThinPlateSplineSurface* );

  void PrintSelf( std::ostream& os, itk::Indent indent ) const;

protected:
  cipLabelMapToLungLobeLabelMapImageFilter();
  virtual ~cipLabelMapToLungLobeLabelMapImageFilter() {}

  typedef itk::ConnectedComponentImageFilter< OutputImageType, OutputImageType >  ConnectedComponentType;

  void GenerateData();

private:
  cipLabelMapToLungLobeLabelMapImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  unsigned short FissureSurfaceValue;

  std::vector< InputImageType::IndexType >  LeftObliqueFissureIndices;
  std::vector< InputImageType::IndexType >  RightObliqueFissureIndices;
  std::vector< InputImageType::IndexType >  RightHorizontalFissureIndices;

  std::vector< double* >  LeftObliqueFissurePoints;
  std::vector< double* >  RightObliqueFissurePoints;
  std::vector< double* >  RightHorizontalFissurePoints;

  cipThinPlateSplineSurface* LeftObliqueThinPlateSplineSurface;
  cipThinPlateSplineSurface* RightObliqueThinPlateSplineSurface;
  cipThinPlateSplineSurface* RightHorizontalThinPlateSplineSurface;
};
  
#ifndef ITK_MANUAL_INSTANTIATION
#include "cipLabelMapToLungLobeLabelMapImageFilter.txx"
#endif

#endif
