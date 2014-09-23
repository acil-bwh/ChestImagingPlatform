/**
 *  \class cipLabelMapToLungLobeLabelMapImageFilter
 *  \ingroup common
 *  \brief This filter takes as input a label map image (adhering to
 *  the labeling conventions laid out in cipChestConventions) as well as
 *  information needed to define thin plate spline boundary surfaces
 *  and produces a label map image with the lung lobes defined.
 *
 *  This filter assumes that the input label map has the left and
 *  right lungs properly defined.
 *
 *  The boundaries between the lobes can be defined in a number of 
 *  ways. A user can specify a set of physical points (or indices,
 *  which are converted points) along the boundary of interest. These
 *  points are used to form a thin plate spline (TPS) surface defining
 *  the boundary between the lobes. Alternatively, a user can define
 *  a TPS surface directly to indicate the lobe boundary. Finally, a
 *  user can specify both a set of points (indices) AND a TPS surface
 *  model. In that case, the TPS surface formed using the points will
 *  be used to form the boundary in regions near those points, but in
 *  regions that are far from those points, the TPS surface specified
 *  using the points will gradually give way (blen into) the surface
 *  that was directly specified.
 */

#ifndef __cipLabelMapToLungLobeLabelMapImageFilter_h
#define __cipLabelMapToLungLobeLabelMapImageFilter_h

#include "itkImageToImageFilter.h"
#include "itkImage.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkImageRegionConstIterator.h"
#include "cipChestConventions.h"
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

  /** Image indices indicating the locations of points along the left 
   *  oblique fissure (along the boundary separating the left upper lobe
   *  from the left lower lobe). */
  void SetLeftObliqueFissureIndices( std::vector< InputImageType::IndexType > );

  /** Image indices indicating the locations of points along the right
   *  oblique fissure (along the boundary separating the right upper lobe
   *  from the right middle and lower lobes). */
  void SetRightObliqueFissureIndices( std::vector< InputImageType::IndexType > );

  /** Image indices indicating the locations of points along the right
   *  horizontal fissure (along the boundary separating the right middle lobe
   *  from the right upper lobe). */
  void SetRightHorizontalFissureIndices( std::vector< InputImageType::IndexType > );

  /** Physical point coorindates indicating the locations of points along 
   *  the left oblique fissure (along the boundary separating the left upper 
   *  lobe from the left lower lobe). */
  void SetLeftObliqueFissurePoints( std::vector< double* >* );

  /** Physical point coordinates indicating the locations of points along 
   *  the right oblique fissure (along the boundary separating the right upper 
   *  lobe from the right middle and lower lobes). */
  void SetRightObliqueFissurePoints( std::vector< double* >* );

  /** Physical point coordinates indicating the locations of points along 
   *  the right horizontal fissure (along the boundary separating the right 
   *  middle lobe from the right upper lobe). */
  void SetRightHorizontalFissurePoints( std::vector< double* >* );

  /** Thin plate spline model of the boundary between the left upper lobe and
   *  the left lower lobe. If a surface is specified AND a set of left oblique
   *  fissure points (indices) is specified, the surface that interpolates through
   *  the points will be used in regions near those points, but in regions that
   *  are farther and farther away from these points, the specified surface model
   *  will gradually be preferred. */
  void SetLeftObliqueThinPlateSplineSurface( cipThinPlateSplineSurface* );

  /** Thin plate spline model of the boundary between the right lower lobe and
   *  the right lower and middle lobes. If a surface is specified AND a set of right 
   *  oblique fissure points (indices) is specified, the surface that interpolates 
   *  through the points will be used in regions near those points, but in regions that
   *  are farther and farther away from these points, the specified surface model
   *  will gradually be preferred. */
  void SetRightObliqueThinPlateSplineSurface( cipThinPlateSplineSurface* );

  /** Thin plate spline model of the boundary between the right upper lobe and
   *  the right middle lobe. If a surface is specified AND a set of right 
   *  horizontal fissure points (indices) is specified, the surface that interpolates 
   *  through the points will be used in regions near those points, but in regions that
   *  are farther and farther away from these points, the specified surface model
   *  will gradually be preferred. */
  void SetRightHorizontalThinPlateSplineSurface( cipThinPlateSplineSurface* );

  void PrintSelf( std::ostream& os, itk::Indent indent ) const;

protected:
  cipLabelMapToLungLobeLabelMapImageFilter();
  virtual ~cipLabelMapToLungLobeLabelMapImageFilter() {}

  typedef itk::ConnectedComponentImageFilter< OutputImageType, OutputImageType >  ConnectedComponentType;
  typedef itk::Image< float, 2 >                                                  BlendMapType;

  void GenerateData();

private:
  cipLabelMapToLungLobeLabelMapImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  int GetBoundaryHeightIndex( cipThinPlateSplineSurface*, cipThinPlateSplineSurface*, BlendMapType::Pointer,
			      unsigned int, unsigned int );
  void UpdateBlendMap( cipThinPlateSplineSurface*, BlendMapType::Pointer );

  unsigned short FissureSurfaceValue;

  double BlendSlope;
  double BlendIntercept;

  BlendMapType::Pointer LeftObliqueBlendMap;
  BlendMapType::Pointer RightObliqueBlendMap;
  BlendMapType::Pointer RightHorizontalBlendMap;

  std::vector< InputImageType::IndexType >  LeftObliqueFissureIndices;
  std::vector< InputImageType::IndexType >  RightObliqueFissureIndices;
  std::vector< InputImageType::IndexType >  RightHorizontalFissureIndices;

  std::vector< double* >  LeftObliqueFissurePoints;
  std::vector< double* >  RightObliqueFissurePoints;
  std::vector< double* >  RightHorizontalFissurePoints;

  cipThinPlateSplineSurface* LeftObliqueThinPlateSplineSurface;
  cipThinPlateSplineSurface* RightObliqueThinPlateSplineSurface;
  cipThinPlateSplineSurface* RightHorizontalThinPlateSplineSurface;

  cipThinPlateSplineSurface* LeftObliqueThinPlateSplineSurfaceFromPoints;
  cipThinPlateSplineSurface* RightObliqueThinPlateSplineSurfaceFromPoints;
  cipThinPlateSplineSurface* RightHorizontalThinPlateSplineSurfaceFromPoints;
};
  
#ifndef ITK_MANUAL_INSTANTIATION
#include "cipLabelMapToLungLobeLabelMapImageFilter.txx"
#endif

#endif
