/** \class CIPSplitLeftAndRightLungsImageFilter
 * \brief This filter takes as input a 3D lung label map image (with
 * labelings assumed to adhere to the conventions described in
 * itkLungConventions.h. The output of this filter is a lung label map
 * image with the left and right lungs split. No relabeling is
 * performed. 
 */

#ifndef __itkCIPSplitLeftLungRightLungImageFilter_h
#define __itkCIPSplitLeftLungRightLungImageFilter_h

#include "itkImageToImageFilter.h"
#include "itkImage.h"
#include "itkConnectedComponentImageFilter.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkExtractImageFilter.h"
#include "cipChestConventions.h"
#include "itkCIPDijkstraGraphTraits.h"
#include "itkGraph.h" 
#include "itkImageToGraphFilter.h"
#include "itkCIPDijkstraImageToGraphFunctor.h"
#include "itkCIPDijkstraMinCostPathGraphToGraphFilter.h"

namespace itk
{
template <class TInputImage>
class ITK_EXPORT CIPSplitLeftLungRightLungImageFilter :
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
  typedef CIPSplitLeftLungRightLungImageFilter                   Self;
  typedef ImageToImageFilter< InputImageType, OutputImageType >  Superclass;
  typedef SmartPointer< Self >                                   Pointer;
  typedef SmartPointer< const Self >                             ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(CIPSplitLeftLungRightLungImageFilter, ImageToImageFilter);
  
  /** Image typedef support. */
  typedef unsigned short                        LabelMapPixelType;
  typedef itk::Image< LabelMapPixelType, 3 >    LabelMapType;
  typedef typename InputImageType::PixelType    InputPixelType;
  typedef OutputImageType::PixelType            OutputPixelType;
  typedef typename InputImageType::RegionType   InputImageRegionType;
  typedef OutputImageType::RegionType           OutputImageRegionType;
  typedef typename InputImageType::SizeType     InputSizeType;

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
   * radius) is used to separate the lungs.  The larger the radius, the
   * more aggressive the splitting. The default value is 3. */
  itkSetMacro( LeftRightLungSplitRadius, int );
  itkGetMacro( LeftRightLungSplitRadius, int );

  void SetLungLabelMap( LabelMapType::Pointer );

  void PrintSelf( std::ostream& os, Indent indent ) const override;

protected:
  typedef itk::Image< LabelMapPixelType, 2 >         LabelMapSliceType;
  typedef typename itk::Image< InputPixelType, 2 >   InputSliceType;
  typedef typename InputSliceType::Pointer           InputSlicePointerType;
  typedef LabelMapSliceType::IndexType               LabelMapSliceIndexType;

  typedef itk::Image< InputPixelType, 2 >                                                        InputImageSliceType;
  typedef itk::ConnectedComponentImageFilter< LabelMapSliceType, LabelMapSliceType >             ConnectedComponent2DType;
  typedef itk::ImageRegionIteratorWithIndex< LabelMapType >                                      LabelMapIteratorType;
  typedef itk::ImageRegionConstIterator< InputImageType >                                        InputIteratorType;
  typedef itk::ImageRegionIteratorWithIndex< LabelMapSliceType >                                 LabelMapSliceIteratorType;
  typedef itk::ExtractImageFilter< InputImageType, InputImageSliceType >                         InputExtractorType;
  typedef itk::ExtractImageFilter< LabelMapType, LabelMapSliceType >                             LabelMapExtractorType;
  typedef unsigned long                                                                          GraphTraitsScalarType;
  typedef itk::CIPDijkstraGraphTraits< GraphTraitsScalarType, 2 >                                GraphTraitsType;
  typedef itk::Graph< GraphTraitsType >                                                          GraphType;
  typedef itk::ImageToGraphFilter< InputSliceType, GraphType >                                   GraphFilterType;
  typedef itk::CIPDijkstraImageToGraphFunctor< InputSliceType, GraphType >                       FunctorType;
  typedef itk::CIPDijkstraMinCostPathGraphToGraphFilter< GraphType, GraphType >                  MinPathType;

  CIPSplitLeftLungRightLungImageFilter();
  virtual ~CIPSplitLeftLungRightLungImageFilter() {}

  void ExtractLabelMapSlice( LabelMapType::Pointer, LabelMapSliceType::Pointer, int );

  void FindMinCostPath();

  bool GetLungsMergedInSliceRegion( int, int, int, int, int );

  /** Given a min cost path through a slice, this function will erase
      all pixels that fall on the path (including those with a specified
      radius of each path pixl). The function also records those slice
      indices that were actualy erased for later use. */
  void EraseConnection( unsigned int );

  /** */
  void SetDefaultGraphROIAndSearchIndices( unsigned int );

  /** */
  void SetLocalGraphROIAndSearchIndices( unsigned int );

  void GenerateData() override;

private:
  CIPSplitLeftLungRightLungImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  LabelMapType::Pointer  m_LungLabelMap;

  double  m_ExponentialCoefficient;
  double  m_ExponentialTimeConstant;
  int     m_LeftRightLungSplitRadius;

  std::vector< LabelMapSliceType::IndexType >  m_MinCostPathIndices;
  std::vector< LabelMapSliceType::IndexType >  m_ErasedSliceIndices;
  LabelMapType::IndexType                      m_StartSearchIndex;
  LabelMapType::IndexType                      m_EndSearchIndex;
  typename InputImageType::SizeType            m_GraphROISize;
  typename InputImageType::IndexType           m_GraphROIStartIndex;
  bool                                         m_UseLocalGraphROI;
};
  
} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkCIPSplitLeftLungRightLungImageFilter.txx"
#endif

#endif
