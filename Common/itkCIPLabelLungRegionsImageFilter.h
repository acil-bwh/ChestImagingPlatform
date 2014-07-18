/** \class CIPLabelLungRegionsImageFilter
 * \brief This filter takes as input a lung label map and produces a
 * lung label map with region designations specified by the user. As
 * an initialization step, the output image is filled with the
 * WHOLELUNG region at all defined lung region locations in the input
 * region. This means that the input label map can have mislabeled
 * lung regions -- it should not affect the output label map.
 *
 * Currently, there are two options for lung labeling: left-right lung
 * labeling and thirds lung labeling. Note that in order for thirds
 * lung labeling to occur, the lungs must first be labeled as being
 * either left or right, so if thirds labeling is indicated,
 * left-right labeling will be carried out as a precursor regardless
 * of what the user has set for the left-right labeling option.
 *
 * By default, the output image will contain the WHOLELUNG region. So
 * the user can use this filter to produce an output image with just
 * the WHOLELUNG indicated (with the types indicated in the input
 * image) by simply supplying an input and getting an output (i.e. not
 * turning on left-right or thirds labeling, which are off by
 * default). Additionally, if the user specifies left-right or thirds
 * labeling, the left-right labeling will be carried out. If it's
 * determined that a left-right distinction can't be made (due to the
 * lungs being merged/connected), then the output label map will only
 * have WHOLELUNG specified.
 *
 *  $Date: 2012-09-06 15:43:20 -0400 (Thu, 06 Sep 2012) $
 *  $Revision: 245 $
 *  $Author: rjosest $
 *
 * Labeling conforms to the conventions specified in itkLungConventions.h
 */


#ifndef __itkCIPLabelLungRegionsImageFilter_h
#define __itkCIPLabelLungRegionsImageFilter_h

#include "itkImageToImageFilter.h"
#include "itkImage.h"
#include "itkConnectedComponentImageFilter.h"
#include "itkRelabelComponentImageFilter.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "cipChestConventions.h"


namespace itk
{
class ITK_EXPORT CIPLabelLungRegionsImageFilter :
  public ImageToImageFilter< ::itk::Image< unsigned short, 3 >, Image< unsigned short, 3 > >
{
public:
  /** Extract dimension from input and output image. */
  itkStaticConstMacro( InputImageDimension, unsigned int, 3 );
  itkStaticConstMacro( OutputImageDimension, unsigned int, 3 );

  /** Convenient typedefs for simplifying declarations. */
  typedef Image< unsigned short, 3 >   InputImageType;
  typedef Image< unsigned short, 3 >   OutputImageType;

  /** Standard class typedefs. */
  typedef CIPLabelLungRegionsImageFilter                          Self;
  typedef ImageToImageFilter< InputImageType, OutputImageType >   Superclass;
  typedef SmartPointer< Self >                                    Pointer;
  typedef SmartPointer< const Self >                              ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(CIPLabelLungRegionsImageFilter, ImageToImageFilter);
  
  /** Image typedef support. */
  typedef unsigned short                      LabelMapPixelType;
  typedef InputImageType::PixelType           InputPixelType;
  typedef OutputImageType::PixelType          OutputPixelType;
  typedef InputImageType::RegionType          InputImageRegionType;
  typedef OutputImageType::RegionType         OutputImageRegionType;
  typedef InputImageType::SizeType            InputSizeType;

  /** This variable indicates whether or not the patient was scanned
   *  in the supine position (default is true) */
  itkSetMacro( Supine, bool );
  itkGetMacro( Supine, bool );

  /** This variable indicates whether or not the patient was scanned
   *  in the head-first position (default is true) */
  itkSetMacro( HeadFirst, bool );
  itkGetMacro( HeadFirst, bool );

  /** We may want to know whether or not the labeling operation was
   *  carried out successfully. We can use this function to test this,
   *  but note that it can't meaningfully be called until Update is
   *  called. Right now only evaluation of left-right and thirds
   *  labeling is evaluated. The method will return true if the
   *  left-right labeling has been carried out successfully (thirds
   *  labeling is trivial after this). Note that the labeling
   *  operation will label large connected components. Currently, if
   *  the left or right lungs are broken into multiple components, not
   *  all components may get labeled. Nevertheless, this function will
   *  return true. */
  itkGetMacro( LabelingSuccess, bool );

  /** Set left-right lung labeling On or Off */
  itkSetMacro( LabelLeftAndRightLungs, bool );
  itkGetConstReferenceMacro( LabelLeftAndRightLungs, bool );
  itkBooleanMacro( LabelLeftAndRightLungs );

  /** Set thirds lung labeling On or Off. Note that in order to label
   * lung thirds, the lungs must first be labeled as left or right. If
   * the user indicates to label thirds, the left-right labeling will
   * be carried out as a precursor regardless of what the user sets
   * for the LabelLeftAndRightLungs option. Further note that the
   * LabelLeftAndRightLungs option need not be set for the thirds
   * labeling to be properly performed. I.e. if you want to label lung
   * thirds, you only need to turn on LabelLungThirds.*/
  itkSetMacro( LabelLungThirds, bool );
  itkGetConstReferenceMacro( LabelLungThirds, bool );
  itkBooleanMacro( LabelLungThirds );

  void PrintSelf( std::ostream& os, Indent indent ) const;

protected:
  typedef Image< LabelMapPixelType, 3 >                                   LabelMapType;
  typedef ConnectedComponentImageFilter< LabelMapType, LabelMapType >     ConnectedComponentType;
  typedef RelabelComponentImageFilter< LabelMapType, LabelMapType >       RelabelComponentType;
  typedef ImageRegionIteratorWithIndex< LabelMapType >                    LabelMapIteratorType;
  typedef ImageRegionConstIterator< InputImageType >                      InputIteratorType;

  CIPLabelLungRegionsImageFilter();
  virtual ~CIPLabelLungRegionsImageFilter() {}

  void SetType( OutputImageType::IndexType, int );
  bool LabelLeftAndRightLungs();
  void SetLungThirds();

  void GenerateData();

private:
  CIPLabelLungRegionsImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  cip::ChestConventions m_LungConventions;
  bool                  m_HeadFirst;
  bool                  m_Supine;
  bool                  m_LabelLungThirds;
  bool                  m_LabelLeftAndRightLungs;
  bool                  m_LabelingSuccess;
  int                   m_NumberLungVoxels;
};
  
} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkCIPLabelLungRegionsImageFilter.txx"
#endif

#endif
