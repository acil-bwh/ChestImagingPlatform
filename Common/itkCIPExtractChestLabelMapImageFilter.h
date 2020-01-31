/**
 *  \class CIPExtractChestLabelMapImageFilter
 *  \ingroup common
 *  \brief This filter takes in a lung label map and producs a lung
 *  label map; it assumes the labeling conventions coded in cipConventions.h.
 *  The user specifies the regions and types he/she is
 *  interested in.  All other regions/types are set to UNDEFINEDTYPE
 *  and UNDEFINEDREGION.  Given that regions are hierarchical, a region
 *  that is higher in the hierarchy will be preferred to one that is
 *  lower.  E.g., if the user specifies both WHOLELUNG and
 *  LEFTSUPERIORLOBE, the region in the LEFTSUPERIORLOBE will be defined as
 *  such, and WHOLELUNG will be used elsewhere.
 *
 *  Precedence will be as follows: types, regions, region-type pairs. In 
 *  other words, if the user requests both LEFTLUNG and AIRWAY (not as a 
 *  pair), then an AIRWAY voxel in the LEFTLUNG will be mapped to LEFTLUNG 
 *  in the output. If the user additionally requests the AIRWAY, LEFTLUNG 
 *  pair, then the entire voxel will be preserved.
 *
 *  Note that this filter is not templated.  This is to enforce the
 *  usage of unsigned shorts as the label map type.  
 *
 *  $Date: 2012-09-06 15:43:20 -0400 (Thu, 06 Sep 2012) $
 *  $Revision: 245 $
 *  $Author: rjosest $
 *
 *  TODO:
 *  
 */

#ifndef __itkCIPExtractChestLabelMapImageFilter_h
#define __itkCIPExtractChestLabelMapImageFilter_h

#include "itkImageToImageFilter.h"
#include "itkImage.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkImageRegionConstIterator.h"
#include "cipChestConventions.h"

namespace itk
{

template <unsigned int TDimension> class ITK_EXPORT CIPExtractChestLabelMapImageFilter :
    public ImageToImageFilter< itk::Image<unsigned short, TDimension>, itk::Image<unsigned short, TDimension> >
{
protected:
  struct REGIONANDTYPE
  {
    unsigned char lungRegionValue;
    unsigned char lungTypeValue;
  };

public:
  /** Extract dimension from input and output image. */
  itkStaticConstMacro( InputImageDimension, unsigned int, TDimension );
  itkStaticConstMacro( OutputImageDimension, unsigned int, TDimension );

  /** Convenient typedefs for simplifying declarations. */
  typedef itk::Image< unsigned short, TDimension >   InputImageType;
  typedef itk::Image< unsigned short, TDimension >   OutputImageType;

  /** Standard class typedefs. */
  typedef CIPExtractChestLabelMapImageFilter                     Self;
  typedef ImageToImageFilter< InputImageType, OutputImageType >  Superclass;
  typedef SmartPointer<Self>                                     Pointer;
  typedef SmartPointer<const Self>                               ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(CIPExtractChestLabelMapImageFilter, ImageToImageFilter);
  
  /** Image typedef support. */
  typedef typename InputImageType::PixelType  InputPixelType;
  typedef typename OutputImageType::PixelType OutputPixelType;

  typedef typename InputImageType::RegionType  InputImageRegionType;
  typedef typename OutputImageType::RegionType OutputImageRegionType;

  typedef typename InputImageType::SizeType InputSizeType;

  typedef itk::ImageRegionIteratorWithIndex< OutputImageType > OutputIteratorType;
  typedef itk::ImageRegionConstIterator< InputImageType >      InputIteratorType;

  void PrintSelf( std::ostream& os, Indent indent ) const override;

  /** This method allows the user to specify region-type pairs.  Only
   *  those input image values satisfying both the specified
   *  region-type pair values will be passed on to the output */
  inline void SetRegionAndType( unsigned char regionValue, unsigned char typeValue )
    {
      REGIONANDTYPE regionAndTypeTemp;
        regionAndTypeTemp.lungRegionValue = regionValue;
        regionAndTypeTemp.lungTypeValue   = typeValue;

      this->m_RegionAndTypeVec.push_back( regionAndTypeTemp );
    };
 
  /** This method allows the user to specify a region to be passed on
   *  to the output label map */
  inline void SetChestRegion( unsigned char regionValue )
    {
      this->m_RegionVec.push_back( regionValue );
    };

  /** This method allows the user to specify a type to be passed on to
   *  the output label map */
  inline void SetChestType( unsigned char typeValue )
    {
      this->m_TypeVec.push_back( typeValue );
    };

protected:
  CIPExtractChestLabelMapImageFilter();
  //virtual ~CIPExtractChestLabelMapImageFilter() {}

  void GenerateData() override;

private:
  CIPExtractChestLabelMapImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  void InitializeMaps();

  std::vector< unsigned char >                m_RegionVec;
  std::vector< unsigned char >                m_TypeVec;
  std::vector< REGIONANDTYPE >                m_RegionAndTypeVec;
  std::map< unsigned char, unsigned char >    m_RegionMap;
  std::map< unsigned char, unsigned char >    m_RegionMapForRegionTypePairs;
  std::map< unsigned short, unsigned short >  m_ValueToValueMap;
  cip::ChestConventions                       m_ChestConventions;
  
};
  
} // end namespace itk

#include "itkCIPExtractChestLabelMapImageFilter.txx"

#endif
