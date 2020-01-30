#ifndef __itkCIPMergeChestLabelMapsImageFilter_h
#define __itkCIPMergeChestLabelMapsImageFilter_h

#include "itkImageToImageFilter.h"
#include "itkImage.h"
#include "itkImageRegionConstIterator.h"
#include "itkImageRegionIterator.h"
#include "cipChestConventions.h"
#include "cipHelper.h"

namespace itk
{
/** \class MergeChestLabelMapsImageFilter
 * \brief This filter merges two label map images. It is assumed that
 * the labels in both images are given with respect to the labeling
 * conventions indicated in cipChestConventions.h. The filter expects
 * two images: the base (input) image and the overlay image. All
 * labels in the base image remain untouched except for the rules
 * specified by the user. The user can choose to override or merge a
 * type, region, or type-region pair from the overlay image onto the
 * base image. The merge option is essentially a union. The override
 * option will first completely eliminate the region, type, or
 * region-type pair in the base image and then write the region, type,
 * or region-type pair locations specified in the overlay image. (Note
 * that in the case of region-type pair overrides, only the type is
 * set to UNDEFINEDTYPE in the base image voxels -- the region is left
 * untouched). The user can also choose to preserve certain regions,
 * types, or region-type pairs in the base image -- these specified
 * structures won't be touched in the merging process. The overlay
 * image can differ in size/extent from the base image; the origins
 * may be different; the spacing is assumed to be the same.
 */
class ITK_EXPORT CIPMergeChestLabelMapsImageFilter :
  public ImageToImageFilter< cip::LabelMapType, cip::LabelMapType >
{
protected:
  struct REGIONANDTYPE
  {
    unsigned char chestRegion;
    unsigned char chestType;
  };

public:
  /** Extract dimension from input and output image. */
  itkStaticConstMacro( InputImageDimension, unsigned int, 3 );
  itkStaticConstMacro( OutputImageDimension, unsigned int, 3 );

  /** Standard class typedefs. */
  typedef CIPMergeChestLabelMapsImageFilter                           Self;
  typedef ImageToImageFilter< cip::LabelMapType, cip::LabelMapType >  Superclass;
  typedef SmartPointer< Self >                                        Pointer;
  typedef SmartPointer< const Self >                                  ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(MergeChestLabelMapsImageFilter, ImageToImageFilter);
  
  /** Image typedef support. */
  typedef cip::LabelMapType::PixelType                        PixelType;
  typedef cip::LabelMapType::RegionType                       RegionType;
  typedef cip::LabelMapType::SizeType                         SizeType;
  typedef itk::ImageRegionConstIterator< cip::LabelMapType >  ConstIteratorType;
  typedef itk::ImageRegionIterator< cip::LabelMapType >       IteratorType;

  /** Setting GraftOverlay to be true will simply replace the base
   * image with the contents of the overlay image within the overlay
   * image region. False by default. Note that if this option is
   * specified, none of the other rules supplied by the user will be
   * applied. Additionally, only one of GraftOverlay or MergeOverlay
   * can be set to true. They can't both be true, but they can both be
   * false (in which case the other rules supplied by the user will be
   * applied) */
  itkSetMacro( GraftOverlay, bool ); 
  itkGetMacro( GraftOverlay, bool );

  /** Setting MergeOverlay to be true will place the contents of the
   * overlay image into the base image at all non-zero overlay voxels.
   * False by default. Note that if this option is specified, none of
   * the other rules supplied by the user witll be
   * applied. Additionally, only one of GraftOverlay or MergeOverlay
   * can be set to true. They can't both be true, but they can both be
   * false (in which case the other rules supplied by the user will be
   * applied) */
  itkSetMacro( MergeOverlay, bool ); 
  itkGetMacro( MergeOverlay, bool );

  /** Setting Union to be true results in the following behavior: for a given
   *  voxel, if either the region or type is undefined for either of the base
   *  or overlay label maps, then the merged voxel will get the defined
   *  region/type. If there is a conflict in defined regions for a given voxel,
   *  if one region is a subset of the other region, then the merged voxel 
   *  will get the region that is more specific. Otherwise, if there is a
   *  region/type conflict that can't be resolved, then the region/type in
   *  the base image is used. This option is false by default, and if set to
   *  true, will trump all other rules.
   */
  itkSetMacro( Union, bool );
  itkGetMacro( Union, bool );

  /** Set the type to override. Any voxel in the base (input) image
   * with this type will be set to UNDEFINEDTYPE first, and then
   * the type specified will be set according to the locations of that
   * type in the overlay image */
  inline void SetOverrideChestType( unsigned char typeValue )
    {
      this->m_OverrideChestTypeVec.push_back( typeValue );
    };

  /** Set the region to override. Any voxel in the base (input) image
   * with this region will be set to UNDEFINEDREGION first, and then
   * the region specified will be set according to the locations of that
   * region in the overlay image */
  inline void SetOverrideChestRegion( unsigned char regionValue )
    {
      this->m_OverrideChestRegionVec.push_back( regionValue );
    };

  /** Set the region-type pair to override. Any voxel in the base
   * (input) image with this region-type pair combination will have
   * the type set to UNDEFINEDTYPE. The region will be untouched. All
   * voxels in the overlay image having the specifed pair will have
   * their type grafted onto the base image. */
  inline void SetOverrideChestRegionTypePair( unsigned char regionValue, unsigned char typeValue )
    {
      REGIONANDTYPE regionTypePair;
        regionTypePair.chestRegion = regionValue;
        regionTypePair.chestType   = typeValue;

      this->m_OverrideChestRegionTypePairVec.push_back( regionTypePair );
    };

  /** Set the type to merge. The type specified will be set according
   * to the locations of that type in the overlay image */
  inline void SetMergeChestType( unsigned char typeValue )
    {
      this->m_MergeChestTypeVec.push_back( typeValue );
    };

  /** Set the region to merge. The region specified will be set
   * according to the locations of that region in the overlay image */
  inline void SetMergeChestRegion( unsigned char regionValue )
    {
      this->m_MergeChestRegionVec.push_back( regionValue );
    };

  /** All voxels having the specified region-type in the overlay image
   * will be merged with the base image -- only the type will be 
   * grafted onto the base image, however. The specified region serves
   * to identify the voxels in the overlay image, but the value of the
   * region itself is not grafted. */
  inline void SetMergeChestRegionTypePair( unsigned char regionValue, unsigned char typeValue )
    {
      REGIONANDTYPE regionTypePair;
        regionTypePair.chestRegion = regionValue;
        regionTypePair.chestType   = typeValue;

      this->m_MergeChestRegionTypePairVec.push_back( regionTypePair );
    };

  /** Set the type to preserve in the base image. No voxel in the base
   * image having the specified type will have its type altered. */
  inline void SetPreserveChestType( unsigned char typeValue )
    {
      this->m_PreserveChestTypeVec.push_back( typeValue );
    };

  /** Set the region to preserve in the base image. No voxel in the
   * base image having the specified region will have its region
   * altered. */
  inline void SetPreserveChestRegion( unsigned char regionValue )
    {
      this->m_PreserveChestRegionVec.push_back( regionValue );
    };

  /** Set the region-type pair to preserve in the base image. No voxel
   * in the base image having the specified pair will have either of
   * its region or type altered. */
  inline void SetPreserveChestRegionTypePair( unsigned char regionValue, unsigned char typeValue )
    {
      REGIONANDTYPE regionTypePair;
        regionTypePair.chestRegion = regionValue;
        regionTypePair.chestType   = typeValue;

      this->m_PreserveChestRegionTypePairVec.push_back( regionTypePair );
    };

  /** Set the overlay image. Regions and types specified by the user
   *  will override or be merged with labels in the input (base)
   *  image. Use of this method assumes that the overlay image has the
   *  same origin and extent as the base image. **/
  void SetOverlayImage( cip::LabelMapType::Pointer );

  /** Set the overlay image. Regions and types specified by the user
   *  will override or be merged with labels in the input (base)
   *  image. Use of this method assumes that the overlay image has a
   *  different origin and extent as the base image. The second
   *  parameter is meant to indicate the starting index in the input
   *  (base) image that corresponds to the origin in the overlay
   *  image. **/
  void SetOverlayImage( cip::LabelMapType::Pointer, cip::LabelMapType::IndexType );
    
  void PrintSelf( std::ostream& os, Indent indent ) const override;

protected:

  CIPMergeChestLabelMapsImageFilter();
  virtual ~CIPMergeChestLabelMapsImageFilter() {}

  void GenerateData() override;

  void MergeOverlay();
  void GraftOverlay();
  void Union();
  void ApplyRules();
  bool GetPermitChestRegionChange( unsigned char );
  bool GetPermitChestTypeChange( unsigned char, unsigned char );

private:
  CIPMergeChestLabelMapsImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  std::vector< unsigned char > m_OverrideChestRegionVec;
  std::vector< unsigned char > m_OverrideChestTypeVec;
  std::vector< REGIONANDTYPE > m_OverrideChestRegionTypePairVec;
  std::vector< unsigned char > m_MergeChestRegionVec;
  std::vector< unsigned char > m_MergeChestTypeVec;
  std::vector< REGIONANDTYPE > m_MergeChestRegionTypePairVec;
  std::vector< unsigned char > m_PreserveChestRegionVec;
  std::vector< unsigned char > m_PreserveChestTypeVec;
  std::vector< REGIONANDTYPE > m_PreserveChestRegionTypePairVec;

  cip::LabelMapType::IndexType m_OverlayImageStartIndex;
  cip::LabelMapType::Pointer   m_OverlayImage;

  bool m_GraftOverlay;
  bool m_MergeOverlay;
  bool m_Union;

  cip::ChestConventions m_ChestConventions;
};
  
} // end namespace itk

#endif
