#include "itkCIPExtractChestLabelMapImageFilter.h"


namespace itk
{


CIPExtractChestLabelMapImageFilter
::CIPExtractChestLabelMapImageFilter()
{
//  this->m_RegionVec.push_back( cip::UNDEFINEDREGION );

//   REGIONANDTYPE regionTypeTemp;
//     regionTypeTemp.lungRegionValue = cip::UNDEFINEDREGION;
//     regionTypeTemp.lungTypeValue   = cip::UNDEFINEDTYPE;
 
//   this->m_RegionAndTypeVec.push_back( regionTypeTemp );
}


void
CIPExtractChestLabelMapImageFilter
::GenerateData()
{
  this->InitializeMaps();

  //
  // Allocate the output buffer
  //
  this->GetOutput()->SetBufferedRegion( this->GetOutput()->GetRequestedRegion() );
  this->GetOutput()->Allocate();
  this->GetOutput()->FillBuffer( 0 );

  //
  // Now assign the regions and types in the output image based on the
  // mapping we determined in 'InitializeMaps'
  //
  OutputIteratorType oIt( this->GetOutput(), this->GetOutput()->GetBufferedRegion() );
  InputIteratorType iIt( this->GetInput(), this->GetInput()->GetBufferedRegion() );

  oIt.GoToBegin();
  iIt.GoToBegin();
  while ( !iIt.IsAtEnd() )
    {
    if ( iIt.Get() != 0 )
      {
      unsigned short outputValue = this->m_ValueToValueMap[ iIt.Get() ];

      oIt.Set( outputValue );
      }
    
    ++oIt;
    ++iIt;
    }
}


void
CIPExtractChestLabelMapImageFilter
::InitializeMaps()
{
  typedef std::pair< unsigned char, unsigned char > UCHAR_PAIR;

  //
  // Create the mappings for each region to one of the regions that
  // the user has specified. Note that m_RegionVec will at least
  // contain UNDEFINEDREGION
  //
  for ( unsigned int i=0; i<this->m_ChestConventions.GetNumberOfEnumeratedChestRegions(); i++ )
    {
    unsigned char cipRegion = this->m_ChestConventions.GetChestRegion( i );
    
    bool regionMapped = false;

    for ( unsigned int j=0; j<this->m_RegionVec.size(); j++ )
      {
      if ( this->m_ChestConventions.CheckSubordinateSuperiorChestRegionRelationship( cipRegion, this->m_RegionVec[j] ) )
        {
        this->m_RegionMap.insert( UCHAR_PAIR( cipRegion, this->m_RegionVec[j] ) );
        
        regionMapped = true;
        }
      }
    if ( !regionMapped )
      {
      this->m_RegionMap.insert( UCHAR_PAIR( cipRegion, cip::UNDEFINEDREGION ) );
      }
    }

  //
  // Create the mappings for each region to one of the regions that
  // the user has specified as a region-type pair. Note that
  // m_RegionAndTypeVec will at least contain the UNDEFINEDREGION,
  // UNDEFINEDTYPE pair
  //
  for ( unsigned int i=0; i<this->m_ChestConventions.GetNumberOfEnumeratedChestRegions(); i++ )
    {
    unsigned char cipRegion = this->m_ChestConventions.GetChestRegion( i );

    bool regionMapped = false;

    for ( unsigned int j=0; j<this->m_RegionAndTypeVec.size(); j++ )
      {
      if ( this->m_ChestConventions.CheckSubordinateSuperiorChestRegionRelationship( cipRegion, this->m_RegionAndTypeVec[j].lungRegionValue ) )
        {
        this->m_RegionMapForRegionTypePairs.insert( UCHAR_PAIR( cipRegion, this->m_RegionAndTypeVec[j].lungRegionValue ) );

        regionMapped = true;
        }
      }
    if ( !regionMapped )
      {
      this->m_RegionMapForRegionTypePairs.insert( UCHAR_PAIR( cipRegion, cip::UNDEFINEDREGION ) );
      }
    }

  //
  // Iterate through the input image and create a list of all values.
  // Sort and unique this list and then compute a mapping of the
  // values to the appropriate region/type pairs.  Using this map will
  // greatly speed computation later.
  //
  std::list< unsigned short > valueList;
  valueList.push_back( 0 );

  InputIteratorType iIt( this->GetInput(), this->GetInput()->GetBufferedRegion() );

  iIt.GoToBegin();
  while ( !iIt.IsAtEnd() )
    {
    if ( iIt.Get() != 0 )
      {
      valueList.push_back( iIt.Get() );
      }

    ++iIt;
    }

  valueList.unique();
  valueList.sort();
  valueList.unique();

  std::list< unsigned short >::iterator listIt;
  listIt = valueList.begin();

  for ( unsigned int i=0; i<valueList.size(); i++, listIt++ )
    {
    unsigned char inputType   = this->m_ChestConventions.GetChestTypeFromValue( *listIt );
    unsigned char inputRegion = this->m_ChestConventions.GetChestRegionFromValue( *listIt );

    REGIONANDTYPE mappedRegionAndType;
      mappedRegionAndType.lungRegionValue = 0;
      mappedRegionAndType.lungTypeValue   = 0;

    //
    // Set the mapped region
    //
    mappedRegionAndType.lungRegionValue = this->m_RegionMap[ inputRegion ];

    //
    // Set the mapped type 
    //
    for ( unsigned int j=0; j<this->m_TypeVec.size(); j++ )
      {
      if ( static_cast< int >( inputType ) == static_cast< int >( this->m_TypeVec[j] ) )
        {      
        mappedRegionAndType.lungTypeValue = this->m_TypeVec[j];
        }
      }

    //
    // If there is a type/region pair, it will take precedence for the
    // mapping
    //
    unsigned char mappedRegionForRegionTypePairs = this->m_RegionMapForRegionTypePairs[ inputRegion ];

    if ( inputType != cip::UNDEFINEDTYPE && inputRegion != cip::UNDEFINEDREGION )
      {
      for ( unsigned int i=0; i<this->m_RegionAndTypeVec.size(); i++ )
        {
        if ( inputType == this->m_RegionAndTypeVec[i].lungTypeValue && mappedRegionForRegionTypePairs == this->m_RegionAndTypeVec[i].lungRegionValue )
          {
          mappedRegionAndType.lungTypeValue   = inputType;
          mappedRegionAndType.lungRegionValue = mappedRegionForRegionTypePairs;
          }
        }
      }

    unsigned short mappedValue = this->m_ChestConventions.GetValueFromChestRegionAndType( mappedRegionAndType.lungRegionValue, 
                                                                                         mappedRegionAndType.lungTypeValue );

    this->m_ValueToValueMap.insert( std::pair< unsigned short, unsigned short >( *listIt, mappedValue ) );
    }
}


/**
 * Standard "PrintSelf" method
 */
void
CIPExtractChestLabelMapImageFilter
::PrintSelf(
  std::ostream& os, 
  Indent indent) const
{
  Superclass::PrintSelf( os, indent );
}

} // end namespace itk
