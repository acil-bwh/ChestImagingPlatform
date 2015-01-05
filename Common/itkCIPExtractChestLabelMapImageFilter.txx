#include "itkCIPExtractChestLabelMapImageFilter.h"


namespace itk
{

template < unsigned int Dimension >
CIPExtractChestLabelMapImageFilter< Dimension >
::CIPExtractChestLabelMapImageFilter()
{
//  this->m_RegionVec.push_back( cip::UNDEFINEDREGION );

//   REGIONANDTYPE regionTypeTemp;
//     regionTypeTemp.lungRegionValue = cip::UNDEFINEDREGION;
//     regionTypeTemp.lungTypeValue   = cip::UNDEFINEDTYPE;
 
//   this->m_RegionAndTypeVec.push_back( regionTypeTemp );
}

template < unsigned int Dimension >
void
CIPExtractChestLabelMapImageFilter< Dimension >
::GenerateData()
{
  this->InitializeMaps();

  //
  // Allocate the output buffer
  //
  this->GetOutput()->SetBufferedRegion( this->GetOutput()->GetRequestedRegion() );
  this->GetOutput()->Allocate();
  this->GetOutput()->FillBuffer( 0 );

  // Now assign the regions and types in the output image based on the
  // mapping we determined in 'InitializeMaps'
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

template < unsigned int Dimension >
void
CIPExtractChestLabelMapImageFilter< Dimension >
::InitializeMaps()
{
  // First collect the values in the label map. We will then figure out
  // how to map them to output values based on the user requests
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

  // Now for each of the requests, we need to figure out how to map
  // each of the values in the input label map. Precedence will be as follows:
  // types, regions, region-type pairs. In other words, if the user requests
  // both LEFTLUNG and AIRWAY (not as a pair), then an AIRWAY voxel in the
  // LEFTLUNG will be mapped to LEFTLUNG in the output. If the user additionally
  // requests the AIRWAY, LEFTLUNG pair, then the entire voxel will be preserved.
  std::list< unsigned short >::iterator listIt;
  listIt = valueList.begin();

  while ( listIt != valueList.end() )
    {
    unsigned char inputType   = this->m_ChestConventions.GetChestTypeFromValue( *listIt );
    unsigned char inputRegion = this->m_ChestConventions.GetChestRegionFromValue( *listIt );

    for ( unsigned int i=0; i<this->m_TypeVec.size(); i++ )
      {
	if ( inputType == this->m_TypeVec[i] )
	  {
	    this->m_ValueToValueMap[*listIt] = 
	      this->m_ChestConventions.GetValueFromChestRegionAndType( (unsigned char)(cip::UNDEFINEDREGION), this->m_TypeVec[i] );
	    break;
	  }
      }

    for ( unsigned int i=0; i<this->m_RegionVec.size(); i++ )
      {
	if ( this->m_ChestConventions.CheckSubordinateSuperiorChestRegionRelationship( inputRegion, this->m_RegionVec[i] ) )
	  {
	    this->m_ValueToValueMap[*listIt] = 
	      this->m_ChestConventions.GetValueFromChestRegionAndType(this->m_RegionVec[i], (unsigned char)(cip::UNDEFINEDTYPE) );
	    break;
	  }
      }

    for ( unsigned int i=0; i<this->m_RegionAndTypeVec.size(); i++ )
      {
	if ( this->m_ChestConventions.CheckSubordinateSuperiorChestRegionRelationship(inputRegion, this->m_RegionAndTypeVec[i].lungRegionValue) &&
	     inputType == this->m_RegionAndTypeVec[i].lungTypeValue )
	  {
	    this->m_ValueToValueMap[*listIt] = 
	      this->m_ChestConventions.GetValueFromChestRegionAndType( this->m_RegionAndTypeVec[i].lungRegionValue, 
								       this->m_RegionAndTypeVec[i].lungTypeValue );
	    break;
	  }
      }

    listIt++;
    }
}


/**
 * Standard "PrintSelf" method
 */
template < unsigned int Dimension >
void
CIPExtractChestLabelMapImageFilter< Dimension >
::PrintSelf(
  std::ostream& os, 
  Indent indent) const
{
  Superclass::PrintSelf( os, indent );
}

} // end namespace itk
