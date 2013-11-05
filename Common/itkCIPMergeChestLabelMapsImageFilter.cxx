#ifndef _itCIPkMergeChestLabelMapsImageFilter_txx
#define _itkCIPMergeChestLabelMapsImageFilter_txx

#include "itkCIPMergeChestLabelMapsImageFilter.h"

namespace itk
{

CIPMergeChestLabelMapsImageFilter
::CIPMergeChestLabelMapsImageFilter()
{
  this->m_OverlayImage = cip::LabelMapType::New();
  this->m_GraftOverlay = false;
  this->m_MergeOverlay = false;
}


void
CIPMergeChestLabelMapsImageFilter
::GenerateData()
{
  // Allocate the output buffer
  this->GetOutput()->SetBufferedRegion( this->GetOutput()->GetRequestedRegion() );
  this->GetOutput()->Allocate();
  this->GetOutput()->FillBuffer( 0 );

  if ( this->m_GraftOverlay )
    {
    this->GraftOverlay();
    }
  else if ( this->m_MergeOverlay )
    {
    this->MergeOverlay();
    }
  else
    {
    this->InitializeOutputWithInputAndOverrides();
    this->ApplyOverlayRules();
    }
}


void
CIPMergeChestLabelMapsImageFilter
::GraftOverlay()
{
  ConstIteratorType iIt( this->GetInput(), this->GetInput()->GetBufferedRegion() );
  IteratorType oIt( this->GetOutput(), this->GetOutput()->GetBufferedRegion() );

  oIt.GoToBegin();
  iIt.GoToBegin();
  while ( !oIt.IsAtEnd() )
    {
    oIt.Set( iIt.Get() );

    ++iIt;
    ++oIt;
    }

  cip::LabelMapType::SizeType overlayImageSize = this->m_OverlayImage->GetBufferedRegion().GetSize();

  cip::LabelMapType::RegionType overlayImageRegion;
    overlayImageRegion.SetSize( overlayImageSize );
    overlayImageRegion.SetIndex( this->m_OverlayImageStartIndex );

  IteratorType ovIt( this->m_OverlayImage, this->m_OverlayImage->GetBufferedRegion() );
  IteratorType orIt( this->GetOutput(), overlayImageRegion );

  ovIt.GoToBegin();
  orIt.GoToBegin();
  while ( !ovIt.IsAtEnd() )
    {
    orIt.Set( ovIt.Get() );

    ++ovIt;
    ++orIt;
    }
}


void
CIPMergeChestLabelMapsImageFilter
::MergeOverlay()
{
  ConstIteratorType iIt( this->GetInput(), this->GetInput()->GetBufferedRegion() );
  IteratorType oIt( this->GetOutput(), this->GetOutput()->GetBufferedRegion() );

  oIt.GoToBegin();
  iIt.GoToBegin();
  while ( !oIt.IsAtEnd() )
    {
    oIt.Set( iIt.Get() );

    ++iIt;
    ++oIt;
    }

  cip::LabelMapType::SizeType overlayImageSize = this->m_OverlayImage->GetBufferedRegion().GetSize();

  cip::LabelMapType::RegionType overlayImageRegion;
    overlayImageRegion.SetSize( overlayImageSize );
    overlayImageRegion.SetIndex( this->m_OverlayImageStartIndex );

  IteratorType ovIt( this->m_OverlayImage, this->m_OverlayImage->GetBufferedRegion() );
  IteratorType orIt( this->GetOutput(), overlayImageRegion );

  ovIt.GoToBegin();
  orIt.GoToBegin();
  while ( !ovIt.IsAtEnd() )
    {
    if ( ovIt.Get() != 0 )
      {
      orIt.Set( ovIt.Get() );
      }

    ++ovIt;
    ++orIt;
    }
}


void
CIPMergeChestLabelMapsImageFilter
::InitializeOutputWithInputAndOverrides()
{
  std::map< unsigned short, unsigned short >::iterator mapIt;

  ConstIteratorType iIt( this->GetInput(), this->GetInput()->GetBufferedRegion() );

  iIt.GoToBegin();
  while ( !iIt.IsAtEnd() )
    {
    this->m_InputLabelsToCleanedLabelsMap.insert( std::pair< unsigned short, unsigned short >(iIt.Get(), iIt.Get()) );

    ++iIt;
    }

  // For all the override regions specified by the user, we want to go
  // through the list of labels in the input image, find the regions
  // in the list that are meant to be overriden, and set them to
  // UNDEFINEDREGION. We DON'T want to do this for any region that is
  // meant to be preserved, however.
  for ( int i=0; i<this->m_OverrideChestRegionVec.size(); i++ )
    {
    // Iterate over the map. For each label in the map, we want to get
    // the corresponding region. If the region is in the override
    // vector and NOT in the preserve region vector, then the label
    // will be mapped to the override region
    for ( mapIt = this->m_InputLabelsToCleanedLabelsMap.begin(); mapIt != this->m_InputLabelsToCleanedLabelsMap.end(); mapIt++ )
      {
      unsigned char lungRegion = this->m_ChestConventions.GetChestRegionFromValue( (*mapIt).first );
      unsigned char lungType   = this->m_ChestConventions.GetChestTypeFromValue( (*mapIt).first );

      if ( lungRegion == this->m_OverrideChestRegionVec[i] && this->GetPermitChestRegionChange( lungRegion ) )
        {
	  this->m_InputLabelsToCleanedLabelsMap[(*mapIt).first] = this->m_ChestConventions.GetValueFromChestRegionAndType( cip::UNDEFINEDREGION, lungType );
        }
      }
    }

  // For all the override types specified by the user, we want to go
  // through the list of labels in the input image, find the types
  // in the list that are meant to be overriden, and set them to
  // UNDEFINEDTYPE. We DON'T want to do this for any type that is
  // meant to be preserved, however.
  for ( int i=0; i<this->m_OverrideChestTypeVec.size(); i++ )
    {
    // Iterate over the map. For each label in the map, we want to get
    // the corresponding type. If the type is in the override
    // vector and NOT in the preserve type vector, then the label
    // will be mapped to the override type
    for ( mapIt = this->m_InputLabelsToCleanedLabelsMap.begin(); mapIt != this->m_InputLabelsToCleanedLabelsMap.end(); mapIt++ )
      {
      unsigned char lungRegion = this->m_ChestConventions.GetChestRegionFromValue( (*mapIt).first );
      unsigned char lungType   = this->m_ChestConventions.GetChestTypeFromValue( (*mapIt).first );

      if ( lungType == this->m_OverrideChestTypeVec[i] && this->GetPermitChestTypeChange( lungRegion, lungType ) )
        {
	  this->m_InputLabelsToCleanedLabelsMap[(*mapIt).first] = this->m_ChestConventions.GetValueFromChestRegionAndType( lungRegion, cip::UNDEFINEDTYPE );         
        }
      }
    }

  // For all the override region-type pairs specified by the user, we
  // want to go through the list of labels in the input image, find
  // the pairs in the list that are meant to be overriden, and set the
  // type to UNDEFINEDTYPE (note that the region will be untouched). 
  for ( int i=0; i<this->m_OverrideChestRegionTypePairVec.size(); i++ )
    {
    // Iterate over the map. For each label in the map, we want to get
    // the corresponding region and type. If the region and type are
    // in the override vector, then the label will be mapped with an
    // UNDEFINEDTYPE (the region will be untouched).
    for ( mapIt = this->m_InputLabelsToCleanedLabelsMap.begin(); mapIt != this->m_InputLabelsToCleanedLabelsMap.end(); mapIt++ )
      {
      unsigned char lungRegion = this->m_ChestConventions.GetChestRegionFromValue( (*mapIt).first );
      unsigned char lungType   = this->m_ChestConventions.GetChestTypeFromValue( (*mapIt).first );

      if ( lungType == this->m_OverrideChestRegionTypePairVec[i].lungTypeValue &&
           lungRegion == this->m_OverrideChestRegionTypePairVec[i].lungRegionValue )
        {
	  this->m_InputLabelsToCleanedLabelsMap[(*mapIt).first] = this->m_ChestConventions.GetValueFromChestRegionAndType( lungRegion, cip::UNDEFINEDTYPE );
        }
      }
    }
}

void
CIPMergeChestLabelMapsImageFilter
::SetOverlayImage( cip::LabelMapType::Pointer overlayImage )
{
  this->m_OverlayImage = overlayImage; 

  this->m_OverlayImageStartIndex[0] = 0;
  this->m_OverlayImageStartIndex[1] = 0;
  this->m_OverlayImageStartIndex[2] = 0;
}

void
CIPMergeChestLabelMapsImageFilter
::SetOverlayImage( cip::LabelMapType::Pointer overlayImage, cip::LabelMapType::IndexType startIndex )
{
  this->m_OverlayImage = overlayImage; 

  this->m_OverlayImageStartIndex[0] = startIndex[0];
  this->m_OverlayImageStartIndex[1] = startIndex[1];
  this->m_OverlayImageStartIndex[2] = startIndex[2];
}

void
CIPMergeChestLabelMapsImageFilter
::ApplyOverlayRules()
{
  // At this point we are ready to start filling in the output image
  // with values from the input (base) image. We have a mapping
  // between labels in the input image and the "cleaned" labels that
  // were generated from the "override" lists -- use this mapping to
  // populate the output image
  ConstIteratorType iIt( this->GetInput(), this->GetInput()->GetBufferedRegion() );
  IteratorType oIt( this->GetOutput(), this->GetOutput()->GetBufferedRegion() );

  oIt.GoToBegin();
  iIt.GoToBegin();
  while ( !oIt.IsAtEnd() )
    {
    oIt.Set( this->m_InputLabelsToCleanedLabelsMap[iIt.Get()] );
    
    ++oIt;
    ++iIt;
    }

  // Now we are ready to transfer the desired regions, types, and
  // region-type pairs from the overlay image to the output
  // image. Only do the transfer for non preserved regions, types, and
  // pairs
  cip::LabelMapType::SizeType overlayImageSize = this->m_OverlayImage->GetBufferedRegion().GetSize();

  cip::LabelMapType::RegionType overlayImageRegion;
    overlayImageRegion.SetSize( overlayImageSize );
    overlayImageRegion.SetIndex( this->m_OverlayImageStartIndex );

  IteratorType ovIt( this->m_OverlayImage, this->m_OverlayImage->GetBufferedRegion() );
  IteratorType orIt( this->GetOutput(), overlayImageRegion );

  ovIt.GoToBegin();
  orIt.GoToBegin();
  while ( !ovIt.IsAtEnd() )
    {
    unsigned char overlayChestRegion = this->m_ChestConventions.GetChestRegionFromValue( ovIt.Get() );
    unsigned char overlayChestType   = this->m_ChestConventions.GetChestTypeFromValue( ovIt.Get() );

    for ( int r=0; r<this->m_OverrideChestRegionVec.size(); r++ )
      {
      if ( overlayChestRegion == this->m_OverrideChestRegionVec[r] )
        {
        unsigned char lungRegion = this->m_ChestConventions.GetChestRegionFromValue( orIt.Get() );
        unsigned char lungType   = this->m_ChestConventions.GetChestTypeFromValue( orIt.Get() );

        if ( this->GetPermitChestRegionChange( lungRegion ) )
          {
          orIt.Set( this->m_ChestConventions.GetValueFromChestRegionAndType( overlayChestRegion, lungType ) );
          }

        break;
        }
      }

    for ( int r=0; r<this->m_MergeChestRegionVec.size(); r++ )
      {
      if ( overlayChestRegion == this->m_MergeChestRegionVec[r] )
        {
        unsigned char lungRegion = this->m_ChestConventions.GetChestRegionFromValue( orIt.Get() );
        unsigned char lungType   = this->m_ChestConventions.GetChestTypeFromValue( orIt.Get() );

        if ( this->GetPermitChestRegionChange( lungRegion ) )
          {
          orIt.Set( this->m_ChestConventions.GetValueFromChestRegionAndType( overlayChestRegion, lungType ) );
          }

        break;
        }
      }

    for ( int t=0; t<this->m_OverrideChestTypeVec.size(); t++ )
      {
      if ( overlayChestType == this->m_OverrideChestTypeVec[t] )
        {
        unsigned char lungRegion = this->m_ChestConventions.GetChestRegionFromValue( orIt.Get() );
        unsigned char lungType   = this->m_ChestConventions.GetChestTypeFromValue( orIt.Get() );

        if ( this->GetPermitChestTypeChange( lungRegion, lungType ) )
          {
          orIt.Set( this->m_ChestConventions.GetValueFromChestRegionAndType( lungRegion, overlayChestType ) );
          }

        break;
        }
      }

    for ( int t=0; t<this->m_MergeChestTypeVec.size(); t++ )
      {
      if ( overlayChestType == this->m_MergeChestTypeVec[t] )
        {
        unsigned char lungRegion = this->m_ChestConventions.GetChestRegionFromValue( orIt.Get() );
        unsigned char lungType   = this->m_ChestConventions.GetChestTypeFromValue( orIt.Get() );

        if ( this->GetPermitChestTypeChange( lungRegion, lungType ) )
          {
          orIt.Set( this->m_ChestConventions.GetValueFromChestRegionAndType( lungRegion, overlayChestType ) );
          }

        break;
        }
      }
        
    ++ovIt;
    ++orIt;
    }
}

bool
CIPMergeChestLabelMapsImageFilter
::GetPermitChestRegionChange( unsigned char lungRegion )
{
  for ( int i=0; i<this->m_PreserveChestRegionVec.size(); i++ )
    {
    if ( lungRegion == this->m_PreserveChestRegionVec[i] )
      {
      return false;
      }
    }

  return true;
}

bool
CIPMergeChestLabelMapsImageFilter
::GetPermitChestTypeChange( unsigned char lungRegion, unsigned char lungType )
{
  for ( int i=0; i<this->m_PreserveChestTypeVec.size(); i++ )
    {
    if ( lungType == this->m_PreserveChestTypeVec[i] )
      {
      return false;
      }
    }

  for ( int i=0; i<this->m_PreserveChestRegionTypePairVec.size(); i++ )
    {
    if ( lungType == this->m_PreserveChestRegionTypePairVec[i].lungTypeValue &&
         lungRegion == this->m_PreserveChestRegionTypePairVec[i].lungRegionValue )
      {
      return false;
      }
    }

  return true;
}

/**
 * Standard "PrintSelf" method
 */
void
CIPMergeChestLabelMapsImageFilter
::PrintSelf(
  std::ostream& os, 
  Indent indent) const
{
  Superclass::PrintSelf( os, indent );
  os << indent << "Printing itkCIPMergeChestLabelMapsImageFilter: " << std::endl;
}

} // end namespace itk

#endif
