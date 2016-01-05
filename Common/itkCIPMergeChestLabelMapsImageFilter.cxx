#ifndef _itkCIPMergeChestLabelMapsImageFilter_txx
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
  this->m_Union        = false;
}


void
CIPMergeChestLabelMapsImageFilter
::GenerateData()
{
  // Allocate the output buffer
  this->GetOutput()->SetBufferedRegion( this->GetOutput()->GetRequestedRegion() );
  this->GetOutput()->Allocate();
  this->GetOutput()->FillBuffer( 0 );

  if ( this->m_Union )
    {
      this->Union();
    }
  else if ( this->m_GraftOverlay )
    {
    this->GraftOverlay();
    }
  else if ( this->m_MergeOverlay )
    {
    this->MergeOverlay();
    }
  else
    {
    this->ApplyRules();
    }
}

void
CIPMergeChestLabelMapsImageFilter
::ApplyRules()
{
  unsigned char iRegion, iType; // For input
  unsigned char oRegion, oType; // For overlay
  bool preserve = false;

  ConstIteratorType iIt( this->GetInput(), this->GetInput()->GetBufferedRegion() );
  IteratorType oIt( this->GetOutput(), this->GetOutput()->GetBufferedRegion() );
  IteratorType ovIt( this->m_OverlayImage, this->m_OverlayImage->GetBufferedRegion() );

  oIt.GoToBegin();
  iIt.GoToBegin();
  ovIt.GoToBegin();
  while ( !oIt.IsAtEnd() )
    {
      // By default the merged label map gets the value of the input label map
      oIt.Set( iIt.Get() );

      // Zero out any override regions, types, or region-type pairs if necessary
      if ( iIt.Get() != 0 )
	{
	  iRegion = this->m_ChestConventions.GetChestRegionFromValue( iIt.Get() );	      
	  iType = this->m_ChestConventions.GetChestTypeFromValue( iIt.Get() );

	  // Only consider override types if the base value actually represents a type
	  if ( iIt.Get() > 255 )
	    {
	      for ( unsigned int i=0; i<this->m_OverrideChestTypeVec.size(); i++ )
		{
		  if ( this->m_OverrideChestTypeVec[i] == iType )
		    {
		      oIt.Set( this->m_ChestConventions.GetValueFromChestRegionAndType( iRegion, (unsigned char)(cip::UNDEFINEDTYPE) ) );
		      break;
		    }
		}
	    }

	  // Now deal with override regions
	  for ( unsigned int i=0; i<this->m_OverrideChestRegionVec.size(); i++ )
	    {
	      if ( this->m_OverrideChestRegionVec[i] == iRegion )
	  	{
	  	  oIt.Set( this->m_ChestConventions.GetValueFromChestRegionAndType( (unsigned char)(cip::UNDEFINEDREGION), iType ) );
	  	  break;
	  	}
	    }

	  // Now consider override region-type pairs
	  for ( unsigned int i=0; i<this->m_OverrideChestRegionTypePairVec.size(); i++ )
	    {
	      if ( this->m_OverrideChestRegionTypePairVec[i].chestRegion == iRegion && this->m_OverrideChestRegionTypePairVec[i].chestType == iType )
	  	{
	  	  oIt.Set( this->m_ChestConventions.GetValueFromChestRegionAndType( iRegion, (unsigned char)(cip::UNDEFINEDTYPE) ) );
	  	  break;
	  	}
	    }
	}

      // Now apply overrides and merges provided that the overlay label map voxel actually has something to apply
      if ( ovIt.Get() != 0 )
      	{
      	  iRegion = this->m_ChestConventions.GetChestRegionFromValue( iIt.Get() );	      
      	  iType = this->m_ChestConventions.GetChestTypeFromValue( iIt.Get() );

      	  oRegion = this->m_ChestConventions.GetChestRegionFromValue( ovIt.Get() );	      
      	  oType = this->m_ChestConventions.GetChestTypeFromValue( ovIt.Get() );
	  
      	  // Only consider override types if the overlay value actually represents a type
      	  if ( ovIt.Get() > 255 )
      	    {
      	      for ( unsigned int i=0; i<this->m_OverrideChestTypeVec.size(); i++ )
      		{
      		  if ( this->m_OverrideChestTypeVec[i] == oType )
      		    {
      		      preserve = false;
      		      for ( unsigned int j=0; j<this->m_PreserveChestTypeVec.size(); j++ )
      			{
      			  if ( this->m_PreserveChestTypeVec[j] == iType )
      			    {
      			      preserve = true;
      			      break;
      			    }
      			}

      		      if ( !preserve )
      			{
      			  oIt.Set( this->m_ChestConventions.GetValueFromChestRegionAndType( iRegion, oType ) );
      			}
      		      break;
      		    }
      		}
      	    }

      	  // Now deal with override regions
      	  for ( unsigned int i=0; i<this->m_OverrideChestRegionVec.size(); i++ )
      	    {
      	      if ( this->m_OverrideChestRegionVec[i] == oRegion )
      		{
      		  preserve = false;
      		  for ( unsigned int j=0; j<this->m_PreserveChestRegionVec.size(); j++ )
      		    {
      		      if ( this->m_PreserveChestRegionVec[j] == iRegion )
      			{
      			  preserve = true;
      			  break;
      			}
      		    }
		  
      		  if ( !preserve )
      		    {
      		      oIt.Set( this->m_ChestConventions.GetValueFromChestRegionAndType( oRegion, iType ) );
      		    }
      		  break;
      		}
      	    }

      	  // Now consider override region-type pairs
      	  for ( unsigned int i=0; i<this->m_OverrideChestRegionTypePairVec.size(); i++ )
      	    {
      	      if ( this->m_OverrideChestRegionTypePairVec[i].chestRegion == oRegion && this->m_OverrideChestRegionTypePairVec[i].chestType == oType )
      		{
      		  preserve = false;
      		  for ( unsigned int j=0; j<this->m_PreserveChestRegionVec.size(); j++ )
      		    {
      		      if ( this->m_PreserveChestRegionTypePairVec[j].chestRegion == iRegion && 
      			   this->m_PreserveChestRegionTypePairVec[j].chestType == iType )
      			{
      			  preserve = true;
      			  break;
      			}
      		    }
		  
      		  if ( !preserve )
      		    {
      		      oIt.Set( this->m_ChestConventions.GetValueFromChestRegionAndType( oRegion, oType ) );
      		    }
      		  break;
      		}
      	    }

      	  // Only consider merge types if the overlay value actually represents a type
      	  if ( ovIt.Get() > 255 )
      	    {
      	      for ( unsigned int i=0; i<this->m_MergeChestTypeVec.size(); i++ )
      		{
      		  if ( this->m_MergeChestTypeVec[i] == oType )
      		    {
      		      preserve = false;
      		      for ( unsigned int j=0; j<this->m_PreserveChestTypeVec.size(); j++ )
      			{
      			  if ( this->m_PreserveChestTypeVec[j] == iType )
      			    {
      			      preserve = true;
      			      break;
      			    }
      			}
		      
      		      if ( !preserve )
      			{
      			  oIt.Set( this->m_ChestConventions.GetValueFromChestRegionAndType( iRegion, oType ) );
      			}
      		      break;
      		    }
      		}
      	    }
	  
      	  // Now deal with merge regions
      	  for ( unsigned int i=0; i<this->m_MergeChestRegionVec.size(); i++ )
      	    {
      	      if ( this->m_MergeChestRegionVec[i] == oRegion )
      		{
      		  preserve = false;
      		  for ( unsigned int j=0; j<this->m_PreserveChestRegionVec.size(); j++ )
      		    {
      		      if ( this->m_PreserveChestRegionVec[j] == iRegion )
      			{
      			  preserve = true;
      			  break;
      			}
      		    }
		  
      		  if ( !preserve )
      		    {
      		      oIt.Set( this->m_ChestConventions.GetValueFromChestRegionAndType( oRegion, iType ) );
      		    }
      		  break;
      		}
      	    }
	  
      	  // Now consider merge region-type pairs
      	  for ( unsigned int i=0; i<this->m_MergeChestRegionTypePairVec.size(); i++ )
      	    {
      	      if ( this->m_MergeChestRegionTypePairVec[i].chestRegion == oRegion && this->m_MergeChestRegionTypePairVec[i].chestType == oType )
      		{
      		  preserve = false;
      		  for ( unsigned int j=0; j<this->m_PreserveChestRegionVec.size(); j++ )
      		    {
      		      if ( this->m_PreserveChestRegionTypePairVec[j].chestRegion == iRegion && 
      			   this->m_PreserveChestRegionTypePairVec[j].chestType == iType )
      			{
      			  preserve = true;
      			  break;
      			}
      		    }
		  
      		  if ( !preserve )
      		    {
      		      oIt.Set( this->m_ChestConventions.GetValueFromChestRegionAndType( oRegion, oType ) );
      		    }
      		  break;
      		}
      	    }      
      	}
      
      ++ovIt;  
      ++iIt;
      ++oIt;
    } 
}

void
CIPMergeChestLabelMapsImageFilter
::Union()
{
  IteratorType ovIt( this->m_OverlayImage, this->m_OverlayImage->GetBufferedRegion() );
  ConstIteratorType iIt( this->GetInput(), this->GetInput()->GetBufferedRegion() );
  IteratorType oIt( this->GetOutput(), this->GetOutput()->GetBufferedRegion() );

  oIt.GoToBegin();
  iIt.GoToBegin();
  while ( !oIt.IsAtEnd() )
    {
      if ( iIt.Get() == 0 && ovIt.Get() == 0 )
	{
	  oIt.Set( 0 );
	}
      else if ( iIt.Get() == 0 && ovIt.Get() != 0 )
	{
	  oIt.Set( ovIt.Get() );
	}
      else if ( iIt.Get() != 0 && ovIt.Get() == 0 )
	{
	  oIt.Set( iIt.Get() );
	}
      else 
	{
	  unsigned char overlayChestRegion = this->m_ChestConventions.GetChestRegionFromValue( ovIt.Get() );
	  unsigned char baseChestRegion    = this->m_ChestConventions.GetChestRegionFromValue( iIt.Get() );
	  unsigned char overlayChestType   = this->m_ChestConventions.GetChestTypeFromValue( ovIt.Get() );
	  unsigned char baseChestType      = this->m_ChestConventions.GetChestTypeFromValue( iIt.Get() );

	  unsigned char mergedChestRegion = 0;
	  unsigned char mergedChestType   = 0;

	  if ( overlayChestRegion == 0 && baseChestRegion != 0 )
	    {
	      mergedChestRegion = baseChestRegion;
	    }
	  else if ( overlayChestRegion != 0 && baseChestRegion == 0 )
	    {
	      mergedChestRegion = overlayChestRegion;
	    }
	  else if ( this->m_ChestConventions.CheckSubordinateSuperiorChestRegionRelationship( baseChestRegion, overlayChestRegion ) )
	    {
	      mergedChestRegion = baseChestRegion;
	    }
	  else if ( this->m_ChestConventions.CheckSubordinateSuperiorChestRegionRelationship( overlayChestRegion, baseChestRegion ) )
	    {
	      mergedChestRegion = overlayChestRegion;
	    }
	  else if ( overlayChestRegion != 0 && baseChestRegion != 0 )
	    {
	      mergedChestRegion = baseChestRegion;
	    }

	  if ( overlayChestType == 0 && baseChestType != 0 )
	    {
	      mergedChestType = baseChestType;
	    }
	  else if ( overlayChestType != 0 && baseChestType == 0 )
	    {
	      mergedChestType = overlayChestType;
	    }
	  else if ( overlayChestType != 0 && baseChestType != 0 )
	    {
	      mergedChestType = baseChestType;
	    }

	  oIt.Set( this->m_ChestConventions.GetValueFromChestRegionAndType( mergedChestRegion, mergedChestType ) );	  
	}
    
      ++ovIt;
      ++iIt;
      ++oIt;
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
    if ( lungType == this->m_PreserveChestRegionTypePairVec[i].chestType &&
         lungRegion == this->m_PreserveChestRegionTypePairVec[i].chestRegion )
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
