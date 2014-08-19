#ifndef _itkCIPLabelLungRegionsImageFilter_txx
#define _itkCIPLabelLungRegionsImageFilter_txx

#include "itkCIPLabelLungRegionsImageFilter.h"
#include "itkImageFileWriter.h"
#include <typeinfo>

namespace itk
{

template < class TInputImage, class TOutputImage >
CIPLabelLungRegionsImageFilter< TInputImage, TOutputImage >
::CIPLabelLungRegionsImageFilter()
{
  this->m_HeadFirst               =  true;
  this->m_Supine                  =  true;
  this->m_LabelLungThirds         =  false;
  this->m_LabelLeftAndRightLungs  =  false;
  this->m_LabelingSuccess         =  false;
  this->m_NumberLungVoxels        =  0;
}

template < class TInputImage, class TOutputImage >
void
CIPLabelLungRegionsImageFilter< TInputImage, TOutputImage >
::GenerateData()
{
  if ( (typeid(OutputPixelType) != typeid(unsigned short)) && 
       (typeid(OutputPixelType) != typeid(unsigned char)) )
    {
      throw cip::ExceptionObject( __FILE__, __LINE__, "CIPLabelLungRegionsImageFilter::GenerateData()", 
				  "Unsupported output pixel type. Must be unsigned short or unsigned char." );
    }
  if ( (typeid(InputPixelType) != typeid(unsigned short)) && 
       (typeid(InputPixelType) != typeid(unsigned char)) )
    {
      throw cip::ExceptionObject( __FILE__, __LINE__, "CIPLabelLungRegionsImageFilter::GenerateData()", 
				  "Unsupported input pixel type. Must be unsigned short or unsigned char." );
    }

  // Allocate space for the output image
  typename Superclass::InputImageConstPointer inputPtr  = this->GetInput();
  typename Superclass::OutputImagePointer     outputPtr = this->GetOutput(0);
    outputPtr->SetRequestedRegion( inputPtr->GetRequestedRegion() );
    outputPtr->SetBufferedRegion( inputPtr->GetBufferedRegion() );
    outputPtr->SetLargestPossibleRegion( inputPtr->GetLargestPossibleRegion() );
    outputPtr->Allocate();
    outputPtr->FillBuffer( 0 );

  // Start by filling the output image with the WHOLELUNG region at
  // every location where the input image has a lung region set. 
  InputIteratorType iIt( this->GetInput(), this->GetInput()->GetBufferedRegion() );
  OutputIteratorType oIt( this->GetOutput(), this->GetOutput()->GetBufferedRegion() );

  iIt.GoToBegin();
  oIt.GoToBegin();
  while ( !iIt.IsAtEnd() )
    {
    if ( iIt.Get() != 0 )
      {
	if ( typeid(InputPixelType) == typeid(unsigned short) )
	  {
	    if ( this->m_Conventions.GetChestRegionFromValue( iIt.Get() ) > 0 )
	      {
		oIt.Set( OutputPixelType(cip::WHOLELUNG) );
		this->m_NumberLungVoxels++;
	      }
	  }
	else
	  {
	    oIt.Set( OutputPixelType(cip::WHOLELUNG) );
	    this->m_NumberLungVoxels++;
	  }
      }

    ++iIt;
    ++oIt;
    }
  
  if ( this->m_LabelLungThirds || this->m_LabelLeftAndRightLungs )
    {
      this->m_LabelingSuccess = this->LabelLeftAndRightLungs();
    }
  if ( this->m_LabelLungThirds && this->m_LabelingSuccess )
    {
    this->SetLungThirds();
    }

  // Now set the types from the input image 
  if ( typeid(InputPixelType) != typeid(unsigned short) && 
       typeid(OutputPixelType) != typeid(unsigned short) )
    {
      iIt.GoToBegin();
      oIt.GoToBegin();
      while ( !oIt.IsAtEnd() )
	{
	  if ( iIt.Get() != 0 )
	    {
	      unsigned char cipRegion = (unsigned char)(oIt.Get());
	      unsigned char cipType = this->m_Conventions.GetChestTypeFromValue( iIt.Get() );
	      unsigned short value = this->m_Conventions.GetValueFromChestRegionAndType( cipRegion, cipType );
	      oIt.Set( value );	      
	    }
	  
	  ++iIt;
	  ++oIt;
	}
    }
}

template < class TInputImage, class TOutputImage >
bool
CIPLabelLungRegionsImageFilter< TInputImage, TOutputImage >
::LabelLeftAndRightLungs()
{
  cip::ChestConventions conventions;

  typename ConnectedComponentType::Pointer connectedComponent = ConnectedComponentType::New();
    connectedComponent->SetInput( this->GetOutput() );
    connectedComponent->Update();

  RelabelComponentType::Pointer relabeler = RelabelComponentType::New();
    relabeler->SetInput( connectedComponent->GetOutput() );
  try
    {
    relabeler->Update();
    }
  catch ( itk::ExceptionObject &excp )
    {
    std::cerr << "Exception caught relabeling:";
    std::cerr << excp << std::endl;
    }

  if ( relabeler->GetNumberOfObjects() < 2 )
    {
    return false;
    }

  unsigned int total = 0;
  for ( unsigned int i=0; i<relabeler->GetNumberOfObjects(); i++ )
    {
    total += relabeler->GetSizeOfObjectsInPixels()[i];
    }

  // If the second largest object doesn't comprise at least 30%
  // (arbitrary) of the foreground region, assume the lungs are
  // connected 
  if ( double( relabeler->GetSizeOfObjectsInPixels()[1] )/double( total ) < 0.3 )
    {
    return false;
    }

  // If we're here, we assume that the left and right have been
  // separated, so label them. First, we need to get the relabel
  // component corresponding to the left and the right. We assume that
  // the relabel component value = 1 corresponds to one of the two
  // lungs and a value of 2 corresponds to the other. Find the
  // left-most and right-most component value. Assuming the scan is
  // supine, head-first, the component value corresponding to the
  // smallest x-index will be the left lung and the other major
  // component will be the right lung.
  unsigned int minX = relabeler->GetOutput()->GetBufferedRegion().GetSize()[0];
  unsigned int maxX = 0;

  unsigned int smallIndexComponentLabel, largeIndexComponentLabel;

  UShortIteratorType rIt( relabeler->GetOutput(), relabeler->GetOutput()->GetBufferedRegion() );

  rIt.GoToBegin();
  while ( !rIt.IsAtEnd() )
    {
    if ( rIt.Get() == 1 || rIt.Get() == 2 )
      {
      if ( rIt.GetIndex()[0] < minX )
        {
        smallIndexComponentLabel = rIt.Get();
        minX = rIt.GetIndex()[0];
        }
      if ( rIt.GetIndex()[0] > maxX )
        {
        largeIndexComponentLabel = rIt.Get();
        maxX = rIt.GetIndex()[0];
        }
      }

    ++rIt;
    }

  unsigned int leftLungComponentLabel, rightLungComponentLabel;
  if ( (this->m_HeadFirst && this->m_Supine) || (!this->m_HeadFirst && !this->m_Supine) )
    {
    leftLungComponentLabel  = largeIndexComponentLabel;
    rightLungComponentLabel = smallIndexComponentLabel;
    }
  else
    {
    leftLungComponentLabel  = smallIndexComponentLabel;
    rightLungComponentLabel = largeIndexComponentLabel;
    }

  OutputIteratorType oIt( this->GetOutput(), this->GetOutput()->GetBufferedRegion() );

  oIt.GoToBegin();
  rIt.GoToBegin();
  while ( !oIt.IsAtEnd() )
    {
    if ( rIt.Get() == leftLungComponentLabel )
      {
	oIt.Set( (unsigned short)( cip::LEFTLUNG ) );
      }
    if ( rIt.Get() == rightLungComponentLabel )
      {
	oIt.Set( (unsigned short)( cip::RIGHTLUNG ) );
      }

    ++rIt;
    ++oIt;
    }

  return true;
}


/**
 * Entering this method, the left and right lung designations have
 * been made.
 */
template < class TInputImage, class TOutputImage >
void
CIPLabelLungRegionsImageFilter< TInputImage, TOutputImage >
::SetLungThirds() 
{
  int voxelTally = 0;

  OutputIteratorType oIt( this->GetOutput(), this->GetOutput()->GetBufferedRegion() );

  oIt.GoToBegin();
  while ( !oIt.IsAtEnd() )
    {
      if ( oIt.Get() != 0 )
	{
	  voxelTally++;
	  
	  if ( voxelTally <= this->m_NumberLungVoxels/3 )
	    {
	      if ( this->m_HeadFirst )
		{
		  if ( oIt.Get() == cip::LEFTLUNG )
		    {
		      oIt.Set( (unsigned char)(cip::LEFTLOWERTHIRD) );
		    }
		  if ( oIt.Get() == cip::RIGHTLUNG )
		    {
		      oIt.Set( (unsigned char)(cip::RIGHTLOWERTHIRD) );
		    }
		}
	      else
		{
		  if ( oIt.Get() == cip::LEFTLUNG )
		    {
		      oIt.Set( (unsigned char)(cip::LEFTUPPERTHIRD) );
		    }
		  if ( oIt.Get() == cip::RIGHTLUNG )
		    {
		      oIt.Set( (unsigned char)(cip::RIGHTUPPERTHIRD) );
		    }
		}
	    }
	  else if ( voxelTally <= 2*this->m_NumberLungVoxels/3 )
	    {
	      if ( oIt.Get() == cip::LEFTLUNG )
		{
		  oIt.Set( (unsigned char)(cip::LEFTMIDDLETHIRD) );
		}
	      if ( oIt.Get() == cip::RIGHTLUNG )
		{
		  oIt.Set( (unsigned char)(cip::RIGHTMIDDLETHIRD) );
		}
	    }
	  else
	    {
	      if ( this->m_HeadFirst )
		{
		  if ( oIt.Get() == cip::LEFTLUNG )
		    {
		      oIt.Set( (unsigned char)(cip::LEFTUPPERTHIRD) );
		    }
		  if ( oIt.Get() == cip::RIGHTLUNG )
		    {
		      oIt.Set( (unsigned char)(cip::RIGHTUPPERTHIRD) );
		    }
		}
	      else
		{
		  if ( oIt.Get() == cip::LEFTLUNG )
		    {
		      oIt.Set( (unsigned char)(cip::LEFTLOWERTHIRD) );
		    }
		  if ( oIt.Get() == cip::RIGHTLUNG )
		    {
		      oIt.Set( (unsigned char)(cip::RIGHTLOWERTHIRD) );
		    }
		}
	    }
	}
      
      ++oIt;
    }
}

/**
 * Standard "PrintSelf" method
 */
template < class TInputImage, class TOutputImage >
void
CIPLabelLungRegionsImageFilter< TInputImage, TOutputImage >
::PrintSelf(
  std::ostream& os, 
  Indent indent) const
{
  Superclass::PrintSelf( os, indent );
  os << indent << "Printing itkCIPLabelLungRegionsImageFilter: " << std::endl;
  os << indent << "HeadFirst: " << this->m_HeadFirst << std::endl;
  os << indent << "Supine: " << this->m_Supine << std::endl;
  os << indent << "LabelLungThirds: " << this->m_LabelLungThirds << std::endl;
  os << indent << "LabelLeftAndRightLungs: " << this->m_LabelLeftAndRightLungs << std::endl;
  os << indent << "NumberLungVoxels: " << this->m_NumberLungVoxels << std::endl;
}

} // end namespace itk

#endif
