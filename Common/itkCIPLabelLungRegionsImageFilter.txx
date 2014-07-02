

#ifndef _itkCIPLabelLungRegionsImageFilter_txx
#define _itkCIPLabelLungRegionsImageFilter_txx

#include "itkCIPLabelLungRegionsImageFilter.h"
#include "itkImageFileWriter.h"


namespace itk
{

CIPLabelLungRegionsImageFilter
::CIPLabelLungRegionsImageFilter()
{
  this->m_HeadFirst               =  true;
  this->m_Supine                  =  true;
  this->m_LabelLungThirds         =  false;
  this->m_LabelLeftAndRightLungs  =  false;
  this->m_LabelingSuccess         =  false;
  this->m_NumberLungVoxels        =  0;
}


void
CIPLabelLungRegionsImageFilter
::GenerateData()
{
  //
  // Allocate space for the output image
  //
  Superclass::InputImageConstPointer inputPtr  = this->GetInput();
  Superclass::OutputImagePointer     outputPtr = this->GetOutput(0);
    outputPtr->SetRequestedRegion( inputPtr->GetRequestedRegion() );
    outputPtr->SetBufferedRegion( inputPtr->GetBufferedRegion() );
    outputPtr->SetLargestPossibleRegion( inputPtr->GetLargestPossibleRegion() );
    outputPtr->Allocate();
    outputPtr->FillBuffer( 0 );

  //
  // Start by filling the output image with the WHOLELUNG region at
  // every location where the input image has a lung region set. Begin
  // by determining which of the input values correspond to a defined
  // lung region.
  //
  std::list< unsigned short > inputValuesList;

  InputIteratorType iIt( this->GetInput(), this->GetInput()->GetBufferedRegion() );

  iIt.GoToBegin();
  while ( !iIt.IsAtEnd() )
    {
    if ( iIt.Get() != 0 )
      {
      inputValuesList.push_back( iIt.Get() );
      }

    ++iIt;
    }
  
  inputValuesList.unique();
  inputValuesList.sort();
  inputValuesList.unique();

  std::vector< unsigned short > definedLungRegionValuesVec;

  std::list< unsigned short >::iterator listIt;

  std::map< unsigned short, unsigned char > valueToTypeMap; 

  for ( listIt = inputValuesList.begin(); listIt != inputValuesList.end(); listIt++ )
    {
    unsigned char lungRegion = this->m_LungConventions.GetChestRegionFromValue( *listIt );
    unsigned char lungType   = this->m_LungConventions.GetChestTypeFromValue( *listIt );

    valueToTypeMap[ *listIt ] = lungType;

    if ( lungRegion != 0 )
      {
      definedLungRegionValuesVec.push_back( *listIt );
      }
    }

  //
  // Now that we've collected all the input values that correspond to
  // defined lung regions, we can initialize the output image with the
  // WHOLELUNG region
  //
  unsigned short wholeLungLabel = this->m_LungConventions.GetValueFromChestRegionAndType( cip::WHOLELUNG, cip::UNDEFINEDTYPE );

  LabelMapIteratorType oIt( this->GetOutput(), this->GetOutput()->GetBufferedRegion() );

  oIt.GoToBegin();
  iIt.GoToBegin();
  while ( !oIt.IsAtEnd() )
    {
    if ( iIt.Get() != 0 )
      {
      for ( unsigned int i=0; i<definedLungRegionValuesVec.size(); i++ )
        {
        if ( iIt.Get() == definedLungRegionValuesVec[i] )
          {
          oIt.Set( wholeLungLabel );

          this->m_NumberLungVoxels++;
          }
        }
      }

    ++oIt;
    ++iIt;
    }

  if ( this->m_LabelLungThirds || this->m_LabelLeftAndRightLungs )
    {
      this->m_LabelingSuccess = this->LabelLeftAndRightLungs();
    }
  if ( this->m_LabelLungThirds && this->m_LabelingSuccess )
    {
    this->SetLungThirds();
    }

  //
  // Now set the types from the input image 
  //
  iIt.GoToBegin();
  oIt.GoToBegin();
  while ( !oIt.IsAtEnd() )
    {
    if ( iIt.Get() != 0 )
      {
      unsigned short lungTypeValue = valueToTypeMap[iIt.Get()];

      if ( lungTypeValue != cip::UNDEFINEDTYPE )
        {
        this->SetType( oIt.GetIndex(), lungTypeValue ); 
        }
      }

    ++iIt;
    ++oIt;
    }
}


void
CIPLabelLungRegionsImageFilter
::SetType( OutputImageType::IndexType index, int lungTypeValue )
{
  unsigned short currentValue = this->GetOutput()->GetPixel( index );

  //
  // Get the binary representation of the current value
  //
  int currentPlaces[16];
  for ( int i=0; i<16; i++ )
    {
    currentPlaces[i] = 0;
    }

  for ( int i=15; i>=0; i-- )
    {
    int power = static_cast< int >( pow( 2.0, i ) );

    if ( power <= currentValue )
      {
      currentPlaces[i] = 1;
      
      currentValue = currentValue % power;
      }
    }

  //
  // Get the binary representation of the type to set
  //
  int typeValue = lungTypeValue;

  int typePlaces[8];
  for ( int i=0; i<8; i++ )
    {
    typePlaces[i] = 0;
    }

  for ( int i=7; i>=0; i-- )
    {
    int power = static_cast< int >( pow( 2.0, i ) );

    if ( power <= typeValue )
      {
      typePlaces[i] = 1;
      
      typeValue = typeValue % power;
      }
    }

  //
  // Compute the new value to assign to the label map voxel 
  //
  unsigned short newValue = 0;

  for ( int i=0; i<16; i++ )
    {
    if ( i < 8 )
      {
      newValue += static_cast< unsigned short >( currentPlaces[i] )*static_cast< unsigned short >( pow( 2.0, i ) );
      }
    else
      {
      newValue += static_cast< unsigned short >( typePlaces[i-8] )*static_cast< unsigned short >( pow( 2.0, i ) );
      }
    }

  this->GetOutput()->SetPixel( index, newValue );
}


/**
 * 
 */
bool
CIPLabelLungRegionsImageFilter
::LabelLeftAndRightLungs()
{
  cip::ChestConventions conventions;

  ConnectedComponentType::Pointer connectedComponent = ConnectedComponentType::New();
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

  //
  // If the second largest object doesn't comprise at least 30%
  // (arbitrary) of the foreground region, assume the lungs are
  // connected 
  //
  if ( static_cast< double >( relabeler->GetSizeOfObjectsInPixels()[1] )/static_cast< double >( total ) < 0.3 )
    {
    return false;
    }

  //
  // If we're here, we assume that the left and right have been
  // separated, so label them. First, we need to get the relabel
  // component corresponding to the left and the right. We assume that
  // the relabel component value = 1 corresponds to one of the two
  // lungs and a value of 2 corresponds to the other. Find the
  // left-most and right-most component value. Assuming the scan is
  // supine, head-first, the component value corresponding to the
  // smallest x-index will be the left lung and the other major
  // component will be the right lung.
  //
  unsigned int minX = relabeler->GetOutput()->GetBufferedRegion().GetSize()[0];
  unsigned int maxX = 0;

  unsigned int smallIndexComponentLabel, largeIndexComponentLabel;

  LabelMapIteratorType rIt( relabeler->GetOutput(), relabeler->GetOutput()->GetBufferedRegion() );

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

  LabelMapIteratorType oIt( this->GetOutput(), this->GetOutput()->GetBufferedRegion() );

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
void
CIPLabelLungRegionsImageFilter
::SetLungThirds() 
{
  unsigned short leftLowerThirdLabel   = this->m_LungConventions.GetValueFromChestRegionAndType( cip::LEFTLOWERTHIRD, cip::UNDEFINEDTYPE );
  unsigned short rightLowerThirdLabel  = this->m_LungConventions.GetValueFromChestRegionAndType( cip::RIGHTLOWERTHIRD, cip::UNDEFINEDTYPE );
  unsigned short leftMiddleThirdLabel  = this->m_LungConventions.GetValueFromChestRegionAndType( cip::LEFTMIDDLETHIRD, cip::UNDEFINEDTYPE );
  unsigned short rightMiddleThirdLabel = this->m_LungConventions.GetValueFromChestRegionAndType( cip::RIGHTMIDDLETHIRD, cip::UNDEFINEDTYPE );
  unsigned short leftUpperThirdLabel   = this->m_LungConventions.GetValueFromChestRegionAndType( cip::LEFTUPPERTHIRD, cip::UNDEFINEDTYPE );
  unsigned short rightUpperThirdLabel  = this->m_LungConventions.GetValueFromChestRegionAndType( cip::RIGHTUPPERTHIRD, cip::UNDEFINEDTYPE );

  int voxelTally = 0;

  LabelMapIteratorType oIt( this->GetOutput(), this->GetOutput()->GetBufferedRegion() );

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
            oIt.Set( leftLowerThirdLabel );
            }
          if ( oIt.Get() == cip::RIGHTLUNG )
            {
            oIt.Set( rightLowerThirdLabel );
            }
          }
        else
          {
          if ( oIt.Get() == cip::LEFTLUNG )
            {
            oIt.Set( leftUpperThirdLabel );
            }
          if ( oIt.Get() == cip::RIGHTLUNG )
            {
            oIt.Set( rightUpperThirdLabel );
            }
          }
        }
      else if ( voxelTally <= 2*this->m_NumberLungVoxels/3 )
        {
        if ( oIt.Get() == cip::LEFTLUNG )
          {
          oIt.Set( leftMiddleThirdLabel );
          }
        if ( oIt.Get() == cip::RIGHTLUNG )
          {
          oIt.Set( rightMiddleThirdLabel );
          }
        }
      else
        {
        if ( this->m_HeadFirst )
          {
          if ( oIt.Get() == cip::LEFTLUNG )
            {
            oIt.Set( leftUpperThirdLabel );
            }
          if ( oIt.Get() == cip::RIGHTLUNG )
            {
            oIt.Set( rightUpperThirdLabel );
            }
          }
        else
          {
          if ( oIt.Get() == cip::LEFTLUNG )
            {
            oIt.Set( leftLowerThirdLabel );
            }
          if ( oIt.Get() == cip::RIGHTLUNG )
            {
            oIt.Set( rightLowerThirdLabel );
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
void
CIPLabelLungRegionsImageFilter
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
