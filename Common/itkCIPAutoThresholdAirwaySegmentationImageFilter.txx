/**
 *
 *
 */


#ifndef _itkCIPAutoThresholdAirwaySegmentationImageFilter_txx
#define _itkCIPAutoThresholdAirwaySegmentationImageFilter_txx

#include "itkCIPAutoThresholdAirwaySegmentationImageFilter.h"
#include "itkNumericTraits.h"
#include "cipExceptionObject.h"

namespace itk
{

template < class TInputImage >
CIPAutoThresholdAirwaySegmentationImageFilter< TInputImage >
::CIPAutoThresholdAirwaySegmentationImageFilter()
{
  // We manually segmented the airway trees from twenty five 
  // inspiratory CT scans acquired from healthy individuals. 
  // 30,000 mm^3 is approximately the smallest of the airway
  // volumes we computed
  this->m_MaxAirwayVolume = 30000;
 
  this->m_MinIntensityThresholdSet = false;
  this->m_MaxIntensityThresholdSet = false;
}


template< class TInputImage >
void
CIPAutoThresholdAirwaySegmentationImageFilter< TInputImage >
::GenerateData()
{
  if ( !this->m_MaxIntensityThresholdSet )
    {
      throw cip::ExceptionObject( __FILE__, __LINE__, "CIPAutoThresholdAirwaySegmentationImageFilter::GenerateData()", 
				  "Max intensity threshold not set" );
    }
  if ( !this->m_MinIntensityThresholdSet )
    {
      throw cip::ExceptionObject( __FILE__, __LINE__, "CIPAutoThresholdAirwaySegmentationImageFilter::GenerateData()", 
				  "Min intensity threshold not set" );
    }

  // Allocate space for the output image
  typename Superclass::InputImageConstPointer inputPtr  = this->GetInput();
  typename Superclass::OutputImagePointer     outputPtr = this->GetOutput(0);
    outputPtr->SetRequestedRegion( inputPtr->GetRequestedRegion() );
    outputPtr->SetBufferedRegion( inputPtr->GetBufferedRegion() );
    outputPtr->SetLargestPossibleRegion( inputPtr->GetLargestPossibleRegion() );
    outputPtr->Allocate();
    outputPtr->FillBuffer( 0 );

  unsigned short airwayLabel = this->m_ChestConventions.GetValueFromChestRegionAndType( cip::UNDEFINEDREGION, cip::AIRWAY );

  typename InputImageType::SpacingType spacing = this->GetInput()->GetSpacing();

  // Set up the connected threshold airway segmentation filter
  typename SegmentationType::Pointer segmentationFilter = SegmentationType::New();
    segmentationFilter->SetInput( this->GetInput() );
    segmentationFilter->SetLower( this->m_MinIntensityThreshold );
    segmentationFilter->SetReplaceValue( airwayLabel );
  for ( unsigned int s=0; s<this->m_SeedVec.size(); s++ )
    {
    segmentationFilter->AddSeed( this->m_SeedVec[s] );
    }

  InputPixelType currentThresh   = this->m_MaxIntensityThreshold;
  InputPixelType lastUpperThresh = this->m_MaxIntensityThreshold;
  InputPixelType lastLowerThresh = this->m_MinIntensityThreshold;

  assert( lastLowerThresh < lastUpperThresh );

  unsigned int inc = 0;
  while ( lastUpperThresh - lastLowerThresh > 5 )
    {
    inc++;
    assert( inc < this->m_MaxIntensityThreshold - itk::NumericTraits< InputPixelType >::min() );

    segmentationFilter->SetUpper( currentThresh );
    segmentationFilter->Update();

    // Compute the volume of the tree
    int count = 0;
    LabelMapIteratorType sIt( segmentationFilter->GetOutput(), segmentationFilter->GetOutput()->GetBufferedRegion() );

    sIt.GoToBegin();
    while ( !sIt.IsAtEnd() )
      {
      if ( sIt.Get() == airwayLabel )
        {
        count++;
        }

      ++sIt;
      }

    double volume = static_cast< double >( count )*spacing[0]*spacing[1]*spacing[2];

    if ( volume > this->m_MaxAirwayVolume )
      {
	InputPixelType tmp = (lastLowerThresh + currentThresh)/2;
	lastUpperThresh = currentThresh;
	currentThresh = tmp;
      }
    else
      {
	InputPixelType tmp = (currentThresh + lastUpperThresh)/2;
	lastLowerThresh = currentThresh;
	currentThresh = tmp;
      }
    }

  segmentationFilter->SetUpper( lastLowerThresh );
  try
    {
      segmentationFilter->Update();
    }
  catch ( itk::ExceptionObject &excp )
    {
      std::cerr << "Exception caught while running airway segmentation:";
      std::cerr << excp << std::endl;
    }

  // Fill holes that might be in the airway mask by performing
  // morphological closing
  ElementType structuringElement;
    structuringElement.SetRadius( 1 );
    structuringElement.CreateStructuringElement();

  typename DilateType::Pointer dilater = DilateType::New();
    dilater->SetInput( segmentationFilter->GetOutput() );
    dilater->SetKernel( structuringElement );
    dilater->SetDilateValue( airwayLabel );
  try
    {
    dilater->Update();
    }
  catch ( itk::ExceptionObject &excp )
    {
    std::cerr << "Exception caught dilating:";
    std::cerr << excp << std::endl;
    }

  typename ErodeType::Pointer eroder = ErodeType::New();
    eroder->SetInput( dilater->GetOutput() );
    eroder->SetKernel( structuringElement );
    eroder->SetErodeValue( airwayLabel );
  try
    {
    eroder->Update();
    }
  catch ( itk::ExceptionObject &excp )
    {
    std::cerr << "Exception caught eroding:";
    std::cerr << excp << std::endl;
    }

  this->GraftOutput( eroder->GetOutput() );
}


template < class TInputImage >
void
CIPAutoThresholdAirwaySegmentationImageFilter< TInputImage >
::AddSeed( OutputImageType::IndexType seed )
{
  this->m_SeedVec.push_back( seed );
}
  

template < class TInputImage >
void
CIPAutoThresholdAirwaySegmentationImageFilter< TInputImage >
::SetMaxIntensityThreshold( InputPixelType threshold ) 
{
  this->m_MaxIntensityThreshold = threshold;
  this->m_MaxIntensityThresholdSet = true;
}


template < class TInputImage >
void
CIPAutoThresholdAirwaySegmentationImageFilter< TInputImage >
::SetMinIntensityThreshold( InputPixelType threshold ) 
{
  this->m_MinIntensityThreshold = threshold;
  this->m_MinIntensityThresholdSet = true;
}


/**
 * Standard "PrintSelf" method
 */
template < class TInputImage >
void
CIPAutoThresholdAirwaySegmentationImageFilter< TInputImage >
::PrintSelf( std::ostream& os, Indent indent ) const
{
  Superclass::PrintSelf( os, indent );
  os << indent << "Printing itkCIPAutoThresholdAirwaySegmentationImageFilter: " << std::endl;
  os << indent << "m_MaxAirwayVolume:\t" << this->m_MaxAirwayVolume << std::endl;
  os << indent << "m_MinIntensityThreshold:\t" << m_MinIntensityThreshold << std::endl;
  os << indent << "m_MaxIntensityThreshold:\t" << m_MaxIntensityThreshold << std::endl;
  os << indent << "m_MinIntensityThresholdSet:\t" << m_MinIntensityThresholdSet << std::endl;
  os << indent << "m_MaxIntensityThresholdSet:\t" << m_MaxIntensityThresholdSet << std::endl;
  for ( unsigned int i=0; i<this->m_SeedVec.size(); i++ )
    {
    os << indent << "Seed " << i << ":\t" << this->m_SeedVec[i] << std::endl;
    }
}

} // end namespace itk

#endif
