/**
 *
 *  $Date:  $
 *  $Revision:  $
 *  $Author: $
 *
 */


#ifndef _itkCIPAutoThresholdAirwaySegmentationImageFilter_txx
#define _itkCIPAutoThresholdAirwaySegmentationImageFilter_txx

#include "itkCIPAutoThresholdAirwaySegmentationImageFilter.h"


namespace itk
{

template < class TInputImage >
CIPAutoThresholdAirwaySegmentationImageFilter< TInputImage >
::CIPAutoThresholdAirwaySegmentationImageFilter()
{
  this->m_MinAirwayVolume             = 10.0;
  this->m_MaxAirwayVolume             = 500.0;
  this->m_MaxAirwayVolumeIncreaseRate = 2.0; 
}


template< class TInputImage >
void
CIPAutoThresholdAirwaySegmentationImageFilter< TInputImage >
::GenerateData()
{
  //
  // Allocate space for the output image
  //
  typename Superclass::InputImageConstPointer inputPtr  = this->GetInput();
  typename Superclass::OutputImagePointer     outputPtr = this->GetOutput(0);
    outputPtr->SetRequestedRegion( inputPtr->GetRequestedRegion() );
    outputPtr->SetBufferedRegion( inputPtr->GetBufferedRegion() );
    outputPtr->SetLargestPossibleRegion( inputPtr->GetLargestPossibleRegion() );
    outputPtr->Allocate();
    outputPtr->FillBuffer( 0 );

  this->Test();

//   unsigned short airwayLabel = this->m_LungConventions.GetValueFromLungRegionAndType( UNDEFINEDREGION, AIRWAY );

//   typename InputImageType::SpacingType spacing = this->GetInput()->GetSpacing();

//   //
//   // Find the darkest value among the seeds
//   //
//   InputPixelType darkestSeedPixel = itk::NumericTraits< InputPixelType >::max();

//   for ( unsigned int i=0; i<this->m_SeedVec.size(); i++ )
//     {
//     InputPixelType value = this->GetInput()->GetPixel( this->m_SeedVec[i] );

//     if ( value < darkestSeedPixel )
//       {
//       darkestSeedPixel = value;
//       }
//     }

//   //
//   // Set up the connected threshold airway segmentation filter
//   //
//   typename SegmentationType::Pointer segmentationFilter = SegmentationType::New();
//     segmentationFilter->SetInput( this->GetInput() );
//     segmentationFilter->SetLower( itk::NumericTraits< InputPixelType >::min() );
//     segmentationFilter->SetReplaceValue( airwayLabel );

//   for ( unsigned int s=0; s<this->m_SeedVec.size(); s++ )
//     {
//     segmentationFilter->AddSeed( this->m_SeedVec[s] );
//     }


//   short  finalThreshold  = darkestSeedPixel;
//   short  threshold       = darkestSeedPixel;
//   short  thresholdInc    = 32;
//   double lastVolume      = -1;
//   bool   leakageDetected = false;

//   while ( !leakageDetected )
//     {
//     segmentationFilter->SetUpper( threshold );
//     segmentationFilter->Update();

//     //
//     // Compute the volume of the tree
//     //
//     int count = 0;

//     LabelMapIteratorType sIt( segmentationFilter->GetOutput(), segmentationFilter->GetOutput()->GetBufferedRegion() );

//     sIt.GoToBegin();
//     while ( !sIt.IsAtEnd() )
//       {
//       if ( sIt.Get() == airwayLabel )
//         {
//         count++;
//         }

//       ++sIt;
//       }

//     double volume = static_cast< double >( count )*spacing[0]*spacing[1]*spacing[2]/1000.0;

//     if ( volume > this->m_MaxAirwayVolume )
//       {
//       leakageDetected = true;

//       finalThreshold = threshold - thresholdInc;
//       }
//     else if ( volume > this->m_MinAirwayVolume && lastVolume == -1 && thresholdInc != 1 )
//       {
//       threshold -= thresholdInc;
      
//       thresholdInc = thresholdInc/2;
//       }
//     else if ( volume > this->m_MinAirwayVolume )
//       {
//       if ( lastVolume == -1 )
//         {
//         lastVolume = volume;

//         thresholdInc = 32;
//         }
//       else
//         {
//         if ( (volume-lastVolume)/static_cast< double >( thresholdInc ) > this->m_MaxAirwayVolumeIncreaseRate )
//           {
//           if ( thresholdInc != 1 )
//             {
//             threshold -= thresholdInc;

//             thresholdInc = thresholdInc/2;
//             }
//           else
//             {
//             leakageDetected = true;

//             finalThreshold = threshold-1;
//             }
//           }
//         else
//           {          
//           lastVolume = volume;
//           }
//         }
//       }

//     threshold += thresholdInc;
//     }

//   segmentationFilter->SetUpper( finalThreshold );
//   try
//     {
//   segmentationFilter->Update();
//     }
//   catch ( itk::ExceptionObject &excp )
//     {
//     std::cerr << "Exception caught while running airway segmentation:";
//     std::cerr << excp << std::endl;
//     }

//   //
//   // Fill holes that might be in the airway mask by performing
//   // morphological closing
//   //
//   ElementType structuringElement;
//     structuringElement.SetRadius( 1 );
//     structuringElement.CreateStructuringElement();

//   typename DilateType::Pointer dilater = DilateType::New();
//     dilater->SetInput( segmentationFilter->GetOutput() );
//     dilater->SetKernel( structuringElement );
//     dilater->SetDilateValue( airwayLabel );
//   try
//     {
//     dilater->Update();
//     }
//   catch ( itk::ExceptionObject &excp )
//     {
//     std::cerr << "Exception caught dilating:";
//     std::cerr << excp << std::endl;
//     }

//   typename ErodeType::Pointer eroder = ErodeType::New();
//     eroder->SetInput( dilater->GetOutput() );
//     eroder->SetKernel( structuringElement );
//     eroder->SetErodeValue( airwayLabel );
//   try
//     {
//     eroder->Update();
//     }
//   catch ( itk::ExceptionObject &excp )
//     {
//     std::cerr << "Exception caught eroding:";
//     std::cerr << excp << std::endl;
//     }

//   this->GraftOutput( eroder->GetOutput() );
}


template < class TInputImage >
void
CIPAutoThresholdAirwaySegmentationImageFilter< TInputImage >
::Test()
{
  unsigned short airwayLabel = this->m_LungConventions.GetValueFromLungRegionAndType( cip::UNDEFINEDREGION, cip::AIRWAY );

  typename InputImageType::SpacingType spacing = this->GetInput()->GetSpacing();

  unsigned int numVoxels = 0; // Will keep track of the number of
                              // voxels as we add to the vector of
                              // indices to go in the output

  //
  // 'indicesVecFinal' will be a container for our final collection of
  // indices to set to foreground in our output label
  // map. 'indicesVecTemp' will be a temporary container that will
  // collect new indices during the iterative process below. After a
  // given iteration, the indices in 'indicesVecTemp' will be dumped
  // into 'indicesVecFinal' (provided we don't go over the max number
  // of allowable indices as deteremined by the 'm_MaxAirwayVolume'.
  //
  std::vector< OutputImageType::IndexType > indicesVecFinal;
  std::vector< OutputImageType::IndexType > indicesVecTemp;

  //
  // Find the darkest value among the seeds
  //
  InputPixelType darkestSeedPixel = itk::NumericTraits< InputPixelType >::max();

  for ( unsigned int i=0; i<this->m_SeedVec.size(); i++ )
    {
    numVoxels++;

    this->GetOutput()->SetPixel( this->m_SeedVec[i], airwayLabel );

    indicesVecFinal.push_back( this->m_SeedVec[i] );

    InputPixelType value = this->GetInput()->GetPixel( this->m_SeedVec[i] );

    if ( value < darkestSeedPixel )
      {
      darkestSeedPixel = value;
      }
    }

  //
  // Out strategy will be to grow the airway region until the max
  // volume is exactly reached. We will do this be considering 3x3x3
  // neighborhoods of the current set of seed points using the
  // darkestSeedPixel value as our initial threshold value. We will
  // keep adding indices to our label map until the max volume is
  // reached. If after considering all neighborhoods in our current
  // set of indices no new indices are added and the max volume has
  // not been reached, increment the threshold value.
  //
  //short threshold = darkestSeedPixel;
  short threshold=-960; //sila
  unsigned int maxNumberVoxels = static_cast< unsigned int >( this->m_MaxAirwayVolume/(spacing[0]*spacing[1]*spacing[2]) );
  unsigned int minNumberVoxels = static_cast< unsigned int >( this->m_MinAirwayVolume/(spacing[0]*spacing[1]*spacing[2]) );

  OutputImageType::IndexType index;

  bool add, addedNew;

  while ( numVoxels < minNumberVoxels )
    {
    indicesVecTemp.clear();

    addedNew = false;

    for ( unsigned int i=0; i<indicesVecFinal.size(); i++ )
      {
      for ( int x=-1; x<=1; x++ )
        {
        index[0] = indicesVecFinal[i][0] + x;
        
        for ( int y=-1; y<=1; y++ )
          {
          index[1] = indicesVecFinal[i][1] + y;
          
          for ( int z=-1; z<=1; z++ )
            {
            index[2] = indicesVecFinal[i][2] + z;

            if ( this->GetInput()->GetBufferedRegion().IsInside( index ) )
              {
              if ( this->GetOutput()->GetPixel( index ) == 0 && this->GetInput()->GetPixel( index ) <= threshold )
                {
                add = true;
                for ( unsigned int j=0; j<indicesVecTemp.size(); j++ )
                  {
                  if ( indicesVecTemp[j][0] == index[0] && indicesVecTemp[j][1] == index[1] && indicesVecTemp[j][2] == index[2] )
                    {
                    add = false;
                    break;
                    }
                  }
                if ( add && numVoxels < maxNumberVoxels )
                  {
                  indicesVecTemp.push_back( index );
                  numVoxels++;

                  addedNew = true;
                  }
                }
              }
            }
          }
        }
      } 
  
    if ( addedNew )
      {
      for ( unsigned int k=0; k<indicesVecTemp.size(); k++ )
        {
        indicesVecFinal.push_back( indicesVecTemp[k] );

        this->GetOutput()->SetPixel( indicesVecTemp[k], airwayLabel );
        }
      }
    else if ( numVoxels < maxNumberVoxels )
      {
      threshold += 10; 
      }
    }

}


template < class TInputImage >
void
CIPAutoThresholdAirwaySegmentationImageFilter< TInputImage >
::AddSeed( OutputImageType::IndexType seed )
{
  this->m_SeedVec.push_back( seed );
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
  os << indent << "MinAirwayVolume:\t" << this->m_MinAirwayVolume << std::endl;
  os << indent << "MaxAirwayVolumeIncreaseRate:\t" << this->m_MaxAirwayVolumeIncreaseRate << std::endl;
  for ( unsigned int i=0; i<this->m_SeedVec.size(); i++ )
    {
    os << indent << "Seed " << i << ":\t" << this->m_SeedVec[i] << std::endl;
    }
}

} // end namespace itk

#endif
