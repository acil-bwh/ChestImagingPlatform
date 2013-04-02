/*=========================================================================

=========================================================================*/
#ifndef _itkCIPOtsuLungCastImageFilter_txx
#define _itkCIPOtsuLungCastImageFilter_txx

#include "itkCIPOtsuLungCastImageFilter.h"
#include "cipConventions.h"


namespace itk
{

template < class TInputImage >
CIPOtsuLungCastImageFilter< TInputImage >
::CIPOtsuLungCastImageFilter()
{

}


template< class TInputImage >
void
CIPOtsuLungCastImageFilter< TInputImage >
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

  //
  // The first step is to run Otsu threshold on the input data.  This
  // classifies each voxel as either "body" or "air"
  //
  typename OtsuThresholdType::Pointer otsuThreshold = OtsuThresholdType::New();
    otsuThreshold->SetInput( this->GetInput() );
    otsuThreshold->Update();

  int ctXDim = (this->GetInput()->GetBufferedRegion().GetSize())[0];
  int ctYDim = (this->GetInput()->GetBufferedRegion().GetSize())[1];

  this->GraftOutput( otsuThreshold->GetOutput() );
  //
  // The next step is to identify all connected components in the
  // thresholded image
  //
  ConnectedComponent3DType::Pointer connectedComponent = ConnectedComponent3DType::New();
    connectedComponent->SetInput( otsuThreshold->GetOutput() );
    connectedComponent->Update();

  //
  // Relabel the connected components
  //
  Relabel3DType::Pointer relabelComponent = Relabel3DType::New();
    relabelComponent->SetInput( connectedComponent->GetOutput() );
    relabelComponent->Update();  

  //
  // Now we want to identify the component labels that correspond to
  // the left lung and the right lung. In some cases, they might not
  // be connected, that's why we need to do a separate search for
  // each.
  //
  std::vector< int >  lungHalf1ComponentCounter;
  std::vector< int >  lungHalf2ComponentCounter;
  for ( unsigned int i=0; i<=relabelComponent->GetNumberOfObjects(); i++ )
    {
    lungHalf1ComponentCounter.push_back( 0 );
    lungHalf2ComponentCounter.push_back( 0 );
    }

  ComponentIteratorType rIt( relabelComponent->GetOutput(), relabelComponent->GetOutput()->GetBufferedRegion() );

  int lowerYBound = static_cast< int >( 0.45*static_cast< double >( ctYDim ) );
  int upperYBound = static_cast< int >( 0.55*static_cast< double >( ctYDim ) );

  int lowerXBound = static_cast< int >( 0.20*static_cast< double >( ctXDim ) );
  int upperXBound = static_cast< int >( 0.80*static_cast< double >( ctXDim ) );

  int middleX =  static_cast< int >( 0.5*static_cast< double >( ctXDim ) );

  rIt.GoToBegin();
  while ( !rIt.IsAtEnd() )
    {
    if ( rIt.Get() != 0 )
      {
      LabelMapType::IndexType index = rIt.GetIndex();

      if ( index[1] >= lowerYBound && index[1] <= upperYBound )
        {
        int whichComponent = static_cast< int >( rIt.Get() );

        if ( index[0] >= lowerXBound && index[0] <= middleX )
          {
          lungHalf1ComponentCounter[whichComponent] = lungHalf1ComponentCounter[whichComponent]+1;
          }
        else if ( index[0] < upperXBound && index[0] > middleX )
          {
          lungHalf2ComponentCounter[whichComponent] = lungHalf2ComponentCounter[whichComponent]+1;
          }
        }
      }

    ++rIt;
    }

  unsigned short lungHalf1Label;
  unsigned short lungHalf2Label;
  int maxLungHalf1Count = 0;
  int maxLungHalf2Count = 0;
  for ( unsigned int i=0; i<=relabelComponent->GetNumberOfObjects(); i++ )
    {
    if ( lungHalf1ComponentCounter[i] > maxLungHalf1Count )
      {
      maxLungHalf1Count = lungHalf1ComponentCounter[i];

      lungHalf1Label = static_cast< unsigned short >( i );
      }
    if ( lungHalf2ComponentCounter[i] > maxLungHalf2Count )
      {
      maxLungHalf2Count = lungHalf2ComponentCounter[i];

      lungHalf2Label = static_cast< unsigned short >( i );
      }
    }

  LabelMapIteratorType mIt( this->GetOutput(), this->GetOutput()->GetBufferedRegion() );

  mIt.GoToBegin();
  rIt.GoToBegin();
  while ( !mIt.IsAtEnd() )
    {
    if ( rIt.Get() == lungHalf1Label || rIt.Get() == lungHalf2Label )
      {
      mIt.Set( static_cast< unsigned short >( WHOLELUNG ) );
      }
    else 
      {
      mIt.Set( 0 );
      }

    ++mIt;
    ++rIt;
    }
}

  
/**
 * Standard "PrintSelf" method
 */
template < class TInputImage >
void
CIPOtsuLungCastImageFilter< TInputImage >
::PrintSelf(
  std::ostream& os, 
  Indent indent) const
{
  Superclass::PrintSelf( os, indent );
  os << indent << "Printing itkCIPOtsuLungCastImageFilter: " << std::endl;
}

} // end namespace itk

#endif

