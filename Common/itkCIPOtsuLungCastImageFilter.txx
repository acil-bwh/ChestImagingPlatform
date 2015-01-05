/**
 *
 */
#ifndef _itkCIPOtsuLungCastImageFilter_txx
#define _itkCIPOtsuLungCastImageFilter_txx

#include "itkCIPOtsuLungCastImageFilter.h"
#include "cipChestConventions.h"
#include "cipExceptionObject.h"
#include "itkImageFileWriter.h" //DEBUG

namespace itk
{

template < class TInputImage, class TOutputImage >
CIPOtsuLungCastImageFilter< TInputImage, TOutputImage >
::CIPOtsuLungCastImageFilter()
{

}

template< class TInputImage, class TOutputImage >
void
CIPOtsuLungCastImageFilter< TInputImage, TOutputImage >
::GenerateData()
{  
  if ( (typeid(OutputPixelType) != typeid(unsigned short)) && 
       (typeid(OutputPixelType) != typeid(unsigned char)) )
    {
      throw cip::ExceptionObject( __FILE__, __LINE__, "CIPOtsuLungCastImageFilter::GenerateData()", 
				  "Unsupported output pixel type. Must be unsigned short or unsigned char." );
    }

  // Allocate space for the output image
  typename Superclass::InputImageConstPointer inputPtr  = this->GetInput();
  typename Superclass::OutputImagePointer     outputPtr = this->GetOutput(0);
    outputPtr->SetRequestedRegion( inputPtr->GetRequestedRegion() );
    outputPtr->SetBufferedRegion( inputPtr->GetBufferedRegion() );
    outputPtr->SetLargestPossibleRegion( inputPtr->GetLargestPossibleRegion() );
    outputPtr->Allocate();
    outputPtr->FillBuffer( 0 );

  int ctXDim = (this->GetInput()->GetBufferedRegion().GetSize())[0];
  int ctYDim = (this->GetInput()->GetBufferedRegion().GetSize())[1];

  {
    // The first step is to run Otsu threshold on the input data.  This
    // classifies each voxel as either "body" or "air"
    typename OtsuThresholdType::Pointer otsuThreshold = OtsuThresholdType::New();
      otsuThreshold->SetInput( this->GetInput() );
      otsuThreshold->Update();
    
    this->GraftOutput( otsuThreshold->GetOutput() );
  }

  // Go slice-by-slice and remove all objects touching one of the four
  // corners
  this->RemoveCornerObjects();  
  
  // The next step is to identify all connected components in the
  // thresholded image
  typename ConnectedComponent3DType::Pointer connectedComponent = ConnectedComponent3DType::New();
    connectedComponent->SetInput( this->GetOutput() );
    connectedComponent->SetFullyConnected( true );
    connectedComponent->Update();

  // Relabel the connected components
  Relabel3DType::Pointer relabelComponent = Relabel3DType::New();
    relabelComponent->SetInput( connectedComponent->GetOutput() );
    relabelComponent->Update();    

  // Now we want to identify the component labels that correspond to
  // the left lung and the right lung. In some cases, they might not
  // be connected, that's why we need to do a separate search for
  // each.
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
      typename OutputImageType::IndexType index = rIt.GetIndex();

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

      lungHalf1Label = (unsigned short)( i );
      }
    if ( lungHalf2ComponentCounter[i] > maxLungHalf2Count )
      {
      maxLungHalf2Count = lungHalf2ComponentCounter[i];

      lungHalf2Label = (unsigned short)( i );
      }
    }

  OutputIteratorType mIt( this->GetOutput(), this->GetOutput()->GetBufferedRegion() );

  mIt.GoToBegin();
  rIt.GoToBegin();
  while ( !mIt.IsAtEnd() )
    {
    if ( rIt.Get() == lungHalf1Label || rIt.Get() == lungHalf2Label )
      {
      mIt.Set( OutputPixelType( cip::WHOLELUNG ) );
      }
    else 
      {
      mIt.Set( 0 );
      }

    ++mIt;
    ++rIt;
    }
}

template< class TInputImage, class TOutputImage >
void CIPOtsuLungCastImageFilter< TInputImage, TOutputImage >
::RemoveCornerObjects()
{
  typename OutputImageType::SizeType size = this->GetOutput()->GetBufferedRegion().GetSize();

  typename OutputImageType::SizeType sliceExtractorSize;
    sliceExtractorSize[0] = size[0];
    sliceExtractorSize[1] = size[1];
    sliceExtractorSize[2] = 0;

  typename OutputImageType::IndexType sliceStartIndex;
    sliceStartIndex[0] = 0;
    sliceStartIndex[1] = 0;

  typename OutputImageType::RegionType sliceExtractorRegion;
    sliceExtractorRegion.SetSize( sliceExtractorSize );

  typename OutputImageExtractorType::Pointer sliceExtractor = OutputImageExtractorType::New();
    sliceExtractor->SetInput( this->GetOutput() );
    sliceExtractor->SetDirectionCollapseToIdentity();

  typename ConnectedComponent2DType::Pointer connectedComponents = ConnectedComponent2DType::New();

  ComponentSliceType::IndexType sliceIndex;
  typename OutputImageType::IndexType index;

  for ( unsigned int z=0; z<size[2]; z++ )
    {
      sliceStartIndex[2] = z;
      sliceExtractorRegion.SetIndex( sliceStartIndex );

      sliceExtractor->SetExtractionRegion( sliceExtractorRegion );
      sliceExtractor->Update();

      connectedComponents->SetInput( sliceExtractor->GetOutput() );
      connectedComponents->Update();

      std::vector< unsigned long > cornerLabels;

      sliceIndex[0] = 0;
      sliceIndex[1] = 0;
      cornerLabels.push_back( connectedComponents->GetOutput()->GetPixel( sliceIndex ) );

      sliceIndex[0] = 0;
      sliceIndex[1] = size[1] - 1;
      cornerLabels.push_back( connectedComponents->GetOutput()->GetPixel( sliceIndex ) );

      sliceIndex[0] = size[0] - 1;
      sliceIndex[1] = 0;
      cornerLabels.push_back( connectedComponents->GetOutput()->GetPixel( sliceIndex ) );

      sliceIndex[0] = size[0] - 1;
      sliceIndex[1] = size[1] - 1;
      cornerLabels.push_back( connectedComponents->GetOutput()->GetPixel( sliceIndex ) );

      ComponentSliceIteratorType cIt( connectedComponents->GetOutput(), connectedComponents->GetOutput()->GetBufferedRegion() );

      cIt.GoToBegin();
      while ( !cIt.IsAtEnd() )
	{
	  if ( cIt.Get() != 0 )
	    {
	      for ( unsigned int i=0; i<cornerLabels.size(); i++ )
		{
		  if ( cIt.Get() == cornerLabels[i] )
		    {
		      index[0] = cIt.GetIndex()[0];
		      index[1] = cIt.GetIndex()[1];
		      index[2] = z;

		      this->GetOutput()->SetPixel( index, 0 );
		      break;
		    }
		}
	    }

	  ++cIt;
	}
    }
}
  
/**
 * Standard "PrintSelf" method
 */
template < class TInputImage, class TOutputImage >
void
CIPOtsuLungCastImageFilter< TInputImage, TOutputImage >
::PrintSelf(
  std::ostream& os, 
  Indent indent) const
{
  Superclass::PrintSelf( os, indent );
  os << indent << "Printing itkCIPOtsuLungCastImageFilter: " << std::endl;
}

} // end namespace itk

#endif

