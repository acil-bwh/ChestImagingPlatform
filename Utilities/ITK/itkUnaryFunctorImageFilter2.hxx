/*=========================================================================
 *
 *  Copyright Insight Software Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
#ifndef __itkUnaryFunctorImageFilter2_hxx
#define __itkUnaryFunctorImageFilter2_hxx

#include "itkUnaryFunctorImageFilter2.h"
#include "itkImageRegionIterator.h"
#include "itkProgressReporter.h"

namespace itk
{
/**
 * Constructor
 */
template< class TInputImage, class TOutputImage >
UnaryFunctorImageFilter2< TInputImage, TOutputImage >
::UnaryFunctorImageFilter2()
{
  this->SetNumberOfRequiredInputs(1);
  this->InPlaceOff();
}

/**
 * UnaryFunctorImageFilter2 can produce an image which is a different resolution
 * than its input image.  As such, UnaryFunctorImageFilter2 needs to provide an
 * implementation for GenerateOutputInformation() in order to inform
 * the pipeline execution model.  The original documentation of this
 * method is below.
 *
 * \sa ProcessObject::GenerateOutputInformaton()
 */
template< class TInputImage, class TOutputImage >
void
UnaryFunctorImageFilter2< TInputImage, TOutputImage >
::GenerateOutputInformation()
{
  // do not call the superclass' implementation of this method since
  // this filter allows the input the output to be of different dimensions

  // get pointers to the input and output
  typename Superclass::OutputImagePointer outputPtr = this->GetOutput();
  typename Superclass::InputImageConstPointer inputPtr  = this->GetInput();

  if ( !outputPtr || !inputPtr )
    {
    return;
    }

  // Set the output image largest possible region.  Use a RegionCopier
  // so that the input and output images can be different dimensions.
  OutputImageRegionType outputLargestPossibleRegion;
  this->CallCopyInputRegionToOutputRegion( outputLargestPossibleRegion,
                                           inputPtr->GetLargestPossibleRegion() );
  outputPtr->SetLargestPossibleRegion(outputLargestPossibleRegion);

  // Set the output spacing and origin
  const ImageBase< Superclass::InputImageDimension > *phyData;

  phyData =
    dynamic_cast< const ImageBase< Superclass::InputImageDimension > * >( this->GetInput() );

  if ( phyData )
    {
    // Copy what we can from the image from spacing and origin of the input
    // This logic needs to be augmented with logic that select which
    // dimensions to copy
    unsigned int i, j;
    const typename InputImageType::SpacingType &
    inputSpacing = inputPtr->GetSpacing();
    const typename InputImageType::PointType &
    inputOrigin = inputPtr->GetOrigin();
    const typename InputImageType::DirectionType &
    inputDirection = inputPtr->GetDirection();

    typename OutputImageType::SpacingType outputSpacing;
    typename OutputImageType::PointType outputOrigin;
    typename OutputImageType::DirectionType outputDirection;

    // copy the input to the output and fill the rest of the
    // output with zeros.
    for ( i = 0; i < Superclass::InputImageDimension; ++i )
      {
      outputSpacing[i] = inputSpacing[i];
      outputOrigin[i] = inputOrigin[i];
      for ( j = 0; j < Superclass::OutputImageDimension; j++ )
        {
        if ( j < Superclass::InputImageDimension )
          {
          outputDirection[j][i] = inputDirection[j][i];
          }
        else
          {
          outputDirection[j][i] = 0.0;
          }
        }
      }
    for (; i < Superclass::OutputImageDimension; ++i )
      {
      outputSpacing[i] = 1.0;
      outputOrigin[i] = 0.0;
      for ( j = 0; j < Superclass::OutputImageDimension; j++ )
        {
        if ( j == i )
          {
          outputDirection[j][i] = 1.0;
          }
        else
          {
          outputDirection[j][i] = 0.0;
          }
        }
      }

    // set the spacing and origin
    outputPtr->SetSpacing(outputSpacing);
    outputPtr->SetOrigin(outputOrigin);
    outputPtr->SetDirection(outputDirection);
    outputPtr->SetNumberOfComponentsPerPixel(  // propagate vector length info
      inputPtr->GetNumberOfComponentsPerPixel() );
    }
  else
    {
    // pointer could not be cast back down
    itkExceptionMacro( << "itk::UnaryFunctorImageFilter2::GenerateOutputInformation "
                       << "cannot cast input to "
                       << typeid( ImageBase< Superclass::InputImageDimension > * ).name() );
    }
}

/**
 * ThreadedGenerateData Performs the pixel-wise addition
 */
template< class TInputImage, class TOutputImage >
void
UnaryFunctorImageFilter2< TInputImage, TOutputImage >
::ThreadedGenerateData(const OutputImageRegionType & outputRegionForThread,
                       ThreadIdType threadId)
{
  InputImagePointer  inputPtr = this->GetInput();
  OutputImagePointer outputPtr = this->GetOutput(0);

  // Define the portion of the input to walk for this thread, using
  // the CallCopyOutputRegionToInputRegion method allows for the input
  // and output images to be different dimensions
  InputImageRegionType inputRegionForThread;

  this->CallCopyOutputRegionToInputRegion(inputRegionForThread, outputRegionForThread);

  // Define the iterators
  ImageRegionConstIterator< TInputImage > inputIt(inputPtr, inputRegionForThread);
  ImageRegionIterator< TOutputImage >     outputIt(outputPtr, outputRegionForThread);

  ProgressReporter progress( this, threadId, outputRegionForThread.GetNumberOfPixels() );

  inputIt.GoToBegin();
  outputIt.GoToBegin();

  while ( !inputIt.IsAtEnd() )
    {
    outputIt.Set( m_Functor->Evaluate( inputIt.Get() ) );
    ++inputIt;
    ++outputIt;
    progress.CompletedPixel();  // potential exception thrown here
    }
}
} // end namespace itk

#endif
