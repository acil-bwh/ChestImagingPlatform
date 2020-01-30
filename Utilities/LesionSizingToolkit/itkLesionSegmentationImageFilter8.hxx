/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    itkLesionSegmentationImageFilter8.hxx
  Language:  C++
  Date:      $Date$
  Version:   $Revision$

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkLesionSegmentationImageFilter8_hxx
#define __itkLesionSegmentationImageFilter8_hxx
#include "itkLesionSegmentationImageFilter8.h"

#include "itkNumericTraits.h"
#include "itkProgressReporter.h"
#include "itkGradientMagnitudeImageFilter.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"

namespace itk
{

template <class TInputImage, class TOutputImage>
LesionSegmentationImageFilter8<TInputImage, TOutputImage>::
LesionSegmentationImageFilter8()
{
  m_CannyEdgesFeatureGenerator = CannyEdgesFeatureGeneratorType::New();
  m_LesionSegmentationMethod = LesionSegmentationMethodType::New();
  m_LungWallFeatureGenerator = LungWallGeneratorType::New();
  m_VesselnessFeatureGenerator = VesselnessGeneratorType::New();
  m_SigmoidFeatureGenerator = SigmoidFeatureGeneratorType::New();
  m_FeatureAggregator = FeatureAggregatorType::New();
  m_SegmentationModule = SegmentationModuleType::New();
  m_CropFilter = CropFilterType::New();
  m_IsotropicResampler = IsotropicResamplerType::New();
  m_InputSpatialObject = InputImageSpatialObjectType::New();

  // Report progress.
  m_CommandObserver    = CommandType::New();
  m_CommandObserver->SetCallbackFunction(
    this, &Self::ProgressUpdate );
  m_LungWallFeatureGenerator->AddObserver(
      itk::ProgressEvent(), m_CommandObserver );
  m_SigmoidFeatureGenerator->AddObserver(
      itk::ProgressEvent(), m_CommandObserver );
  m_VesselnessFeatureGenerator->AddObserver(
      itk::ProgressEvent(), m_CommandObserver );
  m_CannyEdgesFeatureGenerator->AddObserver(
      itk::ProgressEvent(), m_CommandObserver );
  m_SegmentationModule->AddObserver(
      itk::ProgressEvent(), m_CommandObserver );
  m_CropFilter->AddObserver(
      itk::ProgressEvent(), m_CommandObserver );
  m_IsotropicResampler->AddObserver(
      itk::ProgressEvent(), m_CommandObserver );

  // Connect pipeline
  m_LungWallFeatureGenerator->SetInput( m_InputSpatialObject );
  m_SigmoidFeatureGenerator->SetInput( m_InputSpatialObject );
  m_VesselnessFeatureGenerator->SetInput( m_InputSpatialObject );
  m_CannyEdgesFeatureGenerator->SetInput( m_InputSpatialObject );
  m_FeatureAggregator->AddFeatureGenerator( m_LungWallFeatureGenerator );
  m_FeatureAggregator->AddFeatureGenerator( m_VesselnessFeatureGenerator );
  m_FeatureAggregator->AddFeatureGenerator( m_SigmoidFeatureGenerator );
  m_FeatureAggregator->AddFeatureGenerator( m_CannyEdgesFeatureGenerator );
  m_LesionSegmentationMethod->AddFeatureGenerator( m_FeatureAggregator );
  m_LesionSegmentationMethod->SetSegmentationModule( m_SegmentationModule );

  // Populate some parameters
  m_LungWallFeatureGenerator->SetLungThreshold( -400 );
  m_VesselnessFeatureGenerator->SetSigma( 1.0 );
  m_VesselnessFeatureGenerator->SetAlpha1( 0.1 );
  m_VesselnessFeatureGenerator->SetAlpha2( 2.0 );
  m_VesselnessFeatureGenerator->SetSigmoidAlpha( -10.0 );
  m_VesselnessFeatureGenerator->SetSigmoidBeta( 40.0 );
  m_SigmoidFeatureGenerator->SetAlpha( 100.0 );
  m_SigmoidFeatureGenerator->SetBeta( -500.0 );
  m_CannyEdgesFeatureGenerator->SetSigma(1.0);
  m_CannyEdgesFeatureGenerator->SetUpperThreshold( 150.0 );
  m_CannyEdgesFeatureGenerator->SetLowerThreshold( 75.0 );
  m_FastMarchingStoppingTime = 5.0;
  m_FastMarchingDistanceFromSeeds = 0.5;
  m_SigmoidBeta = -500.0;
  m_StatusMessage = "";
  m_SegmentationModule->SetCurvatureScaling(1.0);
  m_SegmentationModule->SetAdvectionScaling(0.0);
  m_SegmentationModule->SetPropagationScaling(500.0);
  m_SegmentationModule->SetMaximumRMSError(0.0002);
  m_SegmentationModule->SetMaximumNumberOfIterations(300);
  m_ResampleThickSliceData = true;
  m_AnisotropyThreshold = 1.0;
  m_UserSpecifiedSigmas = false;
}

template <class TInputImage, class TOutputImage>
void
LesionSegmentationImageFilter8<TInputImage,TOutputImage>
::GenerateInputRequestedRegion() throw(InvalidRequestedRegionError)
{
  // call the superclass' implementation of this method
  Superclass::GenerateInputRequestedRegion();

  if ( !this->GetInput() )
    {
    typename InputImageType::Pointer inputPtr  =
      const_cast< TInputImage *>( this->GetInput() );

    // Request the entire input image
    inputPtr->SetRequestedRegion(inputPtr->GetLargestPossibleRegion());
    }
}

template <class TInputImage, class TOutputImage>
void
LesionSegmentationImageFilter8<TInputImage,TOutputImage>
::SetSigma( SigmaArrayType s )
{
  this->m_UserSpecifiedSigmas = true;
  m_CannyEdgesFeatureGenerator->SetSigmaArray(s);
}

template <class TInputImage, class TOutputImage>
void
LesionSegmentationImageFilter8<TInputImage,TOutputImage>
::GenerateOutputInformation()
{
  // get pointers to the input and output
  typename Superclass::OutputImagePointer      outputPtr = this->GetOutput();
  typename Superclass::InputImageConstPointer  inputPtr  = this->GetInput();
  if ( !outputPtr || !inputPtr)
    {
    return;
    }

  // Minipipeline is :
  //   Input -> Crop -> Resample_if_too_anisotropic -> Segment

  m_CropFilter->SetInput(inputPtr);
  m_CropFilter->SetRegionOfInterest(m_RegionOfInterest);

  // Compute the spacing after isotropic resampling.
  double minSpacing = NumericTraits< double >::max();
  for (int i = 0; i < ImageDimension; i++)
    {
    minSpacing = (minSpacing > inputPtr->GetSpacing()[i] ?
                  inputPtr->GetSpacing()[i] : minSpacing);
    }

  // Try and reduce the anisotropy.
  SpacingType outputSpacing = inputPtr->GetSpacing();
  for (int i = 0; i < ImageDimension; i++)
    {
    if (outputSpacing[i]/minSpacing > m_AnisotropyThreshold && m_ResampleThickSliceData)
      {
      outputSpacing[i] = minSpacing * m_AnisotropyThreshold;
      }
    }

  if (m_ResampleThickSliceData)
    {
    m_IsotropicResampler->SetInput( m_CropFilter->GetOutput() );
    m_IsotropicResampler->SetOutputSpacing( outputSpacing );
    m_IsotropicResampler->GenerateOutputInformation();
    outputPtr->CopyInformation( m_IsotropicResampler->GetOutput() );
    }
  else
    {
    outputPtr->CopyInformation( m_CropFilter->GetOutput() );
    }
}


template< class TInputImage, class TOutputImage >
void
LesionSegmentationImageFilter8< TInputImage, TOutputImage >
::GenerateData()
{
  m_SigmoidFeatureGenerator->SetBeta( m_SigmoidBeta );
  m_SegmentationModule->SetDistanceFromSeeds(m_FastMarchingDistanceFromSeeds);
  m_SegmentationModule->SetStoppingValue(m_FastMarchingStoppingTime);

  // Allocate the output
  this->GetOutput()->SetBufferedRegion( this->GetOutput()->GetRequestedRegion() );
  this->GetOutput()->Allocate();

  // Get the input image
  typename InputImageType::ConstPointer  input  = this->GetInput();

  // Crop and perform thin slice resampling (done only if necessary)
  m_CropFilter->Update();

  typename InputImageType::Pointer inputImage = nullptr;
  if (m_ResampleThickSliceData)
    {
    m_IsotropicResampler->Update();
    inputImage = this->m_IsotropicResampler->GetOutput();
    }
  else
    {
    inputImage = m_CropFilter->GetOutput();
    }

  // Convert the output of resampling (or cropping based on
  // m_ResampleThickSliceData) to a spatial object that can be fed into
  // the lesion segmentation method

  inputImage->DisconnectPipeline();
  m_InputSpatialObject->SetImage(inputImage);

  // Sigma for the canny is the max spacing of the original input (before
  // resampling)

  if (m_UserSpecifiedSigmas == false)
    {
    double maxSpacing = NumericTraits< double >::min();
    for (int i = 0; i < ImageDimension; i++)
      {
      maxSpacing = (maxSpacing < input->GetSpacing()[i] ?
                      input->GetSpacing()[i] : maxSpacing);
      }
    m_CannyEdgesFeatureGenerator->SetSigma( maxSpacing );
    }

  // Seeds

  typename SeedSpatialObjectType::Pointer seedSpatialObject =
    SeedSpatialObjectType::New();
  seedSpatialObject->SetPoints(m_Seeds);
  m_LesionSegmentationMethod->SetInitialSegmentation(seedSpatialObject);

  // Do the actual segmentation.
  m_LesionSegmentationMethod->Update();

  // Graft the output.
  typename SpatialObjectType::Pointer segmentation =
    const_cast< SpatialObjectType * >(m_SegmentationModule->GetOutput());
  typename OutputSpatialObjectType::Pointer outputObject =
    dynamic_cast< OutputSpatialObjectType * >( segmentation.GetPointer() );
  typename OutputImageType::Pointer outputImage =
    const_cast< OutputImageType * >(outputObject->GetImage());
  outputImage->DisconnectPipeline();
  this->GraftOutput(outputImage);

  /* // DEBUGGING CODE
  typedef ImageFileWriter< OutputImageType > WriterType;
  typename WriterType::Pointer writer = WriterType::New();
  writer->SetFileName("output.mha");
  writer->SetInput(outputImage);
  writer->UseCompressionOn();
  writer->Write();*/
}


template <class TInputImage, class TOutputImage>
void LesionSegmentationImageFilter8< TInputImage,TOutputImage >
::ProgressUpdate( Object * caller,
                  const EventObject & e )
{
  if( typeid( itk::ProgressEvent ) == typeid( e ) )
    {
    if (dynamic_cast< CropFilterType * >(caller))
      {
      this->m_StatusMessage = "Cropping data..";
      this->UpdateProgress( m_CropFilter->GetProgress() );
      }

    if (dynamic_cast< IsotropicResamplerType * >(caller))
      {
      this->m_StatusMessage = "Isotropic resampling of data using BSpline interpolation..";
      this->UpdateProgress( m_IsotropicResampler->GetProgress() );
      }

    else if (dynamic_cast< LungWallGeneratorType * >(caller))
      {
      // Given its iterative nature.. a cranky heuristic here.
      this->m_StatusMessage = "Generating lung wall feature by front propagation..";
      this->UpdateProgress( ((double)(((int)(
        m_LungWallFeatureGenerator->GetProgress()*500))%100))/100.0 );
      }

    else if (dynamic_cast< SigmoidFeatureGeneratorType * >(caller))
      {
      this->m_StatusMessage = "Generating intensity feature..";
      this->UpdateProgress( m_SigmoidFeatureGenerator->GetProgress() );
      }

    else if (dynamic_cast< CannyEdgesFeatureGeneratorType * >(caller))
      {
      m_StatusMessage = "Generating canny edge feature..";
      this->UpdateProgress( m_CannyEdgesFeatureGenerator->GetProgress());
      }

    else if (dynamic_cast< VesselnessGeneratorType * >(caller))
      {
      m_StatusMessage = "Generating vesselness feature (Sato et al.)..";
      this->UpdateProgress( m_LungWallFeatureGenerator->GetProgress() );
      }

    else if (dynamic_cast< SegmentationModuleType * >(caller))
      {
      m_StatusMessage = "Segmenting using level sets..";
      this->UpdateProgress( m_SegmentationModule->GetProgress() );
      }
    }
}

template <class TInputImage, class TOutputImage>
void LesionSegmentationImageFilter8< TInputImage,TOutputImage >
::SetAbortGenerateData( bool abort )
{
  this->Superclass::SetAbortGenerateData(abort);
  this->m_CropFilter->SetAbortGenerateData(abort);
  this->m_IsotropicResampler->SetAbortGenerateData(abort);
  this->m_LesionSegmentationMethod->SetAbortGenerateData(abort);
}

template <class TInputImage, class TOutputImage>
void LesionSegmentationImageFilter8< TInputImage,TOutputImage >
::SetUseVesselEnhancingDiffusion( bool b )
{
  this->m_VesselnessFeatureGenerator->SetUseVesselEnhancingDiffusion(b);
}

template <class TInputImage, class TOutputImage>
void
LesionSegmentationImageFilter8<TInputImage,TOutputImage>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);
}

}//end of itk namespace

#endif
