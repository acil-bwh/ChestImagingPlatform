/** \file
 *  \ingroup commandLineTools 
 *  \details This program takes a CT volume and a Lung label map and
 *  crops the input volume and/or label map to the specified
 *  region/type
 *
 */

#include "cipChestConventions.h"
#include "cipHelper.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include <fstream>
#include "itkImageRegionIteratorWithIndex.h"
#include "itkImageRegionIterator.h"
#include "itkRegionOfInterestImageFilter.h"
#include "itkMultiScaleGaussianEnhancementImageFilter.h"
#include "itkFrangiVesselnessFunctor.h"
#include "itkModifiedKrissianVesselnessFunctor.h"
#include "itkStrainEnergyVesselnessFunctor.h"
#include "itkStrainEnergySheetnessFunctor.h"
#include "itkFrangiSheetnessFunctor.h"
#include "itkDescoteauxSheetnessFunctor.h"
#include "itkFrangiXiaoSheetnessFunctor.h"
#include "itkDescoteauxXiaoSheetnessFunctor.h"
#include "ComputeFeatureStrengthCLP.h"

namespace
{
  typedef itk::ImageFileReader< cip::CTType >  CTFileReaderType;

  cip::CTType::Pointer ReadCTFromFile( std::string );    
  cip::CTType::Pointer ReadCTFromFile( std::string fileName )
  {
    CTFileReaderType::Pointer reader = CTFileReaderType::New();
      reader->SetFileName( fileName );
    try
      {
      reader->Update();
      }
    catch ( itk::ExceptionObject &excp )
      {
      std::cerr << "Exception caught reading CT image:";
      std::cerr << excp << std::endl;
      return nullptr;
      }
    
    return reader->GetOutput();
  }  
} //end namespace

typedef double OutputPixelType;
typedef itk::Image<OutputPixelType,3>          OutputImageType;
typedef itk::ImageFileWriter<OutputImageType>  WriterType;
typedef cip::CTType                            InputImageType;

typedef itk::MultiScaleGaussianEnhancementImageFilter< InputImageType, OutputImageType >  MultiScaleFilterType;
typedef MultiScaleFilterType::GradientMagnitudePixelType                                  GradientMagnitudePixelType;
typedef MultiScaleFilterType::EigenValueArrayType                                         EigenValueArrayType;

/** Supported functors. */
typedef itk::Functor::FrangiVesselnessFunctor< EigenValueArrayType, OutputPixelType >                                    FrangiVesselnessFunctorType;
typedef itk::Functor::FrangiSheetnessFunctor< EigenValueArrayType, OutputPixelType >                                     FrangiSheetnessFunctorType;
typedef itk::Functor::DescoteauxSheetnessFunctor< EigenValueArrayType, OutputPixelType >                                 DescoteauxSheetnessFunctorType;
typedef itk::Functor::ModifiedKrissianVesselnessFunctor< EigenValueArrayType, OutputPixelType >                          ModifiedKrissianVesselnessFunctorType;
typedef itk::Functor::StrainEnergyVesselnessFunctor< GradientMagnitudePixelType, EigenValueArrayType, OutputPixelType >  StrainEnergyVesselnessFunctorType;
typedef itk::Functor::StrainEnergySheetnessFunctor< GradientMagnitudePixelType, EigenValueArrayType, OutputPixelType >   StrainEnergySheetnessFunctorType;
typedef itk::Functor::FrangiXiaoSheetnessFunctor< GradientMagnitudePixelType, EigenValueArrayType, OutputPixelType >     FrangiXiaoSheetnessFunctorType;
typedef itk::Functor::DescoteauxXiaoSheetnessFunctor< GradientMagnitudePixelType, EigenValueArrayType, OutputPixelType > DescoteauxXiaoSheetnessFunctorType;

int main( int argc, char *argv[] )
{
  PARSE_ARGS;
  
  // Parse all arguments
  if (ctFileName.length() == 0 )
    {
      std::cerr << "ERROR: No CT image specified" << std::endl;
      return cip::EXITFAILURE;
    }
  
  if (strengthFileName.length() == 0 )
    {
      std::cerr << "ERROR: No output image specified" << std::endl;
      return cip::EXITFAILURE;
    }
  
  /** Sanity checks. */
  if ( ( gaussianStd.size() != 1 && gaussianStd.size() != 3 ) )
    {
      std::cerr << "ERROR: You should specify 1 or 3 values for \"-std\"." << std::endl;
      return cip::EXITFAILURE;
    }
  
  if ( ( sigmaStepMethod != 0 && sigmaStepMethod != 1 ) )
    {
      std::cerr << "ERROR: \"-ssm\" should be one of {0, 1}." << std::endl;
      return EXIT_FAILURE;
    }
  
  /** Get the range of sigma values. */
  double sigmaMinimum = gaussianStd[ 0 ];
  double sigmaMaximum = gaussianStd[ 0 ];
  unsigned int numberOfSigmaSteps = 1;
  if ( gaussianStd.size() == 3 )
    {
      sigmaMaximum = gaussianStd[ 1 ];
      numberOfSigmaSteps = static_cast<unsigned int>( gaussianStd[ 2 ] );
    }
  
  bool generateScalesOutput;
    
  if (scaleFileName.length() == 0 )
    {
      generateScalesOutput = false;
    }
  else
    {
      generateScalesOutput = true;
    }
    
    
  // Set threads
  unsigned int maxThreads = itk::MultiThreaderBase::GetGlobalDefaultNumberOfThreads();
  
  if (threads == 0)
    {
      threads = maxThreads;
    }
  
  itk::MultiThreaderBase::SetGlobalMaximumNumberOfThreads( maxThreads );
  
    
  // Read the input image now that filter is good to go
  cip::CTType::Pointer ctImage;

  std::cout << "Reading CT from file..." << std::endl;
  ctImage = ReadCTFromFile( ctFileName );

  if (ctImage.GetPointer() == nullptr)
    {
        return cip::NRRDREADFAILURE;
    }

  std::cout<<sigmaMinimum<<" "<<sigmaMaximum<<" "<<numberOfSigmaSteps<<std::endl;
    
  // Create multi-scale filter. */
  MultiScaleFilterType::Pointer multiScaleFilter = MultiScaleFilterType::New();
    
  // Filter set up before execution
  multiScaleFilter->SetSigmaMinimum( sigmaMinimum );
  multiScaleFilter->SetSigmaMaximum( sigmaMaximum );
  multiScaleFilter->SetNumberOfSigmaSteps( numberOfSigmaSteps );
  multiScaleFilter->SetNonNegativeHessianBasedMeasure( true );
  multiScaleFilter->SetGenerateScalesOutput( generateScalesOutput );
  multiScaleFilter->SetSigmaStepMethod( sigmaStepMethod );
  multiScaleFilter->SetRescale( !rescaleOff );
  multiScaleFilter->SetInput( ctImage );
   

  // Create Functor function based on selected method and connect to filter
  if ( method == "Frangi" )
    {
      if ( feature == "RidgeLine" )
	{
	  FrangiVesselnessFunctorType::Pointer functor = FrangiVesselnessFunctorType::New();
	    functor->SetAlpha( alpha );
	    functor->SetBeta( beta );
	    functor->SetC( C );
	    functor->SetBrightObject( true );

	  multiScaleFilter->SetUnaryFunctor( functor );
	}
      else if (feature == "ValleyLine" )
	{
	  FrangiVesselnessFunctorType::Pointer functor = FrangiVesselnessFunctorType::New();
	    functor->SetAlpha( alpha );
	    functor->SetBeta( beta );
	    functor->SetC( C );
	    functor->SetBrightObject( false );

	  multiScaleFilter->SetUnaryFunctor( functor );
	}
      else if (feature == "RidgeSurface" )
	{
	  FrangiSheetnessFunctorType::Pointer functor = FrangiSheetnessFunctorType::New();
	    functor->SetAlpha( alpha );
	    functor->SetBeta( beta );
	    functor->SetC( C );
	    functor->SetBrightObject( true );

	  multiScaleFilter->SetUnaryFunctor( functor );
	}
      else if (feature == "ValleySurface" )
	{
	  FrangiSheetnessFunctorType::Pointer functor = FrangiSheetnessFunctorType::New();
	    functor->SetAlpha( alpha );
	    functor->SetBeta( beta );
	    functor->SetC( C );
	    functor->SetBrightObject( false );

	  multiScaleFilter->SetUnaryFunctor( functor );
	}
      else
	{
	  itkGenericExceptionMacro( << "ERROR: unknown feature type " << feature << " for method " << method << "!" );
	  return cip::EXITFAILURE;
	}
      
    }
  else if ( method == "StrainEnergy" )
    {
      if ( feature == "RidgeLine" )
	{
	  StrainEnergyVesselnessFunctorType::Pointer functor = StrainEnergyVesselnessFunctorType::New();
	    functor->SetAlpha( alphase );
	    functor->SetBeta( betase );
	    functor->SetNu( nu );
	    functor->SetKappa( kappa );
	    functor->SetBrightObject( true );
	  
	  multiScaleFilter->SetBinaryFunctor( functor );
	}
      else if ( feature == "ValleyLine" )
	{
	  StrainEnergyVesselnessFunctorType::Pointer functor = StrainEnergyVesselnessFunctorType::New();
	    functor->SetAlpha( alphase );
	    functor->SetBeta( betase );
	    functor->SetNu( nu );
	    functor->SetKappa( kappa );
	    functor->SetBrightObject( false );
	  
	  multiScaleFilter->SetBinaryFunctor( functor );
	}
      else if ( feature == "RidgeSurface" )
	{
	  StrainEnergySheetnessFunctorType::Pointer functor = StrainEnergySheetnessFunctorType::New();
	    functor->SetAlpha( alphase );
	    functor->SetBeta( betase );
	    functor->SetNu( nu );
	    functor->SetKappa( kappa );
	    functor->SetBrightObject( true );
	  
	  multiScaleFilter->SetBinaryFunctor( functor );
	}
      else if ( feature == "ValleySurface" )
	{
	  StrainEnergySheetnessFunctorType::Pointer functor = StrainEnergySheetnessFunctorType::New();
	    functor->SetAlpha( alphase );
	    functor->SetBeta( betase );
	    functor->SetNu( nu );
	    functor->SetKappa( kappa );
	    functor->SetBrightObject( false );
	  
	  multiScaleFilter->SetBinaryFunctor( functor );
	}
      else
	{
	  itkGenericExceptionMacro( << "ERROR: unknown feature type " << feature << " for method " << method << "!" );
	  return cip::EXITFAILURE;
	}
      
    }
  else if ( method == "ModifiedKrissian" )
    {
      if ( feature == "RidgeLine" )
	{
	  ModifiedKrissianVesselnessFunctorType::Pointer functor = ModifiedKrissianVesselnessFunctorType::New();
	    functor->SetBrightObject( true );
	  
	  multiScaleFilter->SetUnaryFunctor( functor );
	}
      else if ( feature == "ValleyLine" )
	{
	  ModifiedKrissianVesselnessFunctorType::Pointer functor = ModifiedKrissianVesselnessFunctorType::New();
	    functor->SetBrightObject( false );
	  
	  multiScaleFilter->SetUnaryFunctor( functor );
	}
      else
	{
	  itkGenericExceptionMacro( << "ERROR: unknown feature type " << feature << " for method " << method << "!" );
	  return cip::EXITFAILURE;
	}
    }
  else if ( method == "Descoteaux" )
    {
      if ( feature == "RidgeSurface" )
	{
	  DescoteauxSheetnessFunctorType::Pointer functor = DescoteauxSheetnessFunctorType::New();
	    functor->SetAlpha( alpha );
	    functor->SetBeta( beta );
	    functor->SetC( C );
	    functor->SetBrightObject( true );
	  
	  multiScaleFilter->SetUnaryFunctor( functor );
	}
      else if ( feature == "ValleySurface" )
	{
	  DescoteauxSheetnessFunctorType::Pointer functor = DescoteauxSheetnessFunctorType::New();
	    functor->SetAlpha( alpha );
	    functor->SetBeta( beta );
	    functor->SetC( C );
	    functor->SetBrightObject( false );
	  
	  multiScaleFilter->SetUnaryFunctor( functor );
	}
      else
	{
	  itkGenericExceptionMacro( << "ERROR: unknown feature type " << feature << " for method " << method << "!" );
	  return cip::EXITFAILURE;
	}
    }
  else if ( method == "FrangiXiao" )
    {
      if ( feature ==  "RidgeSurface" )
	{
	  FrangiXiaoSheetnessFunctorType::Pointer functor = FrangiXiaoSheetnessFunctorType::New();
	    functor->SetAlpha( alpha );
	    functor->SetBeta( beta );
	    functor->SetC( C );
	    functor->SetKappa( kappa );
	    functor->SetBrightObject( true );
	  
	  multiScaleFilter->SetBinaryFunctor( functor );
	}
      else if ( feature == "ValleySurface" )
	{
	  FrangiXiaoSheetnessFunctorType::Pointer functor = FrangiXiaoSheetnessFunctorType::New();
	    functor->SetAlpha( alpha );
	    functor->SetBeta( beta );
	    functor->SetC( C );
	    functor->SetKappa( kappa );
	    functor->SetBrightObject( false );

	  multiScaleFilter->SetBinaryFunctor( functor );
	}
      else
	{
	  itkGenericExceptionMacro( << "ERROR: unknown feature type " << feature << " for method " << method << "!" );
	  return cip::EXITFAILURE;
	}
    }
  else if ( method == "DescoteauxXiao" )
    {
      if ( feature == "RidgeSurface" )
	{
	  DescoteauxXiaoSheetnessFunctorType::Pointer functor = DescoteauxXiaoSheetnessFunctorType::New();
	    functor->SetAlpha( alpha );
	    functor->SetBeta( beta );
	    functor->SetC( C );
	    functor->SetKappa( kappa );
	    functor->SetBrightObject( true );
	  
	  multiScaleFilter->SetBinaryFunctor( functor );
	}
      else if ( feature == "ValleySurface" )
	{
	  DescoteauxXiaoSheetnessFunctorType::Pointer functor = DescoteauxXiaoSheetnessFunctorType::New();
	    functor->SetAlpha( alpha );
	    functor->SetBeta( beta );
	    functor->SetC( C );
	    functor->SetKappa( kappa );
	    functor->SetBrightObject( false );
	  
	  multiScaleFilter->SetBinaryFunctor( functor );
	}
      else
	{
	  itkGenericExceptionMacro( << "ERROR: unknown feature type " << feature << " for method " << method << "!" );
	  return cip::EXITFAILURE;
	}
    }
  else
    {
      itkGenericExceptionMacro( << "ERROR: unknown method " << method << "!" );
      return cip::EXITFAILURE;
    }
  
  try
    {
      multiScaleFilter->Update();
    }
  catch (itk::ExceptionObject &excp)
    {
      std::cerr << "Exception caught executing method";
      std::cerr << excp << std::endl;
      return cip::EXITFAILURE;
    }
  
  // Write feature strenght output
  WriterType::Pointer writer = WriterType::New();
    writer->SetInput( multiScaleFilter->GetOutput() );
    writer->SetFileName( strengthFileName );
    writer->UseCompressionOn();
  try
    {
    writer->Update();
    }
  catch (itk::ExceptionObject &excp )
    {
    std::cerr << "Exception caught writing output image";
    std::cerr << excp << std::endl;
    return cip::NRRDWRITEFAILURE;
    }
  
  // Write the maximum scale response
  if( generateScalesOutput == true )
    {
      writer->SetInput( multiScaleFilter->GetOutput( 1 ) );
      writer->SetFileName( scaleFileName );
      writer->UseCompressionOn();
    try
      {
      writer->Update();
      }
    catch (itk::ExceptionObject &excp )
      {
      std::cerr << "Exception caught writing output image";
      std::cerr << excp << std::endl;
      return cip::NRRDWRITEFAILURE;
      }      
    }
    
  std::cout<< "DONE." << std::endl;
  
  multiScaleFilter = nullptr;
  return cip::EXITSUCCESS;
}


