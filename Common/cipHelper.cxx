/**
*
*  $Date: $
*  $Revision: $
*  $Author: $
*
*  Class cipHelper
*	
*/

#include "cipHelper.h"
#include "itkResampleImageFilter.h"
#include "itkBSplineInterpolateImageFunction.h"
#include "itkNearestNeighborInterpolateImageFunction.h"

//
// Code modified from //http://www.itk.org/Wiki/ITK/Examples/ImageProcessing/Upsampling
//
cip::LabelMapType::Pointer cip::DownsampleLabelMap( short samplingAmount, cip::LabelMapType::Pointer inputLabelMap )
{
  cip::LabelMapType::Pointer outputLabelMap;

  typedef itk::IdentityTransform< double, 3 >                                        TransformType;
  typedef itk::NearestNeighborInterpolateImageFunction< cip::LabelMapType, double >  InterpolatorType;
  typedef itk::ResampleImageFilter< cip::LabelMapType, cip::LabelMapType >           ResampleType;

  //
  // Instantiate the transform, the b-spline interpolator and the resampler
  //
  TransformType::Pointer idTransform = TransformType::New();
    idTransform->SetIdentity();

  InterpolatorType::Pointer imageInterpolator = InterpolatorType::New();

  //
  // Compute and set the output spacing from the input spacing and samplingAmount 
  //     
  const cip::LabelMapType::RegionType& inputRegion = inputLabelMap->GetLargestPossibleRegion();
  const cip::LabelMapType::SizeType& inputSize = inputRegion.GetSize();

  unsigned int originalWidth  = inputSize[0];
  unsigned int originalLength = inputSize[1];
  unsigned int originalHeight = inputSize[2];
  
  unsigned int newWidth  = (unsigned int)(double(originalWidth)/double(samplingAmount));
  unsigned int newLength = (unsigned int)(double(originalLength)/double(samplingAmount));
  unsigned int newHeight = (unsigned int)(double(originalHeight)/double(samplingAmount));

  const cip::LabelMapType::SpacingType& inputSpacing = inputLabelMap->GetSpacing();
  
  double outputSpacing[3];
  outputSpacing[0] = inputSpacing[0]*(double(originalWidth)/double(newWidth));
  outputSpacing[1] = inputSpacing[1]*(double(originalLength)/double(newLength));
  outputSpacing[2] = inputSpacing[2]*(double(originalHeight)/double(newHeight));
  
  std::cout << "old dimensions are: " << originalWidth << " " << originalLength << " " << originalHeight << std::endl;
  std::cout << "spacing is: " << inputSpacing[0] << " " << inputSpacing[1] << " " << inputSpacing[2] << std::endl;

  std::cout << "new dimensions are: " << newWidth << " " << newLength << " " << newHeight << std::endl;
  std::cout << "spacing is: " << outputSpacing[0] << " " << outputSpacing[1] << " " << outputSpacing[2] << std::endl;
	
  //
  // Set the resampler with the calculated parameters and resample
  //
  itk::Size< 3 > outputSize = { {newWidth, newLength, newHeight} };

  ResampleType::Pointer resizeFilter = ResampleType::New();
    resizeFilter->SetTransform( idTransform );
    resizeFilter->SetInterpolator( imageInterpolator );
    resizeFilter->SetOutputOrigin( inputLabelMap->GetOrigin() );
    resizeFilter->SetOutputSpacing( outputSpacing );
    resizeFilter->SetSize( outputSize );
    resizeFilter->SetInput( inputLabelMap );
    resizeFilter->Update();

  //
  // Save the resampled output to the output image and return
  //
  outputLabelMap = resizeFilter->GetOutput();

  std::cout << "sucessully generaed volume" << std::endl;

  return outputLabelMap;
}

cip::LabelMapType::Pointer cip::UpsampleLabelMap( short samplingAmount, cip::LabelMapType::Pointer inputLabelMap )
{
  cip::LabelMapType::Pointer outputLabelMap;

  typedef itk::IdentityTransform<double, 3> identityTransform;

	// Code modified from //http://www.itk.org/Wiki/ITK/Examples/ImageProcessing/Upsampling

	//
	// Interpolate using a 3rd order B-pline
	//
	typedef itk::BSplineInterpolateImageFunction<LabelMapType, double, double> BSplineInterpolator;
	typedef itk::ResampleImageFilter<LabelMapType, LabelMapType> ResampleFilter;

	//
	// Instantiate the transform, the b-spline interpolator and the resampler
	//
	identityTransform::Pointer idTransform = identityTransform::New();
	idTransform->SetIdentity();

	BSplineInterpolator::Pointer imageInterpolator = BSplineInterpolator::New();
	imageInterpolator->SetSplineOrder(3);

	ResampleFilter::Pointer resizeFilter = ResampleFilter::New();
	resizeFilter->SetTransform(idTransform);
	resizeFilter->SetInterpolator(imageInterpolator);
	resizeFilter->SetOutputOrigin(inputLabelMap->GetOrigin());

	//
	// Compute and set the output spacing from the input spacing and samplingAmount 
	//     
	const LabelMapType::RegionType& inputRegion = inputLabelMap->GetLargestPossibleRegion();
	const LabelMapType::SizeType& inputSize = inputRegion.GetSize();

	unsigned int originalWidth = inputSize[0];
	unsigned int originalLength = inputSize[1];
	unsigned int originalHeight = inputSize[2];

	unsigned int newWidth = (int)((double) originalWidth * (double) samplingAmount);
	unsigned int newLength = (int)((double) originalLength * (double) samplingAmount);
	unsigned int newHeight = (int)((double) originalHeight * (double) samplingAmount);

	const LabelMapType::SpacingType& inputSpacing = inputLabelMap->GetSpacing();

	double outputSpacing[3];
	outputSpacing[0] = inputSpacing[0] * ((double) originalWidth / (double) newWidth);
	outputSpacing[1] = inputSpacing[1] * ((double) originalLength / (double) newLength);
	outputSpacing[2] = inputSpacing[2] * ((double) originalHeight / (double) newHeight);

	std::cout<<"old dimensions are: "<< originalWidth<< " "<<originalLength<< " "<<originalHeight<< " "<<std::endl;
	std::cout<<"spacing is: "<< inputSpacing[0]<< " "<<inputSpacing[1]<< " "<<inputSpacing[2]<< " "<<std::endl;

	std::cout<<"new dimensions are: "<< newWidth<< " "<<newLength<< " "<<newHeight<< " "<<std::endl;
	std::cout<<"spacing is: "<< outputSpacing[0]<< " "<<outputSpacing[1]<< " "<<outputSpacing[2]<< " "<<std::endl;
	
	//
	// Set the resampler with the calculated parameters and resample
	//
	resizeFilter->SetOutputSpacing(outputSpacing);
	itk::Size<3> outputSize = { {newWidth, newLength, newHeight} };
	resizeFilter->SetSize(outputSize);
	resizeFilter->SetInput(inputLabelMap);
	resizeFilter->Update();

	//
	// Save the resampled output to the output image and return
	//

	outputLabelMap= resizeFilter->GetOutput();

  return outputLabelMap;
}

cip::CTType::Pointer cip::UpsampleCT( short samplingAmount, cip::CTType::Pointer inputCT )
{
  cip::CTType::Pointer outputCT;

  typedef itk::IdentityTransform<double, 3> identityTransform;

	// Code modified from //http://www.itk.org/Wiki/ITK/Examples/ImageProcessing/Upsampling

	//
	// Interpolate using a 3rd order B-pline
	//
	typedef itk::BSplineInterpolateImageFunction<CTType, double, double> BSplineInterpolator;
	typedef itk::ResampleImageFilter<CTType, CTType> ResampleFilter;

	//
	// Instantiate the transform, the b-spline interpolator and the resampler
	//
	identityTransform::Pointer idTransform = identityTransform::New();
	idTransform->SetIdentity();

	BSplineInterpolator::Pointer imageInterpolator = BSplineInterpolator::New();
	imageInterpolator->SetSplineOrder(3);

	ResampleFilter::Pointer resizeFilter = ResampleFilter::New();
	resizeFilter->SetTransform(idTransform);
	resizeFilter->SetInterpolator(imageInterpolator);
	resizeFilter->SetOutputOrigin(inputCT->GetOrigin());

	//
	// Compute and set the output spacing from the input spacing and samplingAmount 
	//     
	const CTType::RegionType& inputRegion = inputCT->GetLargestPossibleRegion();
	const CTType::SizeType& inputSize = inputRegion.GetSize();

	unsigned int originalWidth = inputSize[0];
	unsigned int originalLength = inputSize[1];
	unsigned int originalHeight = inputSize[2];

	unsigned int newWidth = (int)((double) originalWidth * (double) samplingAmount);
	unsigned int newLength = (int)((double) originalLength * (double) samplingAmount);
	unsigned int newHeight = (int)((double) originalHeight * (double) samplingAmount);

	const CTType::SpacingType& inputSpacing = inputCT->GetSpacing();

	double outputSpacing[3];
	outputSpacing[0] = inputSpacing[0] * ((double) originalWidth / (double) newWidth);
	outputSpacing[1] = inputSpacing[1] * ((double) originalLength / (double) newLength);
	outputSpacing[2] = inputSpacing[2] * ((double) originalHeight / (double) newHeight);

	std::cout<<"old dimensions are: "<< originalWidth<< " "<<originalLength<< " "<<originalHeight<< " "<<std::endl;
	std::cout<<"spacing is: "<< inputSpacing[0]<< " "<<inputSpacing[1]<< " "<<inputSpacing[2]<< " "<<std::endl;

	std::cout<<"new dimensions are: "<< newWidth<< " "<<newLength<< " "<<newHeight<< " "<<std::endl;
	std::cout<<"spacing is: "<< outputSpacing[0]<< " "<<outputSpacing[1]<< " "<<outputSpacing[2]<< " "<<std::endl;
	
	//
	// Set the resampler with the calculated parameters and resample
	//
	resizeFilter->SetOutputSpacing(outputSpacing);
	itk::Size<3> outputSize = { {newWidth, newLength, newHeight} };
	resizeFilter->SetSize(outputSize);
	resizeFilter->SetInput(inputCT);
	resizeFilter->Update();
	//
	// Save the resampled output to the output image and return
	//
	std::cout<<"sucessully generaed volume"<<std::endl;
	outputCT= resizeFilter->GetOutput();

  return outputCT;
}

cip::CTType::Pointer cip::DownsampleCT( short samplingAmount, cip::CTType::Pointer inputCT )
{

  cip::CTType::Pointer outputCT;

	typedef itk::IdentityTransform<double, 3> identityTransform;

	// Code modified from //http://www.itk.org/Wiki/ITK/Examples/ImageProcessing/Upsampling

	//
	// Interpolate using a 3rd order B-pline
	//
	typedef itk::BSplineInterpolateImageFunction<CTType, double, double> BSplineInterpolator;
	typedef itk::ResampleImageFilter<CTType, CTType> ResampleFilter;

	//
	// Instantiate the transform, the b-spline interpolator and the resampler
	//
	identityTransform::Pointer idTransform = identityTransform::New();
	idTransform->SetIdentity();

	BSplineInterpolator::Pointer imageInterpolator = BSplineInterpolator::New();
	imageInterpolator->SetSplineOrder(3);

	ResampleFilter::Pointer resizeFilter = ResampleFilter::New();
	resizeFilter->SetTransform(idTransform);
	resizeFilter->SetInterpolator(imageInterpolator);
	resizeFilter->SetOutputOrigin(inputCT->GetOrigin());

	//
	// Compute and set the output spacing from the input spacing and samplingAmount 
	//     
	const CTType::RegionType& inputRegion = inputCT->GetLargestPossibleRegion();
	const CTType::SizeType& inputSize = inputRegion.GetSize();

	unsigned int originalWidth = inputSize[0];
	unsigned int originalLength = inputSize[1];
	unsigned int originalHeight = inputSize[2];

	unsigned int newWidth = (int)((double) originalWidth / (double) samplingAmount);
	unsigned int newLength = (int)((double) originalLength / (double) samplingAmount);
	unsigned int newHeight = (int)((double) originalHeight / (double) samplingAmount);

	const CTType::SpacingType& inputSpacing = inputCT->GetSpacing();

	double outputSpacing[3];
	outputSpacing[0] = inputSpacing[0] * ((double) originalWidth / (double) newWidth);
	outputSpacing[1] = inputSpacing[1] * ((double) originalLength / (double) newLength);
	outputSpacing[2] = inputSpacing[2] * ((double) originalHeight / (double) newHeight);

	std::cout<<"old dimensions are: "<< originalWidth<< " "<<originalLength<< " "<<originalHeight<< " "<<std::endl;
	std::cout<<"spacing is: "<< inputSpacing[0]<< " "<<inputSpacing[1]<< " "<<inputSpacing[2]<< " "<<std::endl;

	std::cout<<"new dimensions are: "<< newWidth<< " "<<newLength<< " "<<newHeight<< " "<<std::endl;
	std::cout<<"spacing is: "<< outputSpacing[0]<< " "<<outputSpacing[1]<< " "<<outputSpacing[2]<< " "<<std::endl;
	
	//
	// Set the resampler with the calculated parameters and resample
	//
	resizeFilter->SetOutputSpacing(outputSpacing);
	itk::Size<3> outputSize = { {newWidth, newLength, newHeight} };
	resizeFilter->SetSize(outputSize);
	resizeFilter->SetInput(inputCT);
	resizeFilter->Update();

	//
	// Save the resampled output to the output image and return
	//
	outputCT= resizeFilter->GetOutput();
	std::cout<<"returning volume"<<std::endl;
	return outputCT;

}

