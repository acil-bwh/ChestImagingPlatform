#include "itkDiscreteGaussianImageFilter.h"

#include "cipHelper.h"

#include "ExampleCLICLP.h"

int main( int argc, char *argv[] )
{
  PARSE_ARGS;

  // Read the CT image
  cip::CTReaderType::Pointer reader = cip::CTReaderType::New();
  reader->SetFileName( inputFileName );   // CLI parameter
  try
  {
    reader->Update();
  }
  catch ( itk::ExceptionObject &excp ) {
    std::cerr << "Exception caught reading CT image:";
    std::cerr << excp << std::endl;
    return cip::EXITFAILURE;
  }

  // Apply the Gaussian filter
  typedef itk::DiscreteGaussianImageFilter<cip::CTType, cip::CTType>  FilterType;
  FilterType::Pointer filter = FilterType::New();
  filter->SetInput(reader->GetOutput());
  filter->SetVariance( gaussianVariance );    // CLI parameter
  filter->SetMaximumKernelWidth( maxKernelWidth );  // CLI parameter

  // Write the result
  cip::CTWriterType::Pointer writer = cip::CTWriterType::New();
  writer->SetFileName(outputFileName);    // CLI parameter
  writer->SetInput(filter->GetOutput());
  writer->Update();

  std::cout << "File created: " << outputFileName << std::endl;

  return cip::EXITSUCCESS;
  // Something failed in the command

}
