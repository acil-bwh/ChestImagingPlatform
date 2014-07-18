/** \file
 *  \ingroup This program generates a  one-pixel-wide skeleton from a 3D binary mask. 
 *   The input is supposed to be binary. All the non-zero voxels are set to 1.
 * 
 * USAGE: 

 */


#include "GenerateBinaryThinning3DCLP.h"

#include "cipChestConventions.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"

#include "itkConnectedThresholdImageFilter.h"
#include "itkBinaryThinningImageFilter3D.h"

#include <iostream>
#include <stdlib.h>   // for atoi()


typedef itk::Image< unsigned short, 3 >                           ImageType;
typedef itk::ImageFileReader< ImageType >                   ReaderType;
typedef itk::ImageFileWriter< ImageType >                   WriterType;

int main( int argc, char *argv[] )
{

  // Parse the input arguments
  PARSE_ARGS;
  
  
  
  ReaderType::Pointer reader= ReaderType::New();
  reader->SetFileName(inputMask);
  try {
    reader->Update();
  } catch (itk::ExceptionObject &exp) {
    std::cerr << "Exception caught reading label map:";
    std::cerr<< exp << std::endl;
    return cip::LABELMAPREADFAILURE;
  }
  
  // Define the thinning filter
  typedef itk::BinaryThinningImageFilter3D< ImageType, ImageType > ThinningFilterType;
  ThinningFilterType::Pointer thinningFilter = ThinningFilterType::New();
  thinningFilter->SetInput( reader->GetOutput() );
  thinningFilter->Update();
  
  // output to file
  typedef itk::ImageFileWriter< ImageType > WriterType;
  WriterType::Pointer writer = WriterType::New();
  writer->SetInput( thinningFilter->GetOutput() );
  writer->SetFileName( outputMask );
  
  try
  {
    writer->Update();
  }
  catch ( itk::ExceptionObject &excp )
  {
    std::cerr << "Exception caught writing label map:";
    std::cerr << excp << std::endl;
    
    return cip::LABELMAPWRITEFAILURE;
  }
  
  std::cout << "DONE." << std::endl;
  
  return cip::EXITSUCCESS;
  
}
