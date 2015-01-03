/** \file
 *  \ingroup commandLineTools 
 *  \details This program reads a label map and splits the left and
 *  right lungs so that they are uniquely labeled. If the input is
 *  already split, the output will be identical to the input. 
 * 
 *  $Date: 2012-04-24 17:06:09 -0700 (Tue, 24 Apr 2012) $
 *  $Revision: 93 $
 *  $Author: jross $
 *
 */

#ifndef DOXYGEN_SHOULD_SKIP_THIS

#include "cipChestConventions.h"
#include "cipHelper.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkCIPSplitLeftLungRightLungImageFilter.h"
#include "SplitLeftLungRightLungCLP.h"

namespace
{
  typedef itk::CIPSplitLeftLungRightLungImageFilter< cip::CTType >  SplitterType;
}

int main( int argc, char *argv[] )
{  
  PARSE_ARGS;
  
  std::cout << "Reading label map..." << std::endl;
  cip::LabelMapReaderType::Pointer lmReader = cip::LabelMapReaderType::New();
    lmReader->SetFileName( lmFileName );
  try
    {
    lmReader->Update();
    }
  catch ( itk::ExceptionObject &excp )
    {
    std::cerr << "Exception caught reading lung label map:";
    std::cerr << excp << std::endl;

    return cip::LABELMAPREADFAILURE;
    }

  std::cout << "Reading CT..." << std::endl;
  cip::CTReaderType::Pointer ctReader = cip::CTReaderType::New();
    ctReader->SetFileName( ctFileName );
  try
    {
    ctReader->Update();
    }
  catch ( itk::ExceptionObject &excp )
    {
    std::cerr << "Exception caught reading CT:";
    std::cerr << excp << std::endl;

    return cip::NRRDREADFAILURE;
    }

   std::cout << "Splitting..." << std::endl;
   SplitterType::Pointer splitter = SplitterType::New();
     splitter->SetInput( ctReader->GetOutput() );
     splitter->SetLungLabelMap( lmReader->GetOutput() );
     splitter->SetExponentialCoefficient( exponentialCoefficient );
     splitter->SetExponentialTimeConstant( exponentialTimeConstant );
     splitter->SetLeftRightLungSplitRadius( splitRadius );
   try
     {
     splitter->Update();
     }
   catch ( itk::ExceptionObject &excp )
     {
     std::cerr << "Exception caught splitting:";
     std::cerr << excp << std::endl;
     }

   std::cout << "Writing split lung label map..." << std::endl;
   cip::LabelMapWriterType::Pointer writer = cip::LabelMapWriterType::New();
     writer->SetInput( splitter->GetOutput() );
     writer->SetFileName( outFileName );
     writer->UseCompressionOn();
   try
     {
     writer->Update();
     }
   catch ( itk::ExceptionObject &excp )
     {
     std::cerr << "Exception caught while writing lung label map:";
     std::cerr << excp << std::endl;
     return cip::LABELMAPWRITEFAILURE;
     }

   std::cout << "DONE." << std::endl;

  return cip::EXITSUCCESS;
}

#endif
