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

#include <tclap/CmdLine.h>
#include "cipConventions.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkCIPSplitLeftLungRightLungImageFilter.h"
 
 //typedef itk::Image< short, 3 >                                    cip::CTType;
 typedef itk::Image< unsigned short, 3 >                           LabelMapType;
 typedef itk::ImageFileReader< LabelMapType >                      ReaderType;
 typedef itk::ImageFileWriter< LabelMapType >                      WriterType;
typedef itk::CIPSplitLeftLungRightLungImageFilter< cip::CTType >   SplitterType;

int main( int argc, char *argv[] )
{
   //
   // Begin by defining the arguments to be passed
   //
   std::string inLabelMapFileName  = "NA";
   std::string outLabelMapFileName = "NA";
   double exponentialTimeConstant  = -700;
   double exponentialCoefficient   = 200.0;

   //
   // Descriptions for user help
   //
   std::string programDescription = "This program reads a label map and splits the left and\
 right lungs so that they are uniquely labeled. If the input is\
 already split, the output will be identical to the input.";

   std::string inLabelMapFileNameDescription = "Input label map file name";
   std::string outLabelMapFileNameDescription = "Output label map file name";
   std::string exponentialTimeConstantDescription = "Exponential time constant (-700 by default)";
   std::string exponentialCoefficientDescription = "Exponential coefficient (200 by default)";

   //
   // Parse the input arguments
   //
   try
     {
     TCLAP::CmdLine cl( programDescription, ' ', "$Revision: 93 $" );

     TCLAP::ValueArg<std::string> inLabelMapFileNameArg( "i", "in", inLabelMapFileNameDescription, true, inLabelMapFileName, "string", cl );
     TCLAP::ValueArg<std::string> outLabelMapFileNameArg( "o", "out", outLabelMapFileNameDescription, true, outLabelMapFileName, "string", cl );
     TCLAP::ValueArg<double> exponentialTimeConstantArg( "t", "timeConst", exponentialTimeConstantDescription, false, exponentialTimeConstant, "double", cl );
     TCLAP::ValueArg<double> exponentialCoefficientArg( "c", "coefficient", exponentialCoefficientDescription, false, exponentialCoefficient, "double", cl );

     cl.parse( argc, argv );

     inLabelMapFileName      = inLabelMapFileNameArg.getValue();
     outLabelMapFileName     = outLabelMapFileNameArg.getValue();
     exponentialTimeConstant = exponentialTimeConstantArg.getValue();
     exponentialCoefficient  = exponentialCoefficientArg.getValue();
     }
   catch ( TCLAP::ArgException excp )
     {
     std::cerr << "Error: " << excp.error() << " for argument " << excp.argId() << std::endl;
     return cip::ARGUMENTPARSINGERROR;
     }

   std::cout << "Reading label map..." << std::endl;
   ReaderType::Pointer labelMapReader = ReaderType::New();
     labelMapReader->SetFileName( inLabelMapFileName );
   try
     {
     labelMapReader->Update();
     }
   catch ( itk::ExceptionObject &excp )
     {
     std::cerr << "Exception caught reading lung label map:";
     std::cerr << excp << std::endl;

     return cip::LABELMAPREADFAILURE;
     }

 //   std::cout << "Splitting..." << std::endl;
 //   SplitterType::Pointer splitter = SplitterType::New();
 //     splitter->SetInput( reader->GetOutput() );
 //     splitter->SetExponentialCoefficient( exponentialCoefficient );
 //     splitter->SetExponentialTimeConstant( exponentialTimeConstant );
 //     splitter->SetLeftRightLungSplitRadius( leftRightSplitRadius );
 //   if ( aggressiveLeftRightSplitting == 1 )
 //     {
 //     splitter->SetAggressiveLeftRightSplitter( true );
 //     }
 //   try
 //     {
 //     splitter->Update();
 //     }
 //   catch ( itk::ExceptionObject &excp )
 //     {
 //     std::cerr << "Exception caught splitting:";
 //     std::cerr << excp << std::endl;
 //     }

 //   std::cout << "Writing split lung label map..." << std::endl;
 //   WriterType::Pointer writer = WriterType::New();
 //     writer->SetInput( splitter->GetOutput() );
 //     writer->SetFileName( outputFileName );
 //     writer->UseCompressionOn();
 //   try
 //     {
 //     writer->Update();
 //     }
 //   catch ( itk::ExceptionObject &excp )
 //     {
 //     std::cerr << "Exception caught while writing lung label map:";
 //     std::cerr << excp << std::endl;
 //     return cip::LABELMAPWRITEFAILURE;
 //     }

   std::cout << "DONE." << std::endl;

  return cip::EXITSUCCESS;
}

#endif
