/** \file
 *  \ingroup commandLineTools 
 *  \details This simple program reads and writes images, either 
 *  label maps or CT images. It is useful for renaming (obviating 
 *  the need to need to manually modify headers)
 *
 *  USAGE: 
 *
 *  ReadWriteImageData  [--oct <string>] [--ict <string>] [--ol <string>]
 *                      [--il <string>] [--] [--version] [-h]
 *
 *  Where: 
 *
 *   --oct <string>
 *     Output CT file name
 *
 *   --ict <string>
 *     Input CT file name
 *
 *   --ol <string>
 *     Output label map file name
 *
 *   --il <string>
 *     Input label map file name
 *
 *   --,  --ignore_rest
 *     Ignores the rest of the labeled arguments following this flag.
 *
 *   --version
 *     Displays version information and exits.
 *
 *   -h,  --help
 *     Displays usage information and exits.
 *
 *  $Date$
 *  $Revision$
 *  $Author$
 *
 */

#ifndef DOXYGEN_SHOULD_SKIP_THIS

#include <tclap/CmdLine.h>
#include "cipConventions.h"

int main( int argc, char *argv[] )
{
  //
  // Begin by defining the arguments to be passed
  //
  std::string inLabelMapFileName  = "NA";
  std::string outLabelMapFileName = "NA";
  std::string inCTFileName        = "NA";
  std::string outCTFileName       = "NA";

  //
  // Input argument descriptions for user help
  //
  std::string programDesc = "This simple program reads and writes images, either label maps \
or CT images. It is useful for renaming (obviating the need to need to manually modify headers)";

  std::string inLabelMapFileNameDesc  = "Input label map file name";
  std::string outLabelMapFileNameDesc = "Output label map file name";
  std::string inCTFileNameDesc        = "Input CT file name";
  std::string outCTFileNameDesc       = "Output CT file name";

  //
  // Parse the input arguments
  //
  try
    {
    TCLAP::CmdLine cl( programDesc, ' ', "$Revision$" );

    TCLAP::ValueArg<std::string> inLabelMapFileNameArg ( "", "il", inLabelMapFileNameDesc, false, inLabelMapFileName, "string", cl );
    TCLAP::ValueArg<std::string> outLabelMapFileNameArg ( "", "ol", outLabelMapFileNameDesc, false, outLabelMapFileName, "string", cl );
    TCLAP::ValueArg<std::string> inCTFileNameArg ( "", "ict", inCTFileNameDesc, false, inCTFileName, "string", cl );
    TCLAP::ValueArg<std::string> outCTFileNameArg ( "", "oct", outCTFileNameDesc, false, outCTFileName, "string", cl );

    cl.parse( argc, argv );

    inLabelMapFileName  = inLabelMapFileNameArg.getValue();
    outLabelMapFileName = outLabelMapFileNameArg.getValue();
    inCTFileName        = inCTFileNameArg.getValue();
    outCTFileName       = outCTFileNameArg.getValue();
    }
  catch ( TCLAP::ArgException excp )
    {
    std::cerr << "Error: " << excp.error() << " for argument " << excp.argId() << std::endl;
    return cip::ARGUMENTPARSINGERROR;
    }

  if (inLabelMapFileName.compare("NA") != 0)
    {
    std::cout << "Reading label map..." << std::endl;
    LabelMapReaderType::Pointer labelMapReader = LabelMapReaderType::New();
      labelMapReader->SetFileName(inLabelMapFileName);
    try
      {
      labelMapReader->Update();
      }
    catch ( itk::ExceptionObject &excp )
      {
      std::cerr << "Exception caught reading label map:";
      std::cerr << excp << std::endl;
      return cip::LABELMAPREADFAILURE;
      }

    if (outLabelMapFileName.compare("NA") != 0)
      {
      std::cout << "Writing label map..." << std::endl;
      LabelMapWriterType::Pointer labelMapWriter = LabelMapWriterType::New();
        labelMapWriter->SetFileName(outLabelMapFileName);
	labelMapWriter->SetInput(labelMapReader->GetOutput());
	labelMapWriter->UseCompressionOn();
      try
	{
	labelMapWriter->Update();
	}
      catch ( itk::ExceptionObject &excp )
	{
	std::cerr << "Exception caught writing label map:";
	std::cerr << excp << std::endl;
	return cip::LABELMAPWRITEFAILURE;
	}
      }
    }
  
  if (inCTFileName.compare("NA") != 0)
    {
    std::cout << "Reading CT..." << std::endl;
    CTReaderType::Pointer ctReader = CTReaderType::New();
      ctReader->SetFileName(inCTFileName);
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

    if (outCTFileName.compare("NA") != 0)
      {
      std::cout << "Writing CT..." << std::endl;
      CTWriterType::Pointer ctWriter = CTWriterType::New();
        ctWriter->SetFileName(outCTFileName);
	ctWriter->SetInput(ctReader->GetOutput());
	ctWriter->UseCompressionOn();
      try
	{
	ctWriter->Update();
	}
      catch ( itk::ExceptionObject &excp )
	{
	std::cerr << "Exception caught writing CT:";
	std::cerr << excp << std::endl;
	return cip::NRRDWRITEFAILURE;
	}
      }
    }

  std::cout << "DONE." << std::endl;

  return cip::EXITSUCCESS;
}

#endif
